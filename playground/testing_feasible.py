import jax
import jax.numpy as jnp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

import time
def batched_jraph_iteration(graphs):
    nodes = graphs.nodes
    n_node = graphs.n_node
    n_graph = graphs.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    sum_n_node = graphs.nodes.shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
    cum_sum = jnp.concatenate([jnp.array([0]),jax.lax.cumsum(n_node)[:-1]], axis = 0)
    cum_max_sum = jax.lax.cumsum(n_node)

    input_tuple = (graphs,0)
    max_val = int(jnp.max(graphs.n_node))

    nodes = nodes.at[cum_sum, jnp.zeros_like(cum_sum)].set(jnp.ones_like(cum_sum))

    ps = jnp.array(np.random.uniform(0, 1, (graphs.nodes.shape[0], 1)))
    X_0 = 1*(ps > 0.5)
    print("run test CE")
    best_X0 = ConditionalExpectation_test(graphs, X_0)

    batch_size = 10
    ps = np.random.uniform(0, 1, (batch_size, graphs.nodes.shape[0], 1))
    X_0 = 1*(ps > 0.5)

    #vmapped_CE = jax.vmap(lambda a,b,c,d: ConditionalExpectation_v2(a,b,c,d, max_steps= max_val), in_axes=(None, 0, None, None))
    vmapped_CE = jax.vmap(lambda a,b: ConditionalExpectation_v2(a,b), in_axes=(None, 0))
    vmapped_calc_energy = jax.vmap(calc_energy, in_axes=(None, 0))

    ### TODO find out whether violations can occur
    reps = 100
    for rep in range(reps):
        ### TODO vmap this
        start_time = time.time()
        print("run test CE", rep)
        (_, best_X_0, _, _, _, _) = vmapped_CE(graphs, X_0)
        end_time = time.time()
        print(best_X_0.shape)
        print("time", end_time - start_time )
    #best_ps = ps
    Energy, HB_per_node = vmapped_calc_energy(graphs, best_X_0)
    print("final Energies", Energy[0])
    print(jnp.sum(jnp.abs(HB_per_node)))
    # for i in range(graphs.nodes.shape[0]):
    # 	print(i, best_ps[0,i], Energy_per_node[0,i], HB_per_node[0,i])
    # print("here")
    pass

from functools import partial

@jax.jit
def calc_energy(H_graph, bins, A = 1., B = 1.1):
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = H_graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

    raveled_bins = bins
    Energy_messages = (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])

    # print("Energy_per_graph", Energy.shape)
    HA_per_node = - A * raveled_bins
    HB_per_node = B * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_nodes)
    Energy = jax.ops.segment_sum(HA_per_node + HB_per_node, node_graph_idx, n_graph)
    return Energy, HB_per_node

@jax.jit
def ConditionalExpectation_v2(graphs, X_0):
    n_node = graphs.n_node
    cum_sum = jnp.concatenate([jnp.array([0]),jax.lax.cumsum(n_node)[:-1]], axis = 0)
    cum_max_sum = jax.lax.cumsum(n_node)
    max_steps = jnp.max(graphs.n_node)
    Hb = 1.

    def cond(arg):
        step, _ , Hb, cum_sum, cum_max_sum, graphs = arg
        return (Hb > 0.)

    @jax.jit
    def body(arg):
        step, X_0, _, cum_sum, cum_max_sum, graphs = arg


        _, Hb_per_node = calc_energy(graphs, X_0)
        Hb_idxs = sort_violations(graphs, Hb_per_node)
        sorted_cum_sum_idxs = Hb_idxs[cum_sum]

        new_x_values = jnp.where(X_0[sorted_cum_sum_idxs, 0] == 1, jnp.zeros_like(sorted_cum_sum_idxs),
                                 jnp.ones_like(sorted_cum_sum_idxs))

        X_0 = X_0.at[sorted_cum_sum_idxs, 0].set(new_x_values)

        _, Hb_per_node = calc_energy(graphs, X_0)
        Hb = jnp.sum(jnp.abs(Hb_per_node))

        # cum_sum_p_1 = cum_sum + 1
        # cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
        return (step + 1, X_0, Hb, cum_sum, cum_max_sum, graphs)

    return jax.lax.while_loop(
        cond,
        body,
        (0, X_0, Hb, cum_sum, cum_max_sum, graphs)
    )

def ConditionalExpectation_test(graphs, X_0):
    n_node = graphs.n_node
    cum_sum = jnp.concatenate([jnp.array([0]),jax.lax.cumsum(n_node)[:-1]], axis = 0)
    cum_max_sum = jax.lax.cumsum(n_node)
    max_steps = jnp.max(graphs.n_node)
    Hb = 1.

    def flip_value(bins, sign):
        spins = 2*bins-1
        flipped_spins = sign*spins
        flipped_bins = (flipped_spins+1)/2
        return flipped_bins

    def cond(arg):
        step, _ , Hb, cum_sum, cum_max_sum, graphs = arg
        return (Hb > 0.)

    #@jax.jit
    def body(arg):
        step, X_0, _, cum_sum, cum_max_sum, graphs = arg


        _, Hb_per_node = calc_energy(graphs, X_0)
        Hb_idxs = sort_violations(graphs, Hb_per_node)
        sorted_cum_sum_idxs = Hb_idxs[cum_sum]

        flip_per_node = jnp.where(Hb_per_node[sorted_cum_sum_idxs,0] > 0, jnp.ones_like(sorted_cum_sum_idxs), jnp.zeros_like(sorted_cum_sum_idxs))
        flip_sign = jnp.where(flip_per_node, -jnp.ones_like(sorted_cum_sum_idxs),
                                 jnp.ones_like(sorted_cum_sum_idxs))

        flipped_X_0 = flip_value(X_0[sorted_cum_sum_idxs, 0],flip_sign)

        X_0 = X_0.at[sorted_cum_sum_idxs, 0].set(flipped_X_0)

        Energy, Hb_per_node = calc_energy(graphs, X_0)
        Hb = jnp.sum(jnp.abs(Hb_per_node))
        print("Energy", jnp.sum(Energy), jnp.sum(Hb))


        # cum_sum_p_1 = cum_sum + 1
        # cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
        return (step + 1, X_0, Hb, cum_sum, cum_max_sum, graphs)

    input_args = (0, X_0, Hb, cum_sum, cum_max_sum, graphs)
    while(cond(input_args)):
        input_args = body(input_args)

    (_, X_corrected, Hb,  cum_sum, cum_max_sum, graphs) = input_args
    Energy, HB_per_node = calc_energy(graphs, X_corrected)

    print("test Energy", Energy)
    if (jnp.sum(HB_per_node) != 0):
        raise ValueError("Test there is a violation")
    return X_corrected


import networkx as nx
def create_graph(n_nodes):
    # Define a three node graph, each node has an integer as its feature.
    gnx = nx.barabasi_albert_graph(n=n_nodes, m=4)
    edges = list(gnx.edges)

    for el in edges:
        if(el[0] == el[1]):
            raise ValueError("Self loops included")

    node_features = jnp.zeros((n_nodes,1))

    # We will construct a graph for which there is a directed edge between each node
    # and its successor. We define this with `senders` (source nodes) and `receivers`
    # (destination nodes).
    senders = jnp.array([e[0] for e in edges])
    receivers = jnp.array([e[1] for e in edges])

    # You can optionally add edge attributes.

    edges = jnp.ones_like(senders)[:, None]

    # We then save the number of nodes and the number of edges.
    # This information is used to make running GNNs over multiple graphs
    # in a GraphsTuple possible.
    n_node = jnp.array([node_features.shape[0]])
    n_edge = jnp.array([3])

    # Optionally you can add `global` information, such as a graph label.

    global_context = jnp.array([[1]])
    graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
                              edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)
    return graph


def for_loop_CE(graph_list):
    for graph in graph_list:
        nodes = graph.nodes
        ps = 0.5*jnp.ones_like(nodes)

        for node_idx in range(nodes.shape[0]):
            up_nodes = ps.at[node_idx,0].set(1)
            down_nodes = ps.at[node_idx,0].set(0)

            _, up_Energy, _ = calc_energy(graph, up_nodes)
            _, down_Energy, _ = calc_energy(graph, down_nodes)
            ps = jnp.where(up_Energy < down_Energy, up_nodes, down_nodes)

        Energy, Energy_per_node, HB_per_node = calc_energy(graph, ps)

        print("for loop Energy",Energy)
        if(jnp.sum(HB_per_node) != 0):
            raise ValueError("there is a violation")

#@jax.jit
def sort_ps(graphs, ps):

    #ps = np.random.uniform(0,1, (graphs.nodes.shape[0],1))
    nodes = graphs.nodes
    n_node = graphs.n_node
    n_graph = graphs.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

    shifted_ps = -ps + node_graph_idx[:,None]
    #sorted_ps = jax.lax.sort(shifted_ps, dimension = 0)
    p_idxs = jax.numpy.argsort(shifted_ps, axis = 0)

    return p_idxs[:,0]

def sort_violations(graphs, HB_per_node):

    #ps = np.random.uniform(0,1, (graphs.nodes.shape[0],1))
    nodes = graphs.nodes
    n_node = graphs.n_node
    n_graph = graphs.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

    shifted_HB_per_node = -HB_per_node/jnp.max(HB_per_node) + node_graph_idx[:,None]
    #sorted_ps = jax.lax.sort(shifted_ps, dimension = 0)
    HB_per_node_idxs = jax.numpy.argsort(shifted_HB_per_node, axis = 0)

    return HB_per_node_idxs[:,0]

if __name__ == '__main__':
    import igraph
    import numpy as np

    # pmap example
    #https: // github.com / google - deepmind / jraph / blob / master / jraph / ogb_examples / train_pmap.py
    import jax
    #jax.config.update('jax_platform_name', 'cpu')
    from jax import numpy as jnp

    n_state = 5
    n_classes = 10
    spin_logits = jnp.log(jnp.ones((n_state, n_classes)))

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    X_next = jax.random.categorical(key=subkey,
                                    logits=spin_logits,
                                    axis=-1,
                                    shape=spin_logits.shape[:-1])

    one_hot_state = jax.nn.one_hot(X_next, num_classes=n_classes)
    # X_next = jnp.expand_dims(X_next, axis = -1)
    spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)


    import jraph

    #node_list = [100, 120, 80, 40, 300]
    node_list = [10, 12, 8, 6, 30, 500, 5]
    node_list = [ 10000 for i in range(5)]
    graph_list = [create_graph(n) for n in node_list]
    #for_loop_CE(graph_list)
    batched_graph = jraph.batch(graph_list)

    vmap_calc_energy = jax.vmap(calc_energy, in_axes=(None, 0))

    import time

    import numpy as np

    import numpy as np

    # Example array A with shape (N, M)
    A = np.array([[1, 2, 3],
                  [4, 0, 6],
                  [7, 8, 9]])

    # Find the indices of the minimum values along axis 0
    min_idxs = np.argmin(A, axis=0)

    # Access the elements of A at the minimum indices
    min_elements = A[min_idxs, np.arange(A.shape[1])]

    print("Min indices along axis 0:", min_idxs)
    print("Elements at min indices:", min_elements)

    @jax.jit
    def cond(arg):
        step = arg["step"]
        max_steps = arg["max_steps"]
        return step < max_steps  # jax.lax.bitwise_and(Hb > 0., step < max_steps)

    @jax.jit
    def body(arg):

        arg["states"]= arg["states"].at[arg["step"],0].set(1.)
        arg["step"] = arg["step"] + 1

        return arg  # jax.lax.bitwise_and(Hb > 0., step < max_steps)

    @jax.jit
    def while_func(graphs, states):
        n_node = graphs.n_node
        cum_sum = jnp.concatenate([jnp.array([0]), jax.lax.cumsum(n_node)[:-1]], axis=0)
        cum_max_sum = jax.lax.cumsum(n_node)
        max_steps = jnp.max(graphs.n_node)
        scan_dict = {"graphs": graphs, "states": states, "max_steps": max_steps, "step": 0}
        jax.lax.while_loop(cond, body, scan_dict)


    vmap_while_func = jax.vmap(while_func, in_axes=(None, 0))

    reps = np.arange(1,14)
    for rep in reps:

        states = jnp.repeat(batched_graph.nodes[None, ...], 2**rep, axis = 0)

        vmap_while_func(batched_graph, states)

        s = time.time()
        vmap_while_func(batched_graph, states)
        e = time.time()

        print("time", e-s, rep)



