
import jax
import jax.numpy as jnp
import jax.tree_util as tree

def calcEnergy_sparse(H_graph, spins, A = 1., B = 1.1):
    bins = (spins + 1)/2
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

    adjacency = jnp.ones_like(H_graph.edges)
    ### normalise through average number of nodes in dataset
    normalisation = H_graph.edges[0,0]
    A = A*normalisation
    B = B*normalisation

    raveled_bins = jnp.reshape(bins, (bins.shape[0] * bins.shape[1], 1))[:(bins.shape[0] - 1) * bins.shape[1] + 1]
    Energy_messages = adjacency * raveled_bins[H_graph.senders] * raveled_bins[H_graph.receivers]
    Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

    Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))[:-1, 0]

    Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))[:-1, 0]

    Energy = - Ha + Hb

    return Energy

def calcEnergy_test():
    ### TODO return Ha and Hb
    pass

def recalcEnergy_from_gt(H_graphs, gt_spins, A = 1., B = 1.):
    ### TODO to be implemented
    pass


def calcEnergy(H_graph, spins, A = 1., B = 1.1):
    bins = (spins + 1)/2
    bins = jnp.where(spins == 1, jnp.ones_like(spins), jnp.zeros_like(spins))
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

    adjacency = jnp.ones_like(H_graph.edges)

    ### normalise through average number of nodes in dataset
    normalisation = H_graph.edges[0,0]
    A = A*normalisation
    B = B*normalisation

    raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
    Energy_messages = adjacency * raveled_bins[H_graph.senders] * raveled_bins[H_graph.receivers]
    Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

    Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))
    Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))

    Energy = - Ha + Hb

    return Energy

def MVC_Energy(H_graph, bins, A = 1., B = 1.1):
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

    adjacency = jnp.ones_like(H_graph.edges)

    ### normalise through average number of nodes in dataset
    A = A
    B = B

    raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
    Energy_messages = adjacency * (1-raveled_bins[H_graph.senders]) * (1-raveled_bins[H_graph.receivers])
    Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

    Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))
    Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))

    Energy = Ha + Hb

    return Energy

def MIS_Energy(H_graph, bins, A = 1., B = 1.1):
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

    adjacency = jnp.ones_like(H_graph.edges)

    ### normalise through average number of nodes in dataset
    A = A
    B = B

    raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
    Energy_messages = adjacency * (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])
    Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

    Hb = B * (jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph))
    Ha = A * (jax.ops.segment_sum(raveled_bins, node_gr_idx, n_graph))

    Energy = - Ha + Hb

    return Energy