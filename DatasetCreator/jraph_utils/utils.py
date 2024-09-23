import jax.numpy as jnp
import numpy as np
import jraph
import igraph


def get_first_node_idxs(n_node, np_ = jnp):
    node_idx = np_.concatenate([np_.array([0]),n_node], axis = -1)

    global_node_idx = np_.cumsum(node_idx[0:-1])

    return global_node_idx


def pad_graph(jgraph, add_padded_node=False, time_horizon=0, random_node_features = 20):
    if (not add_padded_node):
        num_nodes = jgraph.nodes.shape[0]
    else:
        num_nodes = jgraph.nodes.shape[0]
        rest = num_nodes % max([time_horizon, 1])
        if(rest != 0):
            additional_nodes = time_horizon - rest
            num_nodes = num_nodes + additional_nodes
        else:
            additional_nodes = 0

    nodes = np.zeros((num_nodes, 1), dtype=jnp.float32)

    if (random_node_features > 0):
        random_node_features = np.random.uniform(size=(nodes.shape[0], random_node_features))
        nodes = np.concatenate([nodes, random_node_features], axis=-1)

    mask = np.zeros((num_nodes, 1))
    mask[:num_nodes - additional_nodes] = np.ones((num_nodes - additional_nodes, 1))

    mask_and_nodes = np.concatenate([nodes, mask], axis=-1)

    n_node = np.array([num_nodes])

    jgraph = jgraph._replace(nodes = mask_and_nodes, n_node = n_node)
    return jgraph

def from_pyg_graph_to_igraph(pyg_graph):
    edge_index = pyg_graph.edge_index
    ig = igraph.Graph([ (edge_index[0,i], edge_index[1,i]) for i in range(edge_index.shape[1])])
    ig.simplify()
    ig["gt_Energy"] = None
    return ig

def from_jgraph_to_igraph(jgraph, simplify = True):
    senders = jgraph.senders
    receivers = jgraph.receivers
    Energy = jgraph.globals
    ig = igraph.Graph(directed = True)
    ig.add_vertices(jgraph.nodes.shape[0])
    ig.add_edges([ (s,r) for s,r in zip(senders, receivers)])
    if(simplify):
        ig.simplify()
    ig["gt_Energy"] = Energy

    n_nodes = jgraph.nodes.shape[0]
    if(n_nodes != ig.vcount()):
        raise ValueError("nodes got lost during casting")

    return ig

def from_jgraph_to_igraph_normed(jgraph, directed = False):
    senders = jgraph.senders
    receivers = jgraph.receivers
    Energy = jgraph.globals
    ig = igraph.Graph(directed = directed)
    ig.add_vertices(jgraph.nodes.shape[0])
    ig.add_edges([ (s,r) for s,r in zip(senders, receivers)])
    #ig = igraph.Graph([ (s,r) for s,r in zip(senders, receivers)], directed=True)
    ig.vs["ext_fields"] = jgraph.nodes
    ig.es["couplings"] = jgraph.edges
    #ig.es["couplings"] = jgraph.edges
    ig["gt_Energy"] = Energy
    if(jgraph.nodes.shape[0] != ig.vcount()):
        raise ValueError("Nodes got lost")
    return ig

def from_jgraph_to_dir_igraph_normed(jgraph):
    senders = jgraph.senders
    receivers = jgraph.receivers
    edges = jgraph.edges

    # senders = np.reshape(receivers, (int(receivers.shape[0]/2),2))[:,0]
    # receivers = np.reshape(receivers, (int(receivers.shape[0]/2),2))[:,1]
    # edges = np.expand_dims(np.reshape(jgraph.edges, (int(jgraph.edges.shape[0]/2),2))[:,0], axis = -1)
    from collections import Counter
    edge_list = [(min([s,r]), max([s,r]), float(e)) for s,r,e in zip(senders, receivers, edges) ]
    edge_list = list(Counter(edge_list).keys())

    Energy = jgraph.globals
    ig = igraph.Graph(directed = False)
    ig.add_vertices(jgraph.nodes.shape[0])
    ig.add_edges([(s,r) for (s,r,e) in edge_list])
    #ig = igraph.Graph([ (s,r) for s,r in zip(senders, receivers)], directed=True)
    ig.vs["ext_fields"] = jgraph.nodes

    ig.es["couplings"] = np.expand_dims(np.array([e for (s,r,e) in edge_list]), axis = -1)
    #ig.es["couplings"] = jgraph.edges
    ig["gt_Energy"] = Energy
    if(jgraph.nodes.shape[0] != ig.vcount()):
        raise ValueError("Nodes got lost")
    return ig

def from_igraph_to_dir_jgraph_normed(igraph):
    num_vertices = igraph.vcount()
    edge_arr = np.array(igraph.get_edgelist())
    if(igraph.ecount() > 0):
        receivers = edge_arr[:, 0]
        senders = edge_arr[:, 1]
        edges =  np.array(igraph.es["couplings"])
    else:
        receivers = np.ones((0,), dtype = np.int32)
        senders = np.ones((0,), dtype = np.int32)
        edges =  np.ones((0, 1), dtype = np.float32)

    nodes = np.array(igraph.vs["ext_fields"])
    globals = np.array([num_vertices])
    n_node = np.array([num_vertices])
    n_edge = np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders = senders, receivers = receivers, edges = edges, nodes = nodes, n_edge = n_edge , n_node = n_node, globals = globals )
    return jgraph

def from_igraph_to_jgraph(igraph, zero_edges = True, double_edges = True, _np = np):
    num_vertices = igraph.vcount()
    edge_arr = _np.array(igraph.get_edgelist())
    if(double_edges):
        print("ATTENTION: edges will be dublicated in this method!")
        if(igraph.ecount() > 0):
            undir_receivers = edge_arr[:, 0]
            undir_senders = edge_arr[:, 1]
            receivers = _np.concatenate([undir_receivers[:, np.newaxis], undir_senders[:, np.newaxis]], axis=-1)
            receivers = _np.ravel(receivers)
            senders = _np.concatenate([undir_senders[:, np.newaxis], undir_receivers[:, np.newaxis]], axis=-1)
            senders = _np.ravel(senders)
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.concatenate([edge_weights, edge_weights], axis=0)
    else:
        if(igraph.ecount() > 0):
            senders = edge_arr[:, 0]
            receivers = edge_arr[:, 1]
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.array(edge_weights)

    nodes = _np.zeros((num_vertices, 1))
    globals = _np.array([num_vertices])
    n_node = _np.array([num_vertices])
    n_edge = _np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders = senders, receivers = receivers, edges = edges, nodes = nodes, n_edge = n_edge , n_node = n_node, globals = globals )
    return jgraph

def make_jgraph(n_nodes, edge_arr,double_edges = True, _np = np):
    num_vertices = n_nodes
    edge_arr = _np.array(edge_arr)
    if(double_edges):
        print("ATTENTION: edges will be dublicated in this method!")
        if(len(edge_arr) > 0):
            undir_senders = edge_arr[:, 0]
            undir_receivers = edge_arr[:, 1]
            receivers = _np.concatenate([undir_receivers[:, np.newaxis], undir_senders[:, np.newaxis]], axis=-1)
            receivers = _np.ravel(receivers)
            senders = _np.concatenate([undir_senders[:, np.newaxis], undir_receivers[:, np.newaxis]], axis=-1)
            senders = _np.ravel(senders)
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

    else:
        if(len(edge_arr) > 0):
            senders = edge_arr[:, 0]
            receivers = edge_arr[:, 1]
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

    nodes = _np.zeros((num_vertices, 1))
    globals = _np.array([num_vertices])
    n_node = _np.array([num_vertices])
    n_edge = _np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders = senders, receivers = receivers, edges = edges, nodes = nodes, n_edge = n_edge , n_node = n_node, globals = globals )

    return jgraph

def check_number_of_edge_occurances(jgraph, num_occ = 1):
    from collections import Counter

    senders = jgraph.senders
    receivers = jgraph.receivers

    edge_list = [ (min([s,r]),max([r,s])) for s,r in zip(senders, receivers)]
    values = list(Counter(edge_list).values())
    #print(Counter(edge_list).keys())
    if(np.any(num_occ != np.array(values))):
        print(Counter(edge_list).keys())
        print(Counter(edge_list).values())
        raise ValueError("There are too many edge occurances")

def from_igraph_to_dir_jgraph(igraph):
    num_vertices = igraph.vcount()
    edge_arr = np.array(igraph.get_edgelist())
    if(igraph.ecount() > 0):
        receivers = edge_arr[:, 0]
        senders = edge_arr[:, 1]
        edges =  np.ones((senders.shape[0], 1), dtype = np.float32)
    else:
        receivers = np.ones((0,), dtype = np.int32)
        senders = np.ones((0,), dtype = np.int32)
        edges =  np.ones((0, 1), dtype = np.float32)

    nodes = np.zeros((num_vertices, 1),dtype = np.float32)
    globals = np.array([num_vertices])
    n_node = np.array([num_vertices])
    n_edge = np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders = senders, receivers = receivers, edges = edges, nodes = nodes, n_edge = n_edge , n_node = n_node, globals = globals )


    return jgraph

def from_igraph_to_normed_jgraph(igraph):
    num_vertices = igraph.vcount()
    edge_arr = np.array(igraph.get_edgelist())
    if(igraph.ecount() > 0):
        receivers = edge_arr[:, 0]
        senders = edge_arr[:, 1]
        edges =  np.array(igraph.es["couplings"])
    else:
        receivers = np.ones((0,), dtype = np.int32)
        senders = np.ones((0,), dtype = np.int32)
        edges =  np.ones((0, 1))

    nodes = np.zeros((num_vertices, 1))
    globals = np.array([num_vertices])
    n_node = np.array([num_vertices])
    n_edge = np.array([receivers.shape[0]])

    nodes = np.array(igraph.vs["ext_fields"])
    jgraph = jraph.GraphsTuple(senders = senders, receivers = receivers, edges = edges, nodes = nodes, n_edge = n_edge , n_node = n_node, globals = globals )


    return jgraph

def collate_from_pyg_to_igraph(datas):
    ig_list = [from_pyg_graph_to_igraph(pyg_graph) for pyg_graph in datas]
    Hb_igraph = igraph.disjoint_union(ig_list)
    return Hb_igraph

def collate_from_jgraph_to_igraph_normed(datas):
    ig_list = [from_jgraph_to_igraph_normed(jgraph) for (jgraph, gs) in datas]
    Hb_igraph = igraph.disjoint_union(ig_list)
    Hb_igraph["gt_Energy"] = ig_list[0]["gt_Energy"]
    return Hb_igraph, datas[0][1]


def collate_igraph_normed(datas):
    ig_list = [jgraph for (jgraph, orig_jgraph, gs) in datas]
    orig_ig_list = [orig_jgraph for (jgraph, orig_jgraph, gs) in datas]
    Hb_igraph = igraph.disjoint_union(ig_list)
    if(orig_ig_list[0] != None):
        Hb_orig_igraph = igraph.disjoint_union(orig_ig_list)
    else:
        Hb_orig_igraph = None
    Hb_igraph["gt_Energy"] = ig_list[0]["gt_Energy"]
    Hb_igraph["original_Energy"] = ig_list[0]["original_Energy"]
    Hb_igraph["self_loop_Energy"] = ig_list[0]["self_loop_Energy"]
    return Hb_igraph, Hb_orig_igraph, datas[0][-1]

def collate_simple_return(datas):
    return datas[0]

def collate_from_jgraph_to_igraph(datas):
    ig_list = [from_jgraph_to_igraph(jgraph) for jgraph in datas]
    Hb_igraph = igraph.disjoint_union(ig_list)
    Hb_igraph["gt_Energy"] = ig_list[0]["gt_Energy"]
    return Hb_igraph

def collate_jraphs_to_max_size(datas, random_node_features):

    num_nodes = max([graph.nodes.shape[0] for graph in datas])
    jdata_list = [pad_graph(graph, add_padded_node=True, time_horizon= num_nodes, random_node_features=random_node_features) for graph in datas]
    batched_jdata = jraph.batch_np(jdata_list)
    return (batched_jdata, jdata_list)


def collate_jraphs_to_max_size(datas, random_node_features):

    num_nodes = max([graph.nodes.shape[0] for graph in datas])
    jdata_list = [pad_graph(graph, add_padded_node=True, time_horizon= num_nodes, random_node_features=random_node_features) for graph in datas]
    batched_jdata = jraph.batch_np(jdata_list)
    return (batched_jdata, jdata_list)

def collate_jraphs_to_horizon(datas, time_horizon, random_node_features):

    jdata_list = [pad_graph(graph, add_padded_node=True, time_horizon= time_horizon, random_node_features=random_node_features) for graph in datas]
    batched_jdata = jraph.batch_np(jdata_list)
    return (batched_jdata, jdata_list)

def shuffle_senders_and_receivers(num_nodes, senders, receivers):
    aranged_indeces = np.arange(0, num_nodes)

    np.random.shuffle(aranged_indeces)

    shuffled_senders = aranged_indeces[senders]
    shuffled_receivers = aranged_indeces[receivers]
    return shuffled_senders, shuffled_receivers

def order_senders_and_receivers(order, senders, receivers):
    ordered_indeces = np.asarray(order)

    ordered_senders = ordered_indeces[senders]
    ordered_receivers = ordered_indeces[receivers]
    return ordered_senders, ordered_receivers

def order_jgraph(j_graph, order):

    senders = j_graph.senders
    receivers = j_graph.receivers
    ordered_senders, ordered_receivers = order_senders_and_receivers(order, senders, receivers)
    ordered_j_graph = j_graph._replace(senders = ordered_senders, receivers = ordered_receivers)

    return ordered_j_graph

def igraph_to_jraph(i_graph, np_ = np):
    num_nodes = i_graph.vcount()
    edge_list = i_graph.get_edgelist()
    edge_arr = np_.array(edge_list)

    undir_receivers = edge_arr[:, 0]
    undir_senders = edge_arr[:, 1]
    receivers = np_.concatenate([undir_receivers, undir_senders], axis=-1)
    senders = np_.concatenate([undir_senders, undir_receivers], axis=-1)
    edges = np_.ones((senders.shape[0], 1))
    nodes = np_.zeros((num_nodes, 1))
    j_graph = jraph.GraphsTuple(nodes=nodes, edges=edges, receivers=receivers,
                                          senders=senders,
                                          n_node= np_.array([num_nodes], dtype=np_.int32),
                                          n_edge= np_.array([senders.shape[0]], dtype=np_.int32), globals= np_.array([senders.shape[0]], dtype=np_.int32) )

    return j_graph

# def jgraph_to_igraph(j_graph):
#
#     senders = j_graph.senders
#     receivers = j_graph.receivers
#
#     ig = igraph.Graph()
#     ig.add_vertices(j_graph.nodes.shape[0])
#     ig_too_many_edges = igraph.Graph([ (s,r) for s,r in zip(senders, receivers)])
#     edge_list = list(set(ig_too_many_edges.get_edgelist()))
#     ig.add_edges(edge_list)
#     return ig

def pyg_to_jgraph(num_nodes, edge_index):
    num_edges = edge_index.shape[1]

    nodes = np.zeros((num_nodes, 1), dtype=np.float32)

    senders = np.array(edge_index[0, :])
    receivers = np.array(edge_index[1, :])


    edges = np.ones((num_edges, 1), dtype=np.float32)
    n_node = np.array([num_nodes])
    n_edge = np.array([num_edges])

    jgraph = jraph.GraphsTuple(nodes=nodes, senders=senders, receivers=receivers,
                               edges=edges, n_node=n_node, n_edge=n_edge, globals=np.zeros((1,)))
    return jgraph

def cast_Tuple_to(j_graph, np_ = jnp):

    j_graph = jraph.GraphsTuple(nodes= np_.asarray(j_graph.nodes), edges=np_.asarray(j_graph.edges), receivers=np_.asarray(j_graph.receivers),
                                          senders=np_.asarray(j_graph.senders),
                                          n_node= np_.asarray(j_graph.n_node),
                                          n_edge= np_.asarray(j_graph.n_edge), globals= np_.asarray(j_graph.globals) )

    return j_graph

def cast_Tuple_to_float32(j_graph, np_ = jnp):
    ### TODO check if replace is faster
    j_graph = jraph.GraphsTuple(nodes= np_.asarray(j_graph.nodes, dtype=np_.float32), edges=np_.asarray(j_graph.edges, dtype=np_.float32), receivers=np_.asarray(j_graph.receivers, np_.int32),
                                          senders=np_.asarray(j_graph.senders, np_.int32),
                                          n_node= np_.asarray(j_graph.n_node, np_.int32),
                                          n_edge= np_.asarray(j_graph.n_edge, np_.int32), globals= np_.asarray(j_graph.globals, np_.float32) )

    return j_graph

def cast_List_to( jnp_arr_list, np_ = np):
    np_arr_list = (np_.asarray(arr) for arr in jnp_arr_list)
    return np_arr_list


def _nearest_bigger_power_of_k(x: int, k: float) -> int:
    """Computes the nearest power of two greater than x for padding."""
    ### TODO test this change during training
    if(k == 1.):
        return x
    else:
        if(x == 0):
            x = 1

        exponent = np.log(x) / np.log(k)

        return int(k**(int(exponent) + 1))


def pad_graph_to_nearest_power_of_k(graphs_tuple: jraph.GraphsTuple, k = 1.4, k_n = 1.4, pad_edges_to = None, np_ = jnp) -> jraph.GraphsTuple:
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_node), k_n ) + 1
    if(pad_edges_to == None):
        pad_edges_to = _nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_edge), k )
    else:
        pad_edges_to = pad_edges_to
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                                 pad_graphs_to)

def pad_graph_and_ext_fields_to_nearest_power_of_k(graphs_tuple: jraph.GraphsTuple, Nb_ext_fields, k = 1.4, np_ = jnp, min_nodes = 1) -> jraph.GraphsTuple:
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_node), k ) + min_nodes
    pad_edges_to = _nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_edge), k )
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1

    # print("Graph is going to be padded")
    # print("num of addtional nodes", pad_nodes_to-Nb_ext_fields.shape[1])
    # print("pad nodes to", pad_nodes_to)
    # print("pad edges to",pad_edges_to)
    # print("original graph_size", graphs_tuple.nodes.shape[0])
    padded_ext_fields = np.concatenate([Nb_ext_fields, np.zeros((Nb_ext_fields.shape[0], pad_nodes_to-Nb_ext_fields.shape[1], Nb_ext_fields.shape[-1]))], axis = -2)
    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                                 pad_graphs_to), padded_ext_fields
