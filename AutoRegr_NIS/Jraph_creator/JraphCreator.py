import jax.tree_util
import jraph
import networkx as nx
import igraph as ig
import numpy as np
import jax.numpy as jnp


def generate_graph(n):
    """
    Generate a NxN lattice graph
    """
    gnx = nx.grid_2d_graph(n, n, periodic=True)
    return gnx

def create_graph(n):
    # suppose a 4x4 lattice
    gnx = generate_graph(n)
    weight = {e: 1. for e in gnx.edges()}
    nx.set_edge_attributes(gnx, weight, "weight")

    ### todo needs to be updated since ordering of igraph is not correct anymore
    H_graph, graph_size, density = nx_to_jraph(gnx)
    return jraph.batch([H_graph])



def nx_to_jraph(gnx: nx.Graph) -> (jraph.GraphsTuple, float, int):
    """
    Convert networkx graph to jraph graph via igraph

    :param gnx: networkx graph
    :return: (H_graph, density, graph_size)
    """
    g = ig.Graph.TupleList(gnx.edges(), directed=False)
    density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
    graph_size = g.vcount()
    return from_igraph_to_jgraph(g), density, graph_size


def nx_to_igraph(gnx: nx.Graph) -> ig.Graph:
    """
    Convert networkx graph to igraph graph

    :param gnx: networkx graph
    :return: igraph graph
    """
    return ig.Graph.TupleList(gnx.edges(), directed=False)


def igraph_to_jraph(g: ig.Graph, _np = np) -> (jraph.GraphsTuple, float, int):
    """
    Convert igraph graph to jraph graph

    :param g: igraph graph
    :return: (H_graph, density, graph_size)
    """
    density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
    graph_size = g.vcount()
    return from_igraph_to_jgraph(g, _np = _np), density, graph_size


def igraph_to_jraph_explicit(i_graph, np_ = np):
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

def from_igraph_to_jgraph(igraph, zero_edges = True, double_edges = True, _np = np):
    num_vertices = igraph.vcount()
    edge_arr = _np.array(igraph.get_edgelist())
    if(double_edges):
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


def nx_to_jraph( nx_graph):
    num_vertices = nx_graph.number_of_nodes()

    node_idx = {}
    for i, node in enumerate(nx_graph.nodes):
        node_idx[node] = i

    edge_idx = []
    for i, edge in enumerate(nx_graph.edges):
        sender, receiver = edge
        edge_idx.append([node_idx[sender], node_idx[receiver]])
    edge_idx = np.array(edge_idx)
    undir_senders = edge_idx[:, 0]
    undir_receivers = edge_idx[:, 1]
    receivers = np.concatenate([undir_receivers[:, np.newaxis], undir_senders[:, np.newaxis]], axis=-1)
    receivers = np.ravel(receivers)
    senders = np.concatenate([undir_senders[:, np.newaxis], undir_receivers[:, np.newaxis]], axis=-1)
    senders = np.ravel(senders)
    edges = np.ones((senders.shape[0], 1))

    N = int(np.sqrt(num_vertices))
    x = np.arange(0, N)
    y = np.arange(0, N)
    xv, yv = np.meshgrid(x, y)

    nodes_x = jax.nn.one_hot(xv.flatten(), N)
    nodes_y = jax.nn.one_hot(yv.flatten(), N)
    nodes = np.concatenate([nodes_x, nodes_y], axis=-1)
    nodes = np.array(nodes)

    print("nodes", nodes.shape, num_vertices)
    nodes = np.zeros((num_vertices, 1))

    globals = np.array([num_vertices])
    n_node = np.array([num_vertices])
    n_edge = np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders=senders, receivers=receivers, edges=edges, nodes=nodes, n_edge=n_edge,
                               n_node=n_node, globals=globals)

    density = 2 * n_edge / (n_node * (n_node - 1))
    graph_size = n_node
    return jgraph, graph_size, density