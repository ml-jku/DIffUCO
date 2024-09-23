import sys

import jraph

sys.path.append("..")
import numpy as np
from DatasetCreator.jraph_utils import utils as jutils
import copy
import igraph
from DatasetCreator.utils.Message_passing_test import MeanAggr, MessagePassing, MaxAggr, MaxPool
import jax.numpy as jnp
import jax

def solveMIS(orig_i_graph):
    print("solve DG_MIS")
    if(len(orig_i_graph.get_edgelist()) != len(set(orig_i_graph.get_edgelist()))):
        print(orig_i_graph.get_edgelist())
        raise ValueError("dublicates in edge list")

    i_graph = copy.deepcopy(orig_i_graph)
    ### TODO adapt code so that vertex position can be determined can also be set
    vertex_set = []
    finished = False
    global_idx = list(np.arange(0, i_graph.vcount()))
    if(len(i_graph.get_edgelist()) == 0):
        vertex_set = [i for i in range(i_graph.vcount())]
    else:
        while(not finished):
            degrees = i_graph.degree()
            num_vertices = i_graph.vcount()
            vertices = np.arange(0, num_vertices)
            sorted_vertices = sorted(zip(vertices, degrees, global_idx), key = lambda x: x[1])

            min_degree_node = sorted_vertices[0][0]

            selected_node_idx = sorted_vertices[0][-1]
            vertex_set.append(selected_node_idx)

            neighbours = i_graph.neighbors(i_graph.vs[min_degree_node])

            delete_nodes = [min_degree_node] + neighbours

            i_graph.delete_vertices(delete_nodes)

            delete_nodes_global = [global_idx[min_degree_node]] + [global_idx[neigh] for neigh in neighbours]
            for del_node in delete_nodes_global:
                global_idx.remove(del_node)

            if(i_graph.vcount() != len(global_idx)):
                print(len(global_idx), i_graph.vcount() )
                raise ValueError("not enough nodes deleted")

            if(i_graph.vcount() == 0):
                finished = True

    if(len(set(vertex_set)) != len(vertex_set)):
        print(vertex_set)
        raise ValueError("same node occurs multiple times")

    pred_Energy = -len(vertex_set)
    res_dict = {"vertex_set": vertex_set, "i_graph": i_graph}
    return pred_Energy, res_dict

def make_independent_set_graph(graph ,independent_set):
    max_hops = 2
    edges = []
    #new_vertex_indices = independent_set
    for node1 in independent_set:
        two_hop_neighbors = set(graph.neighborhood(node1, order=max_hops, mode=igraph.ALL))

        for neighbor_node in two_hop_neighbors:
            if(neighbor_node in independent_set):
                if(node1 !=  neighbor_node):
                    new_node1 = list(independent_set).index(node1)
                    new_neighbor_node = list(independent_set).index(neighbor_node)
                    edges.append(( min([new_node1, new_neighbor_node]), max([new_node1, new_neighbor_node])))

    edges = set(edges)

    # if(len(list(edges)) == 0):
    #     pooled_graph = igraph.Graph.Full(len(independent_set))
    # else:
    pooled_graph = igraph.Graph()
    pooled_graph.add_vertices(len(independent_set))
    pooled_graph.add_edges(edges)

    if(len(pooled_graph.get_edgelist()) != len(set(pooled_graph.get_edgelist()))):
        print(pooled_graph.get_edgelist())
        raise ValueError("dublicates in edge list")

    return pooled_graph

def make_upsampling_graph(graph, independent_set):
    ### TODO somehow some nodes are missing, find out why
    ### The upsampling graph will be used to mean aggregate the hidden dimension of the previous layer to newly added nodes
    edges = []
    # new_vertex_indices = independent_set
    for node1 in independent_set:
        one_hop_neighbors = set(graph.neighborhood(node1, order=1, mode=igraph.ALL))

        if(node1 not in one_hop_neighbors):
            raise ValueError("self loop not contained")

        for neighbor_node in one_hop_neighbors:
            edges.append((node1, neighbor_node))
            #edges.append(( min([node1, neighbor_node]), max([node1, neighbor_node])))

            if(neighbor_node in independent_set and node1 != neighbor_node):
                print(neighbor_node, node1)
                print(independent_set)
                print(one_hop_neighbors)
                print(graph.get_edgelist())
                raise ValueError("This should not be possible")

    edges = list(set(edges))

    receivers = [e[1] for e in edges]
    if(len(set(receivers)) != graph.vcount()):
        print(set(receivers) != graph.vcount())
        print(set(receivers), graph.vcount(), len(set(receivers)))
        raise ValueError("")


    if(len(edges) == 0):
        raise ValueError("no edges in upsampling graph")
    # for edge in edges:
    #     flip_edge = (edge[1],edge[0])
    #     if(flip_edge in edges and edge[0] != edge[1]):
    #         print(flip_edge, edge)
    #         raise ValueError("")

    upsampling_graph = jutils.make_jgraph(graph.vcount(), edges, double_edges = False)
    return upsampling_graph

def make_max_aggr_graph(graph, independent_set):
    ### todo only messages from 1 hop neighbors to independent set nodes
    ### The upsampling graph will be used to mean aggregate the hidden dimension of the previous layer to newly added nodes
    edges = []
    # new_vertex_indices = independent_set
    for node1 in independent_set:
        one_hop_neighbors = set(graph.neighborhood(node1, order=1, mode=igraph.ALL))
        for neighbor_node in one_hop_neighbors:
            #edges.append(( min([node1, neighbor_node]), max([node1, neighbor_node])))
            edges.append((neighbor_node, node1))

            if(neighbor_node in independent_set and neighbor_node != node1):
                print(neighbor_node, node1)
                print(independent_set)
                print(one_hop_neighbors)
                print(graph.get_edgelist())
                raise ValueError("This should not be possible")

    edges = list(set(edges))
    # print("max_aggr graph", edges)
    # for edge in edges:
    #     flip_edge = (edge[1],edge[0])
    #     if(flip_edge in edges and edge[0] != edge[1]):
    #         print(flip_edge, edge)
    #         raise ValueError("")

    max_aggr_graph = jutils.make_jgraph(graph.vcount(), edges, double_edges = False)
    print("max aggr", set(max_aggr_graph.receivers))
    return max_aggr_graph

def make_pooling_graph(independent_set, prev_num_nodes, _np = np):
    ## number of nodes of previous graph as global features
    nodes = _np.array(independent_set)[:,None]
    n_node = _np.array([nodes.shape[0]])

    senders = _np.zeros((0))
    receivers = _np.zeros((0))
    edges = _np.zeros((0,1))
    n_edge = _np.array([0])

    glob = _np.array([prev_num_nodes])
    pool_graph = jraph.GraphsTuple(nodes = nodes, senders = senders, receivers = receivers, edges = edges, n_node = n_node, n_edge = n_edge, globals = glob)

    return pool_graph


def plot_graph(graph, indeces):
    import networkx as nx
    import matplotlib.pyplot as plt

    # Convert igraph graph to NetworkX graph
    networkx_graph = nx.Graph()
    networkx_graph.add_nodes_from(np.arange(0, graph.vcount()))
    networkx_graph.add_edges_from(graph.get_edgelist())

    # Plot the NetworkX graph
    node_color = ["blue" for i in range(graph.vcount())]
    for i in indeces:
        node_color[i] = "red"

    pos = nx.spring_layout(networkx_graph)  # You can choose different layout algorithms
    nx.draw(networkx_graph, pos, with_labels=True, node_color=node_color, linewidths=3)

    # Show the plot
    plt.show()

def make_U_net_architecture(graph, n_layers = 4):
    ### TODO upsampling and downsampling graph should be directed graphs

    edges = graph.get_edgelist()
    for edge in edges:
        if(edge[0] == edge[1]):
            print("self loop")
            print(edge)


    _, res_dict =  solveMIS(graph)
    independent_set = res_dict["vertex_set"]
    #plot_graph(graph, independent_set)
    upsampling_graph = make_upsampling_graph(graph, independent_set)
    max_aggr_graph = make_max_aggr_graph(graph, independent_set)
    # Replace these node indices with the actual indices of the two nodes you are interested in
    pooling_graph = make_pooling_graph(independent_set, graph.vcount())

    ### 1. Graph: MP on Input graph -> pooling graph -> MP on Pooled Graph -> pooling Graph -> MP --- -> upsampling Graph
    U_net_graph_dict = {"graphs": [graph], "pooling_graph": [pooling_graph], "max_aggr_graph": [max_aggr_graph],  "upsampling_graph": [upsampling_graph], "bottleneck_graph": []}
    for n in range(n_layers):
        pooled_graph = make_independent_set_graph(graph, independent_set)

        graph = pooled_graph
        _, res_dict = solveMIS(graph)
        independent_set = res_dict["vertex_set"]
        upsampling_graph = make_upsampling_graph(graph, independent_set)
        max_aggr_graph = make_max_aggr_graph(graph, independent_set)
        pooling_graph = make_pooling_graph(independent_set, graph.vcount())

        U_net_graph_dict["graphs"].append(pooled_graph)
        U_net_graph_dict["pooling_graph"].append(pooling_graph)
        U_net_graph_dict["upsampling_graph"].append(upsampling_graph)
        U_net_graph_dict["max_aggr_graph"].append(max_aggr_graph)
        #plot_graph(pooled_graph, independent_set)

    ### TODO make bottle neck graph fully connected?
    pooled_graph = make_independent_set_graph(graph, independent_set)
    U_net_graph_dict["bottleneck_graph"] = jutils.from_igraph_to_jgraph(pooled_graph, _np = np)

    U_net_graph_dict["graphs"] = [jutils.from_igraph_to_jgraph(i_graph, _np = np) for i_graph in U_net_graph_dict["graphs"]]
    U_net_graph_dict["pooling_graph"] = [j_graph for j_graph in U_net_graph_dict["pooling_graph"]]

    # for upsampling_graph in U_net_graph_dict["upsampling_graph"]:
    #     print(set(upsampling_graph.receivers))
    #     nodes= upsampling_graph.nodes
    #     if(len(nodes) != len(set(upsampling_graph.receivers))):
    #         print(set(upsampling_graph.receivers), len(set(upsampling_graph.receivers)), len(nodes))
    #         raise ValueError("")

    return U_net_graph_dict

def Unet_toy_example(jraph_list, pool_graph_list, upsample_graph, max_agg_graph_list, bottleneck_graph):
    ### TODO make it work for batched graph
    key = jax.random.PRNGKey(0)

    node_features = jax.random.normal(key, shape = jraph_list[0].nodes.shape)


    ### Downsampling
    intermediate_downpool_features = []

    for graph, pool_graph, max_aggr_graph in zip(jraph_list, pool_graph_list, max_agg_graph_list):
        #print("before", jnp.mean(node_features))
        updated_nodes = MessagePassing(graph, node_features)

        #print("updated_nodes", jnp.mean(updated_nodes))
        intermediate_downpool_features.append(updated_nodes)

        node_features = MaxPool(max_aggr_graph, pool_graph, updated_nodes)
        # print("MO down", jnp.mean(node_features))
        # print("downpooling", node_features.shape, updated_nodes.shape)

    node_features = MessagePassing(bottleneck_graph, node_features)

    for graph, pool_graph, upsample_graph, intermediate_features in zip(reversed(jraph_list), reversed(pool_graph_list), reversed(upsample_graph), reversed(intermediate_downpool_features)):

        upsample_graph.nodes[pool_graph.nodes[...,0]] = node_features
        updated_nodes = MeanAggr(upsample_graph, upsample_graph.nodes)
        #print("prev_nodes", jnp.mean(node_features))
        #print("updated_nodes", jnp.mean(updated_nodes))
        #print(intermediate_features.shape, updated_nodes.shape, node_features.shape)
        updated_nodes = MessagePassing(graph, updated_nodes + intermediate_features)
        #print("MO up", jnp.mean(node_features))
        node_features = updated_nodes

    return node_features

import jax.tree_util as tree
def shift_node_graph_index(pooling_graph):
    nodes = pooling_graph.nodes
    n_node = pooling_graph.n_node
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

    node_offset = jax.lax.cumsum(pooling_graph.globals) - pooling_graph.globals[0]
    node_offset_per_node = jnp.repeat(
        node_offset, n_node, axis=0, total_repeat_length=sum_n_node)

    shifted_independent_indices = node_offset_per_node[:, None] + nodes
    pooling_graph = pooling_graph._replace(nodes = shifted_independent_indices)
    return pooling_graph

def shift_indices(indices, shift):
    return indices + shift

def batch_U_net_graph_dict(U_net_graph_dict_list):
    keys = U_net_graph_dict_list[0].keys()
    batched_U_net_graph_dict = {key: [U_net_graph_dict_el[key] for U_net_graph_dict_el in U_net_graph_dict_list] for key in keys}


    n_layers = len(batched_U_net_graph_dict["graphs"][0])
    batched_layer_dict = {"graphs": {}, "upsampling_graph": {}, "pooling_graph": {}, "max_aggr_graph": {}}
    for n_layer in range(n_layers):
        batched_layer_dict["graphs"][f"layer_{n_layer}"] = []
        batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = []
        batched_layer_dict["pooling_graph"][f"layer_{n_layer}"] = []
        batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"] = []

        for key in keys:
            if( "bottleneck_graph" != key):
                for el in batched_U_net_graph_dict[key]:
                    batched_layer_dict[key][f"layer_{n_layer}"].append(el[n_layer])

        batched_layer_dict["graphs"][f"layer_{n_layer}"] = jraph.batch_np(batched_layer_dict["graphs"][f"layer_{n_layer}"])
        batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = jraph.batch_np(batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"])
        batched_layer_dict["pooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(jraph.batch_np(batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]))
        batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"] = jraph.batch_np(batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"])

    batched_U_net_graph_dict["graphs"] = [batched_layer_dict["graphs"][key] for key in batched_layer_dict["graphs"]]
    batched_U_net_graph_dict["upsampling_graph"] = [batched_layer_dict["upsampling_graph"][key] for key in batched_layer_dict["upsampling_graph"]]
    batched_U_net_graph_dict["pooling_graph"] = [batched_layer_dict["pooling_graph"][key] for key in batched_layer_dict["pooling_graph"]]
    batched_U_net_graph_dict["max_aggr_graph"] = [batched_layer_dict["max_aggr_graph"][key] for key in batched_layer_dict["max_aggr_graph"]]
    batched_U_net_graph_dict["bottleneck_graph"] = jraph.batch_np(batched_U_net_graph_dict["bottleneck_graph"])

    return batched_U_net_graph_dict

if(__name__ == "__main__"):

    ### TODO add bottleneck graph?
    for i in range(100):
        ### TODO implement max pool aggr.
        graph = igraph.Graph.Erdos_Renyi(n=100, p=0.05)
        U_net_graph_dict = make_U_net_architecture(graph, n_layers=4)

        print("upsample", [graph.nodes.shape for graph in U_net_graph_dict["upsampling_graph"]])
        #print("indices", [len(indices) for indices in U_net_graph_dict["selected_indices"]])
        print("MPP", [graph.nodes.shape for graph in U_net_graph_dict["graphs"]])


        U_net_graph_dict_list = [U_net_graph_dict, U_net_graph_dict]
        ### TODO implement padding
        #### TODO implement batching
        batched_U_net_graph_dict = batch_U_net_graph_dict(U_net_graph_dict_list)

        Unet_toy_example(U_net_graph_dict["graphs"], U_net_graph_dict["pooling_graph"], U_net_graph_dict["upsampling_graph"], U_net_graph_dict["max_aggr_graph"], U_net_graph_dict["bottleneck_graph"])

