
import sys

import jraph

sys.path.append("..")
import numpy as np
from DatasetCreator.jraph_utils import utils as jutils
from DatasetCreator.GreedyAlgorithms import SpectralClustering
import copy
import igraph
from DatasetCreator.utils.Message_passing_test import MeanAggr, MessagePassing, MaxAggr, MaxPool
import jax.numpy as jnp
import jax

### TODO code that creates an inter cluster aggragation graph; probably has to be a piparitte graph, select cluster nodes with indexing
### TODO add code that creates the pooled graph. Only one node per cluster remains and all edges between clusters remain
### TODO create graph that performs an upsampling aggregation; pipartite graph but htsi time in reverse direction

def cluster_aggragation(i_graph, clusters, n_clusters):

    cluster_nodes = i_graph.vcount() + np.arange(0,n_clusters)

    edges = []
    for cluster_node in np.arange(0,n_clusters):
        for node_idx in range(i_graph.vcount()):
            cluster_membership = clusters[node_idx]
            if(cluster_membership == cluster_node):
                edges.append((node_idx, cluster_node))

    downsampling_indices = cluster_nodes
    edges = list(set(edges))

    if(len(edges) > 0):
        print("here")
    cluster_aggr_graph = jutils.make_jgraph(i_graph.vcount() + n_clusters, edges, double_edges = True)

    ### TODO pooling graph is also neccesary here
    return cluster_aggr_graph, downsampling_indices

def k_cut_graph(i_graph, clusters, n_clusters):

    edge_list = []
    for edges in i_graph.get_edgelist():
        n1 = edges[0]
        n2 = edges[1]

        cluster_membership_1 = clusters[n1]
        cluster_membership_2 = clusters[n2]

        if(cluster_membership_2 != cluster_membership_1):
            edge_list.append([cluster_membership_2, cluster_membership_1])


    cutted_graph = jutils.make_jgraph( n_clusters, edge_list, double_edges = True)
    print("n_clusters", n_clusters, cutted_graph.nodes.shape[0])
    return cutted_graph

def cluster_upsampling(cluster_aggragation_j_graph, n_clusters):
    ''' receivers and senders are simply swapped here'''
    n_nodes = cluster_aggragation_j_graph.nodes.shape[0]
    receivers = cluster_aggragation_j_graph.receivers
    senders = cluster_aggragation_j_graph.senders
    cluster_upsampling_j_graph = cluster_aggragation_j_graph._replace(receivers = senders, senders = receivers)
    ### TODO pooling graph is also neccesary here
    upsampling_indices = np.arange(0, n_nodes - n_clusters)

    return cluster_upsampling_j_graph, upsampling_indices

def make_pooling_graph(indices, prev_num_nodes, _np = np):
    ## number of nodes of previous graph as global features
    nodes = _np.array(indices)[:,None]
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

def count_subgraphs(graph):
    subgraphs = graph.decompose()
    return len(subgraphs)

def plot_igraph_list(i_graph_dict):
    plt.figure()
    for key in i_graph_dict.keys():
        for i, g in enumerate(i_graph_dict[key]):
            g = jutils.from_jgraph_to_igraph(g, simplify=False)
            g_nx = nx.from_edgelist(g.get_edgelist())

            #labels = {node: cluster for node, cluster in zip(g_nx.nodes, clusters)}
            # Plot the graph and label nodes with cluster values
            #pos = nx.spring_layout(g_nx)  # You can use other layout algorithms as well
            #plt.subplot( 1, len(i_graph_list), i + 1)
            plt.figure()
            plt.title(f"n_nodes = {g.vcount()}, {key}")
            nx.draw(g_nx, with_labels=True, cmap=plt.cm.rainbow)
        plt.show()
        plt.close("all")

def create_U_net(g, n_layers = 4,plotting = False):
    N = g.vcount()
    num_cluster_list = [int(N/(2*(n+1))) for n in range(n_layers)]
    i_graph_dict = {"graphs": [], "downsampling_graph": [], "downpooling_graph": [],
                    "upsampling_graph": [], "uppooling_graph": []}
    g_init = copy.deepcopy(g)

    for i, num_clusters in enumerate(num_cluster_list):
        clusters = SpectralClustering.spectral_clustering(g, num_clusters)

        cutted_graph = k_cut_graph(g, clusters, num_clusters)

        if (i < len(num_cluster_list) - 1):
            i_graph_dict["graphs"].append(jutils.from_igraph_to_jgraph(g))
            cluster_aggr_j_graph, downpooling_indices = cluster_aggragation(g, clusters, num_clusters)
            downpooling_j_graph = make_pooling_graph(downpooling_indices, cluster_aggr_j_graph.nodes.shape[0])
            cluster_upsampling_j_graph, uppooling_indices = cluster_upsampling(cluster_aggr_j_graph, num_clusters)
            uppooling_j_graph = make_pooling_graph(uppooling_indices, cluster_aggr_j_graph.nodes.shape[0])


            i_graph_dict["downsampling_graph"].append(cluster_aggr_j_graph)

            i_graph_dict["upsampling_graph"].append(cluster_upsampling_j_graph)
            i_graph_dict["downpooling_graph"].append(downpooling_j_graph)
            i_graph_dict["uppooling_graph"].append(uppooling_j_graph)

        else:
            i_graph_dict["bottleneck_graph"] = cutted_graph
        g = jutils.from_jgraph_to_igraph(cutted_graph)

    if(plotting):
        plot_igraph_list(i_graph_dict)

    return i_graph_dict

from DatasetCreator.loadGraphDatasets.RB_graphs import generate_xu_instances
import networkx as nx
from matplotlib import pyplot as plt
from copy import deepcopy
if(__name__ == "__main__"):
    # Example usage
    # Create a sample graph
    np.random.seed(0)
    for rep in range(2):
        min_n, max_n = 0, np.inf
        n = np.random.randint(9, 15)
        k = np.random.randint(8, 11)
        p = np.random.uniform(0.3, 1.0)
        edges = generate_xu_instances.get_random_instance(n, k, p)
        g = igraph.Graph([(edge[0], edge[1]) for edge in edges])

        # Perform spectral clustering with 2 clusters
        create_U_net(g, plotting=True)

