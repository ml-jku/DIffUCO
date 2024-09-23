import networkx as nx

from .BaseDatasetGenerator import BaseDatasetGenerator
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append("..")
from DatasetCreator.jraph_utils import utils as jutils
from matplotlib import pyplot as plt

try:
    from concorde.tsp import TSPSolver
    from concorde.tests.data_utils import get_dataset_path
except:
    pass

import jraph
import igraph
import numpy as np
from scipy.spatial import distance
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class TSPDatasetGenerator(BaseDatasetGenerator):
    """
    Class for generating datasets for the Barabasi-Albert model
    """
    def __init__(self, config):
        super().__init__(config)

        self.graph_config = {
            "n_train": 40000,
            "n_val": 2000,
            "n_test": 2000,
                    }

        print(f'\nGenerating Barabasi-Albert {self.mode} dataset "{self.dataset_name}" with {self.graph_config[f"n_{self.mode}"]} instances!\n')

    def generate_dataset(self):
        """
        Generate a Barabasi-Albert graph instances for the dataset
        """
        solutions = {
            "Energies": [],
            "H_graphs": [],
            "gs_bins": [],
            "graph_sizes": [],
            "densities": [],
            "runtimes": [],
            "upperBoundEnergies": [],
            "compl_H_graphs": [],
        }


        if ("100" in self.dataset_name):
            graph_size = 100
        elif ("20" in self.dataset_name):
            graph_size = 20

        for idx in tqdm(range(self.graph_config[f"n_{self.mode}"])):
            blockPrint()
            if(self.mode != "train"):
                xs, opt_tour = generate_and_solve(graph_size, solve=self.gurobi_solve)
            else:
                xs, opt_tour = generate_and_solve(graph_size, solve=False)
            enablePrint()

            opt_value = calc_optimal_value_from_tour(opt_tour, xs)
            x_mat_opt = create_opt_matrix(opt_tour, xs, opt_value)
            g = create_jraph(xs)
            H_graph = jutils.from_igraph_to_jgraph(g, zero_edges=False)
            H_graph = H_graph._replace(nodes=xs, edges = np.zeros((0,1)), receivers = np.zeros((0,)), senders = np.zeros((0,)))
            ### TODO remove edges from H_graph?

            energy_graph = make_energy_graph(xs, graph_size)
            energy_graph = energy_graph._replace(nodes=xs)

            print("xs", xs.shape, len(opt_tour), opt_tour)
            opt_tour = np.random.randint(0, len(opt_tour), size=len(opt_tour))
            x_mat_opt = create_opt_matrix(opt_tour, xs, opt_value)
            print("xs", xs.shape, len(opt_tour), opt_tour)
            Energy_check, vio_per_node, _ = compute_energy_mat(xs, opt_tour, x_mat_opt)
            print(vio_per_node)
            print("E1", Energy_check)
            Energy_check_2, vio_per_node_2 = compute_energy(make_energy_graph(xs, graph_size, no_edges=False), opt_tour, x_mat_opt)
            print(vio_per_node_2)
            print("E2", Energy_check_2)
            raise ValueError("")

            solutions["Energies"].append(opt_value)
            solutions["H_graphs"].append(H_graph)
            solutions["gs_bins"].append(xs)
            solutions["graph_sizes"].append(graph_size)
            solutions["densities"].append(1.)
            solutions["runtimes"].append(None)
            solutions["upperBoundEnergies"].append(None)
            solutions["compl_H_graphs"].append(energy_graph)

            indexed_solution_dict = {}
            for key in solutions.keys():
                if len(solutions[key]) > 0:
                    indexed_solution_dict[key] = solutions[key][idx]
            self.save_instance_solution(indexed_solution_dict, idx)
        self.save_solutions(solutions)


def compute_energy_mat(positions, X_0_classes, x_mat_prime, A = 1.45):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_platform_name', 'cpu')
    n_bernoulli_features = len(X_0_classes)
    x_mat = jax.nn.one_hot(X_0_classes, num_classes=n_bernoulli_features)
    distance_matrix = jnp.sqrt(jnp.sum((positions[:, None] - positions[None, :])**2, axis = -1))

    x_mat = x_mat_prime
    cycl_perm_mat = jnp.diag(jnp.ones((n_bernoulli_features - 1,)), k=-1)
    cycl_perm_mat = cycl_perm_mat.at[0, -1].set(1)  ###

    x_mat_cycl = jnp.tensordot(x_mat, cycl_perm_mat, axes=[[-1],[0]])
    H_mat = jnp.tensordot(x_mat_cycl, x_mat, axes=[[-1],[-1]])

    X_0_classe_violation = jnp.where(X_0_classes[None, :] == X_0_classes[:, None], 1., 0.)
    X_0_classe_violation_per_node = jnp.sum(X_0_classe_violation, axis = -2) - 1

    Obj1_per_graph = (jnp.sum(x_mat, axis = 0) -1)**2
    Obj2_per_graph = (jnp.sum(x_mat, axis = 1) -1)**2

    HB_per_graph =  A* jnp.sum(Obj1_per_graph)
    H2 = A* jnp.sum(Obj2_per_graph) #A * jnp.sum(jnp.diag(H_mat))
    H3 = jnp.sum(H_mat* distance_matrix)

    Energy = HB_per_graph + H2 + H3
    print(HB_per_graph, H3, H2)

    return Energy[...,None], X_0_classe_violation_per_node, HB_per_graph

def compute_energy(H_graph, X_0_classes, x_mat, A = 1.45):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_platform_name', 'cpu')
    n_bernoulli_features = len(X_0_classes)
    #x_mat_prime = jax.nn.one_hot(X_0_classes, num_classes=n_bernoulli_features)
    # print("normal", x_mat)
    # X_mat = np.concatenate([X_mat, X_mat], axis = 0)
    # X_mat = np.random.randint(0,1, size = X_mat.shape)
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = np.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = np.repeat(graph_idx, n_node, axis=0)

    N = x_mat.shape[-1]
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    sum_n_node = H_graph.nodes.shape[0]

    idxs = jnp.arange(0, N)
    idxs_p1 = (idxs + 1) % (N)

    Obj1_per_graph_per_feature = (1 - jraph.segment_sum(x_mat, node_gr_idx, n_graph)) ** 2
    Obj1_per_graph = jnp.sum(Obj1_per_graph_per_feature, axis=-1, keepdims=True)
    Obj2_per_node = (1 - jnp.sum(x_mat, axis=-1, keepdims=True)) ** 2
    Obj2_per_graph = jraph.segment_sum(Obj2_per_node, node_gr_idx, n_graph)

    edge_features = H_graph.edges
    receivers = H_graph.receivers
    senders = H_graph.senders

    tour_messages = jnp.sum(x_mat[senders[:, jnp.newaxis], idxs[jnp.newaxis, :]] * x_mat[
        receivers[:, jnp.newaxis], idxs_p1[jnp.newaxis, :]], axis=-1)
    tour_messages = jnp.expand_dims(tour_messages, axis=-1)

    Obj3_per_node = jraph.segment_sum(edge_features * tour_messages, H_graph.receivers, sum_n_node)
    Obj3_per_graph = jraph.segment_sum(Obj3_per_node, node_gr_idx, n_graph)

    Energy_per_graph = A * Obj1_per_graph + A * Obj2_per_graph + Obj3_per_graph
    # Energy_per_node = A*Obj1_per_graph_per_feature + A*Obj2_per_node + Obj3_per_node
    # print("here energy",Energy_per_graph.shape, Obj1_per_graph.shape, Obj2_per_graph.shape, Obj3_per_graph.shape)
    # print("Energy_per_graph", Energy_per_graph.shape)
    # print("Objectives", Obj1_per_graph[0], Obj2_per_graph[0], Obj3_per_graph[0])
    X_senders = X_0_classes[senders]
    X_receivers = X_0_classes[receivers]

    n_copy_senders = jnp.where(X_receivers == X_senders, 1., 0.)

    Obj1_per_node = jraph.segment_sum(n_copy_senders, H_graph.receivers, sum_n_node)

    return Energy_per_graph, Obj1_per_node

def generate_and_solve(N , f = 1000, solve = False):
    xs =  np.random.uniform(0, 1, size=(N, 2))
    ys = 1000 *xs
    if(solve):
        solver = TSPSolver.from_data(ys[:, 0], ys[:, 1], 'EUC_2D')
        solution = solver.solve()
        tour = solution.tour
        #### TODO make fiel that deleted created log files
    else:
        tour = np.arange(0,N)

    return xs, tour

def calc_optimal_value_from_tour(opt_tour, xs):
    my_tour_length = 0
    distance_list = []
    for i, el in enumerate(opt_tour):
        idx = opt_tour[i]
        next_idx = opt_tour[(i + 1) % (len(opt_tour))]
        dsit = np.sqrt(np.sum((xs[idx] - xs[next_idx]) ** 2))
        my_tour_length += dsit
        distance_list.append(dsit)

    print("distances", distance_list)
    print("my_tour_length", my_tour_length)
    return my_tour_length

def create_opt_matrix(opt_tour, xs, tour_length):
    N = len(opt_tour)
    x_mat = np.zeros((N,N))

    for idx, el in enumerate(opt_tour):
        x_mat[el, idx] = 1


    A = np.sqrt(2)
    Obj1 = A* np.sum((1-np.sum(x_mat, axis = 0))**2, axis = 0)
    Obj2 = A* np.sum((1-np.sum(x_mat, axis = 1))**2, axis = 0)

    idxs = np.arange(0, N)
    idxs_p1 = (idxs+1)%(N)
    pos_mat = compute_distance(xs)
    # print(opt_tour)
    # print("asdad", np.sum(x_mat[:,np.newaxis, idxs]*x_mat[np.newaxis,:, idxs_p1], axis = -1))
    # print(pos_mat)
    # print(pos_mat[:,:]* np.sum(x_mat[:,np.newaxis, idxs]*x_mat[np.newaxis,:, idxs_p1], axis = -1))

    Obj3= np.sum(pos_mat[:,:]* np.sum(x_mat[:,np.newaxis, idxs]*x_mat[np.newaxis,:, idxs_p1], axis = -1))


    distances = []
    for i in idxs:
        for k in range(x_mat.shape[0]):
            for kk in range(x_mat.shape[1]):
                res = x_mat[k, i] * x_mat[kk, (i+1)%N]
                if(res == 1):
                    distances.append(pos_mat[k,kk])

    print("checking")
    print(Obj1, Obj2, Obj3)
    print("opt_tour", tour_length)
    return x_mat

def create_jraph(points, closest = False):

    distances = distance.cdist(points, points, 'euclidean')
    g = igraph.Graph()
    g.add_vertices(len(points))

    if (closest):
        k = 10  # Replace with your desired value for k
    else:
        k = len(points)

    edge_weights = []
    edges = []
    for i in range(len(points)):
        # Sort distances and get the indices of the k closest neighbors
        closest_neighbors = np.argsort(distances[i])[1:k + 1]

        # Add edges to the graph connecting the current point to its k closest neighbors
        edges.extend([(i, neighbor) for neighbor in closest_neighbors])
        dist = np.sort(distances[i])[1:k + 1]
        edge_weights.extend(list(dist))

    g.add_edges(edges)
    g.es['weight'] = np.expand_dims(np.array(edge_weights), axis = -1)
    return g

def plotting(opt_tour, g, points):

    plt.figure()
    for i, el in enumerate(opt_tour):
        idx = opt_tour[i]
        next_idx = opt_tour[(i + 1) % (len(opt_tour))]
        plt.plot([points[idx, 0], points[next_idx, 0]], [points[idx, 1], points[next_idx, 1]], "-x", color="red")
    plt.show()

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(len(points)))
    nx_graph.add_edges_from(g.get_edgelist())

    # Plot the NetworkX graph
    pos = dict(enumerate(points))  # Use the points as positions for the nodes
    nx.draw(nx_graph, pos, with_labels=False, node_size=10, font_size=8, font_color='black', alpha=0.5)
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='.', label='Points')  # Plot the points
    plt.legend()
    plt.show()
    plt.close("all")

def make_energy_graph(pos, N, double_edges = True, no_edges = True):
    graph = igraph.Graph.Full(N)

    edges = np.array(graph.get_edgelist())
    if(not no_edges):
        if(double_edges):
            senders = np.concatenate( [edges[:,0], edges[:,1]],axis = -1)
            receivers = np.concatenate( [edges[:,1], edges[:,0]],axis = -1)
        else:
            senders = edges[:,0]
            receivers = edges[:,1]
        distances =  np.sqrt(np.sum((pos[senders,:] - pos[receivers,:])**2, axis = -1))
        distances = np.expand_dims(distances, axis = -1)
    else:
        senders = np.zeros((0,))
        receivers = np.zeros((0,))
        distances = np.zeros((0,1))

    energy_graph = jraph.GraphsTuple(nodes = np.zeros((N,1)), senders = senders, receivers= receivers, edges = distances, n_edge=np.array([len(senders)]), n_node = np.array([N]), globals = None)
    return energy_graph


def compute_distance(pos):
    pos_mat = np.sqrt(np.sum((pos[:,np.newaxis,:] - pos[np.newaxis,:,:])**2, axis = -1))
    return pos_mat

def translate_to_matrix(pos, tour, N = 100, f = 100):

    x_mat = np.zeros((N,N))

    for idx, el in enumerate(tour):
        x_mat[el, idx] = 1

    A = np.sqrt(2)
    Obj1 = A* np.sum((1-np.sum(x_mat, axis = 0))**2, axis = 0)
    Obj2 = A* np.sum((1-np.sum(x_mat, axis = 1))**2, axis = 0)

    idxs = np.arange(0, N)
    idxs_p1 = (idxs+1)%(N)
    pos_mat = compute_distance(pos)
    Obj3_2 = np.sum(pos_mat[:,:]* np.sum(x_mat[:,np.newaxis, idxs]*x_mat[np.newaxis,:, idxs_p1], axis = -1))

    distances = []
    for i in idxs:
        for k in range(x_mat.shape[0]):
            for kk in range(x_mat.shape[1]):
                res = x_mat[k, i] * x_mat[kk, (i+1)%N]
                if(res == 1):
                    distances.append(pos_mat[k,kk])
    print(distances)


    print("normal",Obj1, Obj2, Obj3_2)
    import jraph
    import igraph
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    graph = igraph.Graph.Full(N)

    edges = np.array(graph.get_edgelist())
    senders = np.concatenate( [edges[:,0], edges[:,1]],axis = -1)
    receivers = np.concatenate( [edges[:,1], edges[:,0]],axis = -1)
    distances =  np.sqrt(np.sum((pos[senders,:] - pos[receivers,:])**2, axis = -1))
    distances = np.expand_dims(distances, axis = -1)

    j_graph = jraph.GraphsTuple(nodes = x_mat, senders = senders, receivers= receivers, edges = distances, n_edge=np.array([len(senders)]), n_node = np.array([N]), globals = None)


    H_graph = jraph.batch_np([j_graph, j_graph])

    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = np.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = np.repeat(graph_idx, n_node, axis=0)

    x_mat = H_graph.nodes
    Obj1_per_graph_per_feature = (1 - jraph.segment_sum(x_mat, node_gr_idx, n_graph))**2
    Obj1_per_graph = np.sum(Obj1_per_graph_per_feature, axis = -1)
    Obj2_per_node = (1- np.sum(x_mat, axis = -1))**2
    Obj2_per_graph = jraph.segment_sum(Obj2_per_node, node_gr_idx, n_graph)

    Obj1 = np.mean(Obj1_per_graph)
    Obj2 = np.mean(Obj2_per_graph)

    edge_features = H_graph.edges
    receivers = H_graph.receivers
    senders = H_graph.senders
    tour_messages =  np.sum(x_mat[senders[:,np.newaxis], idxs[np.newaxis,:]]*x_mat[receivers[:,np.newaxis], idxs_p1[np.newaxis,:]], axis = -1)
    tour_messages = np.expand_dims(tour_messages, axis = -1)

    Obj3_per_node = jraph.segment_sum(edge_features* tour_messages,  H_graph.receivers, sum_n_node)
    Obj3_per_graph = jraph.segment_sum(Obj3_per_node, node_gr_idx, n_graph)

    ### TODO switch off graph padding in training code in this simple case
    # Obj3 = 0
    # for i in range(N):
    #     for j in range(N):
    #
    #         Obj3 += pos_mat[i,j] * np.sum(x_mat[i, idxs] * x_mat[j, idxs_p1], axis=-1)
    #
    # print(Obj1, Obj2, Obj3/f, Obj3_2/f)
    print("graphwise", Obj1_per_graph, Obj2_per_graph, Obj3_per_graph)
    return Obj3_2/f
