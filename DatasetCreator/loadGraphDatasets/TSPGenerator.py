import sys
sys.path.append("..")
from tqdm import tqdm
from .save_utils import save_indexed_dict
import igraph
import numpy as np
from scipy.spatial import distance
import networkx as nx
from matplotlib import pyplot as plt
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
import jraph
### TODO write code that generates and saves Instances
import sys, os
import pickle
from unipath import Path
import os
from DatasetCreator.jraph_utils import utils as jutils

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def generate_TSP_Instances(dataset_name = f"2D_random_100", EnergyFunction = "TSP", mode = "train", parent = False, seed = 123, n_val_graphs = 1000, n_train_graphs = 40000, save = False):


    if("100" in dataset_name):
        size = 100
    elif ("20" in dataset_name):
        size = 20


    p = Path(os.getcwd())
    if (parent):
        path = p.parent
    else:
        path = p

    if (mode == "val"):
        seed_int = 5
        solve = True
    elif (mode == "test"):
        seed_int = 4
        solve = True
    else:
        seed_int = 0
        solve = False

    np.random.seed(seed + seed_int)

    if (mode == "val"):
        n_graphs = n_val_graphs
    elif (mode == "train"):
        n_graphs = n_train_graphs
    else:
        n_graphs = 10000

    solutions = {}
    solutions["Energies"] = []
    solutions["H_graphs"] = []
    solutions["compl_H_graphs"] = []
    solutions["gs_bins"] = []
    solutions["graph_sizes"] = []
    solutions["densities"] = []
    solutions["runtimes"] = []
    solutions["upperBoundEnergies"] = []
    solutions["p"] = []
    solutions["compl_H_graphs"] = []

    print(dataset_name, "is currently solved with gurobi")
    for i in tqdm(range(n_graphs)):

        ### TODO solve here
        blockPrint()
        xs, opt_tour = generate_and_solve(size, solve = solve)
        enablePrint()
        ### TODO make H_graph here
        opt_value = calc_optimal_value_from_tour(opt_tour,xs)
        x_mat_opt = create_opt_matrix(opt_tour, xs, opt_value)
        g = create_jraph(xs)
        H_graph = jutils.from_igraph_to_jgraph(g, zero_edges = False)

        H_graph = H_graph._replace(nodes = xs)

        energy_graph = make_energy_graph(xs, size)

        compute_energy(energy_graph, x_mat_opt)
        ### TODO add energy graph

        #plotting(opt_tour,g, xs)
        solutions["compl_H_graphs"].append(energy_graph)
        solutions["Energies"].append(opt_value)
        solutions["gs_bins"].append(x_mat_opt)
        solutions["H_graphs"].append(H_graph)
        solutions["graph_sizes"].append(g.vcount())
        solutions["densities"].append(2 * g.ecount() / (g.vcount() * (g.vcount() - 1)))
        solutions["runtimes"].append(0.)
        solutions["p"].append(p)

        indexed_solution_dict = {}
        for key in solutions.keys():
            if (len(solutions[key]) > 0):
                indexed_solution_dict[key] = solutions[key][i]

        save_indexed_dict(path=path, mode=mode, dataset_name=dataset_name, i=i, EnergyFunction=EnergyFunction,
                          seed=seed, indexed_solution_dict=indexed_solution_dict)

    if (save):
        newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
        pickle.dump(solutions, open(save_path, "wb"))

    return solutions["densities"], solutions["graph_sizes"]

def compute_energy(H_graph, X_mat, A = 1.5):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_platform_name', 'cpu')
    # X_mat = np.concatenate([X_mat, X_mat], axis = 0)
    # X_mat = np.random.randint(0,1, size = X_mat.shape)
    print("n_edges ", H_graph.edges.shape)
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = np.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = np.repeat(graph_idx, n_node, axis=0)

    x_mat = X_mat

    N = X_mat.shape[-1]
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
    print(Energy_per_graph)
    return Energy_per_graph, None

def generate_and_solve(N , f = 1000, solve = False):
    xs =  np.random.uniform(0, 1, size=(N, 2))
    ys = 1000 *xs
    if(solve):
        solver = TSPSolver.from_data(ys[:, 0], ys[:, 1], 'EUC_2D')
        solution = solver.solve()
        tour = solution.tour
    else:
        tour = np.arange(0,N)

    return xs, tour

def calc_optimal_value_from_tour(opt_tour, xs):
    my_tour_length = 0
    for i, el in enumerate(opt_tour):
        idx = opt_tour[i]
        next_idx = opt_tour[(i + 1) % (len(opt_tour))]
        dsit = np.sqrt(np.sum((xs[idx] - xs[next_idx]) ** 2))
        my_tour_length += dsit

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
    Obj3= np.sum(pos_mat[:,:]* np.sum(x_mat[:,np.newaxis, idxs]*x_mat[np.newaxis,:, idxs_p1], axis = -1))

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

def make_energy_graph(pos, N, double_edges = True):
    graph = igraph.Graph.Full(N)

    edges = np.array(graph.get_edgelist())
    if(double_edges):
        senders = np.concatenate( [edges[:,0], edges[:,1]],axis = -1)
        receivers = np.concatenate( [edges[:,1], edges[:,0]],axis = -1)
    else:
        senders = edges[:,0]
        receivers = edges[:,1]
    distances =  np.sqrt(np.sum((pos[senders,:] - pos[receivers,:])**2, axis = -1))
    distances = np.expand_dims(distances, axis = -1)

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


if(__name__ == "__main__"):

    from matplotlib import pyplot as plt
    import concorde
    from concorde.tsp import TSPSolver
    from concorde.tests.data_utils import get_dataset_path

    n_instances = 2
    solutions = []
    for instance in range(n_instances):
        N = 100
        f = 1000
        xs = np.round(1000*np.random.uniform(0,1, size = (N,2)), 0)

        solver = TSPSolver.from_data(xs[:,0], xs[:,1], 'EUC_2D')
        solution = solver.solve()

        my_tour_length = 0.

        distances = []

        #print("mean", np.mean(solutions), np.std(solutions)/np.sqrt(len(solutions)))

        if(True):
            plt.figure()
            plt.title(str(solution.optimal_value/f))
            for i, el in enumerate(solution.tour):
                idx = solution.tour[i]
                next_idx = solution.tour[(i + 1)%(len(solution.tour))]
                dsit = np.sqrt(np.sum((xs[idx] - xs[next_idx])**2))
                my_tour_length += dsit
                distances.append(dsit)
                plt.plot([xs[idx, 0] , xs[next_idx, 0]] , [xs[idx, 1] , xs[next_idx, 1]], "-x", color = "red")
            plt.show()
            print("pcconcorde tour lenght ", solution.optimal_value)
            print("my_tour_length",my_tour_length)
            translate_to_matrix(xs, solution.tour, N = N, f= f)

        opt_value = translate_to_matrix(xs, solution.tour, N=N, f=f)
        solutions.append(opt_value)
        test(xs)

