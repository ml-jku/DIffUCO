import sys
sys.path.append("..")
import numpy as np
import itertools
import random
import igraph as ig
from collections import Counter
from .RB_graphs import generate_xu_instances
from tqdm import tqdm
from .save_utils import save_indexed_dict
from DatasetCreator.GreedyAlgorithms import GreedyMIS
from DatasetCreator.UnetUtils.create_graphs import create_U_net

def generate_instance(n, k, r, p):
    '''
    n: number of cliques
    k: number of nodes in each clique
    a: log(k)/log(n)
    s: in each sampling iteration, the number of edges to be added
    iterations: how many iteration to sample
    return: the single-directed edges in numpy array form
    '''
    a = np.log(k) / np.log(n)
    v = k * n
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)
    parts = np.reshape(np.int64(range(v)), (n, k))
    nand_clauses = []

    for i in parts:
        nand_clauses += itertools.combinations(i, 2)
    edges = set()
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges
        edges |= set(random.sample(tuple(all), k=min(s, len(all))))

    nand_clauses += list(edges)
    clauses = np.array(nand_clauses)

    ordered_edge_list =[ (min([edge[0], edge[1]]), max([edge[0], edge[1]])) for edge in nand_clauses]

    # edges = nand_clauses
    # print(Counter(edges).keys())
    # print(Counter(edges).values())
    # print(len(Counter(edges)), len(edges))
    #
    # edges = ordered_edge_list
    # print(Counter(edges).keys())
    # print(Counter(edges).values())
    # print(len(Counter(edges)), len(edges))
    return Counter(ordered_edge_list)


def combinations(z):
    result = []
    for x in range(2, int(z ** 0.5) + 1):
        y = z // x
        if x * y == z:
            result.append((x, y))
            result.append((y,x))
    return result

def generateRB(mode = "val", n_train_graphs = 2000,seed = 123, RB_size = "200", parent = True, EnergyFunction = "MVC",save = False, solve = True, take_p = None, n_val_graphs = 500, val_time_limit = float("inf")):
    import pickle
    from unipath import Path
    import os
    from DatasetCreator.Gurobi import GurobiSolver
    from DatasetCreator.jraph_utils import utils as jutils

    if(take_p == None):
        dataset_name = f"RB_iid_{RB_size}"
    else:
        dataset_name = f"RB_iid_{RB_size}_p_{take_p}"

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    if(mode == "val"):
        seed_int = 5
    elif(mode == "test"):
        seed_int = 4
    else:
        seed_int = 0

    np.random.seed(seed + seed_int)

    if(mode == "val"):
        n_graphs = n_val_graphs
    elif(mode == "train"):
        n_graphs = n_train_graphs
    else:
        if(take_p == None):
            n_graphs = 500
        else:
            n_graphs = 100


    solutions = {}
    solutions["Energies"] = []
    solutions["H_graphs"] = []
    solutions["U_net_graph_dict"] = []
    solutions["compl_H_graphs"] = []
    solutions["gs_bins"] = []
    solutions["graph_sizes"] = []
    solutions["densities"] = []
    solutions["runtimes"] = []
    solutions["upperBoundEnergies"] = []
    solutions["p"] = []

    print(dataset_name, "is currently solved with gurobi")
    for i in tqdm(range(n_graphs)):
        while(True):
            if(RB_size == "200"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(20, 25)
                k = np.random.randint(9, 10)
            elif(RB_size == "500"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(30, 35)
                k = np.random.randint(15, 20)
            elif(RB_size == "1000"):
                solve = False
                min_n, max_n = 0, np.inf
                n = np.random.randint(60, 70)
                k = np.random.randint(15, 20)
            elif(RB_size == "2000"):
                solve = False
                min_n, max_n = 0, np.inf
                n = np.random.randint(120, 140)
                k = np.random.randint(15, 20)
            elif(RB_size == "100"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(9, 15)
                k = np.random.randint(8,11)
            elif (RB_size == "100_dummy"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(9, 15)
                k = np.random.randint(8, 11)
            elif(RB_size == "very_small"):
                min_n, max_n = 0, np.inf
                n = np.random.randint(5,10)
                k = np.random.randint(5,10)
            elif RB_size == "small":
                min_n, max_n = 200, 300
                n = np.random.randint(20, 25)
                k = np.random.randint(5, 12)
            elif RB_size == "large":
                solve = False
                min_n, max_n = 800, 1200
                n = np.random.randint(40, 55)
                k = np.random.randint(20, 25)
            if(RB_size == "small" or RB_size == "large"):
                if(take_p == None):
                    p = np.random.uniform(0.3, 1.0)
                else:
                    p = take_p
            else:
                if (take_p == None):
                    p = np.random.uniform(0.25, 1.0)
                else:
                    p = take_p

            if(mode == "train"):
                time_limit = 0.1
            elif(RB_size != "small" or RB_size != "large" or RB_size != "1000" or RB_size != "2000"):
                time_limit = val_time_limit
            else:
                time_limit = 0.1

            edges = generate_xu_instances.get_random_instance(n, k, p)
            g = ig.Graph([(edge[0], edge[1]) for edge in edges])
            isolated_nodes = [v.index for v in g.vs if v.degree() == 0]
            g.delete_vertices(isolated_nodes)
            num_nodes = g.vcount()
            if min_n <= num_nodes <= max_n:
                print(RB_size, "with", num_nodes, "nodes accepted", "timelimit", time_limit)
                break

        H_graph = jutils.from_igraph_to_jgraph(g)

        if(solve):
            if(EnergyFunction == "MVC"):
                _, Energy, solution, runtime = GurobiSolver.solveMVC_as_MIP(H_graph, time_limit=time_limit)

            elif(EnergyFunction == "MIS"):
                _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph, time_limit=time_limit)
                print("solution", p, "Model ", -n, "Gurobi" , Energy)
                H_graph_compl = jutils.from_igraph_to_jgraph(g, double_edges=False)
                solutions["compl_H_graphs"].append(H_graph_compl)


            elif(EnergyFunction == "MaxCl"):
                H_graph_compl = jutils.from_igraph_to_jgraph(g.complementer(loops=False), double_edges = False)
                _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph_compl, time_limit=time_limit)
                print("MaxCL size", Energy)
                solutions["compl_H_graphs"].append(H_graph_compl)
            elif(EnergyFunction == "MaxCut"):

                _, Energy, boundEnergy, solution, runtime = GurobiSolver.solveMaxCut(H_graph, time_limit=time_limit, bnb = False, verbose=False)

                solutions["upperBoundEnergies"].append(boundEnergy)
        else:
            if(EnergyFunction == "MIS"):
               Energy = -n
               solution = np.ones_like(H_graph.nodes)
               runtime = None
               H_graph_compl = jutils.from_igraph_to_jgraph(g, double_edges=False)
               solutions["compl_H_graphs"].append(H_graph_compl)
            elif(EnergyFunction == "MaxCl"):
                Energy = -n
                solution = np.ones_like(H_graph.nodes)
                runtime = None
                H_graph_compl = jutils.from_igraph_to_jgraph(g.complementer(loops=False), double_edges = False)
                solutions["compl_H_graphs"].append(H_graph_compl)
            elif(EnergyFunction == "MIS"):
               raise ValueError("not implemented")
            else:
                ValueError("Other Energy Functions that are not solved with gurobi are not implmented yet")




        solutions["Energies"].append(Energy)
        solutions["gs_bins"].append(solution)
        solutions["H_graphs"].append(H_graph)
        solutions["graph_sizes"].append(g.vcount())
        solutions["densities"].append(2*g.ecount()/(g.vcount()*(g.vcount()-1)))
        solutions["runtimes"].append(runtime)
        solutions["p"].append(p)

        indexed_solution_dict = {}
        for key in solutions.keys():
            if(len(solutions[key]) > 0):
                indexed_solution_dict[key] = solutions[key][i]

        save_indexed_dict(path=path, mode = mode, dataset_name=dataset_name, i=i, EnergyFunction=EnergyFunction,
                                     seed=seed, indexed_solution_dict=indexed_solution_dict)

    print(f"average {EnergyFunction} size", np.mean(np.array(Energy)))
    if(save):
        newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
        pickle.dump(solutions, open(save_path, "wb"))

    return solutions["densities"], solutions["graph_sizes"]


def plot_graphs():
    ### TODO update this if needed
    from matplotlib import pyplot as plt
    modes = [ "val", "test", "train"]
    sizes = ["200", "500"]
    size2 = "very_small"
    plt.figure()
    for size in sizes:
        plt.figure()
        plt.title(size)
        for mode in modes:
            density, graph_sizes = generateRB(RB_size = size, mode = mode, solve = False)

            plt.plot(density, graph_sizes, "x",label = mode)
            # generateRB(RB_size = "200", mode = mode)
            # generateRB(RB_size = "500", mode = mode)
        plt.xlabel("density")
        plt.ylabel("num nodes")
        plt.legend()
        plt.show()

def create_and_solve_graphs(dataset_name , parent = True, EnergyFunction = "MIS", modes = [ "test", "train","val"], seeds = [123], diff_ps = False):
    if("small" in dataset_name):
        ### Settings from EGN-ANneal
        sizes = ["small"]
        if("MaxCl" in EnergyFunction or "MIS" in EnergyFunction):
            n_train_graphs = 4000
        else:
            n_train_graphs = 2000
    elif ("large" in dataset_name):
        sizes = ["large"]
        n_train_graphs = 4000
    elif("200" in dataset_name):
        sizes = ["200"]
        n_train_graphs = 2000
    elif("100" in dataset_name):
        sizes = ["100"]
        n_train_graphs = 3000
        if("dummy" in dataset_name):
            n_train_graphs = 100#
            sizes = ["100_dummy"]
    elif ("huge" in dataset_name):
        sizes = ["1000"]
        n_train_graphs = 3000
    elif("giant" in dataset_name):
        sizes = ["2000"]
        n_train_graphs = 3000

    for seed in seeds:
        for size in sizes:
            for mode in modes:
                if(mode == "test"):
                    if(diff_ps == "No"):
                        curr_p_list = [None]
                    else:
                        curr_p_list = np.linspace(0.25, 1, num=10)
                else:
                    curr_p_list = [None]

                for curr_p in curr_p_list:
                    generateRB(RB_size = size, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, n_train_graphs=n_train_graphs, take_p = curr_p)

def create_and_solve_graphs_MVC(parent = True, seeds = [123], sizes = ["200"], modes = ["test", "train", "val"]):
    EnergyFunction = "MVC"
    sizes = sizes
    for seed in seeds:
        for size in sizes:
            for mode in modes:

                if(mode != "test"):
                    curr_p_list = [None]
                else:
                    curr_p_list = np.linspace(0.25, 1, num=10)

                for curr_p in curr_p_list:

                    generateRB(RB_size = size, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, take_p = curr_p)

def create_and_solve_graphs_MaxCl(parent = True, seeds = [123], sizes = ["small"], modes = ["test", "train", "val"]):
    EnergyFunction = "MaxCl"
    sizes = sizes
    for seed in seeds:
        for size in sizes:
            for mode in modes:

                if(mode != "test"):
                    curr_p_list = [None]
                else:
                    curr_p_list = np.linspace(0.25, 1, num=10)

                for curr_p in curr_p_list:

                    generateRB(RB_size = size, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, take_p = curr_p)

def plot():
    for p in np.linspace(0.7, 1., 10):
        n = np.random.randint(5, 10)
        k = np.random.randint(5, 10)

        edges = generate_xu_instances.get_random_instance(n, k, p)

        import networkx as nx
        from matplotlib import pyplot as plt
        G = nx.Graph()
        G.add_edges_from(edges)
        nx.draw(G, node_size=30)
        plt.show()

def plot2(solution, H_graph):
    import networkx as nx
    import matplotlib.pyplot as plt

    plt.title(f"p {p}")
    G = nx.Graph()

    # Add nodes with attributes (0 or 1)
    node_attributes = {idx: num for idx, num in enumerate(solution)}
    G.add_nodes_from(node_attributes.keys())

    edges = [(s,r) for s,r in zip(H_graph.senders, H_graph.receivers)]
    G.add_edges_from(edges)

    # Define colors based on node attributes
    node_colors = ['red' if node_attributes[node] == 1 else 'blue' for node in G.nodes]

    # Draw the graph
    pos = nx.spring_layout(G)  # You can use other layout algorithms as well
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10,
            font_color='white')
    # Show the plot
    plt.show()

if(__name__ == "__main__"):
    create_and_solve_graphs_MVC(modes = ["test"])
    #create_and_solve_graphs_MaxCut()
    pass
    #plot_graphs()
    #create_and_solve_graphs_MVC()

