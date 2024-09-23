import sys
sys.path.append("..")
import numpy as np
import igraph as ig
from tqdm import tqdm
from .save_utils import save_indexed_dict
import networkx as nx
import pickle
from unipath import Path
from DatasetCreator.Gurobi import GurobiSolver
from DatasetCreator.jraph_utils import utils as jutils
import os


Gset_cut_sizes = {"1": 11624, "2":11620, "3": 11622, "4":11646, "5":11631, "6":2178, "7":2006, "8":2005, "9":2054, "10":2000, "11":564, "12":556, "13":582, "14":3064, "15":3050, "16":3052, "17":3047, "18":992, "19":906, "20":941,
                  "21":931, "22":13359, "23":13344, "24":13337, "25":13340,"26": 13328, "27":3341, "28":3298, "29":3405, "30":3413, "31":3310, "32":1410, "33":1382, "34":1384, "35":7687, "36":7680, "37":7691, "38":7688, "39":2408, "40":2400,
                  "41":2405, "42":2481, "43":6660, "44":6650, "45":6654, "46":6649, "47":6657, "48":6000, "49":6000, "50":5880, "51":3848, "52":3851, "53":3850, "54":3852, "55":10299, "56":4017, "57":3494, "58":19293, "59":6086, "60":14188, "61":5796,
                  "62":4870, "63":27045, "64":8751, "65":5562, "66":6364, "67":6950, "70":9591, "72": 7006}

def generateER(mode = "val", n_train_graphs = 16000,seed = 123, m = 4, dataset_name = "BA_small", parent = True, EnergyFunction = "MaxCut", save = False, n_val_graphs = 500, val_time_limit = float("inf")):

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
    np.random.seed(seed+seed_int)

    if(mode == "val"):
        n_graphs = n_val_graphs
    elif(mode == "train"):
        n_graphs = n_train_graphs
    elif(mode == "test"):
        n_graphs = n_val_graphs

    if (mode == "train"):
        time_limit = 0.1
    elif (mode == "val"):

        time_limit = 1.
    elif(mode == "test"):
        time_limit = 1.

    if(mode == "train"):
        generate_train_set(dataset_name, n_graphs, time_limit, mode, path,EnergyFunction, seed, N = 100)
    elif(mode == "val"):
        generate_train_set(dataset_name, 200, time_limit, mode, path,EnergyFunction, seed, N = 500)
    elif(mode == "test"):
        directory_in_str = path + "/loadGraphDatasets/GsetData/GSET"

        directory = os.fsencode(directory_in_str)

        occurance_dict = {}
        for idx, file in enumerate(os.listdir(directory)):
            file_num = str(file)[3:5]
            if(np.all([ el.isdigit() for el in file_num])):
                file_num = int(file_num)
            else:
                file_num = int(file_num[0])

            print(file_num, file, len(Gset_cut_sizes), len(os.listdir(directory)))
            filename = os.fsdecode(file)
            H_graph, g = load_mc_file(str(directory_in_str) + "/" +filename, mode)
            _, Energy, boundEnergy, solution, runtime, MC_value = GurobiSolver.solveMaxCut(H_graph,
                                                                                           time_limit=time_limit,
                                                                                           bnb=False, verbose=False)
            solutions = {}
            solutions["upperBoundEnergies"] = boundEnergy

            solutions["Energies"] = Gset_cut_sizes[str(file_num)]
            solutions["gs_bins"] = solution
            solutions["H_graphs"] = H_graph
            solutions["graph_sizes"] = g.vcount()
            solutions["densities"] = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
            solutions["runtimes"] = runtime
            solutions["MCValue"] = Gset_cut_sizes[str(file_num)]
            file_num_key = str(file_num)

            print("Gurobi MC", MC_value, "vs best value", Gset_cut_sizes[str(file_num)])
            n_nodes =  H_graph.nodes.shape[0]
            if(n_nodes >= 3000):
                n_nodes = 3000

            if n_nodes not in occurance_dict:
                occurance_dict[n_nodes] = 0
            else:
                occurance_dict[n_nodes] += 1

            dataset_name_N = dataset_name + f"_{n_nodes}"

            print("occurances", occurance_dict)
            save_indexed_dict(path = path, mode = mode, dataset_name=dataset_name_N, i=occurance_dict[n_nodes], EnergyFunction=EnergyFunction, seed=seed, indexed_solution_dict=solutions)


def load_mc_file(path, mode):
    print(path)
    if(mode == "test"):
        gnx = load_mc(path)
    else:
        gnx = load_mtx(path)
    print("nx", len(gnx.nodes()), len(gnx.edges()))
    g = ig.Graph()
    g.add_vertices(np.arange(0, len(gnx.nodes())))
    g.add_edges(gnx.edges())
    #g = ig.Graph.TupleList(gnx.edges(), directed=False)
    # n = np.random.randint(n_min, high = n_max)
    # g = ig.Graph.Barabasi(n, m)
    H_graph = jutils.from_igraph_to_jgraph(g)
    return H_graph, g

def load_mc(path):
    with open(path, 'r') as f:
        g = nx.Graph()

        nodes_saved = False
        for line in f:
            s = line.split()#
            if(nodes_saved):
                g.add_edge(int(s[0]) - 1, int(s[1]) - 1)
            elif len(s) == 3 and np.all([ el.isdigit() for el in s]):
                g.add_nodes_from(range(int(s[0])))
                nodes_saved = True
            #elif (len(s) == 2 and np.all([ el.isdigit() for el in s])) or nodes_saved:
    return g

def load_mtx(path):
    with open(path, 'r') as f:
        g = nx.Graph()
        weights = []
        first_line = True
        for line in f:
            if not line[0] == '%':
                s = line.split()
                if first_line:
                    g.add_nodes_from(range(int(s[0])))
                    first_line = False
                else:
                    g.add_edge(int(s[0]) - 1, int(s[1]) - 1)
                    if len(s) > 2:
                        weights.append(int(s[2]))

    if len(weights) < g.number_of_edges():
        weights = None
    else:
        weights = np.int64(weights)
    return g

def generate_train_set(dataset_name, n_graphs, time_limit, mode, path,EnergyFunction, seed, save = False, N = 100):
    solutions = {}
    solutions["Energies"] = []
    solutions["H_graphs"] = []
    solutions["gs_bins"] = []
    solutions["graph_sizes"] = []
    solutions["densities"] = []
    solutions["runtimes"] = []
    solutions["upperBoundEnergies"] = []
    solutions["compl_H_graphs"] = []

    MC_list = []

    MC_value_list = []
    print(dataset_name, "is currently solved with gurobi")
    for idx in tqdm(range(n_graphs)):

        cur_n = np.random.randint(200,501)
        p = np.random.uniform(0.05, 0.3)

        gnx = nx.erdos_renyi_graph(cur_n, p, seed=None, directed=False)

        g = ig.Graph.TupleList(gnx.edges(), directed=False)
        # n = np.random.randint(n_min, high = n_max)
        # g = ig.Graph.Barabasi(n, m)
        H_graph = jutils.from_igraph_to_jgraph(g)


        _, Energy, boundEnergy, solution, runtime, MC_value = GurobiSolver.solveMaxCut(H_graph, time_limit=time_limit, bnb = False, verbose=False)
        MC_list.append(MC_value)
        print("mean MC value", cur_n,np.mean(np.array(MC_list)), np.std(np.array(MC_list)) / np.sqrt(len(MC_list)))


        solutions["upperBoundEnergies"].append(boundEnergy)

        solutions["Energies"].append(Energy)
        solutions["gs_bins"].append(solution)
        solutions["H_graphs"].append(H_graph)
        solutions["graph_sizes"].append(g.vcount())
        solutions["densities"].append(2*g.ecount()/(g.vcount()*(g.vcount()-1)))
        solutions["runtimes"].append(runtime)
        #MC_value_list.append(MC_value)
        print("current mean_E",mode, np.mean(np.array(solutions["Energies"])),
              np.std(np.array(solutions["Energies"])) / np.sqrt(len(solutions["Energies"])), len(solutions["Energies"]))

        indexed_solution_dict = {}
        for key in solutions.keys():
            if(len(solutions[key]) > 0):
                indexed_solution_dict[key] = solutions[key][idx]

        save_indexed_dict(path = path, mode = mode, dataset_name=dataset_name, i=idx, EnergyFunction=EnergyFunction, seed=seed, indexed_solution_dict=indexed_solution_dict)


    print("mean_E", np.mean(np.array(solutions["Energies"])), np.std(np.array(solutions["Energies"]))/ np.sqrt(len(solutions["Energies"])), len(solutions["Energies"]))

    if(save):
        newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
        pickle.dump(solutions, open(save_path, "wb"))

    return np.mean(np.array(solutions["Energies"])),solutions["densities"], solutions["graph_sizes"]



def create_and_solve_graphs(dataset_name = "Gset",parent = True, seeds = [123, 124, 125], EnergyFunction = "MaxCut", modes =  [ "test","train", "val"]):

    for seed in seeds:
        for mode in modes:
            generateER(dataset_name = dataset_name, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent)



if(__name__ == "__main__"):
    import os
    import socket
    hostname = socket.gethostname()
    print(os.environ["GRB_LICENSE_FILE"])
    os.environ["GRB_LICENSE_FILE"] = f"/system/user/sanokows/gurobi_{hostname}.lic"

    #create_and_solve_graphs_MaxCut()
    pass
    #plot_graphs()
    #create_and_solve_graphs_MVC()



