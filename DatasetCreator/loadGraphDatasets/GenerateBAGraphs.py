import os
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

def generateRB(mode = "val", n_train_graphs = 4000,seed = 123, m = 4, dataset_name = "BA_small", parent = True, EnergyFunction = "MaxCut", save = False, n_val_graphs = 500, val_time_limit = float("inf")):

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
        if(EnergyFunction == "MaxCut"):
            time_limit = 1.
        elif(EnergyFunction == "MDS"):
            time_limit = float("inf")
    elif(mode == "test"):
        if(EnergyFunction == "MaxCut"):
            if("small" in dataset_name):
                time_limit = 60
            if ("large" in dataset_name):
                time_limit = 300
        elif(EnergyFunction == "MDS"):
            time_limit = float("inf")
        else:
            raise ValueError("err")

    if ("huge" in dataset_name):
        n_graphs = 100
    elif ("giant" in dataset_name):
        n_graphs = 100

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

        if ("small" in dataset_name):
            cur_n = np.random.randint(101) + 200
        elif ("large" in dataset_name):
            cur_n = np.random.randint(401) + 800
        elif ("huge" in dataset_name):
            time_limit = 300
            cur_n = np.random.randint(601) + 1200
        elif ("giant" in dataset_name):
            time_limit = 300
            cur_n = np.random.randint(1001) + 2000
        else:
            raise NotImplementedError

        gnx = nx.barabasi_albert_graph(n=cur_n, m=4)
        #gnx.remove_nodes_from(list(nx.isolates(gnx)))
        weight = {e: np.random.rand() for e in gnx.edges()}
        nx.set_edge_attributes(gnx, weight, "weight")
        e = list(gnx.edges(data=True))
        g = ig.Graph.TupleList(gnx.edges(), directed=False)
        # n = np.random.randint(n_min, high = n_max)
        # g = ig.Graph.Barabasi(n, m)
        H_graph = jutils.from_igraph_to_jgraph(g)

        if(EnergyFunction == "MaxCut"):
            _, Energy, boundEnergy, solution, runtime, MC_value = GurobiSolver.solveMaxCut(H_graph, time_limit=time_limit, bnb = False, verbose=False)
            MC_list.append(MC_value)
            print("mean MC value", np.mean(np.array(MC_list)), np.std(np.array(MC_list)) / np.sqrt(len(MC_list)))
        elif(EnergyFunction == "MDS"):

            model, Energy, solution, runtime = GurobiSolver.solveMDS_as_MIP(H_graph, time_limit=time_limit)
            boundEnergy = Energy
            print("MDS Value", Energy, time_limit)
        else:
            raise ValueError("EnergyFunction ", EnergyFunction, " is not implemented")

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



def create_and_solve_graphs(dataset_name = "BA_small",parent = True, seeds = [123, 124, 125], EnergyFunction = "MaxCut", modes =  [ "test","train", "val"]):

    for seed in seeds:
        for mode in modes:
            generateRB(dataset_name = dataset_name, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, n_val_graphs = 500)

def test_dataset(dataset_name = "BA_small",parent = True, seeds = [123], EnergyFunction = "MaxCut"):
    modes = [ "test"]
    Energy_list = []
    for seed in seeds:
        for mode in modes:
            meanEnergy, _, _ = generateRB(dataset_name = dataset_name, seed = seed, mode = mode, EnergyFunction=EnergyFunction, parent = parent, n_val_graphs = 2000)
            Energy_list.append(meanEnergy)

    print("mean Energy", np.mean(Energy_list), np.std(Energy_list)/np.sqrt(len(Energy_list)))

def load_solutions(parent = True, dataset_name = "BA_small", mode = "test", EnergyFunction = "MaxCut", seed = 123):
    import os
    from unipath import Path
    import pickle

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"

    with open(save_path, "rb") as f:
        res = pickle.load(f)

    Energies = np.array(res["Energies"])
    upper_Bound_Energies = res["upperBoundEnergies"]
    H_graphs = res["H_graphs"]
    gs_bins_per_graph = res["gs_bins"]

    n_nodes = np.array([H_graph.n_node[0] for H_graph in H_graphs])
    n_edges = np.array([H_graph.n_edge[0] for H_graph in H_graphs])

    MC_results_list = []
    for  H_graph, gs_bins in zip(H_graphs, gs_bins_per_graph):
        gs_spins = 2 * gs_bins - 1
        receivers = H_graph.receivers
        senders = H_graph.senders
        MC_result = 0
        for s, r in zip(senders, receivers):
            if (s != r):
                MC_result += (1-gs_spins[ s]*gs_spins[ r])/4
        MC_results_list.append(MC_result)

    MC_results_arr = np.array(MC_results_list)
    print("MC_results arr", np.mean(MC_results_arr))

    MC_value = n_edges/4 - Energies/2
    print("MC_value", np.mean(MC_value))
    print("finished")

if(__name__ == "__main__"):
    import os
    import socket
    hostname = socket.gethostname()
    print(os.environ["GRB_LICENSE_FILE"])
    os.environ["GRB_LICENSE_FILE"] = f"/system/user/sanokows/gurobi_{hostname}.lic"
    test_dataset()
    #create_and_solve_graphs_MaxCut()
    pass
    #plot_graphs()
    #create_and_solve_graphs_MVC()
