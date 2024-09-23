import igraph as ig
from jraph_utils import utils as jutils
from Gurobi import GurobiSolver
import numpy as np
import pickle
from unipath import Path
import os
from GreedyAlgorithms import GreedyMIS

def generate_regular_graph(num_nodes, k):
    regular_graph = ig.Graph.K_Regular(num_nodes, k)
    return regular_graph

def do_bfs(i_graph):
    vc = i_graph.vcount()
    idx = np.random.randint(0, high = vc)
    res = i_graph.bfs(idx)
    order = res[0]
    return order

def do_dfs(i_graph):
    vc = i_graph.vcount()
    idx = np.random.randint(0, high = vc)
    res = i_graph.dfs(idx)
    order = res[0]
    return order


### TODO make datast out of that
def generate_and_solve_regular_graphs(n_graphs = 100, num_nodes = 100, node_range = [80,120], k = 10, EnergyFunction = "MIS", mode = "val", seed = 0, parent = True):

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p


    dataset_name = f"RRG_{num_nodes}_k_={k}"
    solution_dict = {}
    solution_dict["Energies"] = []
    solution_dict["H_graphs"] = []
    solution_dict["gs_bins"] = []
    solution_dict["runtimes"] = []

    save_path = path + f"/loadGraphDatasets/DatasetSolutions/{dataset_name}_{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
    pickle.dump(solution_dict, open(save_path, "wb"))

    if(mode == "val"):
        seed_int = 5
    elif(mode == "test"):
        seed_int = 4
    else:
        seed_int = 0

    np.random.seed(seed + seed_int)

    if(mode == "train"):
        time_limit = 0.01
    elif(mode == "val"):
            time_limit = float("inf")
    else:
            time_limit = float("inf")


    newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for idx in range(n_graphs):
        print("solve Graph", idx, "of", n_graphs)
        if(k == "all"):
            ks = [3, 7, 10, 20]
            if(mode == "test"):
                k_index = idx % len(ks)
                curr_k = ks[k_index]
                curr_num_nodes = np.random.choice(node_range)
                #curr_num_nodes = num_nodes
            else:
                curr_k = np.random.choice(ks)
                curr_num_nodes = np.random.choice(node_range)
                #curr_num_nodes = num_nodes
        else:
            curr_num_nodes = num_nodes
            curr_k = k

        print(f"k = {curr_k} num_nodes = {curr_num_nodes}")


        regular_graph = generate_regular_graph(curr_num_nodes, curr_k)

        j_graph = jutils.igraph_to_jraph(regular_graph, np_ = np)


        if(EnergyFunction == "MaxCut"):
            model, energy, solution, runtime = GurobiSolver.solveMaxCut(j_graph, bnb = False, time_limit=3000)
        elif(EnergyFunction == "MIS"):
            model, energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(j_graph, time_limit = time_limit)
        else:
            ValueError("Energy function not specified")
        print(EnergyFunction)
        print("Energy", energy)
        solution_dict["Energies"].append(energy)
        solution_dict["H_graphs"].append(j_graph)
        solution_dict["gs_bins"].append(solution)
        solution_dict["runtimes"].append(runtime)

        print("save solution dict")
        save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
        pickle.dump(solution_dict, open(save_path, "wb"))

def load_and_greedy_solve(num_nodes = 100, ks = ["all"], EnergyFunction = "MIS", mode = "test", seed = 123, parent = True):
    from matplotlib import pyplot as plt
    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    res_dict = {}

    for k in ks:
        dataset_name = f"RRG_{num_nodes}_k_={k}"
        load_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
        with open(load_path, "rb") as f:
            solution_dict = pickle.load(f)

        print("num_graphs", len(solution_dict["H_graphs"]))

        for idx, (H_graph, Energy) in enumerate(zip(solution_dict["H_graphs"], solution_dict["Energies"])):
            n_edge = H_graph.edges.shape[0]
            n_node = H_graph.nodes.shape[0]
            pred_Energy = GreedyMIS.solveMIS(H_graph)
            gt_Energy = Energy
            rel_err = np.abs(gt_Energy-pred_Energy)/np.abs(gt_Energy)
            print(idx,"rel_error", rel_err)

            k = int(np.round(((n_edge)/n_node)))
            if(k in res_dict.keys()):
                res_dict[k].append(rel_err)
            else:
                res_dict[k] = [rel_err]

    print("keys",res_dict.keys())
    k_list = [int(k) for k in res_dict.keys()]
    rel_error = [np.mean(np.array(res_dict[k])) for k in res_dict]

    plt.figure()
    plt.title(str(num_nodes))
    plt.plot(k_list, rel_error, "x")
    plt.show()

    return k_list, rel_error

def make_dataset(dataset_name,parent = True, seeds = [123], modes = ["val","test", "train"]):
    for seed in seeds:
        if("RRG_100" in dataset_name):

            num_node_list = [100]
            for num_nodes in num_node_list:
                    for mode in modes:
                        if(mode == "train"):
                            n_graphs = 3000
                        elif(mode == "val"):
                            n_graphs = 100
                        else:
                            n_graphs = 1000

                        n_range = int(0.1 * num_nodes)
                        node_range = num_nodes - 2 * np.arange(-n_range, n_range)
                        generate_and_solve_regular_graphs(n_graphs = n_graphs, num_nodes=num_nodes, node_range = node_range, k = "all", mode = mode, parent = parent, seed = seed)
        elif("RRG_150" in dataset_name):
            num_node_list = [150]
            for num_nodes in num_node_list:
                    for mode in modes:

                        if(mode == "train"):
                            n_graphs = 3000
                        elif(mode == "val"):
                            n_graphs = 100
                        else:
                            n_graphs = 1000

                        n_range = int(0.1 * num_nodes)
                        node_range = num_nodes - 2 * np.arange(-n_range, n_range)
                        generate_and_solve_regular_graphs(n_graphs = n_graphs, num_nodes=num_nodes, node_range = node_range, k = "all", mode = mode, parent = parent, seed = seed)
        elif("RRG_200" in dataset_name):
            num_node_list = [200]
            for num_nodes in num_node_list:
                    for mode in modes:

                        if(mode == "train"):
                            n_graphs = 3000
                        elif(mode == "val"):
                            n_graphs = 100
                        else:
                            n_graphs = 100

                        if(mode == "test"):
                            n_range = int(0.1*num_nodes)
                            node_range = [num_nodes + int(2*n_range)]
                            generate_and_solve_regular_graphs(n_graphs = n_graphs, num_nodes=num_nodes, node_range = node_range, k = "all", mode = mode, parent = parent, seed = seed)
                        else:
                            n_range = int(0.1*num_nodes)
                            node_range = num_nodes - 2 * np.arange(-n_range, n_range)
                            generate_and_solve_regular_graphs(n_graphs = n_graphs, num_nodes=num_nodes, node_range = node_range, k = "all", mode = mode, parent = parent, seed = seed)


def generate_only_specific_ks(parent = True):
    ks = [3,7,10,20]
    modes = ["test"]
    num_node_list = [100]
    for num_nodes in num_node_list:
        for k in ks:
            for mode in modes:
                n_graphs = 100

                generate_and_solve_regular_graphs(n_graphs = n_graphs, num_nodes=num_nodes, k = k, mode = mode, parent = parent, seed = 123)

if(__name__ == "__main__"):
    load_and_greedy_solve(num_nodes = 100, ks = ["all"], mode = "test")
    load_and_greedy_solve(num_nodes = 150, ks = ["all"], mode = "test")
    #make_dataset()

