from utils import sympy_utils, SympyHamiltonians
from EnergyFunctions import jraphEnergy
from omegaconf import OmegaConf
from unipath import Path
import os
import pickle
import numpy as np
import time
from jraph_utils import utils as jutils
from GlobalProjectVariables import MVC_A, MVC_B, MaxCl_B
from GreedyAlgorithms import GreedyGeneral, FeasibleGreedy
from tqdm import tqdm

def load_solution_dict(dataset_name, EnergyFunction,seed = 0, mode = "val", parent = False):

    if(EnergyFunction == "MaxCl_compl" or EnergyFunction == "MaxCl_EGN"):
        loadEnergyFunction = "MaxCl"
    else:
        loadEnergyFunction = EnergyFunction

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p

    save_path = str(path)

    path = os.path.join(save_path, "loadGraphDatasets", "DatasetSolutions", "no_norm",
                        dataset_name, f"{mode}_{loadEnergyFunction}_seed_{seed}_solutions.pickle")

    file = open(path, "rb")
    solution_dict = pickle.load(file)
    return solution_dict, save_path

def norm_graph(H_graph, mean_Energy, std_Energy, self_loops = True):
    num_nodes = H_graph.nodes.shape[0]
    nodes = H_graph.nodes/std_Energy
    if(self_loops):
        couplings = H_graph.edges[:-num_nodes]/std_Energy

        self_connections = (H_graph.edges[-num_nodes:]-mean_Energy/num_nodes)/std_Energy
        edges = np.concatenate([couplings,self_connections], axis = 0)
        normed_H_graph = H_graph._replace(nodes = nodes, edges = edges)
    else:
        edges = H_graph.edges/ std_Energy
        self_connections = None
        couplings = None
        n_edges = np.array([edges.shape[0]])
        normed_H_graph = H_graph._replace(nodes=nodes, edges=edges, n_edge = n_edges)

    return normed_H_graph, self_connections, couplings

def remove_double_edges(H_graph, self_loops = False):
    num_nodes = H_graph.nodes.shape[0]

    if(not self_loops):
        edges = H_graph.edges[:-2*num_nodes]
        senders = H_graph.senders[:-2*num_nodes]
        receivers = H_graph.receivers[:-2*num_nodes]

        n_edge = np.array([edges.shape[0]])
        H_graph = H_graph._replace(edges=edges, senders = senders, receivers = receivers, n_edge = n_edge)

    igraph = jutils.from_jgraph_to_dir_igraph_normed(H_graph)
    H_graph = jutils.from_igraph_to_dir_jgraph_normed(igraph)

    return H_graph


def calc_dEnergies(H_graph,spins):
    num_nodes = H_graph.nodes.shape[0]
    mask = np.zeros((H_graph.nodes.shape[0],1))

    Energy_list = []
    for i in range(num_nodes):
        mask[i,0] = 1.
        curr_spins = mask*spins
        Energy = jraphEnergy.compute_Energy_full_graph_np(H_graph, curr_spins, half_edges=True)

        if(i != 0):
            dEnergy = Energy - np.sum(Energy_list)
        else:
            dEnergy = Energy

        Energy_list.append(dEnergy)

    d_Energy = np.array(Energy_list)
    print("overall Energy", np.sum(d_Energy, axis = 0))


def load(dataset_name, train_dataset_name, EnergyFunction, mode = "val", seed = 0, parent = False, self_loops = False):
    from unipath import Path
    ### TODO load one graph after another

    solution_dict, save_path = load_solution_dict(dataset_name, EnergyFunction, mode = mode, seed = seed, parent = parent)
    train_solution_dict, _ = load_solution_dict(train_dataset_name, EnergyFunction, mode = "train", seed = seed, parent = parent)

    print(EnergyFunction)
    print("train",calculate_mean_and_std(train_solution_dict["Energies"]))
    run_path = os.getcwd()

    if(not self_loops):
        half_edges = True
        folder = "no_norm_H_graph_sparse"
    else:
        half_edges = True
        folder = "normed_H_graph_sparse"

    path_list = [save_path, "loadGraphDatasets", "DatasetSolutions", folder,dataset_name,f"{mode}_{EnergyFunction}_seed_{seed}_solutions"]
    for path_el in path_list:
        run_path = os.path.join(run_path, path_el)
        if not os.path.exists(run_path):
            os.mkdir(run_path)

    solution_dict["spin_graph"] = []
    solution_dict["original_Energies"] = []
    solution_dict["self_loop_Energies"] = []

    print("graphs are going to be translated to spin formulation")
    zip_data = list(zip(solution_dict["Energies"], solution_dict["H_graphs"], solution_dict["gs_bins"]))
    for idx, (Energy, j_graph, bins) in enumerate(tqdm(zip_data, total = len(zip_data))):

        solution_dict["original_Energies"].append(Energy)

        if(EnergyFunction == "MaxCl" or EnergyFunction == "MaxCl_compl"):
            i_graph = jutils.from_jgraph_to_igraph(j_graph)
            Compl_igraph = i_graph.complementer(loops = False)

            dir_Compl_jgraph = jutils.from_igraph_to_dir_jgraph(Compl_igraph)
            H_graph = SympyHamiltonians.MIS_sparse(dir_Compl_jgraph, B = MaxCl_B)
            print("prin edges of graphs ",dir_Compl_jgraph.n_edge, H_graph.n_edge)
        elif(EnergyFunction == "MVC"):
            s2 = time.time()
            H_graph = SympyHamiltonians.MVC_sparse(j_graph)
            e2 = time.time()
            #print("MVC", e2-s2)
        elif(EnergyFunction == "MIS"):
            s2 = time.time()
            H_graph = SympyHamiltonians.MIS_sparse(j_graph)
            e2 = time.time()
            #print("MIS", e2-s2)
        elif(EnergyFunction == "WMIS"):
            s2 = time.time()
            H_graph = SympyHamiltonians.WMIS_sparse(j_graph)
            e2 = time.time()
            #print("WMIS", e2-s2)
        elif(EnergyFunction == "MaxCut"):
            s2 = time.time()
            H_graph = SympyHamiltonians.MaxCut(j_graph)
            e2 = time.time()
            spins = 2 * bins - 1
            spins = np.expand_dims(spins, axis=-1)
            Energy_from_graph = jraphEnergy.compute_Energy_full_graph_np(H_graph, spins)
            print("Energy from Gurobi", Energy, "Energy from Graph", np.squeeze(Energy_from_graph))
            ### TEST max Cut
            #print("MaxCut", e2-s2)
        else:
            raise ValueError("No such EnergyFunction is implemented")

        spins = 2*bins -1
        spins = np.expand_dims(spins, axis=-1)

        H_graph = jutils.cast_Tuple_to_float32(H_graph, np_=np)
        if(not self_loops):
            print("self loops previous Energy", solution_dict["Energies"][idx], "new energy", jraphEnergy.compute_Energy_full_graph_np(H_graph, spins))
            self_loop_Energy = jraphEnergy.compute_self_loop_Energy(H_graph, spins)
            print("Energy from self loops", self_loop_Energy)
            new_H_graph = remove_double_edges(H_graph, self_loops = self_loops)
            H_graph = new_H_graph
            Energy_from_graph = jraphEnergy.compute_Energy_full_graph_np(H_graph, spins, half_edges=half_edges)
            print("previous Energy", solution_dict["Energies"][idx], "new energy", Energy_from_graph + self_loop_Energy)
            solution_dict["self_loop_Energies"].append(self_loop_Energy)
            solution_dict["Energies"][idx] = float(Energy_from_graph)
        else:
            print("self loops previous Energy", solution_dict["Energies"][idx], "new energy", jraphEnergy.compute_Energy_full_graph_np(H_graph, spins))
            jutils.check_number_of_edge_occurances(H_graph, num_occ=2)
            new_H_graph = remove_double_edges(H_graph, self_loops = self_loops)
            H_graph = new_H_graph
            Energy_from_graph = jraphEnergy.compute_Energy_full_graph_np(H_graph, spins, half_edges=half_edges)
            print("previous Energy", solution_dict["Energies"][idx], "new energy", Energy_from_graph)
            solution_dict["self_loop_Energies"].append(0.)
        jutils.check_number_of_edge_occurances(H_graph, num_occ = 1)
        solution_dict["spin_graph"].append(H_graph)

    new_solution_dict = {}
    new_solution_dict["gs_bins"] = []
    new_solution_dict["normed_igraph"] = []
    new_solution_dict["normed_Energies"] = []
    new_solution_dict["orig_igraph"] = []
    new_solution_dict["self_loop_Energies"] = solution_dict["self_loop_Energies"]

    if("p" in solution_dict.keys()):
        new_solution_dict["p"] = solution_dict["p"]

    ### TODO always calc this on train set
    if(self_loops):
        if(mode == "train"):

            if(EnergyFunction == "MaxCl" or EnergyFunction == "MaxCl_compl" or EnergyFunction == "WMIS"):
                iter_fraction = 1
            else:
                iter_fraction = 1

            mean_greedy_Energy, std_greedy_Energy = calculate_greedy_mean_and_std(solution_dict["spin_graph"], solution_dict["Energies"], EnergyFunction, iter_fraction=iter_fraction, self_loops=self_loops)
        else:
            path = os.path.join(save_path, "loadGraphDatasets", "DatasetSolutions", folder,
                                train_dataset_name, f"train_{EnergyFunction}_seed_{seed}_solutions.pickle")

            file = open(path, "rb")
            train_sol_dict = pickle.load(file)
            mean_greedy_Energy = train_sol_dict["val_mean_Energy"]
            std_greedy_Energy = train_sol_dict["val_std_Energy"]
    else:
        mean_greedy_Energy = 0.
        std_greedy_Energy = 1.

    new_solution_dict["val_mean_Energy"] = mean_greedy_Energy
    new_solution_dict["val_std_Energy"] = std_greedy_Energy
    new_solution_dict["original_Energies"] = solution_dict["original_Energies"]

    print("Energy scale is going to be standartized")
    zip_data = zip(solution_dict["Energies"], solution_dict["spin_graph"], solution_dict["H_graphs"], solution_dict["gs_bins"], solution_dict["original_Energies"],new_solution_dict["self_loop_Energies"])
    for idx, (Energy, H_graph, orig_graph, bins, original_Energy, self_loop_Energy) in enumerate(tqdm(zip_data, total=len(solution_dict["Energies"]))):

        spins = 2*bins -1
        spins = np.expand_dims(spins, axis = -1)

        normed_H_graph, self_connections, couplings = norm_graph(H_graph, mean_greedy_Energy, std_greedy_Energy, self_loops = self_loops)
        normed_Energy = (Energy - mean_greedy_Energy)/std_greedy_Energy
        normed_ig = jutils.from_jgraph_to_igraph_normed(normed_H_graph)
        #normed_H_graph = jutils.from_igraph_to_normed_jgraph(normed_ig)
        calc_dEnergies(normed_H_graph, spins)
        # calc_dEnergies(H_graph, spins)
        #print(jraphEnergy.compute_Energy_full_graph(H_graph, spins))
        print("normed Energy", normed_Energy, jraphEnergy.compute_Energy_full_graph_np(normed_H_graph, spins, half_edges = half_edges), normed_H_graph.nodes.dtype)

        orig_ig = None



        new_solution_dict["normed_igraph"].append(normed_ig)
        new_solution_dict["normed_Energies"].append(normed_Energy)
        new_solution_dict["gs_bins"].append(bins)
        new_solution_dict["orig_igraph"].append(orig_ig)

        graph_dict = {}
        graph_dict["orig_igraph"] = orig_ig
        graph_dict["compl_igraph"] = orig_ig
        graph_dict["normed_igraph"] = normed_ig
        graph_dict["normed_Energy"] = normed_Energy
        graph_dict["gs_bins"] = bins
        graph_dict["val_mean_Energy"] = mean_greedy_Energy
        graph_dict["val_std_Energy"] = std_greedy_Energy
        graph_dict["n_graphs"] = len(solution_dict["Energies"])
        graph_dict["original_Energy"] = original_Energy
        graph_dict["self_loop_Energy"] = self_loop_Energy

        if ("p" in solution_dict.keys()):
            graph_dict["p"] = solution_dict["p"][idx]

        path = os.path.join(save_path, "loadGraphDatasets", "DatasetSolutions", folder,
                            dataset_name, f"{mode}_{EnergyFunction}_seed_{seed}_solutions", f"_idx_{idx}.pickle")

        file = open(path, "wb")

        pickle.dump(graph_dict, file)

    #path = os.path.join(save_path, "loadGraphDatasets", "DatasetSolutions", folder,
    #                    dataset_name, f"{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle")


    file = open(path, "wb")
    pickle.dump(new_solution_dict, file)

def calculate_mean_and_std(Energy_list):
    Energy_arr = np.array(Energy_list)
    #Energy_arr = np.round(Energy_arr)

    mean_energy = np.mean(Energy_arr)
    std_energy = np.std(Energy_arr)

    if(std_energy < 10**-10):
        std_energy = 1

    return mean_energy, std_energy

def calculate_greedy_mean_and_std(H_graphs, gs_Energies, EnergyFunction, norm_mode = "random_greedy", iter_fraction = 1 , self_loops = True):
    greedy_Energies = []
    print("graphs are solved with RGA")
    zip_data = list(zip(H_graphs, gs_Energies))
    for H_graph, gs_Energy in tqdm(zip_data, total=len(zip_data)):
        if(norm_mode == "autoregressive"):
            Energy, spins = GreedyGeneral.AutoregressiveGreedy(H_graph )
        elif("random_greedy"):
            # Energy, spins = GreedyGeneral.random_greedy(H_graph , iter_fraction= 1)
            # print("frac 1", Energy)
            ### TODO map to feasible solution!
            Energy, spins = GreedyGeneral.random_greedy(H_graph , iter_fraction= iter_fraction)
            # Energy_bins, HA, HB, spins = FeasibleGreedy.check_for_violations(H_graph, spins, EnergyFunction)
            # Energy = jraphEnergy.compute_Energy_full_graph_np(H_graph, spins)
            #print("frac ",iter_fraction,  Energy)
        elif("random"):
            Energy, spins = GreedyGeneral.random(H_graph)
        greedy_Energies.append(Energy)
        #print("greedy Energy", Energy, "vs gs Energy", gs_Energy, H_graph.nodes.shape[0], H_graph.edges.shape[0])
        if(Energy < gs_Energy):
            ValueError("gs Energy is larger than greedy Energy")

    greedy_Energies = np.array(greedy_Energies)
    mean_greedy_Energy = np.mean(greedy_Energies)
    if(self_loops):
        std_greedy_Energy = np.std(greedy_Energies)
    else:
        std_greedy_Energy = np.min(greedy_Energies)
        mean_greedy_Energy = 0.*mean_greedy_Energy
    print(f"{norm_mode} Energy", mean_greedy_Energy, std_greedy_Energy)
    print((gs_Energies - mean_greedy_Energy)/std_greedy_Energy )
    return mean_greedy_Energy, std_greedy_Energy


def make_RB_test_graphs():

    modes = ["test"]
    EnergyFunctions = ["MVC"]

    seeds = [124, 125]
    p_list = np.linspace(0.25, 1, num = 10)
    for p in p_list:
        dataset_name = f"RB_iid_small_p_{p}"
        train_dataset_name = f"RB_iid_small"
        for seed in seeds:
                for mode in modes:
                    for EnergyFunction in EnergyFunctions:
                        print("finished", dataset_name, seed)
                        #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
                        load(dataset_name, train_dataset_name, EnergyFunction, mode = mode, seed = seed , self_loops = True, parent = True)
    pass


def solve(dataset, problem, seeds = [123], parent = False, modes = ["train", "val","test"], self_loops = True):
    dataset_names = [dataset]
    # modes = ["test"]
    # EnergyFunctions = ["MVC"]
    EnergyFunctions = [problem]

    for seed in seeds:
        for dataset_name in dataset_names:
            for mode in modes:
                for EnergyFunction in EnergyFunctions:
                    print("The following data is translated to spin formulation:", dataset_name, seed, mode, EnergyFunction)
                    # compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )

                    if("RB" in dataset_name and mode == "test"):
                        p_list = np.linspace(0.25, 1, num=10)
                        for p in p_list:
                            dataset_name_p = f"{dataset_name}_p_{p}"
                            train_dataset_name = f"{dataset_name}"

                            load(dataset_name_p, train_dataset_name, EnergyFunction, mode=mode, seed=seed, parent=parent,
                                 self_loops=self_loops)
                            load(dataset_name_p, train_dataset_name, EnergyFunction, mode=mode, seed=seed, parent=parent,
                                 self_loops=False)
                    else:
                        load(dataset_name, dataset_name, EnergyFunction, mode=mode, seed=seed, parent = parent, self_loops=self_loops)
                        load(dataset_name, dataset_name, EnergyFunction, mode=mode, seed=seed, parent=parent, self_loops=False)

def overwirte_all_data_to_float32():
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    datasets = ["RB_iid_200"]
    problems = ["MVC"]
    # datasets = ["RB_iid_small"]
    # problems = ["MIS"]

    for dataset, problem in zip(datasets, problems):
        solve(dataset, problem, parent = True)


if(__name__ == "__main__"):
    ### TODO remove self loops in the future --> less memory requirements
    #make_RB_test_graphs()
    solve("RB_iid_200", "MVC" , parent = True, modes = ["test"], seeds=[123])
    #make_RB_test_graphs()
    #overwirte_all_data_to_float32()
    if(False):
        #dataset_names = ["ExtToyMIS_[50, 100]_[2, 10]_0.7_0.8", "ToyMIS_13_3_10_harder"]
        import jax

        jax.config.update('jax_platform_name', 'cpu')
        modes = ["train", "val", "test"]
        dataset_names = ["TWITTER"]
        #modes = ["test"]
        #EnergyFunctions = ["MVC"]
        EnergyFunctions = ["MaxCl_compl"]
        self_loops = True

        seeds = [123]
        for seed in seeds:
            for dataset_name in dataset_names:
                for mode in modes:
                    for EnergyFunction in EnergyFunctions:
                        print("finished", dataset_name, seed, mode, EnergyFunction)
                        #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
                        load(dataset_name, dataset_name, EnergyFunction, mode = mode, seed = seed , parent = True, self_loops=self_loops)


    # dataset_names = ["COLLAB","TWITTER", "IMDB-BINARY"]
    # modes = ["train","val", "test"]
    # #dataset_names = ["ToyMIS_13_3_5", "ToyMIS_13_3_3", "ToyMIS_13_3_10"]
    # #modes = ["val", "test"]
    # EnergyFunctions = ["MVC"]
    #
    # seeds = [124, 125]
    # for seed in seeds:
    #     for dataset_name in dataset_names:
    #         for mode in modes:
    #             for EnergyFunction in EnergyFunctions:
    #                 print("finished", dataset_name, seed)
    #                 #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
    #                 load(dataset_name, EnergyFunction, mode = mode, seed = seed )
    #
    # dataset_names = ["RB_iid_100", "RB_iid_200"]
    # modes = ["train","val", "test"]
    # #dataset_names = ["ToyMIS_13_3_5", "ToyMIS_13_3_3", "ToyMIS_13_3_10"]
    # #modes = ["val", "test"]
    # EnergyFunctions = ["MVC"]
    #
    # seeds = [123]
    # for seed in seeds:
    #     for dataset_name in dataset_names:
    #         for mode in modes:
    #             for EnergyFunction in EnergyFunctions:
    #                 print("finished", dataset_name, seed)
    #                 #compute_loading_time(dataset_name, EnergyFunction, mode = mode, seed = seed )
    #                 load(dataset_name, EnergyFunction, mode = mode, seed = seed )