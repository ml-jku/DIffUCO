from jraph_utils import utils as jutils
from Gurobi import GurobiSolver

import os
import numpy as np
from .DatasetGenerator import Generator
from omegaconf import OmegaConf
from unipath import Path
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from loadGraphDatasets.loadTwitterGraph import TWITTER
import time

def solve_and_save(dataset_name ,mode = "val", EnergyFunction = "MaxCut", parent = True, seed = None, time_limit=float("inf"), norm = False):

    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p
    print(path, os.getcwd())
    cfg = {}
    cfg["Ising_params"] = {}
    cfg["Ising_params"]["IsingMode"] = dataset_name
    cfg["Ising_params"]["shuffle_seed"] = seed
    cfg["Ising_params"]["n_rand_nodes"] = 0
    cfg["Ising_params"]["EnergyFunction"] = EnergyFunction

    if(dataset_name != "TWITTER"):
        gen = Generator(cfg)
        ### TODO test stuff here
        if(mode == "val"):
            dataset = gen.pyg_val_dataset
        elif(mode == "train"):
            dataset = gen.pyg_train_dataset
        elif (mode == "test"):
            dataset = gen.pyg_test_dataset
        else:
            ValueError("Dataset mode does not exist")

        jraph_val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                      collate_fn=gen.collate_from_torch_to_jraph_val_and_test, num_workers=0)
    else:
        dataset = TWITTER(cfg, mode = mode, seed = seed)
        jraph_val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                      collate_fn=lambda x: jutils.collate_jraphs_to_max_size(x,0), num_workers=0)
    # subfolders = [f.path for f in os.scandir(p) if f.is_dir()]
    # print(subfolders)
    solution_dict = {}
    solution_dict["Energies"] = []
    solution_dict["H_graphs"] = []
    solution_dict["gs_bins"] = []
    solution_dict["runtimes"] = []

    print(dataset_name)
    print("Dataset size is", len(jraph_val_loader), "mode =", mode)
    print(EnergyFunction)
    print("time limit", time_limit)
    for (j_graph, _) in tqdm(jraph_val_loader, total = len(jraph_val_loader)):
        if(EnergyFunction == "MaxCut"):
            model, energy, solution, runtime = GurobiSolver.solveMaxCut(j_graph, bnb = False, time_limit = time_limit)
        elif(EnergyFunction == "MIS"):
            model, energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(j_graph)
            #print(energy, runtime)
        elif(EnergyFunction == "MVC"):
            start = time.time()
            model, MIS_energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(j_graph)
            model, energy, solution, runtime = GurobiSolver.solveMVC_as_MIP(j_graph, time_limit = time_limit)

            print("MVC", energy, "MIS", MIS_energy, "complement", j_graph.nodes.shape[0] + MIS_energy)
            end = time.time()
            #print("t2", end-start, energy)
            # start = time.time()
            # model, energy, solution, runtime = GurobiSolver.solveMVC_as_MIP(j_graph, time_limit = float("inf"))
            # end = time.time()
            # print("t2  inf", end-start, energy)
        elif(EnergyFunction == "MaxCl"):
            igraph = jutils.from_jgraph_to_igraph(j_graph)
            j_graph = jutils.from_igraph_to_jgraph(igraph)
            compl_jraph = jutils.from_igraph_to_jgraph(igraph.complementer(loops = False))

            model, compl_energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(compl_jraph, time_limit=time_limit)


            # model, energy, solution, runtime = GurobiSolver.solveMaxClique_log(j_graph, bnb=False, time_limit=time_limit)
            #
            # print("Energy")
            # print(energy, compl_energy)
            # print("next")
            energy = compl_energy
            if(j_graph.nodes.shape[0] != solution.shape[0]):
                print(22)
                print(j_graph.nodes.shape[0], solution.shape[0])
                print("here")


        else:
            ValueError("Energy function not specified")

        # H_graph = sympy_utils.getHamiltonianGraph(j_graph,EnergyFunction=EnergyFunction)
        # bins = solution
        # spins = 2*bins -1
        # spins = np.expand_dims(spins, axis=-1)
        # Energy_from_graph = jraphEnergy.compute_Energy_full_graph(H_graph, spins)

        if(norm):
            energy = energy * j_graph.edges[0,0]

        #print("Energy", energy, Energy_from_graph)
        solution_dict["Energies"].append(energy)
        solution_dict["H_graphs"].append(j_graph)
        solution_dict["gs_bins"].append(solution)
        solution_dict["runtimes"].append(runtime)
        ### TODO add normalisation constant to solution dict

    mean_Energy = np.mean(np.array(solution_dict["Energies"]))
    print("mean_Energy", mean_Energy)

    ### TODO make sure this is saved as np jraph tuple
    if(norm):
        save_path = path + f"/loadGraphDatasets/DatasetSolutions/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
    else:
        newpath = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        save_path = path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{dataset_name}/{mode}_{EnergyFunction}_seed_{seed}_solutions.pickle"
    pickle.dump(solution_dict, open(save_path, "wb"))

def load_solution_dataset(dataset_name = "TWITTER", mode = "test", EnergyFunction = "MIS", seed = 0):
    batch_size = 1
    import jax
    jax.config.update('jax_platform_name', 'cpu')

    p = Path(os.getcwd())
    path = p.parent
    cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")
    cfg["Ising_params"]["IsingMode"] = dataset_name
    cfg["Ising_params"]["EnergyFunction"] = EnergyFunction
    cfg["Ising_params"]["shuffle_seed"] = -1


    dataset = JraphSolutionDataset(cfg, mode,seed = seed)


    gen = Generator(cfg, num_workers = 0)
    jraph_val_loader = DataLoader(dataset, batch_size=batch_size,
                                       collate_fn=jutils.collate_jraphs_to_max_size, num_workers=0)

    DBG_rel_error_list = []
    AG_rel_error_list = []
    RG_rel_error_list = []
    AG_RG_rel_error_list = []
    for (graphs, graph_list) in tqdm(jraph_val_loader, total=len(jraph_val_loader)):
        Energies = graphs.globals[0]
        unnormed_Energies = np.round(graphs.globals*gen.normalisation_factor)[0]
        DBG_pred_Energy = GreedyMIS.solveMIS(graphs)
        # RG_pred_Energy = GreedyMaxCut.random_greedy(graphs, EnergyFunction=EnergyFunction)
        # AG_pred_Energy, AR_spins = GreedyMaxCut.AutoregressiveGreedy(graphs, EnergyFunction=EnergyFunction)
        # AG_RG_pred_Energy = GreedyMaxCut.random_greedy(graphs, spins = AR_spins, iter_fraction = 1., EnergyFunction=EnergyFunction)


        DBG_rel_error = (np.abs(unnormed_Energies-DBG_pred_Energy))/np.abs(unnormed_Energies)
        DBG_rel_error_list.append(DBG_rel_error)

        # AG_rel_error = (np.abs(Energies-AG_pred_Energy))/np.abs(Energies)
        # AG_rel_error_list.append(AG_rel_error)
        #
        # RG_rel_error = (np.abs(Energies-RG_pred_Energy))/np.abs(Energies)
        # RG_rel_error_list.append(RG_rel_error)
        #
        # AG_RG_rel_error = (np.abs(Energies-AG_RG_pred_Energy))/np.abs(Energies)
        # AG_RG_rel_error_list.append(AG_RG_rel_error)

        print("DBG", DBG_rel_error)
        # print("AG", AG_rel_error)
        # print("RG", RG_rel_error)
        # print("AG RG", AG_RG_rel_error)

    rel_error_arr = np.array(DBG_rel_error_list)
    mean_rel_error = np.mean(rel_error_arr)
    print(dataset_name, mode, "AR Degree Greedy", EnergyFunction , ":", 1-mean_rel_error)

    rel_error_arr = np.array(RG_rel_error_list)
    mean_rel_error = np.mean(rel_error_arr)
    print(dataset_name, mode, "AR Random Greedy", EnergyFunction ,":", 1-mean_rel_error)

    rel_error_arr = np.array(AG_rel_error_list)
    mean_rel_error = np.mean(rel_error_arr)
    print(dataset_name, mode, "AR Autoregressive Greedy", EnergyFunction ,":", 1-mean_rel_error)

    rel_error_arr = np.array(AG_RG_rel_error_list)
    mean_rel_error = np.mean(rel_error_arr)
    print(dataset_name, mode, "AR Autoregressive Greedy then Random Greedy", EnergyFunction ,":", 1-mean_rel_error)


def GreedySolve(dataset_name ,mode = "val", EnergyFunction = "MIS", parent = True, seed = None):
    import time
    p = Path(os.getcwd())
    if(parent):
        path = p.parent
    else:
        path = p
    print(path, os.getcwd())
    cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")
    cfg["Ising_params"]["IsingMode"] = dataset_name
    cfg["Ising_params"]["shuffle_seed"] = seed
    cfg["Ising_params"]["n_rand_nodes"] = 0
    cfg["Ising_params"]["EnergyFunction"] = EnergyFunction

    if(dataset_name != "TWITTER"):
        gen = Generator(cfg)
        ### TODO test stuff here
        if(mode == "val"):
            dataset = gen.pyg_val_dataset
        elif(mode == "train"):
            dataset = gen.pyg_train_dataset
        elif (mode == "test"):
            dataset = gen.pyg_test_dataset
        else:
            ValueError("Dataset mode does not exist")

        jraph_val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                      collate_fn=gen.collate_from_torch_to_jraph_val_and_test, num_workers=0)
    else:
        dataset = TWITTER(cfg, mode = mode, seed = seed )
        jraph_val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                      collate_fn=lambda x: jutils.collate_jraphs_to_max_size(x,0), num_workers=0)
    # subfolders = [f.path for f in os.scandir(p) if f.is_dir()]
    # print(subfolders)
    Energy_list = []
    AR_list = []
    time_list = []
    print(dataset_name)
    print("Dataset size is", len(jraph_val_loader), "mode =", mode)
    print(EnergyFunction)
    for (j_graph, _) in tqdm(jraph_val_loader, total = len(jraph_val_loader)):
        start = time.time()
        if(EnergyFunction == "MIS"):
            DBG_pred_Energy = GreedyMIS.solveMIS(j_graph)
            Energy_list.append(DBG_pred_Energy)
            model, gt_energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(j_graph)
        if (EnergyFunction == "MVC"):
            DBG_pred_Energy = j_graph.nodes.shape[0] + GreedyMIS.solveMIS(j_graph)
            Energy_list.append(DBG_pred_Energy)
            model, gt_energy, solution, runtime = GurobiSolver.solveMVC_as_MIP(j_graph)
        if (EnergyFunction == "MaxCl"):
            igraph = jutils.from_jgraph_to_igraph(j_graph)
            compl_jraph = jutils.from_igraph_to_jgraph(igraph.complementer(loops = False))

            model, gt_energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(compl_jraph)

            DBG_pred_Energy = GreedyMIS.solveMIS(compl_jraph)

            if(igraph.complementer(loops = False).vcount() == 0):
                print("here")

            Energy_list.append(DBG_pred_Energy)

        end = time.time()
        AR_list.append(DBG_pred_Energy/gt_energy)
        time_list.append(end-start)

    mean_AR = np.mean(np.array(AR_list))
    print("mean AR", mean_AR)
    mean_time = np.mean(np.array(time_list))
    return mean_AR, mean_time

def solve_greedy(parent = True):
    import jax
    jax.config.update('jax_platform_name', 'cpu')

    ### TODO solve dataset for three different shuffle seeds
    dataset_names = ["ENZYMES", "IMDB-BINARY"]
    #dataset_names = ["COLLAB", "TWITTER", "IMDB-BINARY"]
    # dataset_names = ["COLLAB"]
    modes = ["test"]
    # modes = ["train"]
    EnergyFunctions = ["MaxCl"]
    ### TODO overwrite test set for seeds 1,2
    seeds = [123, 124, 125]
    # seeds = [123]


    for dataset_name in dataset_names:
        ARS = []
        for EnergyFunction in EnergyFunctions:
            for seed in seeds:
                for mode in modes:
                    mean_AR, mean_time = GreedySolve( dataset_name=dataset_name, mode=mode, EnergyFunction=EnergyFunction,
                                   parent=parent, seed=seed)
                    ARS.append(mean_AR)

        ARS = np.array(ARS)
        print("mean AR over seeds", dataset_name, np.mean(ARS), np.std(ARS)/np.sqrt(len(seeds)))
        print("mean time", mean_time)

def solve_datasets(dataset, problem, parent = True, seeds = [123]):
    import jax
    jax.config.update('jax_platform_name', 'cpu')

    dataset_names = [dataset]
    #dataset_names = ["COLLAB"]
    modes = ["train", "val", "test"]
    #modes = ["train"]
    EnergyFunctions = [problem]

    for seed in seeds:
        for EnergyFunction in EnergyFunctions:
            for dataset_name in dataset_names:
                for mode in modes:
                        if(mode == "train"):
                            time_limit = 0.05
                        else:
                            time_limit = 3000
                        solve_and_save(norm=False, dataset_name=dataset_name, mode = mode, EnergyFunction = EnergyFunction, parent = parent, seed = seed, time_limit=time_limit) #100000

if(__name__ == "__main__"):
    solve_greedy()
    #solve_datasets()

    # modes = ["val"]
    # #dataset_names = ["ENZYMES", "PROTEINS", "MUTAG", "IMDB-BINARY", "COLLAB", "RRG_num_nodes_=_100_k_=20", "RRG_num_nodes_=_100_k_=10"]
    # dataset_names = ["TWITTER"]
    # for dataset_name in dataset_names:
    #     for mode in modes:
    #             load_solution_dataset(dataset_name=dataset_name, mode = mode)
    # pass

