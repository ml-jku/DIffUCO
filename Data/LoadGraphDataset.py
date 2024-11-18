import os
import random

import jax
from torch.utils.data import Dataset
import pickle
import numpy as np
import igraph
import jraph
import torch
from torch.utils.data import DataLoader
from unipath import Path
import os
import jraph_utils
from playground.Clusters.Meluxina import data_path


class SolutionDatasetLoader:
    def __init__(self, config = {}, dataset="MIS", problem="MIS", batch_size=32, relaxed=False, seed=123, mode = "train"):
        self.dataset_name = dataset
        self.problem_name = problem
        self.batch_size = batch_size
        self.relaxed = relaxed
        self.seed = seed
        self.num_workers = max([self.batch_size,40])
        self.config = config
        self.mode = mode

        torch.manual_seed(self.seed)
        self._init_mode()

    def _init_mode(self):
        if(self.mode == "train"):
            self.train_dataset = True
            self.val_dataset = True
            self.test_dataset = False
        elif(self.mode == "val"):
            self.train_dataset = False
            self.val_dataset = True
            self.test_dataset = False
        else:
            self.train_dataset = False
            self.val_dataset = False
            self.test_dataset = True

    def pmap_collate(self, batch):
        #batch_transposed = list(zip(*batch))
        batch_dict = {key: [] for key in batch[0].keys()}

        for el in batch:
            for key in batch_dict.keys():
                batch_dict[key].append(el[key])

        # jraph_graphs = [el["input_graph"] for el in batch]
        # energy_graphs = [el["energy_graph"] for el in batch]
        # gt_normed_energies = [el["energies"] for el in batch]
        # gt_spin_states = [el["gs_bins"] for el in batch]
        # U_net_graphs_dict = [el["U_net_graphs_dict"] for el in batch]
        # #print(gt_spin_states)
        return batch_dict

    def dataloaders(self):


        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        TRAIN_DATASET = self.train_dataset
        TEST_DATASET = self.test_dataset
        VAL_DATASET = self.val_dataset

        dataset_train = SolutionDataset_InMemory(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="train", relaxed=self.relaxed, seed=self.seed) if TRAIN_DATASET else None
        dataset_test = SolutionDataset_InMemory(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="test", relaxed=self.relaxed, seed=self.seed) if TEST_DATASET else None
        dataset_val = SolutionDataset_InMemory(config = self.config, dataset=self.dataset_name, problem=self.problem_name, mode="val", relaxed=self.relaxed, seed=self.seed) if VAL_DATASET else None

        if(self.mode == "train"):
            mean_energy = dataset_train.val_mean_energy
            std_energy = dataset_train.val_std_energy
        elif(self.mode == "val"):
            mean_energy = dataset_val.val_mean_energy
            std_energy = dataset_val.val_std_energy
        else:
            mean_energy = dataset_test.val_mean_energy
            std_energy = dataset_test.val_std_energy

        collate_function = self.pmap_collate

        self.dataset_train = dataset_train
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, drop_last = True, collate_fn=collate_function, num_workers=self.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=generator) if TRAIN_DATASET else None
        self.dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, collate_fn=collate_function, num_workers=self.num_workers, worker_init_fn=seed_worker, generator=generator) if dataset_test != None else None
        self.dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, collate_fn=collate_function, num_workers=self.num_workers, worker_init_fn=seed_worker, generator=generator) if VAL_DATASET else None
        if(self.dataloader_train != None):
            self._compute_dataset_statistics(mode = "train")
        if(self.dataloader_val != None):
            self._compute_dataset_statistics(mode = "val")
        if(self.dataloader_test != None):
            self._compute_dataset_statistics(mode = "test")
        return self.dataloader_train, self.dataloader_test, self.dataloader_val, (mean_energy, std_energy)

    def _compute_dataset_statistics(self, mode = "train"):

        if(mode == "train"):
            current_dataloader = self.dataloader_train
        elif(mode == "val"):
            current_dataloader = self.dataloader_val
        else:
            current_dataloader = self.dataloader_test

        statistics_dict = {}
        statistics_dict["input_graph"] = {"n_edges": [], "n_nodes": []}
        statistics_dict["energy_graph"] = {"n_edges": [], "n_nodes": []}

        for batch_dict in current_dataloader:
            input_graph = batch_dict["input_graph"]
            energy_graph = batch_dict["energy_graph"]
            energy_graph_nodes = [int(el.n_node[0]) for el in energy_graph]
            input_graph_nodes = [int(el.n_node[0]) for el in input_graph]
            energy_graph_edges = [int(el.n_edge[0]) for el in energy_graph]
            input_graph_edges = [int(el.n_edge[0]) for el in input_graph]

            statistics_dict["input_graph"]["n_edges"].extend(input_graph_edges)
            statistics_dict["energy_graph"]["n_edges"].extend(energy_graph_edges)
            statistics_dict["input_graph"]["n_nodes"].extend(input_graph_nodes)
            statistics_dict["energy_graph"]["n_nodes"].extend(energy_graph_nodes)

        current_dataloader.smallest_n_edges_input_graph, current_dataloader.largest_n_edges_input_graph = get_x_smallest_and_largest(statistics_dict["input_graph"]["n_edges"], self.batch_size)
        current_dataloader.smallest_n_edges_energy_graph, current_dataloader.largest_n_edges_energy_graph = get_x_smallest_and_largest(statistics_dict["energy_graph"]["n_edges"], self.batch_size)
        current_dataloader.smallest_n_nodes_input_graph, current_dataloader.largest_n_nodes_input_graph = get_x_smallest_and_largest(statistics_dict["input_graph"]["n_nodes"], self.batch_size)
        current_dataloader.smallest_n_nodes_energy_graph, current_dataloader.largest_n_nodes_energy_graph = get_x_smallest_and_largest(statistics_dict["energy_graph"]["n_nodes"], self.batch_size)
        print("dataset statistics",mode, current_dataloader.smallest_n_edges_input_graph, current_dataloader.largest_n_edges_input_graph)




    def reinint_train_dataloader(self, epoch):

        def seed_worker(worker_id):
            worker_seed = (torch.initial_seed() + epoch) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.seed+ epoch)

        dataloader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, collate_fn= self.pmap_collate,
                                      num_workers=self.num_workers, shuffle=True, worker_init_fn=seed_worker,
                                      generator=generator)
        return dataloader_train


def return_jraph(i_graph: igraph.Graph):
    """
    Return current igraph as jraph
    The nodes are the external fields and the edges are the couplings
    """
    edges = np.array(i_graph.get_edgelist())
    n_edges = i_graph.ecount()
    if n_edges > 0:
        couplings = np.array(i_graph.es['couplings'])

        jraph_senders = edges[:, 0]
        jraph_receivers = edges[:, 1]

        external_fields = np.array(i_graph.vs['ext_fields'])
        jraph_graph = jraph.GraphsTuple(nodes=external_fields,
                                        edges=couplings,
                                        senders=jraph_senders,
                                        receivers=jraph_receivers,
                                        n_node=np.array([i_graph.vcount()]),
                                        n_edge=np.array([jraph_receivers.shape[0]]),
                                        globals=None)
    else:
        raise NotImplementedError("graph has no edges")
    #print("jraph", jraph_graph)
    return jraph_graph

class SolutionDataset(Dataset):
    def __init__(self, config = {}, dataset="ENZYMES", problem="MIS", mode="val", relaxed=False, seed=123):
        self.dataset_name = dataset
        self.problem_name = problem
        self.mode = mode
        self.seed = seed
        self.relaxed = relaxed

        print("here")
        print(os.path.exist("/mnt/proj2/dd-23-97/"))
        if(os.path.exist("/mnt/proj2/dd-23-97/")):
            base_path = "/mnt/proj2/dd-23-97/"
        elif(os.path.exist(data_path)):
            base_path = data_path
        else:
            base_path = os.path.dirname(os.getcwd()) + "/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/"

        if self.relaxed:
            self.path = base_path + "no_norm/"

            self.graphs_dict, self.metrics = self.__load_dataset()

        else:
            self.path = base_path + "normed_H_graph_sparse/"

            self.graphs_dict, self.metrics = self.__create_jraph_dataset()

    def __len__(self):
        return len(self.normed_energies)

    def __getitem__(self, item):
        gt_normed_energy = np.array(self.metrics["Energies"][item])
        gt_spin_state = np.array(self.metrics["gs_bins"][item]) * 2 - 1

        #print(item, len(self.graphs_dict["input_graphs"]), len(self.graphs_dict["energy_graphs"]))
        return self.graphs_dict["input_graphs"][item], self.graphs_dict["energy_graphs"][item], gt_normed_energy, gt_spin_state

    def __load_dataset(self):
        if(self.problem_name == "MaxClv2"):
            select_data_name =  "MaxCl"
        else:
            select_data_name =  self.problem_name
        base_path = os.path.join(self.path, self.dataset_name)
        path = os.path.join(base_path, f"{self.mode}_{select_data_name}_seed_{self.seed}_solutions.pickle")
        with open(path, 'rb') as file:
            solution_dict = pickle.load(file)

        if(self.problem_name == "MaxCl" or self.problem_name == "TSP" or self.problem_name == "MIS" or self.problem_name == "MaxClv2"):
            energy_graphs = solution_dict["compl_H_graphs"]
        else:
            energy_graphs = solution_dict["H_graphs"]


        U_net_graph_dict = solution_dict["U_net_graph_dict"]
        Energies = solution_dict["Energies"]
        gs_bins = solution_dict["gs_bins"]
        input_graphs = solution_dict["H_graphs"]

        self.normed_energies = Energies
        self.gs_bin_states = gs_bins
        if self.relaxed:
            self.val_mean_energy = 0
            self.val_std_energy = 1
            metrics_dict = {"Energies": Energies, "gs_bins": gs_bins, "mean_energy": self.val_mean_energy,
                                       "std_energy": self.val_std_energy}
            return {"input_graphs": input_graphs, "energy_graphs": energy_graphs}, metrics_dict
        else:
            self.val_mean_energy = solution_dict["val_mean_Energy"]
            self.val_std_energy = solution_dict["val_std_Energy"]
            return_dict = {"input_graphs": input_graphs, "energy_graphs": energy_graphs, "U_net_graph_dict": U_net_graph_dict,
                           "metrics": {"Energies": Energies, "gs_bins": gs_bins, "mean_energy": self.val_mean_energy,
                                       "std_energy": self.val_std_energy}}
            return return_dict

    def __create_jraph_dataset(self):
        if self.relaxed:
            raise ValueError('__create_jraph_dataset should not be called when using relaxed states as the dataset used is not normed!')
        return_dict = self.__load_dataset()

        graph_dict = {"input_graphs": [], "energy_graphs": []}
        for input_i_graph, energy_i_graph in zip(return_dict["input_graph"], return_dict["energy_graph"]):
            graph_dict["input_graphs"].append(return_jraph(input_i_graph))
            graph_dict["energy_graphs"].append(return_jraph(energy_i_graph))


        return graph_dict, return_dict["metrics"]


class SolutionDataset_InMemory(Dataset):
    def __init__(self, config = {}, dataset="ENZYMES", problem="MIS", mode="val", relaxed=False, seed=123):  ### TODO add orderign to config
        self.config = config
        self.dataset_name = dataset
        self.problem_name = problem
        self.mode = mode
        self.seed = seed
        self.relaxed = relaxed

        self.n_diffusion_steps = self.config["n_diffusion_steps"]+ 1
        self.buffer_size = 1000
        self.N_basis_states = self.config["N_basis_states"]

        self.get_dataset_paths(config, mode=mode, seed=seed)
        self._init_MCMCBuffer()
        #super().__init__(self.base_path, None, None, None)

    def _init_MCMCBuffer(self):
        self.MCMCBuffer = [None for i in range(self.__len__())]

    def _get__MCMCBuffer_item(self, graph, idx):
        ### TODO randomly select indices from buffer
        rand_idxs = np.random.choice(self.buffer_size, self.N_basis_states)
        if isinstance(self.MCMCBuffer[idx], np.ndarray):
            X_sequence = self.MCMCBuffer[idx][:,:,rand_idxs]
        else:
            init_X_sequence = np.zeros((graph.nodes.shape[0], self.n_diffusion_steps, self.buffer_size, 1))
            self.MCMCBuffer[idx] = init_X_sequence
            X_sequence = init_X_sequence[:,:,rand_idxs]

        return X_sequence, rand_idxs

    def update_MCMC_buffer(self, updated_X_sequence, idx, rand_idxs):
        if isinstance(self.MCMCBuffer[idx], np.ndarray):
            self.MCMCBuffer[idx][:, :, rand_idxs] = updated_X_sequence
        else:
            init_X_sequence = np.zeros((updated_X_sequence.shape[0], self.n_diffusion_steps, self.buffer_size, 1))
            self.MCMCBuffer[idx] = init_X_sequence
            self.MCMCBuffer[idx][:, :, rand_idxs] = updated_X_sequence

        return True

    def get_dataset_paths(self, cfg, mode="", seed=None):
        if(self.problem_name == "MaxClv2"):
            select_data_name =  "MaxCl"
        else:
            select_data_name =  self.problem_name


        if(os.path.isdir("/mnt/proj2/dd-23-97/")):
            base_path = "/mnt/proj2/dd-23-97/"
            load_path = base_path + f"no_norm/{self.dataset_name}/{self.mode}/{self.mode}/{self.seed}/{select_data_name}/indexed/"
        elif(os.path.isdir(data_path)):
            base_path = data_path
            load_path = base_path + f"/no_norm/{self.dataset_name}/{self.mode}/{self.mode}/{self.seed}/{select_data_name}/indexed/"
        else:
            base_path = os.path.dirname(os.getcwd()) + "/DiffUCO/DatasetCreator/loadGraphDatasets/DatasetSolutions/"

            load_path = base_path + f"no_norm/{self.dataset_name}/{self.mode}/{self.seed}/{select_data_name}/indexed/"
        with open(load_path+ f"idx_{0}_solutions.pickle", "rb") as file:
            pickle.load(file)
        self.base_path = load_path

        self.val_mean_energy = 0.
        self.val_std_energy = 1.

        _, _, files = next(os.walk(load_path))
        file_count = len(files)
        self.n_graphs = file_count

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):

        with open(self.base_path + f"idx_{idx}_solutions.pickle", "rb") as file:
            graph_dict = pickle.load(file)

        input_graph = graph_dict["H_graphs"]

        if("U_net_graph_dict" in graph_dict.keys()):
            U_net_graph_dict = graph_dict["U_net_graph_dict"]
        else:
            U_net_graph_dict = None

        if("compl_H_graphs" in graph_dict.keys()):
            if( graph_dict["compl_H_graphs"] != None):
                energy_graphs = graph_dict["compl_H_graphs"]
            elif(type(graph_dict["compl_H_graphs"]) == list):
                if(len(graph_dict["compl_H_graphs"]) > 0):
                    energy_graphs = graph_dict["compl_H_graphs"]
            else:
                energy_graphs = input_graph
        else:
            if(self.problem_name == "MaxCl" or self.problem_name == "TSP" or self.problem_name == "MIS" or self.problem_name == "MaxClv2"):
                print(graph_dict.keys())
                raise ValueError("that is not possible")
            energy_graphs = input_graph

        # print("compare edges of input graph and energy graph", energy_graphs.edges.shape, input_graph.edges.shape)
        # print("compare edges of input graph and energy graph", energy_graphs.edges, input_graph.edges.shape)
        input_graph = input_graph._replace(edges = input_graph.edges.astype(np.float32))
        energy_graphs = energy_graphs._replace(edges = energy_graphs.edges.astype(np.float32))

        return_dict = {"input_graph": input_graph, "energy_graph": energy_graphs, "energies": graph_dict["Energies"],
                       "U_net_graph_dict": U_net_graph_dict, "bs_bins": graph_dict["gs_bins"]}
        return return_dict


def get_x_smallest_and_largest(lst, x):

    sorted_lst = sorted(lst)

    # Get the X smallest values
    x_smallest = sorted_lst[:x]

    # Get the X largest values
    x_largest = sorted_lst[-x:]

    return sum(x_smallest), sum(x_largest)