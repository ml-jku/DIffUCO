import pickle
from torch.utils.data import Dataset
import numpy as np
from jraph_utils import utils as jutils
import os

class TWITTER(Dataset):
    def __init__(self, cfg,seed = 0, mode = "train"):
        self.mode = mode
        self.random_node_features = cfg["Ising_params"]["n_rand_nodes"]

        path = os.getcwd() + "/loadGraphDatasets/tmp/TWITTER/TWITTER_SNAP_2.p"
        file = open(path, 'rb')
        self.data = pickle.load(file)

        np.random.seed(seed)
        num_data = len(self.data)
        self.data_idxs = np.arange(0, num_data)
        np.random.shuffle(self.data_idxs)

        self.train_data_len = int(0.7 * num_data)
        self.val_data_len = int(0.1 * num_data)
        self.test_data_len = int(0.2 * num_data)

        self.idxs_list = self.data_idxs[0:self.train_data_len]

        self.num_nodes = []
        for idx in self.idxs_list:
            self.num_nodes.append(self.data[idx]["x"].shape[0])

        self.normalisation_constant = np.mean(np.array(self.num_nodes))

        if(self.mode == "train"):
            self.idxs_list = self.data_idxs[0:self.train_data_len]
        elif(self.mode == "val"):
            self.idxs_list = self.data_idxs[self.train_data_len: self.train_data_len + self.val_data_len]
        elif(self.mode == "test"):
            self.idxs_list = self.data_idxs[self.train_data_len + self.val_data_len:]

    def __len__(self):
        return len(self.idxs_list)

    def __getitem__(self, idx):
        idx = self.idxs_list[idx]
        data = self.data[idx]

        num_nodes = data["x"].shape[0]
        edge_index = data["edge_index"]
        j_graph = jutils.pyg_to_jgraph(num_nodes, edge_index)
        print(edge_index.shape)
        igraph = jutils.from_jgraph_to_igraph(j_graph)
        print("edges", igraph.ecount())
        j_graph = jutils.from_igraph_to_dir_jgraph(igraph)
        print("jedges", j_graph.edges.shape)
        return j_graph

def load_TWITTER():
    from unipath import Path
    import os
    from omegaconf import OmegaConf
    from matplotlib import pyplot as plt
    from collections import Counter
    p = Path( os.getcwd())
    path = p.parent
    print(path)
    cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")
    cfg["Ising_params"]["IsingMode"] = "PROTEINS"
    ### TODO take care of self loops in TWITTER dataset

    plt.figure()
    modes = ["train", "val", "test"]
    for mode in modes:
        dataset = TWITTER(cfg, mode = mode, seed = 1)

        nodes_list = []
        denisty_list = []
        print(len(dataset))
        for data in dataset:

            edge_list = [(min([s,r]),max([s,r])) for (s,r) in zip(data.senders, data.receivers)]

            c_keys = Counter(edge_list).keys()
            c_values = Counter(edge_list).values()

            for el, val in zip(c_keys, c_values):
                if(val != 2):
                    print(val, el)

            if(data.edges.shape[0] %2 != 0):
                unique = set(edge_list)
                sorted_edge_list = [(min([e1,e2]),max([e1,e2])) for (e1,e2) in edge_list]
                print(len(unique), len(sorted_edge_list))

                # print(Counter(sorted_edge_list).keys())
                # print(Counter(sorted_edge_list).values())
                print("uneven")

            num_nodes = data.nodes.shape[0]
            num_edges = data.edges.shape[0]

            density = num_edges/num_nodes**2
            nodes_list.append(num_nodes)
            denisty_list.append(density)


        plt.plot(denisty_list, nodes_list, "x", label = mode)
    plt.legend()
    plt.show()
    pass


if(__name__ == "__main__"):
    load_TWITTER()

    # import torch
    # path = "/system/user/publicwork/sanokows/PPO_CombOpt_SpinDrop/loadGraphDatasets/tmp/RB500_max_clique/data.pt"
    # file = open(path, 'rb')
    # data = torch.load(file)
    #
    # print(data)
    # print(data[0])

