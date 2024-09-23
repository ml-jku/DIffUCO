import torch
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as pygDataLoader
import numpy as np
from matplotlib import pyplot as plt
import pickle
### TODO add shuffled data for data augmentation?

def add_node_features_to_dataset():
    dataset_names = ["IMDB-BINARY", "COLLAB"]

    for name in dataset_names:
        dataset = TUDataset(root=f'{os.getcwd()}/tmp/{name}', name=name)
        loader = pygDataLoader(dataset, batch_size=2, shuffle=True)

        dataset.data.x = torch.zeros((dataset.data.num_nodes, 1))

        pickle.dump(dataset.data.x, open(dataset.processed_paths[0], "wb"))

### TODO visualise test train val splits
def dataset_evaluation():
    dataset_names = ["ENZYMES", "PROTEINS", "MUTAG", "COLLAB", "IMDB-BINARY"]

    for name in dataset_names:
        dataset = TUDataset(root=f'{os.getcwd()}/tmp/{name}', name=name)

        dataset_len = len(dataset)

        loader = pygDataLoader(dataset, batch_size=2, shuffle=True)

        for data in loader:
            print("num nodes", data.num_nodes)
            print("num graphs", data.num_graphs)
            print(data.is_undirected())
            print(data.edge_index)
            senders = data.edge_index[0, :]
            receivers = data.edge_index[1, :]

        print(name, len(dataset))
        n_nodes = []
        n_edges = []
        densities = []
        for data in dataset:
            # print(data.x.shape)
            if (data.x != None):
                num = len(data.x)
                n_nodes.append(len(data.x))
            else:
                ValueError("None happened")
                n_nodes.append(data.num_nodes)
                num = data.num_nodes

            n_edges.append(data.edge_index.shape[1])
            densities.append(data.edge_index.shape[1] / num ** 2)

        # if (name == "COLLAB"):
        #     n_nodes = n_nodes[0:1000]
        #     n_edges = n_edges[0:1000]
        #     densities = densities[0:1000]
        #     dataset_len = 1000

        train_len = int(dataset_len*0.6)
        val_len = int(dataset_len*0.3)
        test_len = int(dataset_len*0.1)

        plt.figure()
        plt.title(f"dataset {name}")
        plt.plot(densities[0:train_len], n_nodes[0:train_len], "x", color = "r", label = "train")
        plt.plot(densities[train_len:train_len+ val_len], n_nodes[train_len:train_len+ val_len], "x", color = "b", label = "val")
        plt.plot(densities[train_len+ val_len:], n_nodes[train_len+ val_len:], "x", color = "g", label = "test")

        plt.xlabel("edge denisty")
        plt.ylabel("number of nodes")
        #plt.xscale("log")
        #plt.yscale("log")
        plt.legend()
        plt.show()

        plt.figure()
        plt.title(name)
        plt.hist(n_nodes)
        plt.show()

        n_nodes = np.array(n_nodes)
        print("avrg nodes", np.mean(n_nodes), np.max(n_nodes), np.min(n_nodes))

        n_edges = np.array(n_edges)
        print("avrg edges", np.mean(n_edges), np.max(n_edges), np.min(n_edges))
        print("density", np.mean(np.array(densities)))

if(__name__ == "__main__"):
    #add_node_features_to_dataset()
    dataset_evaluation()
    pass
