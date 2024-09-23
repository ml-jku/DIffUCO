from torch.utils.data import DataLoader
import os
from torch_geometric.datasets import TUDataset
import numpy as np
import jax.numpy as jnp
import jraph
from jraph_utils import utils as jutils
from loadGraphDatasets.loadTwitterGraph import TWITTER


gpu_np = jnp
cpu_np = np

def get_num_nodes_v1(pyg_graph):
    num_nodes = pyg_graph.x.shape[0]
    return num_nodes

def get_num_nodes_v2(pyg_graph):

    num_nodes = pyg_graph.num_nodes
    return num_nodes

class Generator:

    def __init__(self, cfg, num_workers = 0, shuffle_seed = None):
        self.num_workers = num_workers
        ### TODO try to set num workers
        self.cfg = cfg

        self.dataset_name = cfg["Ising_params"]["IsingMode"]

        self.shuffle_seed = cfg["Ising_params"]["shuffle_seed"]

        if(self.dataset_name == "COLLAB" or self.dataset_name == "IMDB-BINARY"):
            self.get_num_nodes_fuc = get_num_nodes_v2
        else:
            self.get_num_nodes_fuc = get_num_nodes_v1

        self.dataset_names = ["ENZYMES", "PROTEINS", "MUTAG", "COLLAB", "IMDB-BINARY"]

        ### TODO implement some exceptions for COLLAB ind IMDB dataset because they have empty node attributes
        if(self.dataset_name in self.dataset_names):
            pass
        else:
            ValueError("This dataset does not exist")

        self.collate_from_torch_to_jraph_fn = lambda data: self.collate_from_torch_to_jraph(data, add_padded_node=True,
                                                                                  time_horizon=self.time_horizon)

        if(self.dataset_name != "TWITTER"):
            self.init_TUDataset()
        else:
            self.init_TWITTERDataset()


    def init_TUDataset(self):
        self.dataset = TUDataset(root=f'{os.getcwd()}/loadGraphDatasets/tmp/{self.dataset_name}', name=self.dataset_name)

        if(self.dataset_name == "COLLAB" and self.cfg["Ising_params"]["EnergyFunction"] == "MIS"):
            self.dataset = self.dataset[0:1000]
            full_dataset_len = 1000
        else:
            full_dataset_len = len(self.dataset)

        print("dataset name", self.dataset_name, full_dataset_len)

        full_dataset_arganged = np.arange(0, full_dataset_len)
        ### TODO shuffle with jax here to make it deterministic by seed

        if(self.shuffle_seed != -1):
            np.random.seed(self.shuffle_seed)
            np.random.shuffle(full_dataset_arganged)

        if(self.cfg["Ising_params"]["EnergyFunction"] == "MIS" or self.dataset_name == "PROTEINS" or self.dataset_name == "ENZYMES"):
            ts = 0.6
            vs = 0.1
        elif(self.dataset_name == "IMDB-BINARY" and self.cfg["Ising_params"]["EnergyFunction"] == "MaxCl"):
            ts = 0.6
            vs = 0.1
        elif(self.dataset_name == "MUTAG" and self.cfg["Ising_params"]["EnergyFunction"] == "MaxCl"):
            ts = 0.6
            vs = 0.1
        else:
            ts = 0.7
            vs = 0.1

        train_dataset_len = int(ts*full_dataset_len)
        val_dataset_len = int(vs*full_dataset_len)
        test_dataset_len = full_dataset_len - train_dataset_len - val_dataset_len
        self.train_dataset_idxs = full_dataset_arganged[0:train_dataset_len]
        self.val_dataset_idxs = full_dataset_arganged[train_dataset_len:train_dataset_len+val_dataset_len]
        self.test_dataset_idxs = full_dataset_arganged[train_dataset_len+val_dataset_len:]

        self.pyg_train_dataset = self.dataset[self.train_dataset_idxs]
        self.pyg_val_dataset = self.dataset[self.val_dataset_idxs]
        self.pyg_test_dataset = self.dataset[self.test_dataset_idxs]


    def init_TWITTERDataset(self):
        self.reset_collate_func = lambda data: jutils.collate_jraphs_to_horizon(data, self.time_horizon, self.random_node_features)
        self.pyg_train_dataset = TWITTER(self.cfg, mode = self.mode)
        self.pyg_loader = iter(self.pyg_train_dataset)

        self.jraph_dataloader = DataLoader(self.pyg_train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.reset_collate_func, num_workers=self.num_workers)
        self.jraph_loader = iter(self.jraph_dataloader )

        self.batched_graphs, self.graph_list = next(self.jraph_loader)


    def collate_from_torch_to_jraph(self, datas, **kargs):
        jdata_list = [self.from_pyg_to_jraph(data, **kargs) for data in datas]
        batched_jdata = jraph.batch_np(jdata_list)
        return (batched_jdata, jdata_list)

    def collate_from_torch_to_jraph_val_and_test(self, datas):

        num_nodes = max([self.get_num_nodes_fuc(data) for data in datas])

        jdata_list = [self.from_pyg_to_jraph(data, add_padded_node=True, time_horizon= num_nodes) for data in datas]
        batched_jdata = jraph.batch_np(jdata_list)
        return (batched_jdata, jdata_list)

    def from_pyg_to_jraph(self, pyg_graph, add_padded_node=False, time_horizon=0):
        num_nodes = self.get_num_nodes_fuc(pyg_graph)
        num_edges = pyg_graph.edge_index.shape[1]

        nodes = cpu_np.zeros((num_nodes, 1), dtype=cpu_np.float32)

        senders = cpu_np.array(pyg_graph.edge_index[0, :])
        receivers = cpu_np.array(pyg_graph.edge_index[1, :])

        edges = cpu_np.ones((num_edges, 1), dtype=cpu_np.float32)
        n_node = cpu_np.array([num_nodes])
        n_edge = cpu_np.array([num_edges])

        jgraph = jraph.GraphsTuple(nodes=nodes, senders=senders, receivers=receivers,
                                   edges=edges, n_node=n_node, n_edge=n_edge, globals= cpu_np.zeros((1,)))
        return jgraph


if(__name__ == "__main__"):

    pass
    # p = Path( os.getcwd())
    # path = p.parent
    # print(path)
    # cfg = OmegaConf.load(path + "/Experiment_configs/HydraBaseConfig.yaml")
    # cfg["Ising_params"]["IsingMode"] = "PROTEINS"
    # gen = Generator(cfg)
