import jax.numpy as jnp
import numpy as np
import jraph
import igraph as ig
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle

from Solvers.GurobiSolver import solve_iGraph


class ErdosRenyiGraphs:
    def __init__(self, epochs, N_spins, p, batch_size, testset_size, seed):
        self.epochs = epochs
        self.N_spins = N_spins
        self.p = p
        self.seed = seed
        self.batch_size = batch_size
        self.testset_size = testset_size

    def create_dataloader(self):
        def collate_function(batch):
            batched_jraph_graph = jraph.batch_np(batch)
            return batched_jraph_graph

        self.dataset = ErdosRenyiGraphDataset(epochs=self.epochs,
                                              N_spins=self.N_spins,
                                              p=self.p,
                                              batch_size=self.batch_size,
                                              testset_size=self.testset_size,
                                              seed=self.seed)

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_function)
        return dataloader

    def create_test_set(self):
        """
        TODO: Save and load graphs
        """
        gt_energies, gt_spin_states, gt_jraph_graphs = self.dataset.generate_test_set()
        return {"test_batch": jraph.batch_np(gt_jraph_graphs),
                "gt_energies": gt_energies,
                "gt_spin_states": gt_spin_states}


class ErdosRenyiGraphDataset(Dataset):
    def __init__(self, epochs, N_spins, p, batch_size, testset_size, seed):
        self.N_spins = N_spins
        self.p = p
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.testset_size = testset_size

    def __len__(self):
        return self.epochs*self.batch_size

    def __return_jraph(self, i_graph, edges, state):
        """
        Return current igraph as jraph
        The nodes are the external fields and the edges are the couplings
        """
        n_edges = i_graph.ecount()
        if n_edges > 0:
            couplings = np.array(i_graph.es['couplings'])

            directed_senders = edges[:, 0]
            directed_receivers = edges[:, 1]
            jraph_senders = np.ravel(
                np.concatenate([directed_senders[:, np.newaxis], directed_receivers[:, np.newaxis]], axis=1))
            jraph_receivers = np.ravel(
                np.concatenate([directed_receivers[:, np.newaxis], directed_senders[:, np.newaxis]], axis=1))

            # edge values for jraph graph, every coupling value needs to be included twice because the graph is now undirected
            # EXAMPLE:
            # directed graph edge values: [1, 2, 3, 4]
            # undirected graph edge values: [1, 1, 2, 2, 3, 3, 4, 4]
            jraph_edge_values = np.concatenate((couplings, couplings), axis=1).flatten()
            jraph_edge_values = np.expand_dims(jraph_edge_values, axis=-1)

            external_fields = np.array(i_graph.vs['external_field'])
            state = np.expand_dims(state, axis=-1)

            jraph_nodes = np.concatenate((external_fields, state), axis=1)

            jraph_graph = jraph.GraphsTuple(nodes=jraph_nodes,
                                            edges=jraph_edge_values,
                                            senders=jraph_senders,
                                            receivers=jraph_receivers,
                                            n_node=np.array([self.N_spins]),
                                            n_edge=np.array([n_edges]),
                                            globals=None)
        else:
            raise NotImplementedError("graph has no edges")
        return jraph_graph

    def __generate_random_graph(self, seed):
        """
        generate erdos renyi graph
        """
        random.seed(seed)
        np.random.seed(seed)

        i_graph = ig.Graph.Erdos_Renyi(n=self.N_spins, p=self.p, directed=False, loops=False)
        edges = np.array(i_graph.get_edgelist())
        i_graph['senders'] = edges[:, 0]
        i_graph['receivers'] = edges[:, 1]

        # couplings are one for now
        i_graph.es['couplings'] = np.ones((i_graph.ecount(), 1))
        # external fields are zero for now
        i_graph.vs['external_field'] = np.zeros((self.N_spins, 1))

        i_graph['senders'] = edges[:, 0]
        i_graph['receivers'] = edges[:, 1]
        return i_graph, edges

    def __getitem__(self, item):
        """
        generate erdos renyi graph
        """
        # to make sure that every graph has its own seed and that the graphs in the test dataset have different seeds
        seed = self.seed + self.testset_size + item
        np.random.seed(seed)
        random_bin_state = np.random.randint(0, 2, size=self.N_spins)
        i_graph, edges = self.__generate_random_graph(seed=seed)
        return self.__return_jraph(i_graph, edges, random_bin_state)

    def generate_test_set(self):
        """
        generate and return test_set
        """
        gt_energies = []
        gt_spin_states = []
        gt_jraph_graphs = []
        for i in range(self.testset_size):
            seed = self.seed + i
            i_graph, edges = self.__generate_random_graph(seed=seed)
            _, energy, bin_solution, _ = solve_iGraph(i_graph=i_graph)
            spin_solution = bin_solution * 2 - 1
            random_bin_state = np.random.randint(0, 2, size=self.N_spins)
            gt_energies.append(energy)
            gt_spin_states.append(spin_solution)
            gt_jraph_graphs.append(self.__return_jraph(i_graph, edges, random_bin_state))
        return gt_energies, gt_spin_states, gt_jraph_graphs



















