from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp

class MaxCutEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, bins, node_gr_idx):
        '''
        This method assumes that edge dublicates are contained in the graph
        :param H_graph:
        :param bins:
        :param node_gr_idx:
        :param A:
        :param B:
        :return:
        '''

        spins = 2 * bins - 1

        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        raveled_spins = jnp.reshape(spins, (bins.shape[0], 1))
        Energy_messages = (raveled_spins[H_graph.senders]) * (raveled_spins[H_graph.receivers])

        # ones = jnp.ones_like(raveled_spins)
        # Degree = (ones[H_graph.senders]) * (ones[H_graph.receivers])
        ### TODO test this change factor 1/2 is removed
        Energy_per_node = (jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes))

        Energy = jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)


        return Energy, bins, Energy

    @partial(jax.jit, static_argnums=(0,))
    def get_MaxCut_Value(self, H_graph, bins, node_gr_idx):
        '''
        This method assumes that edge dublicates are contained in the graph
        :param H_graph:
        :param bins:
        :param node_gr_idx:
        :param A:
        :param B:
        :return:
        '''
        print("Warning: this function might be stale")

        spins = 2 * bins - 1

        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        raveled_spins = jnp.reshape(spins, (bins.shape[0], 1))
        Energy_messages = (raveled_spins[H_graph.senders]) * (raveled_spins[H_graph.receivers])

        # ones = jnp.ones_like(raveled_spins)
        # Degree = (ones[H_graph.senders]) * (ones[H_graph.receivers])

        Energy_per_node = (jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes))

        Energy = jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)

        num_edges = jnp.expand_dims(H_graph.n_edge, axis=-1)
        MaxCut_Value = num_edges / 2 - Energy / 2
        return -MaxCut_Value


    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.01):
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        p = jnp.exp(logits[...,1])
        return self.calculate_Energy(H_graph, p, node_gr_idx)