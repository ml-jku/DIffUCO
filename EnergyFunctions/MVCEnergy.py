from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp

class MVCEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        '''
        This method assumes that edge dublicates are contained in the graph
        :param H_graph:
        :param bins:
        :param node_gr_idx:
        :param A:
        :param B:
        :return:
        '''

        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
        Energy_messages = (1 - raveled_bins[H_graph.senders]) * (1 - raveled_bins[H_graph.receivers])

        HA_per_node = A * raveled_bins
        HB_per_node = B * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

        violations_per_node = 0.5*(HB_per_node + jax.ops.segment_sum(Energy_messages, H_graph.senders, total_num_nodes))
        HA_per_graph = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)
        HB_per_graph = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)

        Energy = HA_per_graph + HB_per_graph
        return Energy, violations_per_node, HB_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_relaxed(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        '''
        This method assumes that edge dublicates are contained in the graph
        :param H_graph:
        :param bins:
        :param node_gr_idx:
        :param A:
        :param B:
        :return:
        '''

        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
        Energy_messages = (1 - raveled_bins[H_graph.senders]) * (1 - raveled_bins[H_graph.receivers])

        HA_per_node = A * raveled_bins
        HB_per_node = B * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)
        HA_per_graph = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)
        HB_per_graph = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)

        Energy = HA_per_graph + HB_per_graph
        return Energy, HB_per_node, HB_per_graph

    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.2):
        self.calculate_Energy_relaxed(H_graph, bins, node_gr_idx, A = A, B = B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        p = jnp.exp(logits[...,1])
        return self.calculate_Energy(H_graph, p, node_gr_idx)