from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp

class MDSEnergyClass(BaseEnergyClass):

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
        Energy_messages = jnp.log1p(-raveled_bins[H_graph.senders])

        # ones = jnp.ones_like(raveled_spins)
        # Degree = (ones[H_graph.senders]) * (ones[H_graph.receivers])
        HA_per_node = raveled_bins
        HB_per_node = B * jnp.exp(
            (jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes) + jnp.log1p(-raveled_bins)))
        HA_per_graph = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)
        HB_per_graph = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)

        Energy = HA_per_graph + HB_per_graph
        return Energy, HB_per_node, HB_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def MDS_Energy_for_Loss(self, H_graph, logits, node_gr_idx, B=1.2):
        log1p = logits[..., 0]
        bins = jnp.exp(logits[..., 1])
        n_graph = H_graph.n_node.shape[0]
        nodes = H_graph.nodes
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
        Energy_messages = log1p[H_graph.senders]

        # ones = jnp.ones_like(raveled_spins)
        # Degree = (ones[H_graph.senders]) * (ones[H_graph.receivers])
        HA_per_node = raveled_bins
        HB_per_node = B * jnp.exp((jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes) + log1p))
        HB_per_graph = jax.lax.stop_gradient(jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph))

        Energy = jax.ops.segment_sum(HA_per_node + HB_per_node, node_gr_idx, n_graph)
        return Energy, HB_per_node, HB_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx, A = 1., B = 1.01):
        self.calculate_Energy(H_graph, bins, node_gr_idx, A = A, B = B)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
        return self.MDS_Energy_for_Loss(H_graph, logits, node_gr_idx)