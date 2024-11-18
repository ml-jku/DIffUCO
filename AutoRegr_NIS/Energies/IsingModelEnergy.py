from functools import partial
import jax
import jax.numpy as jnp



class IsingModelEnergyClass():
	def __init__(self, config, energy_func = "Ising"):
		self.config = config
		size = self.config["N"]

		if(energy_func == "Ising"):
			self.edges = jnp.ones((size**2*4,1))
		else:
			self.key = jax.random.PRNGKey(0)
			undir_couplings = 2*jax.random.randint(self.key, (size**2*2, 1), 0, 2)-1
			dir_couplings = jnp.concatenate([undir_couplings, undir_couplings], axis=0)
			self.edges = dir_couplings



	@partial(jax.jit, static_argnums=(0,))
	def calculate_Energy(self, H_graph, bins, node_gr_idx):
		spins = bins

		n_graph = H_graph.n_node.shape[0]
		nodes = H_graph.nodes
		total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]


		raveled_spins = jnp.reshape(spins, (bins.shape[0], 1))
		Energy_messages = self.edges*(raveled_spins[H_graph.senders]) * (raveled_spins[H_graph.receivers])
		Energy_per_node = jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)
		Energy = -1/2 * jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)

		return Energy, bins
	def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx):
		self.calculate_Energy(H_graph, bins, node_gr_idx)

	@partial(jax.jit, static_argnums=(0,))
	def calculate_Energy_loss(self, H_graph, logits, node_gr_idx):
		p = jnp.exp(logits[...,1])
		return self.calculate_Energy(H_graph, p, node_gr_idx)