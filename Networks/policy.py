import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import jraph

from .EncodeProcessDecode import EncodeProcessDecode
from .MLPs import ProbMLP


class Policy(nn.Module):
	"""
	Policy Network
	"""
	n_features_list_prob: np.ndarray

	n_features_list_nodes: np.ndarray
	n_features_list_edges: np.ndarray
	n_features_list_messages: np.ndarray

	n_features_list_encode: np.ndarray
	n_features_list_decode: np.ndarray

	edge_updates: bool
	n_message_passes: int
	message_passing_weight_tied: bool = True

	linear_message_passing: bool = True

	def setup(self):
		self.encode_process_decode = EncodeProcessDecode(n_features_list_nodes=self.n_features_list_nodes,
														 n_features_list_edges=self.n_features_list_edges,
														 n_features_list_messages=self.n_features_list_messages,
														 n_features_list_encode=self.n_features_list_encode,
														 n_features_list_decode=self.n_features_list_decode,
									 					edge_updates = self.edge_updates,
														 n_message_passes=self.n_message_passes,
														 weight_tied=self.message_passing_weight_tied,
														 linear_message_passing=self.linear_message_passing)
		self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob)

		self.__vmap_get_log_probs = jax.vmap(self.__get_log_prob, in_axes=(0, None, None), out_axes=(0))

	def __call__(self, jraph_graph: jraph.GraphsTuple, N_basis_states: int, key):
		embeddings = self.encode_process_decode(jraph_graph=jraph_graph)
		spin_logits = self.probMLP(embeddings)

		key, subkey = jax.random.split(key)

		nodes = jraph_graph.nodes
		n_node = jraph_graph.n_node
		n_graph = jraph_graph.n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

		log_probs_down = spin_logits[:, 0]
		log_probs_up = spin_logits[:, 1]
		probs_up = jnp.exp(log_probs_up)

		entropy_term_1 = probs_up * log_probs_up
		entropy_term_2 = (1 - probs_up) * log_probs_down
		entropy_term = entropy_term_1 + entropy_term_2
		relaxed_entropies = - jax.ops.segment_sum(entropy_term, node_graph_idx, n_graph)

		# sample new state based on log probs from the probMLP
		sampled_state = jax.random.categorical(key=subkey,
											   logits=spin_logits,
											   axis=-1,
											   shape=(N_basis_states, total_nodes))

		sampled_spin_state = sampled_state * 2 - 1

		one_hot_state = jax.nn.one_hot(sampled_state, num_classes=2)

		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)

		log_probs = self.__vmap_get_log_probs(spin_log_probs, node_graph_idx, n_graph)
		return sampled_spin_state, log_probs, spin_log_probs, spin_logits, relaxed_entropies, key

	def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
		log_probs = jax.ops.segment_sum(spin_log_probs, node_graph_idx, n_graph)
		return log_probs
