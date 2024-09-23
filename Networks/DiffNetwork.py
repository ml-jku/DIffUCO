
import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import jraph
from functools import partial

from .EncodeProcessDecode import EncodeProcessDecode
from .MLPs import ProbMLP
from typing import Callable


class DiffNetwork(nn.Module):
	"""
	Policy Network
	"""
	n_features_list_prob: np.ndarray

	n_features_list_nodes: np.ndarray
	n_features_list_edges: np.ndarray
	n_features_list_messages: np.ndarray

	n_features_list_encode: np.ndarray
	n_features_list_decode: np.ndarray
	beta_list: list
	energy_function_func: Callable
	noise_potential_func: Callable

	n_diffusion_steps: int
	n_message_passes: int
	message_passing_weight_tied: bool = True

	linear_message_passing: bool = True


	def setup(self):
		self.encode_process_decode = EncodeProcessDecode(n_features_list_nodes=self.n_features_list_nodes,
														 n_features_list_edges=self.n_features_list_edges,
														 n_features_list_messages=self.n_features_list_messages,
														 n_features_list_encode=self.n_features_list_encode,
														 n_features_list_decode=self.n_features_list_decode,
														 n_message_passes=self.n_message_passes,
														 weight_tied=self.message_passing_weight_tied,
														 linear_message_passing=self.linear_message_passing)
		self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob)

		self.__vmap_get_log_probs = jax.vmap(self.__get_log_prob, in_axes=(0, None, None), out_axes=(0))

	def _forward(self, jraph_graph, X_prev):
		jraph_graph = jraph_graph._replace(nodes=X_prev)
		embeddings = self.encode_process_decode(jraph_graph=jraph_graph)
		spin_logits = self.probMLP(embeddings)
		return spin_logits

	def _sample(self, spin_logits, key):
		key, subkey = jax.random.split(key)
		X_next = jax.random.categorical(key=subkey,
										logits=spin_logits,
										axis=-1,
										shape=spin_logits.shape[:-1])

		one_hot_state = jax.nn.one_hot(X_next, num_classes=2)
		X_next = jnp.expand_dims(X_next, axis=-1)

		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
		return X_next, spin_logits, spin_log_probs, key

	def _log_metrics(self, input_dict):
		average_probs = jnp.mean(graph_log_prob)
		prob_over_diff_steps = prob_over_diff_steps.at[i].set(average_probs)
		Noise_loss_over_diff_steps = Noise_loss_over_diff_steps.at[i].set(Loss_noise_repara)
		Energy_over_diff_steps = Energy_over_diff_steps.at[i].set(
			self.__get_energy_loss(graphs, spin_logits_next, jnp.sum(log_p_prev_per_node, axis=0), node_gr_idx)[2])

		output_dict["metrics"]["prob_over_diff_steps"] =None
		return input_dict

	### TODO find out how to jit this
	def __call__(self, input_dict):
		X_prev = input_dict["X_prev"]
		spin_logits_prev = input_dict["spin_logits_prev"]
		spin_log_probs_prev = input_dict["spin_log_probs_prev"]
		log_p_prev_per_node = input_dict["log_p_prev_per_node"]
		jraph_graph = input_dict["jraph_graph"]
		key = input_dict["key"]

		L_noise = input_dict["losses"]["L_noise"]
		L_entropy = input_dict["losses"]["L_entropy"]
		L_energy = input_dict["losses"]["L_energy"]

		diff_step = input_dict["train_state"]["diff_step"]
		gamma_t = self.beta_list[diff_step]

		spin_logits = self._forward(jraph_graph, X_prev)

		node_graph_idx, n_graph, total_num_nodes = self._compute_aggr_utils(jraph_graph)
		# sample new state based on log probs from the probMLP
		X_next, spin_logits_next, spin_log_probs, key = self._sample(spin_logits, key)
		n_node = jraph_graph.n_node

		graph_log_prob = jnp.exp((self.__get_log_prob(spin_log_probs, node_graph_idx, n_graph) / n_node)[:-1])

		### TODO Compute_losses
		Noise_Loss, L_repara, L_repara, L_REINFORCE = self.__get_Noise_energy_loss(jraph_graph, X_prev, spin_logits_prev, spin_logits_next, log_p_prev_per_node,
								gamma_t, node_graph_idx)
		Entropy_Loss, relaxed_entropies, L_repara, L_REINFORCE = self.__get_entropy_loss(jraph_graph, X_prev, spin_logits_prev, spin_logits_next, log_p_prev_per_node,
								gamma_t, node_graph_idx)

		L_noise += Noise_Loss
		L_entropy += Entropy_Loss

		if (diff_step != len(self.beta_list)-1):
			log_p_prev_per_node = log_p_prev_per_node.at[diff_step].set(
				spin_log_probs)
		else:
			L_energy, energies, Loss_energy_repara, Loss_energy_Reinforce = self.__get_energy_loss(jraph_graph,
																								   spin_logits_next,
																								   jnp.sum(
																									   log_p_prev_per_node,
																									   axis=0),
																								   node_graph_idx)


		output_dict = input_dict
		output_dict["X_prev"] = X_next
		output_dict["spin_logits_prev"] = spin_logits_prev
		output_dict["log_p_prev_per_node"] = log_p_prev_per_node
		output_dict["key"] = key
		output_dict["losses"]["L_noise"] = L_noise
		output_dict["losses"]["L_entropy"] = L_entropy
		output_dict["losses"]["L_energy"] = L_energy


		diff_step += 1
		output_dict["train_state"]["diff_step"] = diff_step

		return output_dict


	def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
		log_probs = jax.ops.segment_sum(spin_log_probs, node_graph_idx, n_graph)
		return log_probs

	@partial(jax.jit, static_argnums=(0,))
	def _compute_aggr_utils(self, jraph_graph):
		nodes = jraph_graph.nodes
		n_node = jraph_graph.n_node
		n_graph = jraph_graph.n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
		return node_graph_idx, n_graph, total_num_nodes

	@partial(jax.jit, static_argnums=(0,))
	def __get_Noise_energy_loss(self, jraph_graph, X_prev, spin_logits_prev, spin_logits_next, log_p_prev_per_node,
								gamma_t, node_gr_idx):

		relaxed_state_next = jnp.exp(spin_logits_next[:, :, 1])
		p_next = jnp.expand_dims(relaxed_state_next, axis=-1)

		relaxed_state_prev = jnp.exp(spin_logits_prev[:, :, 1])
		p_prev = jnp.expand_dims(relaxed_state_prev, axis=-1)

		noise_energy_per_node, sum_log_p_prev_per_node = self.noise_potential_func(jraph_graph, p_prev, p_next, X_prev,
																		 log_p_prev_per_node, gamma_t, node_gr_idx)
		noise_energy_per_node = (-1) * noise_energy_per_node

		n_graph = jraph_graph.n_node.shape[0]
		Noise_Energy_per_graph = jax.ops.segment_sum(noise_energy_per_node, node_gr_idx, n_graph)

		L_repara_per_graph = Noise_Energy_per_graph

		noise_energy_per_node_no_grad = jax.lax.stop_gradient(noise_energy_per_node)
		baseline = jnp.mean(noise_energy_per_node_no_grad, axis=-2, keepdims=True)
		# print("baseline noise",noise_energy_per_node_no_grad.shape,  baseline.shape)

		L_REINFORCE_per_Node = (noise_energy_per_node_no_grad - baseline) * sum_log_p_prev_per_node[:, :,
																			jnp.newaxis]  ### TODO check wether this nosie reduction in REINFROCE is correct
		L_REINFORCE_per_graph = jax.ops.segment_sum(L_REINFORCE_per_Node, node_gr_idx, n_graph)

		L_REINFORCE = jnp.mean(L_REINFORCE_per_graph[:-1])

		L_repara = jnp.mean(L_repara_per_graph[:-1])
		Noise_Loss = (L_REINFORCE + L_repara)
		return Noise_Loss, L_repara, L_repara, L_REINFORCE

	@partial(jax.jit, static_argnums=(0,))
	def __get_entropy_loss(self, jraph_graph, spin_logits, sum_log_p_prev_per_node, node_gr_idx):

		log_probs_down = spin_logits[:, :, 0]
		log_probs_up = spin_logits[:, :, 1]
		probs_up = jnp.exp(log_probs_up)
		probs_down = jnp.exp(log_probs_down)

		entropy_term_1 = -probs_up * log_probs_up
		entropy_term_2 = -probs_down * log_probs_down
		entropy_term_per_node = entropy_term_1 + entropy_term_2

		n_graph = jraph_graph.n_node.shape[0]
		relaxed_entropies_per_graph = jax.ops.segment_sum(entropy_term_per_node, node_gr_idx, n_graph)

		entropy_term_per_node_no_grad = jax.lax.stop_gradient(entropy_term_per_node)
		baseline = jnp.mean(entropy_term_per_node_no_grad, axis=-1, keepdims=True)
		# print("baseline entropy",entropy_term_per_node_no_grad.shape,  baseline.shape)

		L_REINFORCE_per_node = (entropy_term_per_node_no_grad - baseline) * sum_log_p_prev_per_node
		L_REINFORCE_per_graph = jax.ops.segment_sum(L_REINFORCE_per_node, node_gr_idx, n_graph)
		L_REINFORCE = jnp.mean(L_REINFORCE_per_graph[:-1])

		L_repara_per_graph = relaxed_entropies_per_graph
		L_repara = jnp.mean(L_repara_per_graph[:-1])

		relaxed_entropies = L_repara

		Entropy_Loss = L_repara + L_REINFORCE
		return Entropy_Loss, relaxed_entropies, L_repara, L_REINFORCE

	@partial(jax.jit, static_argnums=(0,))
	def __get_energy_loss(self, jraph_graph, spin_logits, sum_log_p_prev_per_node, node_gr_idx):

		n_graph = jraph_graph.n_node.shape[0]

		relaxed_state = jnp.exp(spin_logits[:, :, 1])
		relaxed_state = jnp.expand_dims(relaxed_state, axis=-1)
		relaxed_energies_per_graph, relaxed_Energy_per_node = self.energy_function_func(jraph_graph, relaxed_state,
																						  node_gr_idx)

		energy_term_per_node_no_grad = jax.lax.stop_gradient(relaxed_Energy_per_node)
		baseline = jnp.mean(energy_term_per_node_no_grad, axis=-2, keepdims=True)

		L_REINFORCE_per_node = (energy_term_per_node_no_grad - baseline) * sum_log_p_prev_per_node[:, :, jnp.newaxis]

		L_REINFORCE_per_graph = jax.ops.segment_sum(L_REINFORCE_per_node, node_gr_idx, n_graph)
		L_REINFORCE = jnp.mean(L_REINFORCE_per_graph[:-1])

		L_repara_per_graph = relaxed_energies_per_graph[:-1]
		L_repara = jnp.mean(L_repara_per_graph)

		Energy_Loss = L_repara + L_REINFORCE
		return Energy_Loss, L_repara_per_graph, L_repara, L_REINFORCE

	def diffusion_loss(self, params, graphs, T, key):
		### TODO for each graph multiple states should be sampled
		key, subkey = jax.random.split(key)

		nodes = graphs.nodes
		p_uniform = jnp.log(0.5 * jnp.ones((nodes.shape[0], self.N_basis_states, 2)))

		X_prev = jax.random.categorical(key=subkey,
										logits=p_uniform,
										axis=-1,
										shape=(nodes.shape[0], self.N_basis_states))
		X_prev = jnp.expand_dims(X_prev, axis=-1)

		spin_logits_prev = p_uniform

		L_entropy = 0.
		L_noise = 0.

		L_energy_Reinforce = 0.
		L_noise_Reinforce = 0.
		L_entropy_Reinforce = 0.
		L_energy_repara = 0.
		L_noise_repara = 0.
		L_entropy_repara = 0.

		log_p_prev_per_node = jnp.zeros((self.n_diffusion_steps, X_prev.shape[0], X_prev.shape[1]))
		prob_over_diff_steps = jnp.zeros((self.n_diffusion_steps,))
		Noise_loss_over_diff_steps = jnp.zeros((self.n_diffusion_steps,))
		Energy_over_diff_steps = jnp.zeros((self.n_diffusion_steps,))

		node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(graphs)
		# key = jax.random.split(key, num=self.N_basis_states)
		# Energy_over_diff_steps = Energy_over_diff_steps.at[0].set(self.__get_energy_loss(graphs, spin_logits_prev, spin_logits_prev)[2])
		### TODO implement this with scanner?
		for i in range(self.n_diffusion_steps):
			### TODO vmapping has to be done here
			key, subkey = jax.random.split(key)
			batched_key = jax.random.split(subkey, num=self.N_basis_states)
			X_next, spin_log_probs, spin_logits_next, graph_log_prob, _ = self.vmapped_make_one_step(params, graphs,
																									 X_prev,
																									 batched_key)

			# print("relaxed states", jnp.squeeze(jnp.exp(spin_logits)[:10, 0]))
			# print("relaxed states", jnp.squeeze(jnp.exp(spin_logits)[:10, 1]))

			Entropy_Loss, Entropy, Loss_entropy_repara, Loss_entropy_Reinforce = self.__get_entropy_loss(graphs,
																										 spin_logits_next,
																										 jnp.sum(
																											 log_p_prev_per_node,
																											 axis=0),
																										 node_gr_idx)
			L_entropy += Entropy_Loss
			L_entropy_repara += Loss_entropy_repara
			L_entropy_Reinforce += Loss_entropy_Reinforce

			Noise_Loss, Noise_Energy, Loss_noise_repara, Loss_noise_Reinforce = self.__get_Noise_energy_loss(graphs,
																											 X_prev,
																											 spin_logits_prev,
																											 spin_logits_next,
																											 log_p_prev_per_node,
																											 self.beta_arr[
																												 i],
																											 node_gr_idx)
			L_noise += Noise_Loss
			L_noise_repara += Loss_noise_repara
			L_noise_Reinforce += Loss_noise_Reinforce

			if (i != self.n_diffusion_steps - 1):
				log_p_prev_per_node = log_p_prev_per_node.at[i].set(
					spin_log_probs)  ### TODO prob of selected spin should be taken?


			X_prev = X_next
			spin_logits_prev = spin_logits_next

			average_probs = jnp.mean(graph_log_prob)
			prob_over_diff_steps = prob_over_diff_steps.at[i].set(average_probs)
			Noise_loss_over_diff_steps = Noise_loss_over_diff_steps.at[i].set(Loss_noise_repara)
			Energy_over_diff_steps = Energy_over_diff_steps.at[i].set(
				self.__get_energy_loss(graphs, spin_logits_next, jnp.sum(log_p_prev_per_node, axis=0), node_gr_idx)[2])

		L_energy, energies, Loss_energy_repara, Loss_energy_Reinforce = self.__get_energy_loss(graphs, spin_logits_next,
																							   jnp.sum(
																								   log_p_prev_per_node,
																								   axis=0), node_gr_idx)

		L_energy_repara += Loss_energy_repara
		L_energy_Reinforce += Loss_energy_Reinforce
		graph_mean_energy = energies

		X_0 = X_next

		Loss = self.calc_loss(L_entropy, L_noise, L_energy, T)
		log_dict = {"Losses": {"L_entropy": L_entropy, "L_noise": L_noise, "L_energy": L_energy,
							   "L_Energy_Reinfroce": L_energy_Reinforce, "L_Noise_Reinforce": L_noise_Reinforce,
							   "L_entropy_REINFORCE": L_entropy_Reinforce, "L_entropy_repara": L_entropy_repara,
							   "L_noise_repara": L_noise_repara, "L_energy_repara": L_energy_repara,
							   "overall_Loss": Loss},
					"metrics": {"energies": energies, "entropies": 0., "spin_log_probs": spin_log_probs,
								"free_energies": L_entropy, "graph_mean_energies": graph_mean_energy},
					"figures": {"prob_over_diff_steps": {"x_values": jnp.arange(0, self.n_diffusion_steps),
														 "y_values": prob_over_diff_steps},
								"Noise_loss_over_diff_steps": {"x_values": jnp.arange(0, self.n_diffusion_steps),
															   "y_values": Noise_loss_over_diff_steps},
								"Energy_over_diff_steps": {"x_values": jnp.arange(0, self.n_diffusion_steps),
														   "y_values": Energy_over_diff_steps}
								},
					"log_p_0": spin_logits_next,
					"X_0": X_0,
					"spin_log_probs": spin_log_probs,
					}

		return Loss, (log_dict, key)



if(__name__ == "__main__"):
	pass