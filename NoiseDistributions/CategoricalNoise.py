from .BaseNoise import BaseNoiseDistr
import jax.numpy as jnp
import jax
from functools import partial

class CategoricalNoseDistr(BaseNoiseDistr):

    def __init__(self, config):
        super().__init__(config)
        self.n_bernoilli_features = config["n_bernoulli_features"]

    def combine_losses(self, L_entropy, L_noise, L_energy, T):
        return -T * L_entropy + T*L_noise + L_energy

    def calculate_noise_distr_reward(self, noise_distr_step, entropy_reward):
        return -(noise_distr_step - entropy_reward)

    @partial(jax.jit, static_argnums=(0,))
    def get_log_p_T_0(self, jraph_graph, X_prev, X_next, t_idx, T):
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jraph_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        gamma_t = self.get_gamma_t(t_idx)
        beta_t = 2 * gamma_t
        p_change_value = beta_t / (self.n_bernoilli_features)
        p_stay_value = 1 - beta_t + p_change_value

        log_p_i = jnp.where(X_next == X_prev, jnp.log(p_stay_value), jnp.log(p_change_value))

        noise_per_node = jnp.sum(log_p_i, axis=-1)
        n_graph = jraph_graph.n_node.shape[0]

        log_p_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        #graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(noise_per_node, node_gr_idx, n_graph) / (n_node[:, None]))[:-1]))
        # print(t_idx, gamma_t, "gamma_t")
        # print("average prob p T:0", jnp.mean(graph_log_prob))

        return log_p_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_loss(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, gamma_t, node_gr_idx, T):
        raise ValueError("relaxed does not make sense here")

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step_relaxed(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx):
        raise ValueError("relaxed does not make sense here")

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step(self, jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr):
        gamma_t = self.beta_arr[model_step_idx]
        reward_idx = model_step_idx
        beta_t = 2 * gamma_t
        p_change_value = beta_t / (self.n_bernoilli_features)
        p_stay_value = 1 - beta_t + p_change_value

        log_p_i = jnp.where(X_next == X_prev, jnp.log(p_stay_value), jnp.log(p_change_value))

        noise_per_node = jnp.sum(log_p_i, axis=-1)
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        noise_step_value = -T*noise_per_graph
        noise_rewards_arr = noise_rewards_arr.at[reward_idx].set(noise_rewards_arr[reward_idx] - noise_step_value)
        return noise_rewards_arr

    def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
        log_probs = jax.ops.segment_sum(spin_log_probs, node_graph_idx, n_graph)
        return log_probs
