from .BaseNoise import BaseNoiseDistr
import jax.numpy as jnp
import jax
from functools import partial

class BernoulliNoiseDistr(BaseNoiseDistr):

    def __init__(self, config):
        super().__init__(config)

    def combine_losses(self, L_entropy, L_noise, L_energy, T):
        return -T * L_entropy + L_noise + L_energy

    def calculate_noise_distr_reward(self, noise_distr_step, entropy_reward):
        return -(noise_distr_step - entropy_reward)

    @partial(jax.jit, static_argnums=(0))
    def get_log_p_T_0(self, jraph_graph, X_prev, X_next, t_idx, T):
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jraph_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
        log_p_per_node = self.get_log_p_T_0_per_node(X_prev, X_next, t_idx)

        n_graph = jraph_graph.n_node.shape[0]
        log_p_per_graph = jax.ops.segment_sum(log_p_per_node, node_gr_idx, n_graph)

        return log_p_per_graph

    @partial(jax.jit, static_argnums=(0))
    def get_log_p_T_0_per_node(self, X_prev, X_next, t_idx):

        gamma_t = self.get_gamma_t(t_idx)
        X_next_down = 1 - X_next
        noise_per_node = jnp.sum((X_prev * (X_next * jnp.log(1 - gamma_t) + (X_next_down) * jnp.log(gamma_t)) + (
                1 - X_prev) * ((X_next_down) * jnp.log(1 - gamma_t) + X_next * jnp.log(gamma_t))), axis=-1)

        log_p_per_node = noise_per_node

        return log_p_per_node
    @partial(jax.jit, static_argnums=(0))
    def sample_forward_diff_process(self, X_t_m1, t_idx, key):
        gamma_t = self.get_gamma_t(t_idx)
        X_next_down = 1 - X_t_m1

        log_p_up = X_t_m1 * jnp.log(1 - gamma_t) + (X_next_down) * jnp.log(gamma_t)
        log_p_down = (X_next_down) * jnp.log(1 - gamma_t) + X_t_m1 * jnp.log(gamma_t)
        log_p_per_node = jnp.concatenate([log_p_down[..., None], log_p_up[..., None]], axis=-1)

        key, subkey = jax.random.split(key)
        X_next = jax.random.categorical(key=subkey,
                                        logits=log_p_per_node,
                                        axis=-1,
                                        shape=log_p_per_node.shape[:-1])

        one_hot_state = jax.nn.one_hot(X_next, num_classes=2)
        spin_log_probs = jnp.sum(log_p_per_node * one_hot_state, axis=-1)

        return X_next, spin_log_probs, key

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_loss(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, model_step_idx, node_gr_idx, T):
        gamma_t = self.beta_arr[model_step_idx]
        p_next_up = jnp.exp(spin_logits_next[..., 1])
        p_next_down = 1 - p_next_up  # jnp.exp(spin_logits_next[...,0])
        noise_per_node = jnp.sum((X_prev * (p_next_up * jnp.log(1 - gamma_t) + (p_next_down) * jnp.log(gamma_t)) + (
                    1 - X_prev) * ((p_next_down) * jnp.log(1 - gamma_t) + p_next_up * jnp.log(gamma_t))), axis=-1)
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        return T*noise_per_graph, jnp.sum(log_p_prev_per_node, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step_relaxed(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx):
        p_next_up = jnp.exp(spin_logits_next[..., 1])
        p_next_down = 1 - p_next_up  # jnp.exp(spin_logits_next[...,0])
        noise_per_node = jnp.sum((X_prev * (p_next_up * jnp.log(1 - gamma_t) + (p_next_down) * jnp.log(gamma_t)) + (
                    1 - X_prev) * ((p_next_down) * jnp.log(1 - gamma_t) + p_next_up * jnp.log(gamma_t))), axis=-1)
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        return noise_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step(self, jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr):
        gamma_t = self.beta_arr[model_step_idx]
        reward_idx = model_step_idx
        X_next_up = X_next
        X_next_down = 1 - X_next_up  # jnp.exp(spin_logits_next[...,0])
        noise_per_node = jnp.sum((X_prev * (X_next_up * jnp.log(1 - gamma_t) + (X_next_down) * jnp.log(gamma_t)) + (
                    1 - X_prev) * ((X_next_down) * jnp.log(1 - gamma_t) + X_next_up * jnp.log(gamma_t))), axis=-1)
        n_graph = jraph_graph.n_node.shape[0]
        noise_per_graph = jax.ops.segment_sum(noise_per_node, node_gr_idx, n_graph)

        noise_step_value = -T*noise_per_graph
        noise_rewards_arr = noise_rewards_arr.at[reward_idx].set(noise_rewards_arr[reward_idx] - noise_step_value)
        return noise_rewards_arr

    def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
        log_probs = jax.ops.segment_sum(spin_log_probs, node_graph_idx, n_graph)
        return log_probs

