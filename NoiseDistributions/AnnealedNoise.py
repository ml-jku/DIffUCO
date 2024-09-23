from .BaseNoise import BaseNoiseDistr
import jax.numpy as jnp
import jax
from functools import partial

class AnnealedNoiseDistr(BaseNoiseDistr):

    def __init__(self,  config ):
        super().__init__(config)
        self.config["beta_factor"] = 1.
        self.vmapped_relaxed_energy_for_Loss = self.config["vmapped_energy_loss_func"]
        self.vmapped_relaxed_energy = self.config["vmapped_energy_func"]
        ### TODO initialize energy function here

    def beta_t_func(self, t, n_diffusion_steps, k=1.):
        if(self.config["noise_potential"] == "combined__"):

            beta = 1-jnp.cos( jnp.pi/2 * ( n_diffusion_steps - (t + 1 ) )/n_diffusion_steps )
            beta = max([beta, 0.])
            print(beta, t , "beta")
        else:
            beta = k * (1. - 1. * (t + 1) / (n_diffusion_steps))
        return beta

    def combine_losses(self, L_entropy, L_noise, L_energy, T):
        return -T * L_entropy + L_noise + L_energy

    def calculate_noise_distr_reward(self, noise_distr_step, entropy_reward):
        return -(noise_distr_step - entropy_reward)

    def get_log_p_T_0(self, jraph_graph, X_prev, X_next, t_idx, T):
        T = jnp.max(jnp.array([T, 10**-6]))
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jraph_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        gamma_t = self.get_gamma_t(t_idx)
        Noise_Energy_per_graph, _, _ = self.vmapped_relaxed_energy_for_Loss(jraph_graph, X_prev, node_gr_idx)
        Noise_Energy_per_graph = jnp.squeeze(Noise_Energy_per_graph, axis = -1)
        log_p = (-1)*gamma_t/T*Noise_Energy_per_graph
        return log_p

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_loss(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, model_step_idx, node_gr_idx, T):
        gamma_t = self.beta_arr[model_step_idx]
        Noise_Energy_per_graph, _, _ = self.vmapped_relaxed_energy_for_Loss(jraph_graph, spin_logits_prev, node_gr_idx)
        Noise_Energy_per_graph = jnp.squeeze(Noise_Energy_per_graph, axis = -1)
        return (-1)*gamma_t*Noise_Energy_per_graph, jnp.sum(log_p_prev_per_node[:-1], axis = 0)

    @partial(jax.jit, static_argnums=(0,))
    def calc_noise_step(self, jraph_graph, X_prev, X_next, model_step_idx, node_gr_idx, T, noise_rewards_arr):
        gamma_t = self.beta_arr[model_step_idx]
        reward_idx = jnp.where(model_step_idx - 1 < 0, jnp.zeros_like(model_step_idx), model_step_idx - 1)  ### when model_step_idx - 1 == 1 hamma_t shoudl be 0!
        Noise_Energy_per_graph, _, _ = self.vmapped_relaxed_energy(jraph_graph, X_prev, node_gr_idx)
        Noise_Energy_per_graph = jnp.squeeze(Noise_Energy_per_graph, axis = -1)
        noise_step_value = gamma_t*Noise_Energy_per_graph
        noise_rewards_arr = noise_rewards_arr.at[reward_idx].set(noise_rewards_arr[reward_idx] - noise_step_value)
        return noise_rewards_arr




