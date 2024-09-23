import jax
import jax.numpy as jnp
from jax import grad
from flax import linen as nn
from DiffNetwork import DiffNetwork
from typing import Callable
import numpy as np


class DiffScanModel(nn.Module):
    n_features_list_prob: np.ndarray

    n_features_list_nodes: np.ndarray
    n_features_list_edges: np.ndarray
    n_features_list_messages: np.ndarray

    n_features_list_encode: np.ndarray
    n_features_list_decode: np.ndarray
    beta_list: list

    n_diffusion_steps: int
    n_message_passes: int
    message_passing_weight_tied: bool = True

    linear_message_passing: bool = True

    energy_function_func: Callable
    noise_potential_func: Callable
    calc_loss_func: Callable
    beta_list: list
    n_diff_steps = 5

    def setup(self):

        lam_diff_network = lambda x: DiffNetwork(n_features_list_prob=self.n_features_list_prob,
								n_features_list_nodes=self.n_features_list_nodes,
								n_features_list_edges=self.n_features_list_edges,
								n_features_list_messages=self.n_features_list_messages,
								n_features_list_encode=self.n_features_list_encode,
								n_features_list_decode=self.n_features_list_decode,
								n_diffusion_steps = self.n_diffusion_steps,
								n_message_passes=self.n_message_passes,
								message_passing_weight_tied=self.message_passing_weight_tied,
								linear_message_passing=self.linear_message_passing,
                                energy_function_func = self.energy_function_func,
                                noise_potential_func = self.noise_potential_func)

        ### TODO implement tradeoff between remat steps and diff steps
        self.DiffScanner = nn.remat_scan(
            lam_diff_network,
            variable_broadcast="params",
            split_rngs={"params": False}, lengths=(self.n_diff_steps, 1))

    def _init_params(self):
        ### TODO implement
        pass

    def __init__X(self, graphs, key):
        key, subkey = jax.random.split(key)

        nodes = graphs.nodes
        p_uniform = jnp.log(0.5 * jnp.ones((nodes.shape[0], self.N_basis_states, 2)))

        X_init = jax.random.categorical(key=subkey,
                                        logits=p_uniform,
                                        axis=-1,
                                        shape=(nodes.shape[0], self.N_basis_states))

        one_hot_state = jax.nn.one_hot(X_init, num_classes=2)
        X_init = jnp.expand_dims(X_init, axis=-1)

        spin_logits = jnp.log(p_uniform)

        spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
        return X_init, spin_logits, spin_log_probs, key

    def _init_metrics(self, X_init):
        log_p_prev_per_node = jnp.zeros((self.n_diffusion_steps, X_init.shape[0], X_init.shape[1]))
        prob_over_diff_steps = jnp.zeros((self.n_diffusion_steps,))
        Noise_loss_over_diff_steps = jnp.zeros((self.n_diffusion_steps,))
        Energy_over_diff_steps = jnp.zeros((self.n_diffusion_steps,))

        metric_logs = {"prob_over_diff_steps": prob_over_diff_steps,
                       "Noise_loss_over_diff_steps": Noise_loss_over_diff_steps,
                       "Energy_over_diff_steps": Energy_over_diff_steps}
        return log_p_prev_per_node, metric_logs

    def __call__(self, j_graph, T, key):
        ### TODO calculate scan utils here and pass it to x?
        ### TODO pass parameters to DiffNetwork here
        idx = 0
        X_init, spin_logits_init, spin_log_probs_init, key = self.__init__X(j_graph, key)
        log_p_prev_per_node, metric_logs = self._init_metrics(X_init)
        figures = {}

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=self.N_basis_states)

        input_dict = {
            "j_graph": j_graph,
            "X_curr": X_init,
            "key": batched_key,
            "spin_logits": spin_logits_init,
            "spin_log_probs": spin_log_probs_init,
            "log_p_prev_per_node": log_p_prev_per_node,

            "train_state": {
                "temp": T,
                "beta_list": self.beta_list,
                "diff_step": idx
            },
            "losses": {
                "L_entropy": 0.,
                "L_noise": 0.,
                "L_energy": 0.
            },
            "metric_logs": metric_logs,
            "figures": figures,
            "metrics": {},
        }

        output_dict = self.DiffScanner()(input_dict)

        L_entropy = output_dict["losses"]["L_entropy"]
        L_noise = output_dict["losses"]["L_noise"]
        L_energy = output_dict["losses"]["L_energy"]

        # metrics = {key: log_dict_list["metrics"][key][-1] for key in log_dict_list["metrics"].keys()}
        # losses = {key: jnp.sum(log_dict_list["losses"][key]) for key in log_dict_list["losses"].keys()}
        # figures = {key: jnp.sum(log_dict_list["figures"][key]) for key in log_dict_list["figures"].keys()}

        Loss = self.calc_loss_func(L_entropy, L_noise, L_energy, T)

        log_dict = {"Losses": losses,
                    "metrics": metrics,
                    "figures": figures,
                    "log_p_0": spin_logits,
                    "X_0": X_0,
                    "spin_log_probs": spin_log_probs,
                    }

        return Loss, (log_dict, key)
