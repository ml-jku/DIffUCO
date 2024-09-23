from functools import partial

import pickle
import datetime

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import os
import sys
sys.path.append("..")
import numpy as np
from Jraph_creator.JraphCreator import create_graph
from Energies.energy import hamiltonian
from .GNN_modules import EncodeProcessDecode
from .MLPs import ProbMLP
import jraph

class AutoregressiveGraphNN(nn.Module):
    nh: int = 34
    n_message_passes: int = 1

    def setup(self):
        nh = self.nh
        self.n_features_list_prob_MLP = [nh, nh, 2]
        # self.n_features_list_nodes = jnp.asarray([nh, nh])
        # self.n_features_list_edges = jnp.asarray([nh])
        # self.n_features_list_messages = jnp.asarray([nh])
        #
        # self.n_features_list_encode = jnp.asarray([nh])
        # self.n_features_list_decode = jnp.asarray([nh])

        self.GNN = EncodeProcessDecode( n_features_list_nodes = [nh, nh, 2],
                                        n_features_list_edges = [nh, nh],
                                        n_features_list_messages =  [nh, nh],
                                        n_features_list_encode = [nh, nh],
                                        n_features_list_decode = [nh, nh],
                                        dtype = jnp.float32,
                                        edge_updates = False,
                                        linear_message_passing = True,
                                        n_message_passes = self.n_message_passes,
                                        weight_tied = False,
                                        mean_aggr = True,
                                        graph_norm = True)
        self.ProbMLP = ProbMLP(self.n_features_list_prob_MLP, jnp.float32)

    #@partial(flax.linen.jit, static_argnums=(0, 1,))
    def __call__(self, H_graph, x):
        node_features = self.GNN(H_graph, x)
        out_probs = self.ProbMLP(node_features)
        return out_probs

    def vmap_apply(self):
        #vmap_apply = jax.vmap(lambda x: self.apply(params, H_graph, x), in_axes=(0,))
        vmap_apply = jax.vmap(self.apply, in_axes=(None, None, 0))
        return vmap_apply

    def generate(self, params, H_graph, x):
        return self.apply(params, H_graph, x)


class AutoregressiveTrainer():

    def __init__(self, model, batch_size):
        ### TODO init vmap generate
        self.model = model
        self.batch_size = batch_size
        self.vmap_generate = jax.vmap(model.generate, in_axes = (None, None, 0))
        pass

    @partial(jax.jit, static_argnums=(0,))
    def logprobs(self, H_Graph, sample, params):
        batch_size, num_spins = sample.shape
        xhat = jnp.zeros((batch_size, num_spins, 2))
        x = jnp.zeros((batch_size, num_spins))
        for i in range(num_spins):
            prob = self.vmap_generate(params, H_Graph, x[...,None])[:,i]
            x = x.at[:, i].set(sample[:, i])
            xhat = xhat.at[:, i, 0].set(prob[:, 0])
            xhat = xhat.at[:, i, 1].set(prob[:, 1])
        return xhat

    def generate_sample_step(self, carry, x):
        s, s2, params, H_Graph, key, epsilon = carry
        i =  x
        x += 1

        x_hat__ = self.vmap_generate(params, H_Graph, s)
        x_hat = x_hat__[:, i]

        clipped_log_prob_x_hat = jnp.clip(x_hat, jnp.log(epsilon), jnp.log(1 - epsilon))
        key, subkey = jax.random.split(key)
        sampled_value = jax.random.bernoulli(subkey, jnp.exp(clipped_log_prob_x_hat[:, 0])).astype(jnp.float32) * 2 - 1

        s = s.at[:, i, 0].set(sampled_value)
        s2 = s2.at[:, i, 0].set(jnp.squeeze(clipped_log_prob_x_hat[:, 0]))
        s2 = s2.at[:, i, 1].set(jnp.squeeze(clipped_log_prob_x_hat[:, 1]))

        return (s, s2, params, H_Graph, key, epsilon), x

    @partial(jax.jit, static_argnums=(0,))
    def generate_sample(self, H_Graph, params, key, epsilon):
        N = jax.tree_util.tree_leaves(H_Graph.nodes)[0].shape[0]
        s = jnp.zeros((self.batch_size, N, 1))
        s2 = jnp.zeros((self.batch_size, N, 2))

        init_carry = (s, s2, params, H_Graph, key, epsilon)
        (s, s2, _, _, _, _), _ = jax.lax.scan(self.generate_sample_step, init_carry, jnp.arange(0, N))

        return s[...,0], s2

    @partial(jax.jit, static_argnums=(0,))
    def log_likelihood(self, sample, log_probs):
        mask = (sample + 1) / 2
        log_prob = (log_probs[:, :, 0] * mask +
                    log_probs[:, :, 1] * (1 - mask))
        log_prob = log_prob.reshape(log_prob.shape[0], -1).sum(axis=1)
        return log_prob

    @staticmethod
    def save_params(params: dict, config, wandb_id, path_to_models = "./AutoRegr_NIS/models"):
        path_folder = f"{path_to_models}/"

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)


        filename = f"{wandb_id}_weights.pickle"

        with open(os.path.join(path_folder, filename), 'wb') as f:
            pickle.dump(params, f)


        filename = f"{wandb_id}_config.pickle"

        with open(os.path.join(path_folder, filename), 'wb') as f:
            pickle.dump(config, f)

    @staticmethod
    def load_params(wandb_id = "", path_to_models="models"):
        cur_path = os.getcwd()
        path_folder = f"{os.path.join(cur_path, path_to_models)}/"
        if wandb_id:
            filename = os.path.join(path_folder, wandb_id)
        else:
            from pathlib import Path
            paths = sorted(Path(path_folder).iterdir(), key=os.path.getmtime)
            filename = paths[-1]

        print("Loading model {}".format(filename))
        with open(filename + "_weights.pickle", "rb") as f:
            params = pickle.load(f)

        with open(filename+ "_config.pickle", "rb") as f:
            config = pickle.load(f)
        return params, config



