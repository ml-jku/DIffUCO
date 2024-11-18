from functools import partial

import pickle
import datetime

import jax
import jax.numpy as jnp
from flax import linen as nn
import os
import sys
sys.path.append("..")

from Jraph_creator.JraphCreator import create_graph
from Energies.energy import hamiltonian
from .U_Net import UNet


class AutoregressiveNN(nn.Module):
    grid_size: int
    epsilon: float = 1e-7
    features: int = 64
    cnn_features: int = 32
    n_layers: int = 1

    @nn.compact
    def __call__(self, x):
        L = self.grid_size
        x = x.reshape(( x.shape[0],L, L, 1))
        size = L
        pos_x, pos_y = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing='ij')
        pos_x = jnp.repeat(pos_x[None,...,jnp.newaxis], x.shape[0], axis = 0)
        pos_y = jnp.repeat(pos_y[None,...,jnp.newaxis], x.shape[0], axis = 0)

        x = jnp.concatenate([x, pos_x/size - 0.5, pos_y/size - 0.5], axis = -1)

        #print("x.shape", x.shape)
        #print(x)
        x = UNet(n_layers = self.n_layers, features=self.cnn_features )(x)

        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))
        x = jnp.mean(x, axis = -2)

        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)
        x = nn.log_softmax(x)
        #print("log softmax", x)
        return x


    @partial(jax.jit, static_argnums=(0,))
    def logprobs(self, sample, params):
        batch_size, num_spins = sample.shape
        xhat = jnp.zeros((batch_size, num_spins, 2))
        x = jnp.zeros((batch_size, num_spins))
        for i in range(num_spins):
            prob = self.apply(params, x)
            x = x.at[:, i].set(sample[:, i])
            xhat = xhat.at[:, i, 0].set(prob[:, 0])
            xhat = xhat.at[:, i, 1].set(prob[:, 1])
        return xhat


    # @partial(jax.jit, static_argnums=(0, 1, 2))
    # def generate_sample(self, num_spins, batch_size, params, key, epsilon):
    #     s = jnp.zeros((batch_size, num_spins))
    #     s2 = jnp.zeros((batch_size, num_spins, 2))
    #     for i in range(num_spins):
    #         x_hat = self.apply(params, s)
    #         clipped_log_prob_x_hat = jnp.clip(x_hat, jnp.log(epsilon), jnp.log(1 - epsilon))
    #         key, subkey = jax.random.split(key)
    #         s = s.at[:, i].set(jax.random.bernoulli(subkey, jnp.exp(clipped_log_prob_x_hat[:, 0])).astype(jnp.float32) * 2 - 1)
    #         s2 = s2.at[:, i, 0].set(jnp.squeeze(clipped_log_prob_x_hat[:, 0]))
    #         s2 = s2.at[:, i, 1].set(jnp.squeeze(clipped_log_prob_x_hat[:, 1]))
    #     return s, s2

    def generate_sample_step(self, carry, x):
        s, s2, params, key, epsilon = carry
        i =  x

        x_hat__ = self.apply(params, s)
        x_hat = x_hat__
        #print("model out", x_hat__.shape, s.shape)
        clipped_log_prob_x_hat = jnp.clip(x_hat, jnp.log(epsilon), jnp.log(1 - epsilon))
        key, subkey = jax.random.split(key)
        #print(clipped_log_prob_x_hat.shape, "here")
        sampled_value = jax.random.bernoulli(subkey, jnp.exp(clipped_log_prob_x_hat[:, 0])) * 2 - 1

        s = s.at[:, i, 0].set(sampled_value)
        s2 = s2.at[:, i, 0].set(jnp.squeeze(clipped_log_prob_x_hat[:, 0]))
        s2 = s2.at[:, i, 1].set(jnp.squeeze(clipped_log_prob_x_hat[:, 1]))

        x += 1
        return (s, s2, params, key, epsilon), s

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def generate_sample(self, N, batch_size, params, key, epsilon):
        s = jnp.zeros((batch_size, N, 1), dtype=jnp.int32)
        s2 = jnp.zeros((batch_size, N, 2))

        all_sample_list = [s]
        init_carry = (s, s2, params, key, epsilon)
        (s, s2, _, _, _), sample_list = jax.lax.scan(self.generate_sample_step, init_carry, jnp.arange(0, N))
        all_sample_list.extend(sample_list)
        all_sample_arr = jnp.array(all_sample_list)
        prev_sample_arr = all_sample_arr[:-1]
        next_sample_arr = all_sample_arr[1:]

        return s[...,0], s2, prev_sample_arr, next_sample_arr

    @partial(jax.jit, static_argnums=(0,))
    def log_likelihood(self, sample, log_probs):
        mask = (sample + 1) / 2
        log_prob = jnp.where(mask == 1, log_probs[:, :, 0], log_probs[:, :, 1])
                    #(log_probs[:, :, 0] * mask + log_probs[:, :, 1] * (1 - mask)))
        log_prob = log_prob.reshape(log_prob.shape[0], -1).sum(axis=1)
        return log_prob


    @partial(jax.jit, static_argnums=(0,))
    def compute_log_likelihood_of_sample(self, params, prev_sample_arr, samples):
        log_probs = self.apply(params, prev_sample_arr)
        mask = (samples + 1) / 2
        log_prob = jnp.where(mask == 1, log_probs[:, 0], log_probs[:, 1])
        state_log_prob = log_prob.sum(axis=-1)
        return state_log_prob

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
    def save_dict(unbiased_dict, wandb_id, seed, path_to_models = "/models", dict_name = "unbiased_dict"):
        current_file_path = os.path.abspath(__file__)
        # Get the parent directory of the current file
        parent_folder = os.path.dirname(os.path.dirname(current_file_path))
        path_folder = f"{parent_folder}{path_to_models}/"

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        filename = f"{wandb_id}_{dict_name}_{seed}.pickle"

        with open(os.path.join(path_folder, filename), 'wb') as f:
            print("save dict to", os.path.join(path_folder, filename))
            pickle.dump(unbiased_dict, f)

    @staticmethod
    def load_dict( wandb_id, seed, path_to_models = "/models", dict_name = "unbiased_dict"):
        
        current_file_path = os.path.abspath(__file__)

        # Get the parent directory of the current file
        parent_folder = os.path.dirname(os.path.dirname(current_file_path))

        # print(f"Current File Path: {current_file_path}")
        # print(f"Parent Folder: {parent_folder}")
        path_folder = f"{parent_folder}{path_to_models}/"

        filename = f"{wandb_id}_{dict_name}_{seed}.pickle"

        with open(os.path.join(path_folder, filename), 'rb') as f:
            unbaised_dict = pickle.load( f)
        return unbaised_dict


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



if __name__ == "__main__":
    n = 9
    batch_size = 3
    rng = jax.random.PRNGKey(0)
    ann = AutoregressiveNN()

    # Initialize parameters
    sample_input = jnp.ones((1, n))
    params = ann.init(rng, sample_input)['params']

    # Generate a sample
    generated_sample, log_probs = ann.generate_sample(n, batch_size, params, rng, 0.01)
    print(generated_sample)

    # x_hat2 = ann.probs(generated_sample, params)

    # assert jnp.array_equal(x_hat1, x_hat2)

    # Example log probabilities computation
    batch_sample = jnp.ones((10, n))
    # xhat = ann.probs(batch_sample, params)
    log_prob = ann.log_likelihood(generated_sample, log_probs)
    print(log_prob)

    # calculate energy
    generated_sample = (generated_sample + 1) / 2
    n = 3
    graph = create_graph(n)
    energy= hamiltonian(graph, generated_sample)
    print(energy)
