import math

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
from Network.AutoregressiveNN import AutoregressiveNN
from Jraph_creator.JraphCreator import create_graph
from Energies.energy import hamiltonian
from ising_free_energy import calculate_ising_free_energy_exact


def free_energy(beta, z_hat, L):
    return -jnp.log(z_hat) / beta / L ** 2

def entropy(beta, samples, z_hat, H_Graph):
    return beta * hamiltonian(H_Graph, samples) + jnp.log(z_hat)

def calc_w_hat(log_prob_per_state, target_temp, H_Graph, sample):
    log_prob_boltz = -(1 / target_temp) * hamiltonian(H_Graph, sample)
    log_w_hat = log_prob_boltz - log_prob_per_state
    w_hat = jnp.exp(log_w_hat)
    return w_hat

class ImportanceSampler():
    def run(self, sample, log_probs):
        loglikelihood_vals = ann.log_likelihood(sample, log_probs)
        ######
        # x_hat_ = jnp.exp(x_hat[:, :, 0].sum(axis=1))
        # w_hat = jnp.exp(-(1/target_temp) * hamiltonian(H_Graph, sample)) / x_hat_

        ######
        w_hat = calc_w_hat(loglikelihood_vals, target_temp, H_Graph, sample)


        Z_hat = w_hat.mean() # eq. (10)

        std_error = jnp.std(w_hat) / jnp.sqrt(len(sample))

        # eq. (6)
        O_F = free_energy(1 / target_temp, Z_hat, int(math.sqrt(num_spins)))
        O_F_p = free_energy(1/target_temp, Z_hat + std_error, int(math.sqrt(num_spins)))
        O_F_m = free_energy(1 / target_temp, Z_hat - std_error, int(math.sqrt(num_spins)))


        #jax.debug.print("Free energy estimate: {}", O_F)
        O_F_exact = calculate_ising_free_energy_exact(1/target_temp, int(math.sqrt(num_spins)))
        #jax.debug.print("Free energy exact: {}", O_F_exact)
        #jax.debug.print("Difference: {}", O_F - O_F_exact)


        # O_F_nis = (jnp.repeat(O_F, w_hat.shape[0]) * w_hat).mean() / Z_hat  # eq. (9)

        O_S = entropy(target_temp, sample, Z_hat, H_Graph) # eq. (7)
        O_S_nis = (O_S * w_hat).mean() / Z_hat  # eq. (9)
        return O_F - O_F_exact, O_F_p - O_F_exact, O_F_m - O_F_exact

if __name__ == "__main__":
    ## apply importance sampling
    from jax import config

    seed = 1705
    key = jax.random.key(seed)
    n = 10
    num_spins = n * n
    target_temp = 1 / 0.44
    H_Graph = create_graph(int(jnp.sqrt(num_spins)))
    ann = AutoregressiveNN()
    params = ann.load_params("20240819191157_10_40000.pickle")

    config.update("jax_enable_x64", True)
    importance = ImportanceSampler()

    batch_size = 600
    iterations = 400

    epsilon = [0, 0.001, 0.01, 0.05, 0.1]

    fig, ax = plt.subplots(1, 1)
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(epsilon)))
    x = np.arange(iterations)

    for i, eps in enumerate(epsilon):
        O_F_diffs = []
        O_F_diffs_p = []
        O_F_diffs_m = []

        sample_acc = None
        log_probs_acc = None
        for rep in range(iterations):
            print(rep)
            key, subkey = jax.random.split(key)
            sample, log_probs = ann.generate_sample(num_spins, batch_size, params, subkey, eps)

            if sample_acc is not None:
                sample_acc = jnp.concat((sample_acc, sample))
                log_probs_acc = jnp.concat((log_probs_acc, log_probs))
            else:
                sample_acc = sample
                log_probs_acc = log_probs

            O_F_diff, O_F_diff_p, O_F_diff_m = importance.run(sample_acc, log_probs_acc)
            O_F_diffs.append(float(O_F_diff))
            O_F_diffs_p.append(float(O_F_diff_p))
            O_F_diffs_m.append(float(O_F_diff_m))

        ax.plot(x, O_F_diffs, label=str(eps), color=colors[i])
        plt.fill_between(x, O_F_diffs_m, O_F_diffs_p, color=colors[i], alpha=0.1)

    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.legend()
    plt.show()