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
from ising_free_energy import calculate_ising_free_energy_exact
from internal_energy import calculate_ising_internal_energy


def free_energy(beta, log_w_hat, L):
    free_energy = -jax.scipy.special.logsumexp(log_w_hat, axis = 0, b =1 / log_w_hat.shape[0]) / beta / L ** 2
    return free_energy

def free_energy_bounds(beta, log_w_hat, L):
    w_hat = jnp.exp(log_w_hat)
    Z_hat = w_hat.mean()  # eq. (10)

    std_error = jnp.std(w_hat) / jnp.sqrt(log_w_hat.shape[0])

    return -jnp.log(Z_hat + std_error) / beta / L ** 2, -jnp.log(Z_hat - std_error) / beta / L ** 2

def calc_log_w_hat(log_prob_per_state, target_temp, H_Graph, sample, hamiltonian):
    log_prob_boltz = -(1 / target_temp) * hamiltonian(H_Graph, sample)
    log_w_hat = log_prob_boltz - log_prob_per_state
    return log_w_hat

def calculate_eff_sample_size(log_w_hat):
    w_hat = jax.nn.softmax(log_w_hat, axis = -1)
    n_eff = (jnp.sum(w_hat))**2/jnp.sum(w_hat**2)/w_hat.shape[0]
    return n_eff

def calculate_inner_energy(H_Graph, sample, log_w_hat, L, hamiltonian):
    Energy = hamiltonian(H_Graph, sample)
    inner_energy = jnp.sum(jax.nn.softmax(log_w_hat, axis = -1)*Energy)/ L ** 2
    return inner_energy

def calculate_entropy( beta, internal_energy, free_energy):
        """
        Calculate the entropy of the NxN lattice
        """
        return (internal_energy - free_energy) / (1 / beta)

class ImportanceSampler():

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        pass
    def run(self, H_Graph, target_temp, num_spins, ann, sample, log_probs):#
        beta = 1/target_temp
        loglikelihood_vals = ann.log_likelihood( sample, log_probs)
        ######
        # x_hat_ = jnp.exp(x_hat[:, :, 0].sum(axis=1))
        # w_hat = jnp.exp(-(1/target_temp) * hamiltonian(H_Graph, sample)) / x_hat_

        ######
        log_w_hat = calc_log_w_hat(loglikelihood_vals, target_temp, H_Graph, sample, self.hamiltonian)

        # eq. (6)
        O_F = free_energy(1 / target_temp, log_w_hat, int(math.sqrt(num_spins)))
        O_F_p, O_F_m = free_energy_bounds(1/target_temp, log_w_hat, int(math.sqrt(num_spins)))


        #jax.debug.print("Free energy estimate: {}", O_F)
        O_F_exact = calculate_ising_free_energy_exact(1/target_temp, int(math.sqrt(num_spins)))
        O_E_exact = calculate_ising_internal_energy(1/target_temp, int(math.sqrt(num_spins)))
        O_S_exact = calculate_entropy(beta, O_E_exact, O_F_exact)
        #jax.debug.print("Free energy exact: {}", O_F_exact)
        #jax.debug.print("Difference: {}", O_F - O_F_exact)
        Inner_Energy = calculate_inner_energy(H_Graph, sample, log_w_hat, int(math.sqrt(num_spins)), self.hamiltonian)
        Entropy_Estimate = calculate_entropy(beta, Inner_Energy, O_F)
        effective_sample_size = calculate_eff_sample_size(log_w_hat)

        # O_F_nis = (jnp.repeat(O_F, w_hat.shape[0]) * w_hat).mean() / Z_hat  # eq. (9)

        # O_S = entropy(target_temp, sample, Z_hat, H_Graph) # eq. (7)
        # O_S_nis = (O_S * w_hat).mean() / Z_hat  # eq. (9)
        out_dict = {"Free_Energy": O_F, "Free_Energy_lower_bound": O_F_p, "Free_Energy_upper_bound": O_F_m, "Free_Energy_exact": O_F_exact,
                    "Entropy": Entropy_Estimate, "Entropy_exact": O_S_exact, "Inner_Energy": Inner_Energy, "Inner_Energy_exact": O_E_exact,
                    "effective_sample_size": effective_sample_size}
        return out_dict

