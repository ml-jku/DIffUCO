from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
def calculate_ising_free_energy_exact(beta, L):
    """
    Calculate the free energy per spin of the NxN lattice
    """
    Z, _ = calculate_ising_partition_function(beta, L)
    return -1 / beta * np.log(Z) / L ** 2


def calculate_ising_partition_function(beta, L):
    """
    Calculate the partition function of the NxN lattice

    https://journals.aps.org/pr/pdf/10.1103/PhysRev.185.832
    """
    Z_1 = 1
    for r in range(0, L):
        gamma = calculate_gamma(2 * r + 1, beta, L)
        Z_1 *= 2 * np.cosh(1 / 2 * L * gamma, dtype=np.float64)

    Z_2 = 1
    for r in range(0, L):
        gamma = calculate_gamma(2 * r + 1, beta, L)
        Z_2 *= 2 * np.sinh(1 / 2 * L * gamma, dtype=np.float64)

    Z_3 = 1
    for r in range(0, L):
        gamma = calculate_gamma(2 * r, beta, L)
        Z_3 *= 2 * np.cosh(1 / 2 * L * gamma, dtype=np.float64)

    Z_4 = 1
    for r in range(0, L):
        gamma = calculate_gamma(2 * r, beta, L)
        Z_4 *= 2 * np.sinh(1 / 2 * L * gamma, dtype=np.float64)

    return 1 / 2 * (2 * np.sinh(2 * beta, dtype=np.float64)) ** (L ** 2 / 2) * (Z_1 + Z_2 + Z_3 + Z_4), (Z_1, Z_2, Z_3, Z_4)


def calculate_gamma(r, beta, L):
    if r == 0:
        gamma = 2 * beta + np.log(np.tanh(beta, dtype=np.float64))
    else:
        c_r = np.cosh(2 * beta, dtype=np.float64) * coth(2 * beta) - np.cos(np.pi * r / L)
        gamma = np.log(c_r + np.sqrt(c_r ** 2 - 1))
    return gamma


def coth(x):
    """
    Coth function
    """
    return np.cosh(x, dtype=np.float64) / np.sinh(x, dtype=np.float64)