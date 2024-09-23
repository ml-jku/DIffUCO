import numpy as np


def coth(x):
    """
    Coth function
    """
    return np.cosh(x, dtype=np.float64) / np.sinh(x, dtype=np.float64)

def csch(x):
    """
    Csch function
    """
    return 1 / np.sinh(x, dtype=np.float64)

def calculate_gamma_prime(r, beta, L):
    if r == 0:
        gamma_prime = csch(beta) / np.cosh(beta) + 2
    else:
        cl = np.cosh(2 * beta, dtype=np.float64) * coth(2 * beta) - np.cos(np.pi * r / L)

        nenner = 2 * (np.cosh(2 * beta) - coth(2 * beta) * csch(2 * beta))
        zaehler = np.sqrt((cl) ** 2 - 1)
        gamma_prime = nenner / zaehler
    return gamma_prime

def calculate_gamma(r, beta, L):
    if r == 0:
        gamma = 2 * beta + np.log(np.tanh(beta, dtype=np.float64))
    else:
        c_r = np.cosh(2 * beta, dtype=np.float64) * coth(2 * beta) - np.cos(np.pi * r / L)
        gamma = np.log(c_r + np.sqrt(c_r ** 2 - 1))
    return gamma
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

def calculate_ising_internal_energy(beta, L):
    """
    Calculate the internal energy per spin of the NxN lattice

    https://journals.aps.org/pr/pdf/10.1103/PhysRev.185.832
    """
    Z, (Z_1, Z_2, Z_3, Z_4) = calculate_ising_partition_function(beta, L)

    Z_1_prime = 0
    for r in range(0, L):
        gamma = calculate_gamma(2 * r + 1, beta, L)
        gamma_prime = calculate_gamma_prime(2 * r + 1, beta, L)
        Z_1_prime += gamma_prime * np.tanh(1 / 2 * L * gamma, dtype=np.float64)
    Z_1_prime = 1 / 2 * L * Z_1_prime * Z_1

    Z_2_prime = 0
    for r in range(0, L):
        gamma = calculate_gamma(2 * r + 1, beta, L)
        gamma_prime = calculate_gamma_prime(2 * r + 1, beta, L)
        Z_2_prime += gamma_prime * coth(1 / 2 * L * gamma)
    Z_2_prime = 1 / 2 * L * Z_2_prime * Z_2

    Z_3_prime = 0
    for r in range(0, L):
        gamma = calculate_gamma(2 * r, beta, L)
        gamma_prime = calculate_gamma_prime(2 * r, beta, L)
        Z_3_prime += gamma_prime * np.tanh(1 / 2 * L * gamma, dtype=np.float64)
    Z_3_prime = 1 / 2 * L * Z_3_prime * Z_3

    Z_4_prime = 0
    for r in range(0, L):
        gamma = calculate_gamma(2 * r, beta, L)
        gamma_prime = calculate_gamma_prime(2 * r, beta, L)
        Z_4_prime += gamma_prime * coth(1 / 2 * L * gamma)
    Z_4_prime = 1 / 2 * L * Z_4_prime * Z_4

    return -coth(2 * beta) - 1 / L ** 2 * (
            np.sum([Z_1_prime, Z_2_prime, Z_3_prime, Z_4_prime]) / np.sum([Z_1, Z_2, Z_3, Z_4]))