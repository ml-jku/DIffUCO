import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from AutoRegr_NIS.Network.AutoregressiveNN import AutoregressiveNN
from AutoRegr_NIS.Jraph_creator.JraphCreator import create_graph
from AutoRegr_NIS.Energies.energy import hamiltonian
from AutoRegr_NIS.internal_energy import calculate_ising_internal_energy


def log_likelihood(sample, log_probs):
    mask = (sample + 1) / 2
    log_prob = (log_probs[:, :, 0] * mask +
                log_probs[:, :, 1] * (1 - mask))
    log_prob = log_prob.reshape(log_prob.shape[0], -1).sum(axis=1)
    return log_prob

if __name__ == "__main__":
    ## apply importance sampling
    jax.config.update("jax_enable_x64", True)

    seed = 1705
    key = jax.random.key(seed)
    n = 10
    num_spins = n * n
    target_temp = 1 / 0.44
    H_Graph = create_graph(int(jnp.sqrt(num_spins)))
    ann = AutoregressiveNN()
    params = ann.load_params()
    batch_size = 2000
    epsilon = 0.0
    log_p_list = []
    log_q_list = []
    sample_list = []
    iterations = 2000
    x = np.arange(iterations)

    sample, q_s = ann.generate_sample(num_spins, batch_size, params, key, epsilon)
    log_q = log_likelihood(sample, q_s)
    log_p = -(1 / target_temp) * hamiltonian(H_Graph, sample)
    for i in range(iterations):
        print(i)
        key, subkey = jax.random.split(key)
        sample_prime, q_s_prime = ann.generate_sample(num_spins, batch_size, params, subkey, epsilon)
        log_q_prime = log_likelihood(sample_prime, q_s_prime)
        log_p_prime = -(1 / target_temp) * hamiltonian(H_Graph, sample_prime)

        ratio = -((log_q_prime + log_p) - (log_q + log_p_prime))
        acceptance_prob = jnp.minimum(1, jnp.exp(ratio))

        key, subkey = jax.random.split(key)
        uniform_random = jax.random.uniform(subkey, shape=acceptance_prob.shape)
        accept = uniform_random <= acceptance_prob
        log_q = jnp.where(accept, log_q_prime, log_q)
        log_p = jnp.where(accept, log_p_prime, log_p)
        accepted_samples = jnp.where(accept[..., None], sample_prime, sample)
        sample = accepted_samples

        log_q_list.append(log_q)
        sample_list.append(accepted_samples)


    internal_energy = calculate_ising_internal_energy(1/target_temp, n)
    O_Energy = jnp.array([hamiltonian(H_Graph, x) for x in sample_list]).mean(axis=1) / n**2

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, O_Energy, label="Inner Energy")
    plt.axhline(y=internal_energy, color='r', linestyle='-')
    plt.show()



