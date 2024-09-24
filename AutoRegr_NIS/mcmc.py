import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from Network.AutoregressiveNN import AutoregressiveNN
from Jraph_creator.JraphCreator import create_graph
from Energies.energy import hamiltonian
from internal_energy import calculate_ising_internal_energy
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--wandb_ids', default=["np8r5ikl"], type = str, nargs = "+")
parser.add_argument('--GPU', default="3", type = str)
parser.add_argument('--batch_size', default=1400, type = int)
parser.add_argument('--iterations', default=400, type = int)
parser.add_argument('--epsilons', default=[0.], type = str, nargs = "+")
parser.add_argument('--seeds', default=3, type = int)

args = parser.parse_args()

def log_likelihood(sample, log_probs):
    mask = (sample + 1) / 2
    log_prob = (log_probs[:, :, 0] * mask +
                log_probs[:, :, 1] * (1 - mask))
    log_prob = log_prob.reshape(log_prob.shape[0], -1).sum(axis=1)
    return log_prob

if __name__ == "__main__":
    ## apply importance sampling
    import os
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.96"
    jax.config.update("jax_enable_x64", True)

    seed = 1705
    key = jax.random.key(seed)
    n = 24
    num_spins = n * n
    target_temp = 1 / 0.4407
    H_Graph = create_graph(int(jnp.sqrt(num_spins)))
    ann = AutoregressiveNN(grid_size = 24)
    wandb_id = args.wandb_ids[0]
    params, config = ann.load_params(wandb_id=wandb_id )

    n = config["Ising_size"]
    num_spins = n * n
    target_temp = config["target_temp"]
    H_Graph = create_graph(int(jnp.sqrt(num_spins)))
    ann = AutoregressiveNN(grid_size = n, n_layers = config["n_layers"], features = config["nh_MLP"], cnn_features = config["nh_conv"])
    batch_size = args.batch_size
    epsilon = 0.0
    log_p_list = []
    log_q_list = []
    sample_list = []
    iterations = args.iterations
    x = np.arange(iterations)

    sample, q_s  ,_ , _ = ann.generate_sample(num_spins, batch_size, params, key, epsilon)
    log_q = log_likelihood(sample, q_s)
    log_p = -(1 / target_temp) * hamiltonian(H_Graph, sample)
    for seed in range(args.seeds):
        MCMC_dict = {}
        for i in range(iterations):
            print(i)
            key, subkey = jax.random.split(key)
            sample_prime, q_s_prime ,_ , _ = ann.generate_sample(num_spins, batch_size, params, subkey, epsilon)
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
        MCMC_dict["MCMC_energy"] = O_Energy
        ann.save_dict(MCMC_dict, wandb_id, seed, dict_name="MCMC")

        # fig, ax = plt.subplots(1, 1)
        # ax.plot(x, O_Energy, label="Inner Energy")
        # plt.axhline(y=internal_energy, color='r', linestyle='-')
        # plt.show()



