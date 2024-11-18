import openjij as oj
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import networkx as nx
import jax

def calculate_energy(samples, J):
    """
    Calculate the energy of given samples from the sampler.

    Args:
    - samples (array): Array of shape (batch_size, n) where n is the number of spins, and each sample is a spin configuration.
    - J (dict): Dictionary where the keys are tuples (i, j) representing the pair of spins and the values are the coupling constants J_{ij}.

    Returns:
    - energy (array): Energy for each sample.
    """
    energy = []
    
    # Iterate over each sample
    for sample in samples:
        sample_energy = 0
        # For each edge in the Ising model, calculate the interaction term
        for (i, j), Jij in J.items():
            # Get the spins of i and j from the sample
            spin_i = sample[i]
            spin_j = sample[j]
            # Accumulate the energy contribution
            sample_energy += Jij * spin_i * spin_j
        energy.append(sample_energy)
    
    return np.array(energy)


# Parameters for the grid
L = 16           # Grid size (LxL)
n = L * L        # Total number of spins

# Create the 2D grid with periodic boundary conditions
nx_graph = nx.grid_2d_graph(L, L, periodic=True)

# Create a mapping from node (x, y) to a 1D index
node_idx = {}
for i, node in enumerate(nx_graph.nodes):
    node_idx[node] = i

# Define the edges in the grid and create the J matrix
print(nx_graph.edges)
print(len(nx_graph.edges))
print(4*L**2)
edge_idx = []
for i, edge in enumerate(nx_graph.edges):
    sender, receiver = edge
    edge_idx.append([node_idx[sender], node_idx[receiver]])
edge_idx = np.array(edge_idx)

# Prepare the interaction (J) matrix
undir_senders = edge_idx[:, 0]
undir_receivers = edge_idx[:, 1]
receivers = np.ravel(undir_receivers)
senders = np.ravel(undir_senders)
edges = np.ones((senders.shape[0], 1))

# Initialize random couplings
key = jax.random.PRNGKey(0)
undir_couplings = -np.array(2 * jax.random.randint(key, (senders.shape[0],), 0, 2) - 1)
print(undir_couplings)
# Set up the Ising model with random couplings
h = {i: 0 for i in range(n)}  # Initialize h with zeros
J = {}
factor = 1.
for idx, (r, s) in enumerate(zip(receivers, senders)):
    J[(r, s)] = factor * undir_couplings[idx]

# Define the annealing parameters
N = 2000
reps = 1
T_end = 1.5
N_warmup = 10
N_equil = 10
steps = N + N_equil
epochs = reps * steps

def exp_annealing(epoch, epochs, T_target, N_equil):
    if epoch < epochs - N_equil - 1:
        T_curr = T_target * 1 / (1 - 0.998 ** (1 * epoch + 1))
    else:
        T_curr = T_target
    return T_curr

# Temperature schedule
temperatures = []
for rep in range(reps):
  temperatures.extend([exp_annealing(i, steps, T_end, N_equil) for i in range(steps)])

# Initialize the SA sampler
sampler = oj.SASampler()

# Perform sampling and log the spin configuration and energy at each step
batch_size = 100
sample_history = []
energy_history = {}
energy_history["mean"] = []
energy_history["min"] = []
energy_history["max"] = []

for T in tqdm.tqdm(temperatures):
    # Define a single-step schedule for the current temperature
    temp_schedule = [[1 / T, 1]]  # 1 / T is the inverse temperature (beta)
    
    # Run MCMC update for the current temperature step
    response = sampler.sample_ising(h, J, schedule=temp_schedule, num_reads=batch_size)
    energy = response.record['energy']
    # Store the first sample and its energy from the response
    mean_energy = np.mean(energy)  # Corresponding energy
    #print(response.record['energy'].shape,response.record['sample'].shape)
    #print(calculate_energy([response.record['sample'][0]], J), response.record['energy'][0])
    # Append the results to the history lists
    energy_history["mean"].append(mean_energy)
    energy_history["min"].append(np.min(energy))
    energy_history["max"].append(np.max(mean_energy))
    sample = response.record['sample']
    sample_history.append(np.std(sample))

# Plot the energy progression over the annealing process
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), sample_history, marker='o', linestyle='-', color='b')
plt.title("Energy Progression Over MCMC Updates")
plt.xlabel("MCMC Step")
plt.ylabel("std sample")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), np.array(energy_history["mean"])/L**2, marker='o', linestyle='-', color='b')
plt.title("Energy Progression Over MCMC Updates")
plt.xlabel("MCMC Step")
plt.ylabel("Energy mean / L**2")
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(range(epochs), np.array(energy_history["mean"]), marker='o', linestyle='-', color='b')
plt.title("Energy Progression Over MCMC Updates")
plt.xlabel("MCMC Step")
plt.ylabel("Energy mean")
plt.grid()
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(range(epochs), np.array(energy_history["min"]), marker='o', linestyle='-', color='b')
plt.title("Energy Progression Over MCMC Updates")
plt.xlabel("MCMC Step")
plt.ylabel("Energy min")
plt.grid()
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(range(epochs), np.array(energy_history["max"]), marker='o', linestyle='-', color='b')
plt.title("Energy Progression Over MCMC Updates")
plt.xlabel("MCMC Step")
plt.ylabel("Energy max")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), temperatures, marker='o', linestyle='-', color='b')
plt.axhline(y = T_end)
plt.xlabel("MCMC Step")
plt.ylabel("Temps")
plt.yscale("log")
plt.grid()
plt.show()