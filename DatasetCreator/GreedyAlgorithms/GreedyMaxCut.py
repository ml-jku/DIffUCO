import jax
import numpy as np
import jax.tree_util as tree
from ..EnergyFunctions import calculateEnergy
### TODO implement greedy MaxCut where we randomly sample conifg and randomly flip if improvement

def AutoregressiveGreedy_old(H_graph):
    ### TODO adapt this to a general Energy function
    nodes = H_graph.nodes
    spins = np.zeros((nodes.shape[0], 1))
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = np.arange(n_graph)
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = np.repeat(graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)

    for i in range(spins.shape[0]):
        External_field_edges = H_graph.edges * spins[H_graph.senders]
        External_field_per_node = jax.ops.segment_sum(External_field_edges, H_graph.receivers, sum_n_node)

        External_field_at_i = External_field_per_node[i]
        if(External_field_at_i == 0):
            spins = spins.at[i].set(1.)
        elif(External_field_at_i > 0):
            spins = spins.at[i].set(-1.)
        else:
            spins = spins.at[i].set(1.)

    Energy_messages = H_graph.edges * spins[H_graph.senders] * spins[H_graph.receivers]
    Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, sum_n_node)
    Energy = jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)

    return Energy

def AutoregressiveGreedy(H_graph, EnergyFunction = "MaxCut"):

    ### TODO Do this with BFS or DFS
    ### TODO adapt this to a general Energy function
    nodes = H_graph.nodes
    spins = np.zeros((nodes.shape[0], 1))

    for i in range(spins.shape[0]):

        spins_up = spins.copy()
        spins_up[i,0] = 1
        Energy_up = calculateEnergy.calc(H_graph, spins_up, EnergyFunction)

        spins_down = spins.copy()
        spins_down[i,0] = -1
        Energy_down = calculateEnergy.calc(H_graph, spins_down, EnergyFunction)

        if(Energy_down < Energy_up):
            spins = spins_down
        else:
            spins = spins_up

    best_Energy = float(calculateEnergy.calc(H_graph, spins, EnergyFunction))

    return best_Energy, spins


def random_greedy(H_graph, iter_fraction = 2., spins = np.array([None]), EnergyFunction = "MaxCut"):
    num_nodes = H_graph.nodes.shape[0]
    num_iterations =int(iter_fraction*num_nodes) + 1

    if (spins.any() == None):
        bins = np.random.randint(0, high = 2, size = (num_nodes,1))
        spins = 2*bins - 1

    best_Energy = calculateEnergy.calc(H_graph, spins, EnergyFunction) ### TODO compute Energy here
    for i in range(num_iterations):
        sampled_site = np.random.randint(0, high = num_nodes)

        new_spins = spins.copy()
        new_spins[sampled_site, 0] = -1*new_spins[sampled_site, 0]

        ### TODO computeEnegryHere
        new_Energy = calculateEnergy.calc(H_graph, new_spins, EnergyFunction)

        if(new_Energy <= best_Energy):
            spins = new_spins
            best_Energy = new_Energy
        else:
            pass

    return float(best_Energy)