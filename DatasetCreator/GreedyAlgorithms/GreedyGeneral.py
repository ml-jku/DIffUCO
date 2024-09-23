import numpy as np

def random_greedy(H_graph, iter_fraction = 1):
    num_nodes = H_graph.nodes.shape[0]
    num_iterations = int(iter_fraction*num_nodes)


    bins = np.random.randint(0, high = 2, size = (num_nodes,1))
    spins = 2*bins - 1

    best_Energy = calculateEnergy(H_graph, spins) ### TODO compute Energy here
    for i in range(num_iterations):
        sampled_site = np.random.randint(0, high = num_nodes)

        new_spins = spins.copy()
        new_spins[sampled_site, 0] = -1*new_spins[sampled_site, 0]

        ### TODO computeEnegryHere
        new_Energy = calculateEnergy(H_graph, new_spins)

        if(new_Energy <= best_Energy):
            spins = new_spins
            best_Energy = new_Energy
        else:
            pass

    return float(best_Energy), spins


def random(H_graph):

    ### TODO Do this with BFS or DFS
    ### TODO adapt this to a general Energy function
    nodes = H_graph.nodes
    spins = np.random.randint(0,2, size=(nodes.shape[0],1))*2-1

    best_Energy = float(calculateEnergy(H_graph, spins))

    return best_Energy, spins

def AutoregressiveGreedy(H_graph):

    ### TODO Do this with BFS or DFS
    ### TODO adapt this to a general Energy function
    nodes = H_graph.nodes
    spins = np.zeros((nodes.shape[0], 1))

    for i in range(spins.shape[0]):

        spins_up = spins.copy()
        spins_up[i,0] = 1
        Energy_up = calculateEnergy(H_graph, spins_up)

        spins_down = spins.copy()
        spins_down[i,0] = -1
        Energy_down = calculateEnergy(H_graph, spins_down)

        if(Energy_down < Energy_up):
            spins = spins_down
        else:
            spins = spins_up

    best_Energy = float(calculateEnergy(H_graph, spins))

    return best_Energy, spins


def calculateEnergy(H_graph, spins):

    Energy_messages = H_graph.edges*spins[H_graph.senders] * spins[H_graph.receivers]

    HB =  np.sum(Energy_messages, axis = 0)
    HA = np.sum(H_graph.nodes*spins, axis = 0)

    return HA + HB