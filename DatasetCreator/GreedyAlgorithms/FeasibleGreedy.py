import numpy as  np
from jraph_utils import utils as jutils

def check_for_violations( H_graph, Spins, EnergyFunction):
    Nb_bins = np.expand_dims((Spins +1 ) /2, axis = 0)
    HB = 1
    while(HB != 0):
        if(EnergyFunction == "MVC"):
            Energy, HA, HB, HB_per_node = MVC_Energy(H_graph, Nb_bins)
            if(HB != 0):
                violation_idx = np.argmax(HB_per_node, axis=-1)
                Nb_bins[np.arange(0, Nb_bins.shape[0]), violation_idx] = 1
        elif(EnergyFunction == "MIS" or "MaxCl" in EnergyFunction):
            Energy, HA, HB, HB_per_node = MIS_Energy(H_graph, Nb_bins)
            if(HB != 0):
                violation_idx = np.argmax(HB_per_node, axis=-1)
                Nb_bins[np.arange(0, Nb_bins.shape[0]), violation_idx] = 0

    if(EnergyFunction == "MVC"):
        Energy, HA, HB, HB_per_node = MVC_Energy(H_graph, Nb_bins)
    elif (EnergyFunction == "MIS" or "MaxCl" in EnergyFunction):
        Energy, HA, HB, HB_per_node = MIS_Energy(H_graph, Nb_bins)

    return Energy, HA, HB, np.squeeze(2*Nb_bins - 1, axis = 0)

def MVC_Energy(H_graph, Nb_bins, A= 1, B = 1.1):
    ### remove self loops that were not there in original graph
    iH_graph = jutils.from_jgraph_to_igraph(H_graph)
    H_graph = jutils.from_igraph_to_jgraph(iH_graph)
    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = B * (1-Nb_bins[:,H_graph.senders]) * (1-Nb_bins[:,H_graph.receivers])
    ### TODO add HB_per_node
    Nb_Energy_messages = np.squeeze(Energy_messages, axis=-1)
    Energy_messages_Nb = np.swapaxes(Nb_Energy_messages, 0, 1)
    HB_per_node_Nb = np.zeros((H_graph.nodes.shape[0], Nb_bins.shape[0]))
    np.add.at(HB_per_node_Nb, H_graph.receivers, Energy_messages_Nb)

    HB =  0.5 * np.sum(Energy_messages, axis = -2)
    HA = A* np.sum(Nb_bins, axis = -2)
    #HB = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
    #HA = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)

    return float(HB + HA), float(HA), float(HB), np.swapaxes(HB_per_node_Nb, 0,1)

def MIS_Energy(H_graph, Nb_bins, A= 1, B = 1.1):
    ### remove self loops that were not there in original graph
    iH_graph = jutils.from_jgraph_to_igraph(H_graph)
    H_graph = jutils.from_igraph_to_jgraph(iH_graph)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = B * Nb_bins[:,H_graph.senders] * Nb_bins[:,H_graph.receivers]

    Nb_Energy_messages = np.squeeze(Energy_messages, axis=-1)
    Energy_messages_Nb = np.swapaxes(Nb_Energy_messages, 0, 1)
    HB_per_node_Nb = np.zeros((H_graph.nodes.shape[0], Nb_bins.shape[0]))
    np.add.at(HB_per_node_Nb, H_graph.receivers, Energy_messages_Nb)

    HB =  0.5 * np.sum(Energy_messages, axis = -2)
    HA = -A* np.sum(Nb_bins, axis = -2)
    #HB = jax.ops.segment_sum(HB_per_node, node_gr_idx, n_graph)
    #HA = jax.ops.segment_sum(HA_per_node, node_gr_idx, n_graph)
    return float(HB + HA), float(HA), float(HB), np.swapaxes(HB_per_node_Nb, 0,1)