from sympy import Sum, symbols, Indexed, lambdify, simplify, expand, IndexedBase, Idx
import numpy as np
from jraph_utils import utils as jutils

def get_two_body_corr(Energy, x,s,r):
    edge_value = Energy.coeff(x[s] * x[r])
    return edge_value

def get_one_body_corr(Energy, x,n):
    node_value = Energy.coeff(x[n])
    node_value = node_value.coeff(x, 0)
    return node_value

def get_constant(Energy, x):
    constant = Energy.coeff(x, 0)
    return constant

def replace_bins_by_spins(Energy, X, S, idx):
    Energy = Energy.subs(X[idx], (S[idx] + 1) / 2)
    return Energy

def Hamiltonian_to_spin(Sym_H, X, N):
    S = IndexedBase('S')

    for n in range(N):
        Sym_H = replace_bins_by_spins(Sym_H, X,S, n)

    return expand(Sym_H), S

def getHamiltonianGraph(Jgraph, EnergyFunction = "MVC"):
    from utils import SympyHamiltonians
    '''
    This function returns a graph with a Hamiltonian description in spin formulation
    :param Jgraph:
    :param EnergyFunction:
    :return:
    '''
    if (EnergyFunction == "MVC"):
        Sympy_H, X = SympyHamiltonians.MVC(Jgraph)

    num_nodes = Jgraph.nodes.shape[0]
    Spin_SymPy_H, S = Hamiltonian_to_spin(Sympy_H, X, num_nodes)

    coupling_func = lambda s,r: get_two_body_corr(Spin_SymPy_H, S, s,r)
    external_fields = lambda idx: get_one_body_corr(Spin_SymPy_H, S, idx)
    constant = get_constant(Spin_SymPy_H, S)

    external_field_map = map(external_fields, np.arange(0, num_nodes))
    coupling_map = map(coupling_func, Jgraph.senders, Jgraph.receivers)

    edges = np.expand_dims(list(coupling_map), axis = -1)
    edges = np.array(edges, dtype = np.float32)
    nodes = np.expand_dims(list(external_field_map), axis = -1)
    nodes = np.array(nodes, dtype = np.float32)

    self_loops = np.arange(0, num_nodes)
    ### factor two because in energy calc messages are divided by 2
    self_couplings = 2*constant/num_nodes*np.ones((num_nodes,1))
    self_couplings = np.array(self_couplings, dtype=np.float32)

    senders = Jgraph.senders
    receivers = Jgraph.receivers

    senders = np.concatenate([senders, self_loops])
    receivers = np.concatenate([receivers, self_loops])
    edges = np.concatenate([edges, self_couplings])

    n_edge = np.array([edges.shape[0]])
    H_graph = Jgraph._replace(edges = edges, nodes = nodes, senders = senders, receivers = receivers, n_edge = n_edge)

    return H_graph

def multinomial_theorem(vec_1, vec_2, is_variable_1, is_variable_2):

    vec_concat = np.concatenate([vec_1, -vec_2], axis = 0)

    res = np.tensordot(vec_concat[:,np.newaxis], vec_concat[np.newaxis,:], axes=[[1],[0]])

    diagonal_idxs = np.arange(0, vec_concat.shape[0])
    self_corr = res[diagonal_idxs, diagonal_idxs]

    senders = []
    receivers = []
    edges = []
    for i in range(vec_concat.shape[0]):
        for j in range(vec_concat.shape[0]):
            if(i != j):
                senders.extend([i,j])
                receivers.extend([j,i])
                coupling = res[i,j] + res[j,i]
                edges.extend([coupling, coupling])

    senders = np.concatenate([senders, diagonal_idxs, diagonal_idxs])
    receivers = np.concatenate([receivers, diagonal_idxs, diagonal_idxs])
    edges = np.concatenate([edges, self_corr, self_corr])

    return senders, receivers, edges



if(__name__ == "__main__"):
    from utils import SympyHamiltonians
    from jraph_utils import utils as jutils
    import igraph as ig
    import time

    D = 10
    N = 20

    multinomial_theorem(np.arange(2,D), np.ones((N)))
