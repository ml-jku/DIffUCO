import copy
import jraph
from GlobalProjectVariables import MVC_A, MVC_B
from sympy import Sum, symbols, Indexed, lambdify, simplify, expand, IndexedBase, Idx, poly
from utils.sympy_utils import get_two_body_corr, get_one_body_corr, get_constant, replace_bins_by_spins
import numpy as np
from collections import Counter

def MVC(H_graph, B = MVC_B, A = MVC_A):
    '''
    input is an undirected jraph
    :param H_graph:
    :param B:
    :param A:
    :return:
    '''

    senders = H_graph.senders
    receivers = H_graph.receivers
    n_nodes = H_graph.nodes.shape[0]

    X = IndexedBase('X')
    Hb = [ 0.5* B*(1-X[s]) *(1-X[r]) for s,r in zip(senders,receivers)]
    Ha = [ A*X[n] for n in range(n_nodes)]
    E = Hb + Ha
    return expand(sum(E)), X

def MVC_sparse(j_graph, B = MVC_B, A = MVC_A):
    ### todo save factor of two by using directed graph
    ### TODO save factor of two by using map
    senders = j_graph.senders
    receivers = j_graph.receivers
    n_nodes = j_graph.nodes.shape[0]
    n_edges = j_graph.edges.shape[0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = 0.5 * B * (1 - X[i]) * (1 - X[j])

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros((n_edges,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        J_sr = J_ij

        couplings[idx] += float(J_sr)
        external_fields[s] += float(external_field_on_i)
        external_fields[r] += float(external_field_on_j)
        constant += float(self_connection)

    expression = A * X[n]
    spin_expression = replace_bins_by_spins(expression, X, S, n)
    spin_expression = expand(spin_expression)
    ext_field = get_one_body_corr(spin_expression, S, n)
    constant_per_spin = get_constant(spin_expression, S)

    for n in range(n_nodes):
        external_fields[n] += float(ext_field)
        constant += float(constant_per_spin)

    self_connections = constant/n_nodes*np.ones((2*n_nodes, 1))### TODO check if this factor of two is correct here
    new_nodes = external_fields
    new_edges = np.concatenate([ couplings, self_connections ], axis = 0)
    n_edge = np.array([new_edges.shape[0]])
    new_senders = np.concatenate([senders, self_senders, self_senders], axis = -1)
    new_receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = -1)
    H_graph = j_graph._replace(nodes = new_nodes, edges = new_edges, n_edge = n_edge, senders = new_senders, receivers = new_receivers)
    return H_graph

def MaxCut(j_graph):
    n_nodes = j_graph.nodes.shape[0]
    edges = j_graph.edges
    senders = j_graph.senders
    receivers = j_graph.receivers

    self_receivers = np.arange(0, n_nodes)
    self_senders = np.arange(0, n_nodes)
    self_edges = np.zeros((n_nodes, 1))

    senders = np.concatenate([senders, self_senders, self_senders], axis = 0)
    receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = 0)
    full_edges = np.concatenate([edges, self_edges, self_edges], axis = 0)
    n_edge = np.array([full_edges.shape[0]])

    H_graph = j_graph._replace(edges = full_edges, senders = senders, receivers = receivers, n_edge = n_edge)
    return H_graph


def WMIS_sparse(j_graph, B = MVC_B, A = MVC_A):
    ### todo save factor of two by using directed graph
    ### TODO save factor of two by using map
    senders = j_graph.senders
    receivers = j_graph.receivers
    n_nodes = j_graph.nodes.shape[0]
    n_edges = j_graph.edges.shape[0]
    weight = j_graph.nodes[:,0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = 0.5 * B * X[i] * X[j]

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros((n_edges,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        J_sr = J_ij

        couplings[idx] += float(J_sr)
        external_fields[s] += float(external_field_on_i)
        external_fields[r] += float(external_field_on_j)
        constant += float(self_connection)

    expression = - A * X[n]
    spin_expression = replace_bins_by_spins(expression, X, S, n)
    spin_expression = expand(spin_expression)
    ext_field = get_one_body_corr(spin_expression, S, n)
    constant_per_spin = get_constant(spin_expression, S)

    for n in range(n_nodes):
        external_fields[n] += weight[n]*float(ext_field)
        constant += weight[n]*float(constant_per_spin)

    self_connections = constant/n_nodes*np.ones((2*n_nodes, 1))### TODO check if this factor of two is correct here
    new_nodes = external_fields
    new_edges = np.concatenate([ couplings, self_connections ], axis = 0)
    n_edge = np.array([new_edges.shape[0]])
    new_senders = np.concatenate([senders, self_senders, self_senders], axis = -1)
    new_receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = -1)
    H_graph = j_graph._replace(nodes = new_nodes, edges = new_edges, n_edge = n_edge, senders = new_senders, receivers = new_receivers)
    return H_graph

def MIS_sparse(j_graph, B = MVC_B, A = MVC_A):
    ### todo save factor of two by using directed graph
    ### TODO save factor of two by using map
    senders = j_graph.senders
    receivers = j_graph.receivers
    n_nodes = j_graph.nodes.shape[0]
    n_edges = j_graph.edges.shape[0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    n = symbols("n")
    expression = 0.5 * B * X[i] * X[j]

    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = 2*get_two_body_corr(spin_expression, S, i,j)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    couplings = np.zeros((n_edges,1))
    constant = 0
    self_senders = np.arange(0, n_nodes)
    self_receivers = self_senders

    for idx, (s, r) in enumerate(zip(senders, receivers)):
        J_sr = J_ij

        couplings[idx] += float(J_sr)
        external_fields[s] += float(external_field_on_i)
        external_fields[r] += float(external_field_on_j)
        constant += float(self_connection)

    expression = - A * X[n]
    spin_expression = replace_bins_by_spins(expression, X, S, n)
    spin_expression = expand(spin_expression)
    ext_field = get_one_body_corr(spin_expression, S, n)
    constant_per_spin = get_constant(spin_expression, S)

    for n in range(n_nodes):
        external_fields[n] += float(ext_field)
        constant += float(constant_per_spin)

    self_connections = constant/n_nodes*np.ones((2*n_nodes, 1))### TODO check if this factor of two is correct here
    new_nodes = external_fields
    new_edges = np.concatenate([ couplings, self_connections ], axis = 0)
    n_edge = np.array([new_edges.shape[0]])
    new_senders = np.concatenate([senders, self_senders, self_senders], axis = -1)
    new_receivers = np.concatenate([receivers, self_receivers, self_receivers], axis = -1)
    H_graph = j_graph._replace(nodes = new_nodes, edges = new_edges, n_edge = n_edge, senders = new_senders, receivers = new_receivers)
    return H_graph

def add_to_graph(ig, senders, receivers, couplings):

    for (s,r,e) in zip(senders, receivers, couplings):
        if(ig.are_connected(s,r)):
            existing_edge  = ig.es.select(_source=s, _target=r)[0]
            existing_edge["weight"] += e
        else:
            ig.add_edge(s,r,weight = e)
    return ig


def build_C_term(H_graph, B):
    C = B

    n_nodes = H_graph.nodes.shape[0]
    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    expression = - C* X[i]
    spin_expression = replace_bins_by_spins(expression, X, S, i)

    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((n_nodes,1))
    constant = 0
    for i in range(n_nodes):
        external_fields[i] += float(external_field_on_i)
        constant += float(self_connection)

    return external_fields, constant


def multinomial_theorem(vec_1, vec_2, A=1.):
    vec_concat = np.concatenate([vec_1, -vec_2], axis=0)

    res = np.tensordot(vec_concat[:, np.newaxis], vec_concat[np.newaxis, :], axes=[[1], [0]])
    overall_nodes = vec_concat.shape[0]

    X = IndexedBase('X')
    S = IndexedBase('S')
    i = symbols("i")
    j = symbols("j")
    expression = 0.5 * A * X[i] * X[j]
    expression = replace_bins_by_spins(expression, X, S, i)
    expression = replace_bins_by_spins(expression, X, S, j)
    spin_expression = expand(expression)
    J_ij = float(2 * get_two_body_corr(spin_expression, S, i, j))
    external_field_on_i = get_one_body_corr(spin_expression, S, i)
    external_field_on_j = get_one_body_corr(spin_expression, S, j)
    self_connection = get_constant(spin_expression, S)

    external_fields = np.zeros((overall_nodes, 1))
    senders = []
    receivers = []
    edges = []
    constant = 0
    for i in range(overall_nodes):
        for j in range(overall_nodes):
            constant += float(self_connection) * (res[i, j] + res[j, i])
            external_fields[i, 0] += 2 * float(external_field_on_i) * res[i, j]
            external_fields[j, 0] += 2 * float(external_field_on_j) * res[j, i]
            if(i != j):
                senders.extend([j])
                receivers.extend([i])
                edges.extend([2*res[i,j]*J_ij])
            else:
                constant += res[j, i]*J_ij

    edges = np.array(edges)
    edges = np.expand_dims(edges, axis = -1)
    senders = np.array(senders)
    receivers = np.array(receivers)
    return external_fields, senders, receivers, edges, constant


def construct_graph(external_fields, senders, receivers, edges, constant, num_nodes):
    self_senders = np.arange(0, num_nodes)
    self_receivers = np.arange(0, num_nodes)
    all_senders = np.concatenate([senders, self_senders, self_receivers])
    all_receivers = np.concatenate([receivers, self_receivers, self_senders])
    self_loops = constant / num_nodes * np.ones((num_nodes, 1))

    edges = np.concatenate([edges, self_loops, self_loops], axis=0)
    n_node = np.array([num_nodes])
    n_edge = np.array([all_senders.shape[0]])
    nodes = external_fields

    jgraph = jraph.GraphsTuple(nodes=nodes, edges=edges, senders=all_senders, receivers=all_receivers, n_node=n_node,
                                 n_edge=n_edge, globals=n_edge)
    return jgraph





if(__name__ == "__main__"):

    pass

