import numpy as np
import itertools
import scipy.sparse as sp
from . import csp_utils
import random
import networkx as nx


def generate_instance(n, k, r, p):
    a = np.log(k) / np.log(n)
    v = k * n
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)

    parts = np.reshape(np.int64(range(v)), (n, k))
    nand_clauses = []

    for i in parts:
        nand_clauses += itertools.combinations(i, 2)

    edges = set()
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges
        edges |= set(random.sample(tuple(all), k=min(s, len(all))))

    nand_clauses += list(edges)
    clauses = {'NAND': nand_clauses}

    instance = csp_utils.CSP_Instance(language=csp_utils.is_language,
                                      n_variables=v,
                                      clauses=clauses)
    return instance

from collections import Counter
def get_random_instance(n,k,p):

    a = np.log(k) / np.log(n)
    r = - a / np.log(1 - p)

    i = generate_instance(n, k, r, p)

    ordered_edge_list =[ (min([edge[0], edge[1]]), max([edge[0], edge[1]])) for edge in i.clauses['NAND']]
    edges = ordered_edge_list
    # print(Counter(edges).keys())
    # print(Counter(edges).values())
    # print(len(Counter(edges)), i.clauses['NAND'].shape)
    # G = nx.Graph()
    # G.add_edges_from(i.clauses['NAND'])

    return Counter(edges)

if(__name__ == "__main__"):
    get_random_instance()