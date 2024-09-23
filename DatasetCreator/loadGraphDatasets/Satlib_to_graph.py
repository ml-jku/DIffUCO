
from pathlib import Path
from pysat.formula import CNF
import networkx as nx
import numpy as np
import pickle
import os
from jraph_utils import utils as jutils
import igraph as ig
from Gurobi import GurobiSolver

class SATGraphDataGenerator():

    def __init__(self):
        self.directory = os.getcwd() + "/loadGraphDatasets/tmp/SATLIB/"
        self.Energy_list = []

        self.interate()
        ### TODO make train test val split
        self.num_graphs = len(os.listdir(self.directory))
        self.n_val_graphs = 500
        self.n_test_graphs = 500


    def interate(self):

        for file_path in os.listdir(self.directory):
            self._build_graph(self.directory + file_path)


    def _build_graph(self, cnf_file_path, time_limit = float("inf")):
        cnf = CNF()
        print(cnf_file_path)
        string = ""
        with open(cnf_file_path, 'r') as f:
            cnf_data = f.readlines()

        for line in cnf_data:
            if(has_four_numbers(line)):
                string += line

        cnf.from_string(string)

        #cnf = CNF.from_file(fname = cnf_file_path)
        nv = cnf.nv
        clauses = list(filter(lambda x: x, cnf.clauses))
        ind = { k:[] for k in np.concatenate([np.arange(1, nv+1), -np.arange(1, nv+1)]) }
        edges = []
        for i, clause in enumerate(clauses):
            a = clause[0]
            b = clause[1]
            c = clause[2]
            aa = 3 * i + 0
            bb = 3 * i + 1
            cc = 3 * i + 2
            ind[a].append(aa)
            ind[b].append(bb)
            ind[c].append(cc)
            edges.append((aa, bb))
            edges.append((aa, cc))
            edges.append((bb, cc))

        for i in np.arange(1, nv+1):
            for u in ind[i]:
                for v in ind[-i]:
                    edges.append((u, v))

        G = nx.from_edgelist(edges)
        g = ig.Graph.TupleList(G.edges(), directed=False)
        j_graph = jutils.from_igraph_to_jgraph(g)

        _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(j_graph, time_limit=time_limit)
        self.Energy_list.append(Energy)
        print("mean Energy",np.mean(self.Energy_list), time_limit)
        return j_graph


    def generate(self, gen_labels = False,  weighted = False):
        for f in self.input_path.rglob("*.cnf"):
            self._build_graph(f, self.output_path / (f.stem + ".gpickle"), gen_labels, weighted)



def has_four_numbers(input_string):
    try:
        # Attempt to convert the input string to a list of numbers
        numbers = list(map(float, input_string.split()))

        # Check if there are exactly four numbers in the list
        if len(numbers) >= 2:
            return True
        else:
            return False
    except ValueError:
        # ValueError occurs if the conversion to float fails
        return False