import networkx as nx

from .BaseDatasetGenerator import BaseDatasetGenerator
from tqdm import tqdm
import numpy as np

class BADatasetGenerator(BaseDatasetGenerator):
	"""
	Class for generating datasets for the Barabasi-Albert model
	"""
	def __init__(self, config):
		super().__init__(config)


		self.graph_config = {
			"n_train": 4000,
			"n_val": 500,
			"n_test": 1000,
					}

		if("huge" in self.dataset_name or "giant" in self.dataset_name):
			self.graph_config["n_test"] = 100

		if("dummy" in self.dataset_name):
			self.graph_config["n_train"] = 300

		print(f'\nGenerating Barabasi-Albert {self.mode} dataset "{self.dataset_name}" with {self.graph_config[f"n_{self.mode}"]} instances!\n')
	def generate_dataset(self):
		"""
		Generate a Barabasi-Albert graph instances for the dataset
		"""
		solutions = {
			"Energies": [],
			"H_graphs": [],
			"gs_bins": [],
			"graph_sizes": [],
			"densities": [],
			"runtimes": [],
			"upperBoundEnergies": [],
			"compl_H_graphs": [],
		}

		for idx in tqdm(range(self.graph_config[f"n_{self.mode}"])):
			if "small" in self.dataset_name:
				curr_n = np.random.randint(101) + 200
			elif "large" in self.dataset_name:
				curr_n = np.random.randint(401) + 800
			elif "huge" in self.dataset_name:
				curr_n = np.random.randint(601) + 1200
			elif "giant" in self.dataset_name:
				curr_n = np.random.randint(1001) + 2000
			elif "dummy" in self.dataset_name:
				curr_n = np.random.randint(40) + 80
			else:
				raise NotImplementedError('Dataset name must contain either "small", "large", "huge" or "giant" to infer the number of nodes')

			gnx = nx.barabasi_albert_graph(curr_n, 4)
			weight = {e: np.random.rand() for e in gnx.edges()}
			nx.set_edge_attributes(gnx, weight, "weight")
			g = self.nx_to_igraph(gnx)
			H_graph, density, graph_size = self.igraph_to_jraph(g)

			Energy, boundEnergy, solution, runtime, H_graph_compl = self.solve_graph(H_graph, g)

			# graphs = H_graph_compl
			# energies = np.squeeze(Energy, axis = -1)
			# num_edges = 2*graphs.n_edge
			# MaxCut_Value = num_edges/4 - energies/2
			# print(MaxCut_Value)
			# raise ValueError("")

			solutions["Energies"].append(Energy)
			solutions["H_graphs"].append(H_graph)
			solutions["gs_bins"].append(solution)
			solutions["graph_sizes"].append(graph_size)
			solutions["densities"].append(density)
			solutions["runtimes"].append(runtime)
			solutions["upperBoundEnergies"].append(boundEnergy)
			solutions["compl_H_graphs"].append(H_graph_compl)

			indexed_solution_dict = {}
			for key in solutions.keys():
				if len(solutions[key]) > 0:
					indexed_solution_dict[key] = solutions[key][idx]
			self.save_instance_solution(indexed_solution_dict, idx)
		self.save_solutions(solutions)





