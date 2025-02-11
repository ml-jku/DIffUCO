import jax.nn
import networkx as nx
import jraph
import numpy as np
from tqdm import tqdm
from DatasetCreator.Gurobi import GurobiSolver

from .BaseDatasetGenerator import BaseDatasetGenerator

class SpinGlassUniformDataset(BaseDatasetGenerator):
	"""
	Class for generating datasets for the NxN lattice
	"""

	def __init__(self, config):
		super().__init__(config)
		if config['mode'] == "train":
			self.n_graphs = 50
		else:
			self.n_graphs = 1
		print(f'\nGenerating NxN lattice {self.mode} dataset "{self.dataset_name}" with {self.n_graphs} instances!\n')

	def generate_graph(self, n):
		"""
		Generate a NxN lattice graph
		"""
		gnx = nx.grid_2d_graph(n, n, periodic=True)
		return gnx

	def generate_dataset(self):
		print(f"Note: This generates a Dataset with the same NxN lattice graph {self.n_graphs} times!")

		if "4x4" in self.dataset_name:
			self.size = 4
		elif "8x8" in self.dataset_name:
			self.size = 8
		elif "10x10" in self.dataset_name:
			self.size = 10
		elif "16x16" in self.dataset_name:
			self.size = 16
		elif "24x24" in self.dataset_name:
			self.size = 24
		elif "32x32" in self.dataset_name:
			self.size = 32
		else:
			raise NotImplementedError('Dataset name must contain either "4x4", "8x8" or "16x16" to infer the number of nodes')

		gnx = self.generate_graph(self.size)
		# weight = {e: 1. for e in gnx.edges()}
		# nx.set_edge_attributes(gnx, weight, "weight")
		# g = self.nx_to_igraph(gnx)
		H_graph, graph_size, density = self.nx_to_jraph(gnx)
		_, Energy, _, _, _ = GurobiSolver.solveSpinGlass(H_graph)
		print("optimal Energy is: ", Energy)
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

		for idx in tqdm(range(self.n_graphs)):
			solutions["Energies"].append(Energy)
			solutions["H_graphs"].append(H_graph)
			solutions["gs_bins"].append(np.ones_like(H_graph.nodes))
			solutions["graph_sizes"].append(graph_size)
			solutions["densities"].append(density)
			solutions["runtimes"].append(None)
			solutions["upperBoundEnergies"].append(0.)
			solutions["compl_H_graphs"].append(None)

			indexed_solution_dict = {}
			for key in solutions.keys():
				if len(solutions[key]) > 0:
					indexed_solution_dict[key] = solutions[key][idx]
			self.save_instance_solution(indexed_solution_dict, idx)
		self.save_solutions(solutions)

	def nx_to_jraph(self, nx_graph):
		num_vertices = nx_graph.number_of_nodes()

		node_idx = {}
		for i, node in enumerate(nx_graph.nodes):
			node_idx[node] = i

		edge_idx = []
		for i, edge in enumerate(nx_graph.edges):
			sender, receiver = edge
			edge_idx.append([node_idx[sender], node_idx[receiver]])
		edge_idx = np.array(edge_idx)
		undir_senders = edge_idx[:, 0]
		undir_receivers = edge_idx[:, 1]
		receivers = np.concatenate([undir_receivers[:, np.newaxis], undir_senders[:, np.newaxis]], axis=-1)
		receivers = np.ravel(receivers)
		senders = np.concatenate([undir_senders[:, np.newaxis], undir_receivers[:, np.newaxis]], axis=-1)
		senders = np.ravel(senders)
		np.random.seed(self.seed - 123)
		half_edges = 2*np.random.uniform((undir_senders.shape[0], 1))-1
		edges =  np.concatenate([half_edges, half_edges], axis=-1)
		edges = np.ravel(edges)[:,None]

		N = int(np.sqrt(num_vertices))
		x = np.arange(0, N)
		y = np.arange(0, N)
		xv, yv = np.meshgrid(x, y)

		nodes_x = jax.nn.one_hot(xv.flatten(), N)
		nodes_y = jax.nn.one_hot(yv.flatten(), N)
		nodes = np.concatenate([nodes_x, nodes_y], axis = -1)
		nodes = np.array(nodes)

		nodes = np.zeros((num_vertices, 1))

		globals = np.array([num_vertices])
		n_node = np.array([num_vertices])
		n_edge = np.array([receivers.shape[0]])

		jgraph = jraph.GraphsTuple(senders=senders, receivers=receivers, edges=edges, nodes=nodes, n_edge=n_edge,
								   n_node=n_node, globals=globals)

		density = 2 * n_edge / (n_node * (n_node - 1))
		graph_size = n_node
		return jgraph, graph_size, density