import sys
sys.path.append("..")
import warnings
from abc import ABC, abstractmethod
import os
import socket
import jraph
import pickle
import numpy as np
from pathlib import Path
from DatasetCreator.Gurobi import GurobiSolver
from DatasetCreator.jraph_utils import utils as jutils
from .save_utils import save_indexed_dict, load_indexed_dict
import igraph as ig
import networkx as nx

class BaseDatasetGenerator(ABC):
	"""
	Base Class for generating datasets
	"""
	def __init__(self, config):
		"""
		:param config: config file to specify the dataset
			{
				dataset_name: (string) name of the dataset,
				problem: (string) problem to be solved (e.g.: MaxCut, MDS, ...),
				mode: (string) [train, val, test],
				seed: (int) seed for random number generator,
				parent: (bool) whether to use parent directory,
				diff_ps: (bool) whether to use different p values,
				gurobi_solve: (bool) whether to solve instances with gurobi or not,
				time_limit: (float) gurobi time limit for solving the problem,
				n_graphs: (int) number of graphs to generate,
			}
		"""
		self.config = config

		self.seed = config["seed"]
		self.mode = config["mode"]
		self.save = config["save"]
		self.dataset_name = config["dataset_name"]
		self.problem = config["problem"]
		self.diff_ps = config["diff_ps"]
		# self.IsingFormulation = config["IsingFormulation"]
		self.gurobi_solve = config["gurobi_solve"]
		self.licence_base_path = config["licence_base_path"]
		self.time_limit = config["time_limit"]
		self.thread_fraction = config["thread_fraction"]

		# set path
		p = Path(os.getcwd())
		if config["parent"]:
			self.path = str(p.parent)
		else:
			self.path = str(p)

		# set seed
		if self.mode == "val":
			seed_int = 5
		elif self.mode == "test":
			seed_int = 4
		else:
			seed_int = 0
		np.random.seed(self.seed + seed_int)

		# set gurobi licence
		if self.gurobi_solve:
			hostname = socket.gethostname()
			device_licence_path = os.path.join(self.licence_base_path, f"gurobi_{hostname}.lic")
			if os.path.exists(device_licence_path):
				os.environ["GRB_LICENSE_FILE"] = device_licence_path
			else:
				gurobi_licence_path = os.path.join(self.licence_base_path, "gurobi.lic")
				if os.path.exists(gurobi_licence_path):
					os.environ["GRB_LICENSE_FILE"] = gurobi_licence_path
				else:
					warnings.warn("No gurobi licence has been found in the given base path. Gurobi will not be used to solve the dataset!")
					self.gurobi_solve = False
		else:
			warnings.warn("Gurobi will not be used to solve the dataset which might lead to wrong solutions for the dataset!")

	@abstractmethod
	def generate_dataset(self):
		"""
		Generate the graph instances for the dataset

		- use self.solve_graph(H_graph) to solve the graph instance
		- use self.save_instance_solution(indexed_solution_dict, idx) to save the graph instance
		- use self.save_solutions(solutions) to save the solutions
		"""
		raise NotImplementedError("generate_graph method not implemented")

	def solve_graph(self, H_graph, g) -> (float, float, list, float, jraph.GraphsTuple):
		"""
		Solve the graph instance for the dataset using gurobi if self.gurobi_solve is True, otherwise return None Tuple

		:param H_graph: jraph graph instance
		:param g: igraph graph instance
		:return: (Energy, boundEnergy, solution, runtime, H_graph_compl)
		"""
		if self.gurobi_solve:
			if self.problem == "MaxCut":
				H_graph_compl = jutils.from_igraph_to_jgraph(g, double_edges=False)
				_, Energy, boundEnergy, solution, runtime, MC_value = GurobiSolver.solveMaxCut(H_graph,
																							   time_limit=self.time_limit,
																							   bnb=False, verbose=False, thread_fraction = self.thread_fraction)
				return Energy, boundEnergy, solution, runtime, H_graph_compl

			elif self.problem == "MDS":
				_, Energy, solution, runtime = GurobiSolver.solveMDS_as_MIP(H_graph, time_limit=self.time_limit, thread_fraction = self.thread_fraction)
				boundEnergy = Energy
				return Energy, boundEnergy, solution, runtime, None

			elif self.problem == "MaxCl":
				H_graph_compl = jutils.from_igraph_to_jgraph(g.complementer(loops=False), double_edges=False)
				_, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph_compl, time_limit=self.time_limit, thread_fraction = self.thread_fraction)
				return Energy, None, solution, runtime, H_graph_compl

			elif self.problem == "MIS":
				H_graph_compl = jutils.from_igraph_to_jgraph(g, double_edges=False)
				_, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph, time_limit=self.time_limit, thread_fraction = self.thread_fraction)
				return Energy, None, solution, runtime, H_graph_compl

			elif self.problem == "MVC":
				H_graph_compl = jutils.from_igraph_to_jgraph(g, double_edges=False)
				_, Energy, solution, runtime = GurobiSolver.solveMVC_as_MIP(H_graph, time_limit=self.time_limit, thread_fraction = self.thread_fraction)
				return Energy, None, solution, runtime, H_graph_compl

			else:
				raise NotImplementedError(f"Problem {self.problem} is not implemented. Choose from [MaxCut, MDS]")
		else:
			# in case gurobi is not used, arbitrary values are returned and for MaxCl, the complement graph is returned
			Energy = 0.
			boundEnergy = 0.
			solution = np.ones_like(H_graph.nodes)
			runtime = None

			if self.problem == "MaxCl":
				H_graph_compl = jutils.from_igraph_to_jgraph(g.complementer(loops=False), double_edges=False)
			elif self.problem == "MIS" or self.problem == "MVC" or self.problem == "MaxCut":
				H_graph_compl = jutils.from_igraph_to_jgraph(g, double_edges=False)
			else:
				H_graph_compl = None
			return Energy, boundEnergy, solution, runtime, H_graph_compl

	def igraph_to_jraph(self, g: ig.Graph) -> (jraph.GraphsTuple, float, int):
		"""
		Convert igraph graph to jraph graph

		:param g: igraph graph
		:return: (H_graph, density, graph_size)
		"""
		density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
		graph_size = g.vcount()
		return jutils.from_igraph_to_jgraph(g), density, graph_size

	def nx_to_jraph(self, gnx: nx.Graph) -> (jraph.GraphsTuple, float, int):
		"""
		Convert networkx graph to jraph graph via igraph

		:param gnx: networkx graph
		:return: (H_graph, density, graph_size)
		"""
		g = ig.Graph.TupleList(gnx.edges(), directed=False)
		density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
		graph_size = g.vcount()
		return jutils.from_igraph_to_jgraph(g), density, graph_size

	def nx_to_igraph(self, gnx: nx.Graph) -> ig.Graph:
		"""
		Convert networkx graph to igraph graph

		:param gnx: networkx graph
		:return: igraph graph
		"""
		return ig.Graph.TupleList(gnx.edges(), directed=False)

	def save_instance_solution(self, indexed_solution_dict, idx):
		"""
		Save the graph instance solution to a file

		:param indexed_solution_dict: dictionary containing the solution
		:param idx: index of the solution
		"""
		indexed_solution_dict["time_limit"] = self.time_limit
		save_indexed_dict(path=self.path, mode=self.mode, dataset_name=self.dataset_name, i=idx,
						EnergyFunction=self.problem, seed=self.seed, indexed_solution_dict=indexed_solution_dict)

	def save_solutions(self, solutions):
		"""
		save the solutions to a file

		:param solutions: dictionary containing the solutions
		"""
		if self.save:
			new_path = self.path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{self.dataset_name}"
			if not os.path.exists(new_path):
				os.makedirs(new_path)

			save_path = self.path + f"/loadGraphDatasets/DatasetSolutions/no_norm/{self.dataset_name}/{self.mode}_{self.problem}_seed_{self.seed}_solutions.pickle"
			pickle.dump(solutions, open(save_path, "wb"))


	def load_solutions(self):
		solutions_list = []
		for idx in range(self.graph_config[f"n_{self.mode}"]):
			solutions_list.append( load_indexed_dict(path=self.path, mode=self.mode, dataset_name=self.dataset_name, i=idx,
							EnergyFunction=self.problem, seed=self.seed))

		avrg_runtimes = np.sum([el["runtimes"] for el in solutions_list])
		avrg_Energies = np.mean([el["Energies"] for el in solutions_list])

		return_dict = {"runtimes": avrg_runtimes,"Energies": avrg_Energies}
		print("number of graphs", len(solutions_list))
		return return_dict
