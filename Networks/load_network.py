import os.path

import numpy as np
from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
import pickle
import optax
import jraph
from tqdm import tqdm
import wandb

from Networks.policy import Policy
from jraph_utils import pad_graph_to_nearest_power_of_k, add_random_node_features
from Data.LoadGraphDataset import SolutionDatasetLoader


class LoadNetwork:
	def __init__(self, wandb_id):
		self.wandb_id = wandb_id

	def __load_params(self, T, best_run):
		if best_run:
			file_name = f"best_{self.wandb_id}.pickle"
		else:
			file_name = f"{self.wandb_id}_T_{T}.pickle"

		with open(f'./Checkpoints/{self.wandb_id}/{file_name}', 'rb') as f:
			params, config = pickle.load(f)
		return params, config

	def __load_network(self, T, best_run):
		self.params, self.config = self.__load_params(T, best_run)

		n_features_list_prob = self.config["n_features_list_prob"]
		n_features_list_nodes = self.config["n_features_list_nodes"]
		n_features_list_edges = self.config["n_features_list_edges"]
		n_features_list_messages = self.config["n_features_list_messages"]
		n_features_list_encode = self.config["n_features_list_encode"]
		n_features_list_decode = self.config["n_features_list_decode"]
		n_message_passes = self.config["n_message_passes"]
		message_passing_weight_tied = self.config["message_passing_weight_tied"]
		linear_message_passing = self.config["linear_message_passing"]

		self.dataset_name = self.config["dataset_name"]
		self.problem_name = self.config["problem_name"]

		self.batch_size = 1
		self.N_basis_states = 100

		self.T_max = self.config["T_max"]

		self.random_node_features = self.config["random_node_features"]

		self.seed = self.config["seed"]
		self.key = jax.random.PRNGKey(self.seed)

		self.model = Policy(n_features_list_prob=n_features_list_prob,
							n_features_list_nodes=n_features_list_nodes,
							n_features_list_edges=n_features_list_edges,
							n_features_list_messages=n_features_list_messages,
							n_features_list_encode=n_features_list_encode,
							n_features_list_decode=n_features_list_decode,
							n_message_passes=n_message_passes,
							message_passing_weight_tied=message_passing_weight_tied,
							linear_message_passing=linear_message_passing)

	def __init_dataset(self):
		data_generator = SolutionDatasetLoader(dataset=self.dataset_name, problem=self.problem_name, batch_size=self.batch_size, seed=self.seed)
		self.dataloader_train, self.dataloader_test, self.dataloader_val, (
			self.mean_energy, self.std_energy) = data_generator.dataloaders()

	def sample_states(self, T, best_run=False):
		self.__load_network(T, best_run)
		self.__init_dataset()

		dict_T = {
			"graphs": [],
			"basis_states": [],
		}

		for iter, (graph_batch, gt_normed_energies, gt_spin_states) in tqdm(enumerate(self.dataloader_val)):
			# gt_spin_states = np.expand_dims(gt_spin_states, axis=0)
			# gt_spin_states = np.repeat(gt_spin_states, self.N_basis_states, axis=0)
			if self.random_node_features:
				graph_batch = add_random_node_features(graph_batch, self.seed)
			graph_batch = pad_graph_to_nearest_power_of_k(graph_batch)

			states, log_probs, spin_log_probs, self.key = self.model.apply(self.params, graph_batch,
																		   self.N_basis_states,
																		   self.key)

			dict_T["graphs"].append(graph_batch)
			dict_T["basis_states"].append(states)
		return dict_T

	def create_dataset(self):
		dict_final = {}
		for T in [0.25, 0.19775, 0.10025, 0.0]:
			dict_T = self.sample_states(T)
			dict_final[f"T: {T}"] = dict_T

		dict_T = self.sample_states(None, best_run=True)
		dict_final[f"best"] = dict_T
		self.__save_dict(dict_final)

	def __save_dict(self, dict_final):
		to_save = (dict_final, self.config)
		path_folder = f"SampledNetworks/{self.wandb_id}/"

		if not os.path.exists(path_folder):
			os.makedirs(path_folder)

		with open(os.path.join(path_folder, f"{self.wandb_id}.pickle"), 'wb') as f:
			pickle.dump(to_save, f)
