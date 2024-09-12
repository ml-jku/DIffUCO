import copy
import os
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import jraph
from tqdm import tqdm
from functools import partial
import wandb
import time

from train import TrainMeanField
from Data.LoadGraphDataset import SolutionDatasetLoader
from jraph_utils import pad_graph_to_nearest_power_of_k, add_random_node_features
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



class ConditionalExpectation:
    def __init__(self, wandb_id, n_different_random_node_features, k = 1, load_best_network = True, eval_step_factor = 2, project_name = ""):
        self.eval_step_factor = eval_step_factor
        self.load_best_network = load_best_network
        self.project_name = project_name

        self.wandb_id = wandb_id
        self.wandb_id_CE = wandb.util.generate_id()

        self.ST_k = k

        if(os.getcwd() == "/code/DiffUCO/"):
            base_path = "/mnt/proj2/dd-23-97/"
        else:
            base_path = os.path.dirname(os.getcwd())

        self.path_results = base_path + "/DiffUCO/CE_results"

        self.path_to_models = base_path + "/DiffUCO/Checkpoints"
        # self.path_to_models = "/system/user/publicwork/sanokows/meanfield_annealing/Checkpoints"

        self.n_different_random_node_features = n_different_random_node_features

        WANDB = True
        if WANDB:
            self.wandb_mode = "online"
        else:
            self.wandb_mode = "disabled"


        self.__load_network()
        self.model.eval_step_factor = self.eval_step_factor
        #self.__init_dataset()
        self.__vmap_get_energy = jax.vmap(self.__get_energy, in_axes=(None, 0), out_axes=(1))

    def __init_wandb(self):
        wandb.init(project=self.wandb_project, name=self.wandb_run, id=self.wandb_id_CE, mode=self.wandb_mode)

    def load_train_curve(self):
        path_folder = f"{self.path_to_models}/{self.wandb_id}/"
        file_name = f"{self.wandb_id}_last_epoch.pickle"
        with open(path_folder + file_name, "rb") as f:
            loaded_dict = pickle.load(f)
        train_curve = loaded_dict["logs"]
        rel_error_curve = train_curve["eval/rel_error"]

        return np.array(rel_error_curve)

    def __load_params(self):
        if(self.load_best_network):
            file_name = f"best_{self.wandb_id}.pickle"
            with open(f'{self.path_to_models}/{self.wandb_id}/{file_name}', 'rb') as f:
                params, config, eval_dict = pickle.load(f)
            for key in eval_dict.keys():
                print(key, eval_dict[key])

            return params, config
        else:
            path_folder = f"{self.path_to_models}/{self.wandb_id}/"
            file_name = f"{self.wandb_id}_last_epoch.pickle"
            with open(path_folder + file_name, "rb") as f:
                loaded_dict = pickle.load(f)
            return loaded_dict["params"], loaded_dict["config"]


    def __load_network(self):
        self.params, self.config  = self.__load_params()
        print("loaded", jax.tree_map(lambda x: x.shape, self.params))
        self.params = jax.tree_map(lambda x: x[0], self.params)
        self.params = jax.device_put_replicated(self.params, list(jax.devices()))

        print(f"wandb ID: {self.wandb_id}\nDataset: {self.config['dataset_name']} | Problem: {self.config['problem_name']}")
        self.path_dataset = self.config['dataset_name']

        self.dataset_name = self.config["dataset_name"]
        self.problem_name = self.config["problem_name"]

        self.batch_size = 32
        self.N_basis_states = 100

        self.T_max = self.config["T_max"]

        self.seed = self.config["seed"]

        #self.wandb_project = f"{self.config['dataset_name']}-{self.config['problem_name']}_ConditionalExpectation"
        self.wandb_project = f"{self.config['dataset_name']}_{self.config['problem_name']}_ConditionalExpectation_REWORKED_ST_{self.ST_k}{self.project_name}"
        # self.wandb_run = f"{self.config['dataset_name']}_{self.seed}_Tmax{self.T_max}_{self.n_different_random_node_features}_originalrun_{self.wandb_id}_currentrun_{self.wandb_id_CE}"
        self.wandb_run = f"{self.config['dataset_name']}_{self.seed}_Tmax{self.T_max}_originalrun_{self.wandb_id}_ST_k_{self.ST_k}_best_checkpoint_{self.load_best_network}_eval_step_factor_{self.eval_step_factor}"
        self.wandb_group = ""
        self.config["wandb"] = False

        self.random_node_features = self.config["random_node_features"]

        self.key = jax.random.PRNGKey(self.seed)
        self.n_random_node_features = self.config["n_random_node_features"] if "n_random_node_features" in self.config.keys() else 1

        self.config["N_basis_states"] = 1
        self.__init_wandb()

        self.model = TrainMeanField(self.config, load_wandb_id = self.wandb_id, eval_step_factor = self.eval_step_factor)
        self.model.params = self.params



    def init_dataset(self, dataset_name, mode = "train"):
        self.mode = mode
        if not isinstance(dataset_name, type(None)):
            self.dataset_name = dataset_name

        data_generator = SolutionDatasetLoader(config = self.model.config, dataset=self.dataset_name, problem=self.problem_name,
                                               batch_size=self.batch_size, relaxed=True, seed=self.seed,
                                               mode = mode)
        self.dataloader_train, self.dataloader_test, self.dataloader_val, (
            self.mean_energy, self.std_energy) = data_generator.dataloaders()

        print(f"T_max: {self.T_max}, {self.dataset_name} - {self.problem_name}")

    @partial(jax.jit, static_argnums=(0,))
    def __get_energy(self, jraph_graph, state):
        state = jnp.expand_dims(state, axis=-1)
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jraph_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        sum_n_node = jraph_graph.nodes.shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

        energy_messages = jraph_graph.edges * state[jraph_graph.senders] * state[jraph_graph.receivers]
        energy_per_node = 0.5 * jax.ops.segment_sum(energy_messages, jraph_graph.receivers,
                                                    sum_n_node) + state * jnp.expand_dims(jraph_graph.nodes[:, 0],
                                                                                          axis=-1)
        energy = jax.ops.segment_sum(energy_per_node, node_graph_idx, n_graph)
        return energy


    def _from_class_to_spins(self, k = 1):
        sampled_class = np.arange(0, 2**k)
        bin_arr = np.unpackbits(sampled_class.reshape(-1,1).view(np.uint8), axis=1,
                                count=k, bitorder="little")
        bin_arr = np.array(bin_arr, dtype=np.float32)
        return bin_arr

    def __conditional_expectation_relaxed(self, jraph_graph, probs, k = 10, _np = np):
        #jax.config.update('jax_platform_name', 'cpu')
        orig_probs = copy.copy(probs)
        probs = np.reshape(probs, (probs.shape[0]*probs.shape[1]))
        # shape = probs.shape
        # probs = np.full(shape, 0.5)
        #probs_idx = probs.copy()
        jit_repeat = lambda x: _np.repeat(x[np.newaxis, :], 2 ** k, axis = 0)
        node_graph_idx, _, _ = self.model.TrainerClass._compute_aggr_utils(jraph_graph)

        def calc_energy(vmapped_probs):
            resh_vmapped_probs = _np.reshape(vmapped_probs, (vmapped_probs.shape[0], orig_probs.shape[0],orig_probs.shape[1]))
            vmapped_energies, _, _ = self.model.TrainerClass.vmapped_relaxed_energy(jraph_graph, jnp.swapaxes(resh_vmapped_probs, 0,1), node_graph_idx)
            vmapped_energies = jnp.swapaxes(vmapped_energies, 0, 1)
            max_idx = _np.argmin(vmapped_energies.flatten(), axis = 0)
            return vmapped_energies, vmapped_probs, max_idx

        #jitted_energy_func = jax.jit(calc_energy)
        vmapped_probs = _np.repeat(probs[np.newaxis, :], 2 ** k, axis=0)
        calc_energy(vmapped_probs)
        rem_spins = len(probs) - int(len(probs)/k)*k
        calc_energy(vmapped_probs[0:2**rem_spins])

        start_Overall = time.time()
        bin_configuration = self._from_class_to_spins(k = k)
        highest_indices = np.argsort(-probs)
        vmapped_probs = _np.repeat(probs[np.newaxis, :], 2 ** k, axis=0)

        for s_i in range(int(len(probs)/k)+1):
            len_probs = len(probs)
            rem_spins = len_probs - s_i*k
            if(rem_spins < k and rem_spins != 0):
                num_spins = len_probs - s_i*k
                bin_configuration = bin_configuration[:2**num_spins, 0:num_spins]
                vmapped_probs = _np.repeat(vmapped_probs[0][np.newaxis, :], 2 ** num_spins, axis=0)
            elif(rem_spins == 0):
                break
            else:
                num_spins = k

            next_indices = _np.array(highest_indices[s_i*k: s_i*k+ num_spins])
            #start_e = time.time()
            # vmapped_probs = _np.repeat(probs[np.newaxis, :], 2 ** num_spins, axis = 0)
            #
            # vmapped_probs[:, next_indices] = bin_configuration
            # vmapped_energies = self.__vmap_relaxed_energy(jraph_graph, vmapped_probs).flatten()
            #vmapped_probs = _np.repeat(probs[np.newaxis, :], 2 ** num_spins, axis = 0)
            #print(bin_configuration.shape, vmapped_probs[:, next_indices].shape)
            vmapped_probs[:, next_indices] = bin_configuration
            vmapped_energies,vmapped_probs, max_idx = calc_energy(vmapped_probs)
            # end_e = time.time()
            #
            # print("tiem for energy", end_e - start_e)

            #max_idx = _np.argmin(vmapped_energies, axis = 0)
            vmapped_probs[:, next_indices] = vmapped_probs[max_idx, next_indices]

        #jax.config.update('jax_platform_name', 'gpu')
        probs = vmapped_probs[max_idx]
        end_Overall = time.time()
        time_needed = end_Overall - start_Overall
        #print("overall_time", end_Overall-start_Overall, int(len(probs)/k)+1,(end_e - start_e)*int(len(probs)/k)+1)
        resh_probs = jnp.reshape(probs, (orig_probs.shape[0], orig_probs.shape[1]))
        state = resh_probs * 2 - 1
        # print("relaxed ",self.__relaxed_energy(jraph_graph, orig_probs).flatten()[0])
        # print("discrete",self.__relaxed_energy(jraph_graph, resh_probs).flatten()[0])
        return self.model.relaxed_energy(jraph_graph, resh_probs, node_graph_idx)[0].flatten()[0], state, time_needed


    def __unbatch(self, graph_batch, gt_normed_energies_batch, states_batch, spin_log_probs_batch, spin_logits_batch):
        nodes = graph_batch.nodes
        n_node = graph_batch.n_node
        n_graph = graph_batch.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        sum_n_node = graph_batch.nodes.shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

        states_batch = np.array(states_batch)
        to_unbatch_graph_batch = copy.deepcopy(graph_batch)
        graphs_list = jraph.unbatch_np(to_unbatch_graph_batch)
        graphs_list = graphs_list[:-1]
        states_list = []
        spin_log_probs_list = []
        gt_normed_energies_list = []
        spin_logits_list = []

        for idx, graph in enumerate(graphs_list):
            gt_normed_energies_list.append(gt_normed_energies_batch[idx])
            states_list.append(states_batch[:, node_graph_idx == idx])
            spin_log_probs_list.append(spin_log_probs_batch[:, node_graph_idx == idx])
            spin_logits_list.append(spin_logits_batch[node_graph_idx == idx])

        return graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list

    def __generate_random_node_feature_batch(self, graph_batch, iter):
        graph_list = []
        for i in range(self.n_different_random_node_features):
            _graph_batch = add_random_node_features(graph_batch, n_random_node_features=self.n_random_node_features,
                                                    seed=self.seed + iter + i)
            graph_list.append(_graph_batch)
        return jraph.batch(graph_list)

    #@partial(jax.jit, static_argnums=(0, 2))
    def forward(self, params, batch_dict, key, seed = (0,0,0)):

        graph_batch, energy_graph_batch = self.model._prepare_graphs(batch_dict)

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))

        loss, (log_dict, _) = self.model.TrainerClass.evaluation_step(params, graph_batch, energy_graph_batch, 0., batched_key)
        return loss, (log_dict, key)

    def run(self, p=None, measure_time = False, dataset_name=f"RB_iid_small", mode="train", break_after_time = None):

        ### TODO use train.eval here
        ### TODO reining test dataset
        ### TODO increase eval diffusion step factor

        self.model.n_basis_states = self.n_different_random_node_features
        if(measure_time):
            self.model.batch_size = 1
            self.batch_size = 1
        self.init_dataset(dataset_name=dataset_name, mode=mode)
        self.model.dataloader_test = self.dataloader_test
        self.model.TrainerClass.eval_step_factor = self.eval_step_factor

        test_log_dict = self.model.test()
        wandb.log(test_log_dict)
        wandb.finish()
        print(test_log_dict.keys())
        return test_log_dict


    def runG_set_(self, p=None, measure_time=True, dataset_name=f"RB_iid_small", mode="train", break_after_time=180, n_instances = 20):

        if (measure_time):
            self.model.n_basis_states = 1
            self.batch_size = 1
        self.init_dataset(dataset_name=dataset_name, mode=mode)

        self.n_different_random_node_features = 1

        if (self.mode == "test"):
            DATALOADER = self.dataloader_test
        elif (self.mode == "val"):
            DATALOADER = self.dataloader_val

        dataset_len = len(DATALOADER.dataset)

        dataset_len_active = 0

        print(f"\nLength of dataset: {dataset_len}\n")

        state_probs = np.zeros(shape=(dataset_len))
        mean_probs = np.zeros(shape=(dataset_len))
        best_rel_errors = np.ones(shape=(dataset_len)) * float('inf')
        mean_ps = []

        forward_time_list = []
        CE_time_list = []
        best_MC_value_list = []
        gt_MC_value_list = []
        for iter, (batch_dict) in tqdm(enumerate(DATALOADER), total=len(DATALOADER)):
            dataset_len_active += len(batch_dict["energies"])
            # one batch
            graph_batch, energy_graph_batch, _, _ = self.model._prepare_graphs(batch_dict)
            gt_normed_energies_batch = batch_dict["energies"]

            #for i in range(self.n_different_random_node_features):
            spent_time = 0
            i = 0
            best_MC_value = float("-inf")
            while(spent_time/n_instances <= break_after_time):
                # one batch with one set of random node features
                if (measure_time):
                    loss, (log_dict, self.key) = self.forward(self.params, batch_dict, self.key)

                start_forward = time.time()
                loss, (log_dict, self.key) = self.forward(self.params, batch_dict, self.key)
                end_forward = time.time()
                forward_time = end_forward - start_forward

                forward_time_list.append(forward_time)

                log_dict = jax.tree_map(lambda x: x[0], log_dict)
                spin_logits_batch = log_dict["log_p_0"][:, 0, :, :]
                # spin_logits_batch = jnp.reshape(spin_logits_batch, (spin_logits_batch.shape[0]*spin_logits_batch.shape[2], spin_logits_batch.shape[-1]))
                states_batch = jnp.expand_dims(log_dict["X_0"][:, 0], axis=0)
                spin_log_probs_batch = jnp.expand_dims(log_dict["spin_log_probs"][:, 0], axis=0)
                mean_ps.append(log_dict["figures"]["prob_over_diff_steps"]["y_values"][-1])

                _energy_graph_batch = jax.tree_map(lambda x: x[0], energy_graph_batch)

                graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list = self.__unbatch(
                    _energy_graph_batch, gt_normed_energies_batch, states_batch, spin_log_probs_batch,
                    spin_logits_batch)
                for idx, (graph, states, spin_log_probs, gt_normed_energies, spin_logits) in enumerate(
                        zip(graphs_list, states_list, spin_log_probs_list, gt_normed_energies_list, spin_logits_list)):

                    probs = np.exp(spin_logits[..., 1])
                    # energy, state, time_needed = self.__conditional_expectation_relaxed(graph, probs, k = self.ST_k)
                    energy, state, time_needed = self.__conditional_expectation_relaxed(graph, probs, k=self.ST_k)
                    MC_value = (graph.n_edge[0] / 4 - energy / 2)

                    CE_time_list.append(time_needed)
                    spent_time += time_needed + forward_time

                    gt_MC_value = batch_dict["energies"][0]

                    print("-----------------------------")
                    print("num graphs finised", len(gt_MC_value_list))
                    print("MC_value", gt_MC_value, MC_value, best_MC_value)
                    print("time", spent_time/n_instances)
                    if(MC_value > best_MC_value):
                        best_MC_value = MC_value
                        print(f"MC_value improved from {best_MC_value} to {MC_value}")
                    print("best diff", gt_MC_value - best_MC_value)
                    if(len(gt_MC_value_list) > 0):
                        print("average_diff", np.mean(np.array(gt_MC_value_list)- np.array(best_MC_value_list)))

            best_MC_value_list.append(best_MC_value)
            gt_MC_value_list.append(gt_MC_value)

        best_MC_value_arr = np.array(best_MC_value_list)
        gt_MC_value_arr = np.array(gt_MC_value_list)
        average_best_MC = np.mean(best_MC_value_arr)
        std_best_MC = np.std(best_MC_value_arr)/np.sqrt(len(best_MC_value_arr))
        diff = gt_MC_value_arr - best_MC_value_arr
        average_difference = np.mean(diff)
        std_difference = np.std(diff)/np.sqrt(len(diff))

        wandb.log({"best_MC_avr": average_best_MC, "std_MC_avr": std_best_MC, "avrg_diff": average_difference, "std_diff": std_difference})
        wandb.finish()

    def __save_result(self, results, p):
        result_dict = {
            "wandb_run_id": self.wandb_id,
            "dataset_name": self.dataset_name,
            "problem_name": self.problem_name,
            "T": self.T_max,
            "n_different_random_node_features": self.n_different_random_node_features,
            "p": p,
            "results": results
            }

        path_folder = f"{self.path_results}/{self.path_dataset}/"
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        print(f'\nsaving to {os.path.join(path_folder, f"{self.ST_k}_{self.wandb_id}_{self.dataset_name}_{self.seed}_.pickle")}\n')

        with open(os.path.join(path_folder, f"{self.ST_k}_{self.wandb_id}_{self.dataset_name}_{self.seed}_.pickle"), 'wb') as f:
            pickle.dump(result_dict, f)

        with open(os.path.join(path_folder, f"{self.ST_k}_{self.wandb_id}_{self.dataset_name}_{self.seed}_eval_step_factor_{self.eval_step_factor}.pickle"), 'wb') as f:
            pickle.dump(result_dict, f)

    def load_results(self, eval_step_factor = None):
        path_folder = f"{self.path_results}/{self.path_dataset}/"
        if(eval_step_factor == None):
            with open(os.path.join(path_folder, f"{self.ST_k}_{self.wandb_id}_{self.dataset_name}_{self.seed}_.pickle"), 'rb') as f:
                result_dict = pickle.load(f)
        else:
            with open(os.path.join(path_folder,f"{self.ST_k}_{self.wandb_id}_{self.dataset_name}_{self.seed}_eval_step_factor_{self.eval_step_factor}.pickle"),'rb') as f:
                result_dict = pickle.load(f)
        return result_dict




def eval_on_dataset(config):
    CE = ConditionalExpectation(wandb_id=config["wandb_id"], n_different_random_node_features=config["n_samples"], k=1, eval_step_factor=config["evaluation_factor"])
    # CE.init_dataset(dataset_name=f"RB_iid_200_p_{p}", mode="test")
    test_log_dict = CE.run(p=None, dataset_name=config["dataset"], mode="test")

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--wandb_id', default="kj0bihnz", type = str)
parser.add_argument('--dataset', default="RB_iid_small", type = str)
parser.add_argument('--GPU', default=7, type = int)
parser.add_argument('--evaluation_factor', default=3, type = int)
parser.add_argument('--n_samples', default=8, type = int, help = "number of samples for each graph")

args = parser.parse_args()

if __name__ == "__main__":

    device = args.GPU
    print("Measure Time")
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

    config = {
        "wandb_id": args.wandb_id,
        "dataset": args.dataset,
        "evaluation_factor": args.evaluation_factor,
        "n_samples": args.n_samples,
    }

    eval_on_dataset(config)









