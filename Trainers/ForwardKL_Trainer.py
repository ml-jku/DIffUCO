import jax.numpy as jnp
from functools import partial
import jax
from .BaseTrainer import Base, repeat_along_nodes
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import optax
import time
from scipy.special import softmax as np_softmax

vmap_repeat_along_nodes = jax.vmap(repeat_along_nodes, in_axes=(0, 0, 0))
@partial(jax.jit, static_argnums=())
def select_time_idxs(graphs, data_buffer_dict, split_diff_arr, key):
    max_diff_steps = data_buffer_dict["log_p_0_T"].shape[1]
    n_devices = data_buffer_dict["log_p_0_T"].shape[0]
    diff_idx_mat = jnp.transpose(split_diff_arr, (0,3, 1, 2))

    device_idx_mat = jnp.arange(0, n_devices)[:,None, None, None, None]

    node_idx_mat = jnp.arange(0, data_buffer_dict["bin_sequence"].shape[2])[None, None, :,  None, None]
    state_idx_mat = jnp.arange(0, data_buffer_dict["bin_sequence"].shape[3])[None, None, None, :, None]
    ones_idx_mat = jnp.arange(0, 1)[None, None, None, None, :]
    rand_node_idx_mat = jnp.arange(0, data_buffer_dict["rand_node_features_diff_steps"].shape[-1])[None, None, None, None, :]

    log_p_0_T = data_buffer_dict["log_p_0_T"]  # device batch, diff_step_batch, graph batch, basisi state batch
    log_q_0_T = data_buffer_dict["log_q_0_T"]

    diff_idx_mat_per_node = vmap_repeat_along_nodes(graphs.nodes, graphs.n_node, jnp.swapaxes(diff_idx_mat, 1, 2))
    diff_idx_mat_per_node = jnp.swapaxes(diff_idx_mat_per_node, 1, 2)[..., None]

    X_prev = data_buffer_dict["bin_sequence"][device_idx_mat, diff_idx_mat_per_node, node_idx_mat, state_idx_mat, ones_idx_mat]
    X_next = data_buffer_dict["bin_sequence"][device_idx_mat, diff_idx_mat_per_node + 1, node_idx_mat, state_idx_mat, ones_idx_mat]
    rand_node_features = data_buffer_dict["rand_node_features_diff_steps"][device_idx_mat, diff_idx_mat_per_node, node_idx_mat, state_idx_mat, rand_node_idx_mat]

    out_dict = {}
    out_dict["log_p_0_T"] = log_p_0_T
    out_dict["log_q_0_T"] = log_q_0_T
    out_dict["X_prev"] = X_prev
    out_dict["rand_node_features"] = rand_node_features
    out_dict["X_next"] = X_next

    t_idx_per_node = diff_idx_mat_per_node
    t_idx_per_node = jnp.swapaxes(t_idx_per_node, -2, -3)
    t_idx_per_node = t_idx_per_node.reshape((t_idx_per_node.shape[0], t_idx_per_node.shape[1] * t_idx_per_node.shape[2], t_idx_per_node.shape[3], 1))
    out_dict["t_idx_per_node"] = t_idx_per_node

    return out_dict, key

def collate_function(batch):
    batch_dict = {key: [] for key in batch[0].keys()}

    for el in batch:
        for key in batch_dict.keys():
            batch_dict[key].append(el[key])

    for key in batch_dict.keys():
        batch_dict[key] = np.array(np.concatenate(batch_dict[key], axis = 1))

    return batch_dict

class ForwardKL(Base):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        super(ForwardKL, self).__init__(config, EnergyClass, NoiseClass, model)
        self.N_basis_states = self.config["N_basis_states"]
        self.n_graphs = self.config["batch_size"] + 1
        self.n_bernoulli_features = self.config["n_bernoulli_features"]
        self.n_diffusion_steps = self.config["n_diffusion_steps"]

        self.inner_loop_steps = self.config["inner_loop_steps"]

        self.vmapped_calc_log_q = jax.vmap(self.model.calc_log_q, in_axes=(None, None, 0, 0, 0, 0, 0))
        self.vmapped_calc_log_p = jax.vmap(self.NoiseDistrClass.get_log_p_T_0, in_axes=(None, 1, 1, None, None))
        self.forw_KL_loss_grad = jax.jit(jax.value_and_grad(self.get_loss, has_aux=True))
        self.pmap_forw_KL_loss_backward = jax.pmap(self.loss_backward, in_axes=(0, 0, 0, 0,0), axis_name="device")
        self.pmap_sample_X_sequence = jax.pmap(lambda a,b,c,d,e: self.sample_X_sequence(a,b,c,d,e, "train"), in_axes=(0, 0, 0, None, 0) )
        self.sample_X_sequence_eval = lambda a,b,c,d,e: self.sample_X_sequence(a,b,c,d,e, "eval")

        self.pmap_environment_steps = jax.pmap(lambda a,b,c,d,e: self._environment_steps_scan(a,b,c,d,e, "train"), in_axes=(0, 0, 0, None, 0))
        self._environment_steps_scan_eval = lambda a,b,c,d,e: self._environment_steps_scan(a,b,c,d,e, "eval")

        ### TODO add exceptions when self.n_diffusion_steps / self.diff_step_batch_size not an int
        self.diff_step_batch_size = min([self.config["minib_diff_steps"], self.n_diffusion_steps])
        self.n_diff_batches = self.n_diffusion_steps // self.diff_step_batch_size
        self.inner_update_steps = int(self.n_diff_batches)*self.inner_loop_steps
        self.n_devices = len(jax.devices())
        self._init_index_arrays()

    #@partial(jax.jit, static_argnums=(0,))
    def _init_index_arrays(self):
        self.n_graphs = int(self.config["batch_size"]/self.n_devices) + 1
        diff_step_arr = jnp.arange(0,self.n_diffusion_steps)
        #Nb_arr = jnp.repeat(diff_step_arr[None, ...], self.N_basis_states, axis=0)
        Nb_diff_step_arr = jnp.repeat(diff_step_arr[None, ...], self.N_basis_states, axis = 0)
        Gb_Nb_diff_step_arr = jnp.repeat(Nb_diff_step_arr[None, ...], self.n_graphs, axis = 0)
        self.Db_Nb_arr = jnp.repeat(Gb_Nb_diff_step_arr[None, ...], self.n_devices, axis = 0)

    def _compute_importance_weights(self, out_dict):
        #print("p maped shape?", out_dict["log_p_0_T"].shape)
        #t_idx = out_dict["t_idx"]
        #idx_dim_2 = np.arange(out_dict["log_p_0_T"].shape[1] - 1)
        #where_true = idx_dim_2[None,:, None, None] < t_idx + 1

        log_p_0_T = np.array(out_dict["log_p_0_T"], dtype=np.float64)
        log_q_0_T = np.array(out_dict["log_q_0_T"], dtype=np.float64)

        importance_weights = np_softmax(np.sum(log_p_0_T - log_q_0_T, axis=1), axis=-1)
        #print("importance_weights", importance_weights.dtype, log_q_0_T.dtype)
        out_dict["importance_weights"] = jnp.array(importance_weights)
        return out_dict

    @partial(jax.jit, static_argnums=(0,))
    def _shuffle_index_array(self, key):
        key, subkey = jax.random.split(key)
        perm_diff_array = jax.random.permutation(subkey, self.Db_Nb_arr, axis=-1, independent=True)
        return perm_diff_array, key

    @partial(jax.jit, static_argnums=(0,2,3))
    def _split_arrays(self, arr, n_splits, axis ):
        arr_list = jnp.split(arr, n_splits, axis=axis)
        return arr_list

    @partial(jax.jit, static_argnums=(0,))
    def loss_backward(self, params, opt_state, graphs, batch_dict, key):
        (loss, (log_dict, key)), grad = self.forw_KL_loss_grad(params, graphs, batch_dict, key)
        grad = jax.lax.pmean(grad, axis_name='device')
        params, opt_state = self.__update_params(params, grad, opt_state)
        return (loss, (log_dict, key)), params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def __update_params(self, params, grads, opt_state):
        grad_update, opt_state = self.opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, grad_update)
        return params, opt_state

    def sample(self, params, graphs, energy_graph_batch, T, key):
        (log_dict, _) =  self._environment_steps_scan_eval(params, graphs, energy_graph_batch, T, key)
        loss = 0.
        return loss, (log_dict, _)

    def train_step(self, params, opt_state, jraph_graph_list, energy_graph_batch, T, key):
        sampling_start_time = time.time()
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))
        out_dict, _ = self.pmap_environment_steps(params, jraph_graph_list, energy_graph_batch, T, batched_key)
        sampling_end_time = time.time()
        sampling_time = sampling_end_time - sampling_start_time

        start_update_policy_time = time.time()
        (loss, (log_dict, _)), params, opt_state = self._update_policy(params, opt_state, jraph_graph_list, out_dict, key)
        end_update_policy_time = time.time()

        log_dict["time"]["sampling_time"] = sampling_time
        log_dict["time"]["update_policy"] = end_update_policy_time - start_update_policy_time
        #log_dict["time"]["cast_to_numpy"] = cast_to_numpy_end_time - cast_to_numpy_start_time
        #print(log_dict["time"])
        return params, opt_state, loss, (log_dict, energy_graph_batch, key)

    def _update_policy(self, params, opt_state, graphs, out_dict, key):

        ### TODO add figures that visualize losses over inner loop steps
        log_dict = {"Losses": {"forward_KL": []}, "time": {"grad_step": [], "dataloading_time": []}}
        for i in range(self.inner_loop_steps):
            start_dataloading_time = time.time()
            perm_diff_array, key = self._shuffle_index_array(key)
            split_diff_array_list = self._split_arrays(perm_diff_array, self.n_diff_batches, -1)
            for split_diff_arr in split_diff_array_list:

                batch_dict, key = select_time_idxs(graphs["graphs"][0], out_dict["DataBuffer"], split_diff_arr, key)
                batch_dict = self._compute_importance_weights(batch_dict)

                #print(jax.tree_map(lambda x: x.shape, batch_dict))
                end_dataloading_time = time.time()
                log_dict["time"]["dataloading_time"].append(end_dataloading_time - start_dataloading_time)

                start_loss_time = time.time()
                key, subkey = jax.random.split(key)
                batched_key = jax.random.split(subkey, num=len(jax.devices()))
                (loss, (loss_dict, _)), params, opt_state = self.pmap_forw_KL_loss_backward(params, opt_state, graphs, batch_dict, batched_key)
                end_loss_time = time.time()

                log_dict["time"]["grad_step"].append(end_loss_time - start_loss_time)

                for dict_key in log_dict["Losses"].keys():
                    log_dict["Losses"][dict_key].append(loss_dict["Losses"][dict_key])
                start_dataloading_time = time.time()

        for dict_key in log_dict["Losses"].keys():
            log_dict["Losses"][dict_key] = np.mean(log_dict["Losses"][dict_key])

        for dict_key in log_dict["time"].keys():
            log_dict["time"][dict_key] = np.sum(log_dict["time"][dict_key])

        return (loss, (log_dict, key)), params, opt_state

    def get_loss(self, params, jraph_graph_list, batch_dict, key):
        loss, (log_dict, key) = self._forward_KL_loss(params, jraph_graph_list, batch_dict, key)
        return loss, (log_dict, key)

    @partial(jax.jit, static_argnums=(0,))
    def _forward_KL_loss(self, params, jraph_graph_list,  batch_dict, key):
        keys = ["X_prev", "X_next", "rand_node_features"]
        orig_shape = batch_dict["X_prev"]

        for dict_key in keys:
            arr = batch_dict[dict_key]
            arr = jnp.swapaxes(arr, 1, 2)
            arr_shape = arr.shape
            new_shape = (arr_shape[0] * arr_shape[1], arr_shape[2], arr_shape[3])
            reshaped_arr = jnp.reshape(arr, new_shape)
            batch_dict[dict_key] = reshaped_arr
        X_prev = batch_dict["X_prev"]
        X_next = batch_dict["X_next"]
        rand_node_features = batch_dict["rand_node_features"]

        t_idx_per_node = batch_dict["t_idx_per_node"]

        log_q_0_T = batch_dict["log_q_0_T"]

        importance_weights = batch_dict["importance_weights"]

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=t_idx_per_node.shape[0])

        out_dict, _ = self.vmapped_calc_log_q( params, jraph_graph_list, X_prev, rand_node_features, X_next, t_idx_per_node, batched_key)
        log_q_t = out_dict["state_log_probs"]

        new_shape = (orig_shape.shape[0], log_q_0_T.shape[2], log_q_0_T.shape[1])

        reshaped_log_q_t = jnp.reshape(log_q_t, new_shape)

        reshaped_log_q_t = jnp.transpose(reshaped_log_q_t, (0, 2, 1))
        reshaped_log_q_t = reshaped_log_q_t[:,:-1]
        n_time_steps = self.diff_step_batch_size


        weights = importance_weights[:-1]

        loss_per_graph_per_state = -n_time_steps* weights * jnp.mean(reshaped_log_q_t, axis = 0)
        loss_per_graph = jnp.sum(loss_per_graph_per_state, axis = -1)
        loss = jnp.mean(loss_per_graph)
        #print(loss)
        log_dict = {"Losses": {"forward_KL": loss}}
        return loss, (log_dict, key)

    @partial(jax.jit, static_argnums=(0,-1))
    def sample_X_sequence(self, params, graphs, energy_graph_batch, T, key, mode):
        print("function is being jitted")
        if(mode == "train"):
            N_basis_states = self.N_basis_states
        else:
            N_basis_states = self.N_test_basis_states

        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, log_q_T, one_hot_state, log_p_uniform, key  = self.model.sample_prior_w_probs(energy_graph_batch, N_basis_states,
                                                                            key)

        n_graphs = energy_graph_batch.n_node.shape[0]

        Xs_over_different_steps = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], 1))
        prob_over_diff_steps = jnp.zeros((self.n_diffusion_steps + 1,))
        log_q_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))
        log_p_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))
        rand_node_features_diff_steps = jnp.zeros((overall_diffusion_steps, X_prev.shape[0], X_prev.shape[1], self.n_random_node_features))

        log_q_0_T = log_q_0_T.at[0].set(log_q_T)
        prob_over_diff_steps = prob_over_diff_steps.at[0].set(0.5)
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)

        for i in range(overall_diffusion_steps):
            model_step_idx = jnp.array([i / self.eval_step_factor], dtype=jnp.int16)
            model_step_idx_per_node = model_step_idx[0] * jnp.ones((energy_graph_batch.nodes.shape[0], 1), dtype=jnp.int16)
            key, subkey = jax.random.split(key)
            batched_key = jax.random.split(subkey, num=N_basis_states)

            out_dict, _ = self.vmapped_make_one_step(params, graphs, X_prev, model_step_idx_per_node, batched_key)

            X_next = out_dict["X_next"]
            spin_log_probs = out_dict["spin_log_probs"]
            spin_logits_next = out_dict["spin_logits"]
            graph_log_prob = out_dict["graph_log_prob"]
            state_log_probs = out_dict["state_log_probs"]
            rand_node_features = out_dict["rand_node_features"]

            rand_node_features_diff_steps = rand_node_features_diff_steps.at[i].set(rand_node_features)

            log_q_t = state_log_probs

            log_p_t = self.NoiseDistrClass.get_log_p_T_0(energy_graph_batch, X_prev, X_next, model_step_idx, T)
            X_prev = X_next
            log_q_0_T = log_q_0_T.at[i+1].set(log_q_t)
            log_p_0_T = log_p_0_T.at[i].set(log_p_t)
            Xs_over_different_steps = Xs_over_different_steps.at[i + 1].set(X_next)

            average_probs = jnp.mean(graph_log_prob[:-1])
            prob_over_diff_steps = prob_over_diff_steps.at[i + 1].set(average_probs)

        X_0 = X_next
        energies, _, _ = self.vmapped_relaxed_energy(energy_graph_batch, X_0, node_gr_idx)
        log_p_0 = self.EnergyClass.get_log_p_0_from_energy(energies, T)
        log_p_0_T = log_p_0_T.at[i+1].set(log_p_0)

        x_axis = jnp.arange(0, overall_diffusion_steps)

        weights = jax.nn.softmax(jnp.sum(log_p_0_T - log_q_0_T, axis = 0), axis = -1)

        forward_KL_per_graph = -jnp.sum(weights* jnp.sum(log_q_0_T, axis = 0) , axis = -1)
        forward_KL = jnp.mean(forward_KL_per_graph[:-1])

        sum_log_p = jnp.sum(log_p_0_T, axis = 0)
        sum_log_q = jnp.sum(log_q_0_T, axis = 0)
        diff = jnp.sum(log_p_0_T - log_q_0_T, axis = 0)
        diff_max = diff - np.max(diff)
        diff_min = diff - np.min(diff)
        diff_mean = diff - np.mean(diff)
        metric_energies = energies[:-1]
        log_dict = {"Losses": {"forward_KL": forward_KL},
                    "metrics": {"energies": metric_energies, "entropies": 0., "spin_log_probs": spin_log_probs,
                                "free_energies": 0., "graph_mean_energies": metric_energies},
                    "energies": {"HA": metric_energies},
                    "figures": {"prob_over_diff_steps": {"x_values": x_axis, "y_values": prob_over_diff_steps},
                                "sum_log_p": {"x_values": jnp.arange(0, sum_log_p[0].shape[-1]), "y_values": sum_log_p[0]},
                                "sum_log_q": {"x_values": jnp.arange(0, sum_log_q[0].shape[-1]), "y_values": sum_log_q[0]},
                                "diff_max": {"x_values": jnp.arange(0, diff_max[0].shape[-1]),
                                              "y_values": diff_max[0]},
                                "diff_min": {"x_values": jnp.arange(0, diff_min[0].shape[-1]),
                                              "y_values": diff_min[0]},
                                "diff_mean": {"x_values": jnp.arange(0, diff_mean[0].shape[-1]),
                                              "y_values": diff_mean[0]},
                                "weights": {"x_values": jnp.arange(0, weights[0].shape[-1]), "y_values": weights[0]}
                                },
                    "DataBuffer": {"log_p_0_T": log_p_0_T,
                                    "log_q_0_T": log_q_0_T,
                                    "bin_sequence": Xs_over_different_steps, "rand_node_features_diff_steps": rand_node_features_diff_steps},
                    "log_p_0": spin_logits_next,
                    "X_0": X_0,
                    "bin_sequence": Xs_over_different_steps,
                    "spin_log_probs": spin_log_probs,
                    }

        return forward_KL, (log_dict, key)

    @partial(jax.jit, static_argnums=(0,))
    def scan_body(self, scan_dict, y):
        i = scan_dict["step"]
        T = scan_dict["T"]
        params = scan_dict["params"]
        X_prev = scan_dict["X_prev"]
        graphs = scan_dict["graphs"]
        node_gr_idx = scan_dict["node_gr_idx"]
        energy_graph_batch = scan_dict["energy_graph_batch"]

        model_step_idx = jnp.array([i / self.eval_step_factor], dtype=jnp.int16)
        model_step_idx_per_node = model_step_idx[0] * jnp.ones((energy_graph_batch.nodes.shape[0], 1), dtype=jnp.int16)

        key, subkey = jax.random.split(scan_dict["key"])
        scan_dict["key"] = key

        batched_key = jax.random.split(subkey, num=X_prev.shape[1])

        out_dict, _ = self.vmapped_make_one_step(params, graphs, X_prev, model_step_idx_per_node,
                                                 batched_key)

        X_next = out_dict["X_next"]
        state_log_probs = out_dict["state_log_probs"]

        log_q_t = state_log_probs
        log_p_t = self.NoiseDistrClass.get_log_p_T_0(energy_graph_batch, X_prev, X_next, model_step_idx, T)

        scan_dict["log_q_0_T"] = scan_dict["log_q_0_T"].at[i + 1].set(log_q_t)
        scan_dict["log_p_0_T"] = scan_dict["log_p_0_T"].at[i].set(log_p_t)

        X_next = out_dict["X_next"]
        spin_log_probs = out_dict["spin_log_probs"]
        spin_logits_next = out_dict["spin_logits"]
        graph_log_prob = out_dict["graph_log_prob"]
        rand_node_features = out_dict["rand_node_features"]

        scan_dict["rand_node_features_diff_steps"] = scan_dict["rand_node_features_diff_steps"].at[i].set(rand_node_features)

        X_prev = X_next
        scan_dict["Xs_over_different_steps"] = scan_dict["Xs_over_different_steps"].at[i + 1].set(X_next)

        average_probs = jnp.mean(graph_log_prob[:-1])
        scan_dict["prob_over_diff_steps"] = scan_dict["prob_over_diff_steps"].at[i + 1].set(average_probs)

        scan_dict["X_prev"] = X_prev
        scan_dict["step"] += 1

        out_dict = {}
        out_dict["spin_log_probs"] = spin_log_probs
        out_dict["spin_logits_next"] = spin_logits_next
        return scan_dict, out_dict

    @partial(jax.jit, static_argnums=(0,-1))
    def _environment_steps_scan(self, params, graphs, energy_graph_batch, T, key, mode):
        ### TDOD cahnge rewards to non exact expectation rewards
        print("scan function is being jitted")
        if(mode == "train"):
            N_basis_states = self.N_basis_states
        else:
            N_basis_states = self.N_test_basis_states

        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, log_q_T, one_hot_state, log_p_uniform, key = self.model.sample_prior_w_probs(energy_graph_batch,
                                                                                             N_basis_states,
                                                                                             key)

        n_graphs = energy_graph_batch.n_node.shape[0]

        Xs_over_different_steps = jnp.zeros((overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], 1))
        prob_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1,), dtype=jnp.float32)
        rand_node_features_diff_steps = jnp.zeros(
            (overall_diffusion_steps, X_prev.shape[0], X_prev.shape[1], self.n_random_node_features), dtype=jnp.float32)
        log_q_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))
        log_p_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))

        prob_over_diff_steps = prob_over_diff_steps.at[0].set(1/self.n_bernoulli_features)
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)
        log_q_0_T = log_q_0_T.at[0].set(log_q_T)

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)
        scan_dict = {"log_q_0_T": log_q_0_T, "log_p_0_T":log_p_0_T, "Xs_over_different_steps": Xs_over_different_steps, "prob_over_diff_steps": prob_over_diff_steps, "rand_node_features_diff_steps":rand_node_features_diff_steps,
                    "step": 0, "node_gr_idx": node_gr_idx, "params": params, "key": key, "X_prev": X_prev, "graphs": graphs, "energy_graph_batch": energy_graph_batch, "T": T}

        scan_dict, out_dict_list = jax.lax.scan(self.scan_body, scan_dict, None, length = overall_diffusion_steps)

        key = scan_dict["key"]
        spin_log_probs = out_dict_list["spin_log_probs"][-1]
        spin_logits_next = out_dict_list["spin_logits_next"][-1]

        log_p_0_T = scan_dict["log_p_0_T"]
        log_q_0_T = scan_dict["log_q_0_T"]
        X_next = scan_dict["X_prev"]#

        X_0 = X_next

        Xs_over_different_steps = scan_dict["Xs_over_different_steps"]
        rand_node_features_diff_steps = scan_dict["rand_node_features_diff_steps"]
        prob_over_diff_steps = scan_dict["prob_over_diff_steps"]

        energies, _, _ = self.vmapped_relaxed_energy(energy_graph_batch, X_0, node_gr_idx)
        log_p_0 = self.EnergyClass.get_log_p_0_from_energy(energies, T)
        log_p_0_T = log_p_0_T.at[-1].set(log_p_0)


        forward_KL = self._forward_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])
        reverse_KL = self._reverse_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])

        # sum_log_p = jnp.sum(log_p_0_T, axis = 0)
        # sum_log_q = jnp.sum(log_q_0_T, axis = 0)
        # diff = jnp.sum(log_p_0_T - log_q_0_T, axis = 0)
        # diff_max = diff - np.max(diff)
        # diff_min = diff - np.min(diff)
        # diff_mean = diff - np.mean(diff)
        metric_energies = energies[:-1]

        x_axis = jnp.arange(0, overall_diffusion_steps)
        log_dict = {"Losses": {"forward_KL": forward_KL, "reverse_KL": reverse_KL},
                    "metrics": {"energies": metric_energies, "entropies": 0., "spin_log_probs": spin_log_probs,
                                "free_energies": 0., "graph_mean_energies": metric_energies},
                    "energies": {"HA": metric_energies},
                    "figures": {"prob_over_diff_steps": {"x_values": x_axis, "y_values": prob_over_diff_steps},
                                # "sum_log_p": {"x_values": jnp.arange(0, sum_log_p[0].shape[-1]),
                                #               "y_values": sum_log_p[0]},
                                # "sum_log_q": {"x_values": jnp.arange(0, sum_log_q[0].shape[-1]),
                                #               "y_values": sum_log_q[0]},
                                # "diff_max": {"x_values": jnp.arange(0, diff_max[0].shape[-1]),
                                #              "y_values": diff_max[0]},
                                # "diff_min": {"x_values": jnp.arange(0, diff_min[0].shape[-1]),
                                #              "y_values": diff_min[0]},
                                # "diff_mean": {"x_values": jnp.arange(0, diff_mean[0].shape[-1]),
                                #               "y_values": diff_mean[0]},
                                # "weights": {"x_values": jnp.arange(0, weights[0].shape[-1]), "y_values": weights[0]}
                                },
                    "DataBuffer": {"log_p_0_T": log_p_0_T,
                                   "log_q_0_T": log_q_0_T,
                                   "bin_sequence": Xs_over_different_steps,
                                   "rand_node_features_diff_steps": rand_node_features_diff_steps},
                    "log_p_0": spin_logits_next,
                    "X_0": X_0,
                    "bin_sequence": Xs_over_different_steps,
                    "spin_log_probs": spin_log_probs,
                    }
        return log_dict, key


