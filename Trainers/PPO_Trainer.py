import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from .BaseTrainer import Base, repeat_along_nodes
import time
import optax
from utils import MovingAverages
### TODO use RL environments to make it possible to project solutions onto feasible solutions!

vmap_repeat_along_nodes = jax.vmap(repeat_along_nodes, in_axes=(0, 0, 0))
@partial(jax.jit, static_argnums=())
def select_time_idxs(graph, data_buffer_dict, rand_diff_steps, rand_states, key):
    n_graphs = data_buffer_dict["policies"].shape[-2]
    n_nodes = data_buffer_dict["states"].shape[-3]
    n_devices = data_buffer_dict["policies"].shape[0]

    # key, subkey = jax.random.split(key)
    # rand_diff_steps = jax.random.choice(subkey, jnp.arange(0, max_diff_steps), shape = (n_devices, n_diff_idxs, n_state_idxs))
    # rand_states = jax.random.choice(subkey, jnp.arange(0, max_states), shape = (n_devices, n_state_idxs))

    D_mat = jnp.arange(0, n_devices)[:, None, None, None]
    graph_mat = jnp.arange(0, n_graphs)[None, None, :, None]
    node_mat = jnp.arange(0, n_nodes)[None, None, :, None]

   #print("asdasd", rand_diff_steps.shape, rand_states.shape)
    rand_diff_steps_original = rand_diff_steps
    rand_diff_steps = jnp.transpose(rand_diff_steps, (0, -1, -3, -2))
    rand_states = jnp.transpose(rand_states, (0, 2,1))

    rand_diff_steps_per_node = vmap_repeat_along_nodes(graph.nodes, graph.n_node, jnp.swapaxes(rand_diff_steps, 1, 2))
    rand_diff_steps_per_node = jnp.swapaxes(rand_diff_steps_per_node, 1, 2)

    out_dict = {}
    for dict_key in data_buffer_dict.keys():
        if(dict_key == "states" or dict_key == "actions" or dict_key == "rand_node_features"):
            el = data_buffer_dict[dict_key][D_mat, rand_diff_steps_per_node, node_mat, rand_states[..., None, :]]
        else:
            el = data_buffer_dict[dict_key][D_mat, rand_diff_steps, graph_mat, rand_states[..., None, :]]

        el = jnp.swapaxes(el, 2,3)
        el = jnp.reshape(el, (el.shape[0], el.shape[1]*el.shape[2]) + el.shape[3:])

        out_dict[dict_key] = el

    ### TODO flatten along step and state dimension
    rand_diff_steps_original = jnp.swapaxes(rand_diff_steps_original, -1, -2)
    rand_diff_steps_resh = jnp.reshape(rand_diff_steps_original, (rand_diff_steps_original.shape[0], rand_diff_steps_original.shape[1], rand_diff_steps_original.shape[2]*rand_diff_steps_original.shape[3],1))
    out_dict["time_index"] = jnp.swapaxes(rand_diff_steps_resh, 1, 2)

    rand_diff_steps_per_node = jnp.swapaxes(rand_diff_steps_per_node, -1, -2)
    rand_diff_steps_per_node = jnp.reshape(rand_diff_steps_per_node, (rand_diff_steps_per_node.shape[0], rand_diff_steps_per_node.shape[1]* rand_diff_steps_per_node.shape[2], rand_diff_steps_per_node.shape[3], 1))

    out_dict["time_index_per_node"] = rand_diff_steps_per_node
    return out_dict, key


class PPO(Base):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        super(PPO, self).__init__(config, EnergyClass, NoiseClass, model)
        self.N_basis_states = self.config["N_basis_states"]
        self.n_bernoulli_features = self.config["n_bernoulli_features"]
        self.n_diffusion_steps = self.config["n_diffusion_steps"]
        self.time_horizon = self.n_diffusion_steps

        ### TODO add these things to config
        k = self.config["TD_k"] # 3
        lam = np.exp(-np.log(k)/self.time_horizon) # float(np.round((1 - 1 / self.time_horizon), decimals=3))
        #lam = float(np.round((1 - 1 / self.time_horizon), decimals=3))
        self.lam = lam
        self.gamma = 1.
        self.inner_loop_steps = self.config["inner_loop_steps"]
        self.clip_value = self.config["clip_value"] # 0.2
        self.n_diff_bs = min([self.config["minib_diff_steps"], self.n_diffusion_steps])
        self.n_state_bs = min([self.config["minib_basis_states"], self.N_basis_states])
        self.inner_update_steps = int(self.n_diffusion_steps*self.N_basis_states/(self.n_diff_bs*self.n_state_bs))*self.inner_loop_steps
        self.c1 = self.config["value_weighting"] # 0.65
        self.proj_method = self.config["proj_method"]

        self.calc_noise_step = self.NoiseDistrClass.calc_noise_step

        self.pmap_calc_traces = jax.pmap(self._calc_traces, in_axes=(0,0))
        self.pmap_environment_steps = jax.pmap(lambda a,b,c,d,e: self._environment_steps_scan(a,b,c,d,e, "train"), in_axes=(0, 0, 0, None, 0))
        self._environment_steps_scan_eval = lambda a,b,c,d,e: self._environment_steps_scan(a,b,c,d,e, "eval")

        self.PPO_loss_grad = jax.jit(jax.value_and_grad(self.PPO_loss, has_aux=True))
        self.pmap_PPO_loss_backward = jax.pmap(self.loss_backward, in_axes=(0, 0, 0, 0,0), axis_name="device")
        self.vmapped_calc_log_q = jax.vmap(self.model.calc_log_q, in_axes = (None, None, 0, 0, 0, 0, 0))

        self.alpha = self.config["mov_average"]
        ### TODO moving averages should ideally be loaded from checkpoint
        self.MovingAverageClass = MovingAverages.MovingAverage(self.alpha, self.alpha)

        self.n_diff_batches = self.n_diffusion_steps // self.n_diff_bs
        self.n_state_batches = self.N_basis_states // self.n_state_bs
        self.n_devices = len(jax.devices())
        self._init_index_arrays()

    #@partial(jax.jit, static_argnums=(0,))
    def _init_index_arrays(self):
        self.n_graphs = int(self.config["batch_size"]/self.n_devices) + 1
        diff_step_arr = jnp.arange(0,self.n_diffusion_steps)
        Nb_arr = jnp.repeat(diff_step_arr[None, ...], self.N_basis_states, axis=0)
        Gb_Nb_arr = jnp.repeat(Nb_arr[None, ...], self.n_graphs, axis=0) #Nb_arr#

        self.Db_Gb_Nb_arr = jnp.repeat(Gb_Nb_arr[None, ...], self.n_devices, axis = 0)

        basis_state_arr = jnp.arange(0, self.N_basis_states)
        self.Db_basis_state_arr = jnp.repeat(basis_state_arr[None, ...], self.n_devices, axis=0)
        self.Db_basis_n_diff_state_arr = jnp.repeat(self.Db_basis_state_arr[...,None], self.n_diffusion_steps, axis=-1)
        print("TODO also add random permutation along graph batch size!")

    @partial(jax.jit, static_argnums=(0,))
    def _shuffle_index_array(self, key):
        key, subkey = jax.random.split(key)
        perm_diff_array = jax.random.permutation(subkey, self.Db_Gb_Nb_arr, axis=-1, independent=True)

        key, subkey = jax.random.split(key)
        perm_state_array = jax.random.permutation(subkey, self.Db_basis_n_diff_state_arr, axis=-2, independent=True)

        return perm_diff_array, perm_state_array, key

    @partial(jax.jit, static_argnums=(0,2,3))
    def _split_arrays(self, arr, n_splits, axis ):
        arr_list = jnp.split(arr, n_splits, axis=axis)
        return arr_list

    @partial(jax.jit, static_argnums=(0,))
    def loss_backward(self, params, opt_state, graphs, batch_dict, key):
        (loss, (log_dict, key)), grad = self.PPO_loss_grad(params, graphs, batch_dict, key)
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
        ### TODO log reverse KL here?
        return loss, (log_dict, _)

    def train_step(self, params, opt_state, graphs, energy_graph_batch, T, key):
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))

        start_env_step_time = time.time()
        out_dict, _ = self.pmap_environment_steps(params, graphs, energy_graph_batch, T, batched_key)
        end_env_step_time = time.time()

        overall_env_step_time = end_env_step_time - start_env_step_time


        if(False):
            energy_graph_batch_copy = jax.tree_util.tree_map(lambda x: x[0], energy_graph_batch)
            graphs_ = jax.tree_util.tree_map(lambda x: x[0], graphs["graphs"][0])
            params_no = jax.tree_util.tree_map(lambda x: x[0], params)

            s_cast = time.time()
            energy_graph_batch_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), energy_graph_batch_copy)
            graphs_ = jax.tree_util.tree_map(lambda x: jnp.array(x), graphs_)
            e_cast = time.time()

            graph_list = {"graphs": [graphs_]}

            start_env_step_time = time.time()
            _, _ = self._environment_steps(params_no, graph_list, energy_graph_batch_copy, T, key)
            end_env_step_time = time.time()

            overall_env_step_time_2 = end_env_step_time - start_env_step_time
            print("pmap compare", overall_env_step_time_2, overall_env_step_time + e_cast - s_cast , e_cast-s_cast)

            energy_graph_batch_copy = jax.tree_util.tree_map(lambda x: x[0],energy_graph_batch)
            node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch_copy)
            # relaxed_energies_per_graph, _, Hb_per_graph = self.vmapped_relaxed_energy(energy_graph_batch_copy, out_dict["best_X_0"][0], node_gr_idx)
            #
            # Hb = jnp.mean(jnp.abs(Hb_per_graph[:-1]))
            #
            # print(jnp.any(out_dict["energies"]["HA"][0] == relaxed_energies_per_graph[:-1]))
            # print("Hb", out_dict["energies"]["Hb"], float(Hb))

            #print(Hb_per_graph[:-1])

            vmap_debug = jax.vmap(self.EnergyClass.calculate_CE_debug, in_axes=(None, 1, None))
            vmap_debug = self.vmapped_energy_CE

            reps = [2,4,6,8,10]
            for rep in reps:
                X_in = out_dict["X_0"][0,:, 0:rep]
                print("X_in", X_in.shape)

                s = time.time()
                _ = vmap_debug(energy_graph_batch_copy, X_in,node_gr_idx)
                e = time.time()

                print("time_needed", e-s, "reps", rep)

            # best_X_0, _, HB_per_graph_recal, HB_per_graph = vmap_debug(energy_graph_batch_copy, out_dict["X_0"][0], node_gr_idx)
            # print(jnp.mean(HB_per_graph_recal), jnp.mean(HB_per_graph), jnp.mean(out_dict["energies"]["Hb"]))
            # if (jnp.sum(HB_per_graph_recal) != jnp.sum(HB_per_graph)):
            #     print(jnp.sum(HB_per_graph_recal, jnp.sum(HB_per_graph)))
            #     raise ValueError("not equal")


            # relaxed_energies_per_graph, Hb_per_node, Hb_per_graph = self.vmapped_relaxed_energy(energy_graph_batch_copy,
            #                                                                           best_X_0,
            #                                                                           node_gr_idx)
            # Hb = jnp.mean(jnp.abs(Hb_per_graph[:-1]))
            #
            # print("is equal",jnp.mean(1*(best_X_0 == out_dict["best_X_0"][0])))
            # print("Hb")
            # print(Hb_per_graph[:-1])
            # print(Hb_per_graph[:-1].shape)
            # print(out_dict["X_0"].shape, best_X_0.shape, out_dict["X_0"][0].shape, "here")
            # print(relaxed_energies_per_graph.shape, Hb_per_node.shape, Hb_per_graph.shape)
            #
            # if(float(Hb) != 0):
            #     raise ValueError("")
        out_dict = self._calculate_advantages(out_dict)

        start_backprob_time = time.time()
        (loss, (log_dict, key)), params, opt_state = self._update_policy(params, opt_state, graphs, out_dict["RL"], key)
        end_backprob_time = time.time()
        # out_dict = jax.tree_map(lambda x: np.array(x), out_dict)
        # self.__init_Dataloader(out_dict)
        overall_backprob_time = end_backprob_time - start_backprob_time

        for dict_key in log_dict["Losses"].keys():
            log_dict["Losses"][dict_key] = np.mean(log_dict["Losses"][dict_key])

        # for dict_key in log_dict["Losses"].keys():
        #     out_dict["Losses"][dict_key] = log_dict["Losses"][dict_key]
        out_dict.update(log_dict)

        time_dict = {"time": {"buffer_backprob": overall_backprob_time, "env_steps": overall_env_step_time}}
        out_dict.update(time_dict)

        return params, opt_state, loss, (out_dict, energy_graph_batch, key)

    #@partial(jax.jit, static_argnums=(0,))
    def _update_policy(self, params, opt_state, graphs, RL_buffer, key):

        log_dict = {"Losses": {"overall_loss": [], "critic_loss": [], "actor_loss": [], "max_ratios": [], "min_ratios": [], "mean_ratios": [], "perc_clipped": []}}

        for i in range(self.inner_loop_steps):

            perm_diff_array, perm_state_array, key = self._shuffle_index_array(key)
            split_diff_array_list = self._split_arrays(perm_diff_array, self.n_diff_batches, -1)
            split_state_array_list = self._split_arrays(perm_state_array, self.n_diff_batches, -1)

            for split_diff_arr, split_state_arr in zip(split_diff_array_list, split_state_array_list):
                split_split_diff_arr_list = self._split_arrays(split_diff_arr, self.n_state_batches, -2)
                split_split_state_arr_list = self._split_arrays(split_state_arr, self.n_state_batches, -2)
                for split_split_diff_arr, split_split_state_arr in zip(split_split_diff_arr_list, split_split_state_arr_list):
                    batch_dict, key = select_time_idxs(graphs["graphs"][0], RL_buffer, split_split_diff_arr, split_split_state_arr, key)

                    key, subkey = jax.random.split(key)
                    batched_key = jax.random.split(subkey, num=len(jax.devices()))
                    (loss, (loss_dict, _)), params, opt_state = self.pmap_PPO_loss_backward(params, opt_state, graphs, batch_dict, batched_key)

                    for dict_key in log_dict["Losses"].keys():
                        log_dict["Losses"][dict_key].append(loss_dict[dict_key])

        return (loss, (log_dict, key)), params, opt_state


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
        spin_log_probs = out_dict["spin_log_probs"]
        state_log_probs = out_dict["state_log_probs"]
        spin_logits_next = out_dict["spin_logits"]
        graph_log_prob = out_dict["graph_log_prob"]
        Values = out_dict["Values"]
        rand_node_features = out_dict["rand_node_features"]

        scan_dict["rand_node_features_diff_steps"] = scan_dict["rand_node_features_diff_steps"].at[i].set(rand_node_features)

        entropy_step = self._get_entropy_step(energy_graph_batch, state_log_probs, node_gr_idx)
        ### TODO is this still correct for annealed noise distr? Anneled reward should be given to step i-1?!
        scan_dict["noise_rewards"] = self._get_noise_distr_step(energy_graph_batch, X_prev, X_next, model_step_idx, node_gr_idx, T, scan_dict["noise_rewards"])

        X_prev = X_next
        scan_dict["Xs_over_different_steps"] = scan_dict["Xs_over_different_steps"].at[i + 1].set(X_next)

        scan_dict["entropy_rewards"] = scan_dict["entropy_rewards"].at[i].set(T * entropy_step)

        scan_dict["log_policies"] = scan_dict["log_policies"].at[i].set(state_log_probs)
        scan_dict["Values_over_diff_steps"] = scan_dict["Values_over_diff_steps"].at[i].set(Values)

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

        log_policies = jnp.zeros((overall_diffusion_steps, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        Xs_over_different_steps = jnp.zeros((overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], 1))
        prob_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1,), dtype=jnp.float32)
        noise_rewards = jnp.zeros((overall_diffusion_steps , n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        entropy_rewards = jnp.zeros((overall_diffusion_steps , n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        Values_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        rand_node_features_diff_steps = jnp.zeros(
            (overall_diffusion_steps, X_prev.shape[0], X_prev.shape[1], self.n_random_node_features), dtype=jnp.float32)

        prob_over_diff_steps = prob_over_diff_steps.at[0].set(1/self.n_bernoulli_features)
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)


        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)
        scan_dict = {"log_policies": log_policies, "Xs_over_different_steps": Xs_over_different_steps, "prob_over_diff_steps": prob_over_diff_steps, "noise_rewards": noise_rewards, "entropy_rewards": entropy_rewards,
                    "Values_over_diff_steps": Values_over_diff_steps, "rand_node_features_diff_steps":rand_node_features_diff_steps,
                    "step": 0, "node_gr_idx": node_gr_idx, "params": params, "key": key, "X_prev": X_prev, "graphs": graphs, "energy_graph_batch": energy_graph_batch, "T": T}

        scan_dict, out_dict_list = jax.lax.scan(self.scan_body, scan_dict, None, length = overall_diffusion_steps)

        key = scan_dict["key"]
        spin_log_probs = out_dict_list["spin_log_probs"][-1]
        spin_logits_next = out_dict_list["spin_logits_next"][-1]

        X_next = scan_dict["X_prev"]#
        energy_step, Hb, best_X_0, key = self._get_energy_step(energy_graph_batch, X_next, node_gr_idx, key)
        energy_reward = -energy_step

        noise_rewards = scan_dict["noise_rewards"]
        entropy_rewards = scan_dict["entropy_rewards"]
        combined_reward = self.NoiseDistrClass.calculate_noise_distr_reward(-noise_rewards, entropy_rewards)

        rewards = combined_reward
        rewards = rewards.at[-1].set(rewards[-1] + energy_reward)

        graph_energies = energy_step[:-1]
        graph_energies = graph_energies[..., None]
        X_0 = X_next

        Xs_over_different_steps = scan_dict["Xs_over_different_steps"]
        rand_node_features_diff_steps = scan_dict["rand_node_features_diff_steps"]
        Values_over_diff_steps = scan_dict["Values_over_diff_steps"]
        log_policies = scan_dict["log_policies"]
        entropy_rewards = scan_dict["entropy_rewards"]
        prob_over_diff_steps = scan_dict["prob_over_diff_steps"]
        noise_rewards = scan_dict["noise_rewards"]

        log_q_0_T = jnp.zeros((overall_diffusion_steps+1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        log_p_0_T = jnp.zeros((overall_diffusion_steps+1, n_graphs, X_prev.shape[1]), dtype=jnp.float32)
        log_q_0_T = log_q_0_T.at[0].set(log_q_T)
        log_q_0_T = log_q_0_T.at[1:].set(log_policies)
        log_p_0_T = log_p_0_T.at[:-1].set(-1/(T* 10**-6)*noise_rewards)
        log_p_0_T = log_p_0_T.at[-1].set(1/(T* 10**-6)*energy_step)

        forward_KL = self._forward_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])
        reverse_KL = self._reverse_KL_loss_value(log_q_0_T[:,:-1], log_p_0_T[:,:-1])

        x_axis = jnp.arange(0, overall_diffusion_steps)
        log_dict = {"RL": {"states": Xs_over_different_steps[0:-1], "rand_node_features": rand_node_features_diff_steps,
                           "actions": Xs_over_different_steps[1:], "policies": log_policies, "rewards": rewards,
                           "values": Values_over_diff_steps},
                    "Losses": {"forward_KL": forward_KL, "reverse_KL": reverse_KL, "noise_rewards": jnp.mean(jnp.sum(noise_rewards[0:, :-1], axis=0)),
                               "energy_rewards": jnp.mean(energy_reward[:-1]),
                               "neg_entropy_rewards": jnp.mean(jnp.sum(entropy_rewards[0:, :-1], axis=0)),
                               "overall_rewards": jnp.mean(jnp.sum(rewards[0:, :-1], axis=0))},
                    "metrics": {"energies": graph_energies, "entropies": 0., "spin_log_probs": spin_log_probs,
                                "graph_mean_energies": energy_reward, "free_energies": 0.},
                    "figures": {"prob_over_diff_steps": {"x_values": x_axis, "y_values": prob_over_diff_steps},
                                "overall_rewards_over_diff_steps": {"x_values": x_axis,
                                                                    "y_values": jnp.mean(
                                                                        jnp.mean(rewards[:, :-1], axis=-1), axis=-1)},
                                "entropy_rewards": {"x_values": x_axis,
                                                    "y_values": jnp.mean(jnp.mean(entropy_rewards[:, :-1], axis=-1),
                                                                         axis=-1)},
                                "noise_rewards": {"x_values": x_axis,
                                                  "y_values": jnp.mean(jnp.mean(noise_rewards[:, :-1], axis=-1),
                                                                       axis=-1)}
                                },
                    "energies": {"HA": graph_energies, "Hb": Hb},
                    "log_p_0": spin_logits_next,
                    "X_0": X_0,
                    "best_X_0": best_X_0,
                    "bin_sequence": Xs_over_different_steps,
                    "spin_log_probs": spin_log_probs,
                    }
        return log_dict, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_noise_distr_step(self, jraph_graph, X_prev, X_next, t_idx, node_gr_idx, T,  noise_rewards_arr):
        noise_rewards_arr = self.calc_noise_step( jraph_graph, X_prev, X_next, t_idx, node_gr_idx, T, noise_rewards_arr)
        return noise_rewards_arr

    @partial(jax.jit, static_argnums=(0,))
    def _get_noise_distr_step_relaxed(self, jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx):
        Noise_Energy_per_graph = self.calc_noise_step_relaxed( jraph_graph, spin_logits_prev, spin_logits_next, X_prev, gamma_t, node_gr_idx)
        return (-1)*Noise_Energy_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def _get_entropy_step(self, jraph_graph, state_log_probs, node_gr_idx):
        return -state_log_probs

    @partial(jax.jit, static_argnums=(0,))
    def _get_entropy_step_relaxed(self, jraph_graph, spin_logits, node_gr_idx):
        log_probs_down = spin_logits[..., 0]
        log_probs_up = spin_logits[..., 1]
        probs_up = jnp.exp(log_probs_up)
        probs_down = jnp.exp(log_probs_down)

        entropy_term_1 = -probs_up * log_probs_up
        entropy_term_2 = -probs_down * log_probs_down
        entropy_term_per_node = entropy_term_1 + entropy_term_2

        n_graph = jraph_graph.n_node.shape[0]
        relaxed_entropies_per_graph = jax.ops.segment_sum(jnp.sum(entropy_term_per_node, axis=-1, keepdims=True),
                                                          node_gr_idx, n_graph)
        return relaxed_entropies_per_graph[...,0]

    @partial(jax.jit, static_argnums=(0,))
    def _get_energy_step(self, jraph_graph, X_0, node_gr_idx, key):
        if(self.proj_method == "feasible"):
            best_X_0, Hb, relaxed_energies_per_graph = self.vmapped_energy_feasible(jraph_graph, X_0)
            Hb = jnp.mean(jnp.abs(Hb))
        elif(self.proj_method == "CE"):
            best_X_0, relaxed_energies_per_graph, Hb_per_graph = self.vmapped_energy_CE(jraph_graph, X_0, node_gr_idx)
            Hb = jnp.mean(jnp.abs(Hb_per_graph[:-1]))
        else:
            relaxed_energies_per_graph, _, Hb_per_graph = self.vmapped_relaxed_energy(jraph_graph, X_0, node_gr_idx)
            best_X_0 = X_0
            Hb = jnp.mean(jnp.abs(Hb_per_graph)[:-1])

        return relaxed_energies_per_graph[...,0], Hb, best_X_0, key

    @partial(jax.jit, static_argnums=(0,))
    def _get_energy_reward_relaxed(self, jraph_graph, spin_logits, node_gr_idx):
        relaxed_energies_per_graph, _, _ = self.vmapped_relaxed_energy_for_Loss(jraph_graph, spin_logits, node_gr_idx)
        return relaxed_energies_per_graph[...,0]



    ### TODO add scanner
    @partial(jax.jit, static_argnums=(0,))
    def _calc_traces(self, values, rewards):
        max_steps = self.n_diffusion_steps
        advantage = jnp.zeros_like(values)
        for t in range(max_steps):
            idx = max_steps - t - 1
            delta = rewards[idx] + self.gamma * values[idx + 1] - values[idx]
            advantage = advantage.at[idx].set(delta + self.gamma * self.lam * advantage[idx + 1])

        value_target = (advantage + values)[0:max_steps]
        return value_target, advantage[0:max_steps]

    def _calculate_advantages(self, log_dict):

        rewards = log_dict["RL"]["rewards"]
        reduced_rewards = rewards[:,:,:-1]

        mov_average_reward, mov_std_reward =  self.MovingAverageClass.update_mov_averages(reduced_rewards)
        normed_rewards = self.MovingAverageClass.calculate_average(rewards, mov_average_reward, mov_std_reward)

        log_dict["RL"]["normed_rewards"] = normed_rewards
        log_dict["energies"]["normed_rewards"] = jnp.swapaxes(normed_rewards[:,:,:-1], 1, 2)
        log_dict["energies"]["reduced_rewards"] = jnp.swapaxes(reduced_rewards, 1, 2)
        log_dict["energies"]["mov_average_reward"] = mov_average_reward
        log_dict["energies"]["mov_std_reward"] = mov_std_reward

        value_target, advantages = self.pmap_calc_traces(log_dict["RL"]["values"], log_dict["RL"]["normed_rewards"])
        normed_advantages = self._normalize_advantages(advantages)
        log_dict["RL"]["value_target"] = value_target
        log_dict["RL"]["advantages"] = normed_advantages
        return log_dict

    @partial(jax.jit, static_argnums=(0,))
    def _normalize_advantages(self, advantages):
        unpadded_adv = advantages[:,:,:-1]
        normed_advantages = (advantages - jnp.mean(unpadded_adv))/(jnp.std(unpadded_adv)+10**-10)
        return normed_advantages

    def get_loss(self, params, jraph_graph_list, batch_dict, key):
        return self.PPO_loss(params, jraph_graph_list, batch_dict, key)

    @partial(jax.jit, static_argnums=(0,))
    def PPO_loss(self, params, jraph_graph_list, batch_dict, key):
        Sb_Hb_Nb_A_k = batch_dict["advantages"]
        Sb_Hb_Nb_X_prev = batch_dict["states"]
        Sb_Hb_Nb_rand_node_features = batch_dict["rand_node_features"]
        Sb_Hb_Nb_X_next = batch_dict["actions"]

        Sb_Nb_t_idx_per_node = batch_dict["time_index_per_node"]
        Sb_Hb_Nb_state_log_probs = batch_dict["policies"]
        Sb_Hb_Nb_rtg = batch_dict["value_target"]

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=Sb_Hb_Nb_A_k.shape[0])

        out_dict, _ = self.vmapped_calc_log_q(params, jraph_graph_list, Sb_Hb_Nb_X_prev, Sb_Hb_Nb_rand_node_features, Sb_Hb_Nb_X_next, Sb_Nb_t_idx_per_node, batched_key)

        out_values = out_dict["Values"]
        state_log_probs = out_dict["state_log_probs"]

        ratios = jnp.exp(state_log_probs - Sb_Hb_Nb_state_log_probs)

        surr1 = ratios * Sb_Hb_Nb_A_k
        ### TODO replace it with KL diergence?
        surr2 = jnp.clip(ratios, 1 - self.clip_value, 1 + self.clip_value) * Sb_Hb_Nb_A_k

        actor_loss = jnp.mean((-jnp.minimum(surr1, surr2)[:,:-1]))
        critic_loss = jnp.mean((out_values - Sb_Hb_Nb_rtg)[:,:-1] ** 2)
        overall_loss = (1-self.c1) * actor_loss + self.c1 *critic_loss

        max_ratios = jnp.max(ratios)
        min_ratios = jnp.min(ratios)
        mean_ratios = jnp.mean(ratios)
        clip_fraction = jnp.mean(1*(ratios < 1-self.clip_value)+1*(ratios > 1+self.clip_value))


        loss_dict = {"actor_loss": actor_loss, "critic_loss": critic_loss, "overall_loss": overall_loss,
                     "max_ratios": max_ratios, "min_ratios": min_ratios, "mean_ratios": mean_ratios, "perc_clipped": clip_fraction}
        return overall_loss, (loss_dict, key)