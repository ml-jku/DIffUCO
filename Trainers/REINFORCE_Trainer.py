import jax.numpy as jnp
from functools import partial
import jax
from .BaseTrainer import Base

class Reinforce(Base):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        super(Reinforce, self).__init__(config, EnergyClass, NoiseClass, model)
        self.N_basis_states = self.config["N_basis_states"]
        self.n_bernoulli_features = self.config["n_bernoulli_features"]
        self.n_diffusion_steps = self.config["n_diffusion_steps"]
        self.inner_update_steps = 1
        self.diffusion_loss_eval = lambda a,b,c,d,e: self.diffusion_loss(a, b, c, d, e, "eval")
        self.diffusion_loss_train = lambda a,b,c,d,e: self.diffusion_loss(a, b, c, d, e,  "train")

        # self.diffusion_loss_eval = lambda a,b,c,d,e: self._environment_steps_scan(a, b, c, d, e, "eval")
        # self.diffusion_loss_train = lambda a,b,c,d,e: self._environment_steps_scan(a, b, c, d, e,  "train")

        # self.vmapped_make_one_step = jax.vmap(self.model.make_one_step, in_axes=(None, None, 1, None, 0),
        #                                       out_axes=(1, 1, 1, 1, 0))

    def get_loss(self, params, graphs, energy_graph_batch, T, key):
        return self.diffusion_loss_train(params, graphs, energy_graph_batch, T, key)

    def sample(self, params, graphs, energy_graph_batch, T, key):
        return self.diffusion_loss_eval(params, graphs, energy_graph_batch, T, key)

    @partial(jax.jit, static_argnums=(0,))
    def __get_entropy_loss(self, jraph_graph, spin_logits, sum_log_p_prev_per_node, node_gr_idx):

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

        entropy_term_per_node_no_grad = jax.lax.stop_gradient(entropy_term_per_node)
        # baseline = jnp.mean(entropy_term_per_node_no_grad, axis = -1, keepdims=True)
        baseline_per_graph = jnp.sum(
            jax.lax.stop_gradient(jnp.mean(relaxed_entropies_per_graph, axis=1, keepdims=True)), axis=-1)
        # print("baseline entropy",entropy_term_per_node_no_grad.shape,  baseline.shape)

        # L_REINFORCE_per_node = (entropy_term_per_node_no_grad -  baseline)*sum_log_p_prev_per_node
        # L_REINFORCE_per_graph = jax.ops.segment_sum(L_REINFORCE_per_node, node_gr_idx, n_graph)

        log_state_prob = jnp.sum(jax.ops.segment_sum(sum_log_p_prev_per_node, node_gr_idx, n_graph), axis=-1)
        entropy_per_State = jnp.sum(jax.ops.segment_sum(entropy_term_per_node_no_grad, node_gr_idx, n_graph), axis=-1)
        # print("here entropy", baseline_per_graph.shape, entropy_per_State.shape, log_state_prob.shape, relaxed_entropies_per_graph.shape)
        L_REINFORCE_per_graph = (entropy_per_State - baseline_per_graph) * log_state_prob
        L_REINFORCE = jnp.mean(L_REINFORCE_per_graph[:-1])

        L_repara_per_graph = relaxed_entropies_per_graph
        L_repara = jnp.mean(L_repara_per_graph[:-1])
        relaxed_entropies = L_repara

        Entropy_Loss = L_repara + L_REINFORCE
        return Entropy_Loss, jax.lax.stop_gradient(relaxed_entropies), jax.lax.stop_gradient(L_repara), L_REINFORCE

    @partial(jax.jit, static_argnums=(0,))
    def __get_energy_loss(self, jraph_graph, spin_logits, sum_log_p_prev_per_node, node_gr_idx):

        n_graph = jraph_graph.n_node.shape[0]

        relaxed_energies_per_graph, HA_per_graph, HB_per_graph = self.vmapped_relaxed_energy_for_Loss(jraph_graph,
                                                                                                      spin_logits,
                                                                                                      node_gr_idx)

        HA = jnp.mean(HA_per_graph[:-1])
        HB = jnp.mean(HB_per_graph[:-1])
        Energy_dict = {"HA": HA, "HB": HB, "HA + HB": HA + HB}

        relaxed_energies_per_graph_no_grad = jax.lax.stop_gradient(relaxed_energies_per_graph)
        baseline_per_graph = jax.lax.stop_gradient(jnp.mean(relaxed_energies_per_graph, axis=1, keepdims=True))

        log_state_prob = jnp.sum(jax.ops.segment_sum(sum_log_p_prev_per_node, node_gr_idx, n_graph), axis=-1,
                                 keepdims=True)
        # print("here energy", relaxed_energies_per_graph_no_grad.shape, baseline_per_graph.shape, log_state_prob.shape)
        L_REINFORCE_per_graph = (relaxed_energies_per_graph_no_grad - baseline_per_graph) * log_state_prob

        L_REINFORCE = jnp.mean(L_REINFORCE_per_graph[:-1])

        L_repara_per_graph = relaxed_energies_per_graph[:-1]
        L_repara = jnp.mean(L_repara_per_graph)

        Energy_Loss = L_repara + L_REINFORCE
        return Energy_Loss, jax.lax.stop_gradient(L_repara_per_graph), jax.lax.stop_gradient(
            L_repara), jax.lax.stop_gradient(L_REINFORCE), Energy_dict

    @partial(jax.jit, static_argnums=(0,))
    def __get_Noise_energy_loss(self, jraph_graph, X_prev, spin_logits_prev, spin_logits_next, log_p_prev_per_node, model_step_idx, node_gr_idx, T):
        Noise_Energy_per_graph, sum_log_p_prev_per_node = self.Noise_func(jraph_graph, spin_logits_prev, spin_logits_next, X_prev, log_p_prev_per_node, model_step_idx, node_gr_idx, T)
        Noise_Energy_per_graph = (-1)*Noise_Energy_per_graph

        n_graph = jraph_graph.n_node.shape[0]
        L_repara_per_graph = Noise_Energy_per_graph

        noise_energy_per_graph_no_grad = jax.lax.stop_gradient(Noise_Energy_per_graph)
        #baseline = jnp.mean(noise_energy_per_node_no_grad, axis = -2, keepdims=True)
        baseline_per_graph = jax.lax.stop_gradient(jnp.mean(Noise_Energy_per_graph, axis = 1, keepdims=True))
        #print("baseline noise",noise_energy_per_node_no_grad.shape,  baseline.shape)


        log_state_prob = jnp.sum(jax.ops.segment_sum(sum_log_p_prev_per_node, node_gr_idx, n_graph), axis = -1)
        #print("here noise engery", noise_energy_per_graph_no_grad.shape,log_state_prob,  baseline_per_graph.shape)

        L_REINFORCE_per_graph = (noise_energy_per_graph_no_grad-baseline_per_graph)*log_state_prob

        L_REINFORCE = jnp.mean(L_REINFORCE_per_graph[:-1])

        L_repara = jnp.mean(L_repara_per_graph[:-1])
        Noise_Loss = (L_REINFORCE + L_repara)
        return Noise_Loss, jax.lax.stop_gradient(L_repara), jax.lax.stop_gradient(L_repara), jax.lax.stop_gradient(L_REINFORCE)

    @partial(jax.jit, static_argnums=(0,-1))
    def diffusion_loss(self, params, graphs, energy_graph_batch, T, key, mode):
        print("function is being jitted")
        if(mode == "train"):
            N_basis_states = self.N_basis_states
        else:
            N_basis_states = self.N_test_basis_states
        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, one_hot_state, log_p_uniform, key = self.model.sample_prior(energy_graph_batch, N_basis_states,
                                                                            key)

        spin_logits_prev = log_p_uniform
        spin_log_probs_prev = jnp.sum(spin_logits_prev * one_hot_state, axis=-1)

        L_entropy = 0.
        L_noise = 0.

        L_energy_Reinforce = 0.
        L_noise_Reinforce = 0.
        L_entropy_Reinforce = 0.
        L_energy_repara = 0.
        L_noise_repara = 0.
        L_entropy_repara = 0.

        log_p_prev_per_node = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], self.n_bernoulli_features))
        Xs_over_different_steps = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], self.n_bernoulli_features))
        prob_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1,))
        Noise_loss_over_diff_steps = jnp.zeros((overall_diffusion_steps,))
        Energy_over_diff_steps = jnp.zeros((overall_diffusion_steps,))
        n_graphs = energy_graph_batch.n_node.shape[0]
        log_p_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs -1, X_prev.shape[1]))

        prob_over_diff_steps = prob_over_diff_steps.at[0].set(0.5)
        log_p_prev_per_node = log_p_prev_per_node.at[0].set(spin_log_probs_prev)
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)
        # key = jax.random.split(key, num=self.N_basis_states)
        # Energy_over_diff_steps = Energy_over_diff_steps.at[0].set(self.__get_energy_loss(graphs, spin_logits_prev, spin_logits_prev)[2])

        for i in range(overall_diffusion_steps):
            model_step_idx = jnp.array([i / self.eval_step_factor], dtype=jnp.int16)
            model_step_idx_per_node = model_step_idx[0] * jnp.ones((energy_graph_batch.nodes.shape[0], 1),
                                                                   dtype=jnp.int16)
            key, subkey = jax.random.split(key)
            batched_key = jax.random.split(subkey, num=X_prev.shape[1])


            out_dict, _ = self.vmapped_make_one_step(params, graphs, X_prev, model_step_idx_per_node, batched_key)

            X_next = out_dict["X_next"]
            spin_log_probs = out_dict["spin_log_probs"]
            spin_logits_next = out_dict["spin_logits"]
            graph_log_prob = out_dict["graph_log_prob"]

            log_p_t = self.NoiseDistrClass.get_log_p_T_0(energy_graph_batch, X_prev, X_next, model_step_idx, T)
            log_p_0_T = log_p_0_T.at[i].set(log_p_t[:-1])

            Entropy_Loss, Entropy, Loss_entropy_repara, Loss_entropy_Reinforce = self.__get_entropy_loss(
                energy_graph_batch, spin_logits_next, jnp.sum(log_p_prev_per_node, axis=0), node_gr_idx)
            L_entropy += Entropy_Loss
            L_entropy_repara += Loss_entropy_repara
            L_entropy_Reinforce += Loss_entropy_Reinforce

            Noise_Loss, Noise_Energy, Loss_noise_repara, Loss_noise_Reinforce = self.__get_Noise_energy_loss(
                energy_graph_batch, X_prev, spin_logits_prev, spin_logits_next, log_p_prev_per_node, model_step_idx,
                node_gr_idx, T)
            L_noise += Noise_Loss
            L_noise_repara += Loss_noise_repara
            L_noise_Reinforce += Loss_noise_Reinforce

            log_p_prev_per_node = log_p_prev_per_node.at[i + 1].set(spin_log_probs)

            X_prev = X_next
            Xs_over_different_steps = Xs_over_different_steps.at[i + 1].set(X_next)
            spin_logits_prev = spin_logits_next

            average_probs = jnp.mean(graph_log_prob[:-1])
            # print("train", average_probs, prob_over_diff_steps.shape, average_probs.shape, i)
            prob_over_diff_steps = prob_over_diff_steps.at[i + 1].set(average_probs)
            Noise_loss_over_diff_steps = Noise_loss_over_diff_steps.at[i].set(Loss_noise_repara)
        # Energy_over_diff_steps = Energy_over_diff_steps.at[i].set(self.__get_energy_loss(energy_graph_batch, spin_logits_next, jnp.sum(log_p_prev_per_node, axis = 0), node_gr_idx)[2])

        L_energy, energies, Loss_energy_repara, Loss_energy_Reinforce, Energy_dict = self.__get_energy_loss(
            energy_graph_batch, spin_logits_next, jnp.sum(log_p_prev_per_node[:-1], axis=0), node_gr_idx)

        log_p_0 = self.EnergyClass.get_log_p_0_from_energy(energies, T)
        log_p_0_T = log_p_0_T.at[i+1].set(log_p_0)
        mean_log_p_0_T = jnp.mean(jnp.sum(log_p_0_T[:], axis = 0))
        mean_p_0_T = jnp.mean(jnp.exp(jnp.sum(log_p_0_T[:], axis = 0)))
        mean_log_p_1_T = jnp.mean(jnp.sum(log_p_0_T[:-1], axis = 0))
        mean_p_1_T = jnp.mean(jnp.exp(jnp.sum(log_p_0_T[:-1], axis = 0)))

        L_energy_repara += Loss_energy_repara
        L_energy_Reinforce += Loss_energy_Reinforce
        graph_mean_energy = energies

        X_0 = X_next

        Loss = self.calc_loss(L_entropy, L_noise, L_energy, T)

        x_axis = jnp.arange(0, overall_diffusion_steps)

        log_dict = {"Losses": {"L_entropy": L_entropy, "L_noise": L_noise, "L_energy": L_energy,
                               "L_Energy_Reinfroce": L_energy_Reinforce, "L_Noise_Reinforce": L_noise_Reinforce,
                               "L_entropy_REINFORCE": L_entropy_Reinforce, "L_entropy_repara": L_entropy_repara,
                               "L_noise_repara": L_noise_repara, "L_energy_repara": L_energy_repara,
                               "overall_Loss": Loss, "log_p_0_T": mean_log_p_0_T, "log_p_1_T": mean_log_p_1_T
                               ,"p_0_T": mean_p_0_T, "p_1_T": mean_p_1_T},
                    "metrics": {"energies": energies, "entropies": 0., "spin_log_probs": spin_log_probs,
                                "free_energies": L_entropy, "graph_mean_energies": graph_mean_energy},
                    "energies": Energy_dict,
                    "figures": {"prob_over_diff_steps": {"x_values": x_axis, "y_values": prob_over_diff_steps},
                                "Noise_loss_over_diff_steps": {"x_values": x_axis,
                                                               "y_values": Noise_loss_over_diff_steps},
                                "Energy_over_diff_steps": {"x_values": x_axis,
                                                           "y_values": Energy_over_diff_steps}
                                },
                    "log_p_0": spin_logits_next,
                    "X_0": X_0,
                    "bin_sequence": Xs_over_different_steps,
                    "spin_log_probs": spin_log_probs,
                    "figs": {}
                    }

        return Loss, (log_dict, key)

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
        spin_logits_next = out_dict["spin_logits"]
        graph_log_prob = out_dict["graph_log_prob"]

        log_p_t = self.NoiseDistrClass.get_log_p_T_0(energy_graph_batch, X_prev, X_next, model_step_idx, T)
        scan_dict["log_p_0_T"] = scan_dict["log_p_0_T"].at[i].set(log_p_t[:-1])

        Entropy_Loss, Entropy, Loss_entropy_repara, Loss_entropy_Reinforce = self.__get_entropy_loss(
            energy_graph_batch, spin_logits_next, jnp.sum(scan_dict["log_p_prev_per_node"], axis=0), node_gr_idx)
        scan_dict["L_entropy"] += Entropy_Loss
        scan_dict["L_entropy_repara"] += Loss_entropy_repara
        scan_dict["L_entropy_Reinforce"] += Loss_entropy_Reinforce

        Noise_Loss, Noise_Energy, Loss_noise_repara, Loss_noise_Reinforce = self.__get_Noise_energy_loss(
            energy_graph_batch, X_prev, scan_dict["spin_logits_prev"], spin_logits_next, scan_dict["log_p_prev_per_node"], model_step_idx,
            node_gr_idx, T)
        scan_dict["L_noise"] += Noise_Loss
        scan_dict["L_noise_repara"] += Loss_noise_repara
        scan_dict["L_noise_Reinforce"] += Loss_noise_Reinforce

        scan_dict["log_p_prev_per_node"] = scan_dict["log_p_prev_per_node"].at[i + 1].set(spin_log_probs)

        scan_dict["Noise_loss_over_diff_steps"] = scan_dict["Noise_loss_over_diff_steps"].at[i].set(Loss_noise_repara)

        X_prev = X_next
        scan_dict["Xs_over_different_steps"] = scan_dict["Xs_over_different_steps"].at[i + 1].set(X_next)


        average_probs = jnp.mean(graph_log_prob[:-1])
        scan_dict["prob_over_diff_steps"] = scan_dict["prob_over_diff_steps"].at[i + 1].set(average_probs)

        scan_dict["X_prev"] = X_prev
        scan_dict["step"] += 1
        scan_dict["spin_logits_prev"] = spin_logits_next

        out_dict = {}
        out_dict["spin_log_probs"] = spin_log_probs
        out_dict["spin_logits_next"] = spin_logits_next
        return scan_dict, out_dict

    @partial(jax.jit, static_argnums=(0, -1))
    def _environment_steps_scan(self, params, graphs, energy_graph_batch, T, key, mode):
        ### TDOD cahnge rewards to non exact expectation rewards
        print("function is being jitted")
        if (mode == "train"):
            N_basis_states = self.N_basis_states
        else:
            N_basis_states = self.N_test_basis_states
        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, one_hot_state, log_p_uniform, key = self.model.sample_prior(energy_graph_batch, N_basis_states,
                                                                            key)

        spin_logits_prev = log_p_uniform
        spin_log_probs_prev = jnp.sum(spin_logits_prev * one_hot_state, axis=-1)

        L_entropy = 0.
        L_noise = 0.

        L_energy_Reinforce = 0.
        L_noise_Reinforce = 0.
        L_entropy_Reinforce = 0.
        L_energy_repara = 0.
        L_noise_repara = 0.
        L_entropy_repara = 0.

        log_p_prev_per_node = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], self.n_bernoulli_features))
        Xs_over_different_steps = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], self.n_bernoulli_features))
        prob_over_diff_steps = jnp.zeros((overall_diffusion_steps + 1,))
        Noise_loss_over_diff_steps = jnp.zeros((overall_diffusion_steps,))
        Energy_over_diff_steps = jnp.zeros((overall_diffusion_steps,))
        n_graphs = energy_graph_batch.n_node.shape[0]
        log_p_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs - 1, X_prev.shape[1]))

        prob_over_diff_steps = prob_over_diff_steps.at[0].set(0.5)
        log_p_prev_per_node = log_p_prev_per_node.at[0].set(spin_log_probs_prev)
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)

        scan_dict = { "L_entropy": L_entropy,
                    "L_noise": L_noise,
                    "L_energy_Reinforce": L_energy_Reinforce,
                    "L_noise_Reinforce": L_noise_Reinforce,
                    "L_entropy_Reinforce": L_entropy_Reinforce,
                    "L_energy_repara": L_energy_repara,
                    "L_noise_repara": L_noise_repara,
                    "L_entropy_repara": L_entropy_repara, "log_p_0_T": log_p_0_T, "spin_logits_prev": spin_logits_prev,
                    "Xs_over_different_steps": Xs_over_different_steps, "Noise_loss_over_diff_steps": Noise_loss_over_diff_steps,
                     "prob_over_diff_steps": prob_over_diff_steps, "log_p_prev_per_node": log_p_prev_per_node,
                     "step": 0, "node_gr_idx": node_gr_idx, "params": params, "key": key, "X_prev": X_prev,
                     "graphs": graphs, "energy_graph_batch": energy_graph_batch, "T": T}

        scan_dict, out_dict_list = jax.lax.scan(self.scan_body, scan_dict, None, length=overall_diffusion_steps)


        Xs_over_different_steps = scan_dict["Xs_over_different_steps"]
        prob_over_diff_steps = scan_dict["prob_over_diff_steps"]
        spin_logits_next = out_dict_list["spin_logits_next"][-1]
        spin_log_probs = out_dict_list["spin_log_probs"][-1]

        L_energy, energies, Loss_energy_repara, Loss_energy_Reinforce, Energy_dict = self.__get_energy_loss(
            energy_graph_batch, spin_logits_next, jnp.sum(log_p_prev_per_node[:-1], axis=0), node_gr_idx)

        log_p_0 = self.EnergyClass.get_log_p_0_from_energy(energies, T)
        log_p_0_T = log_p_0_T.at[-1].set(log_p_0)
        mean_log_p_0_T = jnp.mean(jnp.sum(log_p_0_T[:], axis = 0))
        mean_p_0_T = jnp.mean(jnp.exp(jnp.sum(log_p_0_T[:], axis = 0)))
        mean_log_p_1_T = jnp.mean(jnp.sum(log_p_0_T[:-1], axis = 0))
        mean_p_1_T = jnp.mean(jnp.exp(jnp.sum(log_p_0_T[:-1], axis = 0)))

        L_energy_repara += Loss_energy_repara
        L_energy_Reinforce += Loss_energy_Reinforce
        graph_mean_energy = energies

        X_0 = scan_dict["X_prev"]

        Loss = self.calc_loss(L_entropy, L_noise, L_energy, T)


        x_axis = jnp.arange(0, overall_diffusion_steps)

        log_dict = {"Losses": {"L_entropy": L_entropy, "L_noise": L_noise, "L_energy": L_energy,
                               "L_Energy_Reinfroce": L_energy_Reinforce, "L_Noise_Reinforce": L_noise_Reinforce,
                               "L_entropy_REINFORCE": L_entropy_Reinforce, "L_entropy_repara": L_entropy_repara,
                               "L_noise_repara": L_noise_repara, "L_energy_repara": L_energy_repara,
                               "overall_Loss": Loss, "log_p_0_T": mean_log_p_0_T, "log_p_1_T": mean_log_p_1_T
                               ,"p_0_T": mean_p_0_T, "p_1_T": mean_p_1_T},
                    "metrics": {"energies": energies, "entropies": 0., "spin_log_probs": spin_log_probs,
                                "free_energies": L_entropy, "graph_mean_energies": graph_mean_energy},
                    "energies": Energy_dict,
                    "figures": {"prob_over_diff_steps": {"x_values": x_axis, "y_values": prob_over_diff_steps},
                                "Noise_loss_over_diff_steps": {"x_values": x_axis,
                                                               "y_values": Noise_loss_over_diff_steps},
                                "Energy_over_diff_steps": {"x_values": x_axis,
                                                           "y_values": Energy_over_diff_steps}
                                },
                    "log_p_0": spin_logits_next,
                    "X_0": X_0,
                    "bin_sequence": Xs_over_different_steps,
                    "spin_log_probs": spin_log_probs,
                    "figs": {}
                    }

        return Loss, (log_dict, key)

