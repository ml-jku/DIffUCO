from abc import ABC, abstractmethod
from functools import partial
import jax
import jraph
import optax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import flax
import scipy.special
import wandb
from tqdm import tqdm
import time

class Base(ABC):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        ### TODO implement learning rate schedule correctly
        self.config = config
        self.bfloat16 = self.config["bfloat16"]
        self.N_test_basis_states = self.config["n_test_basis_states"]
        self.n_random_node_features = self.config["n_random_node_features"]
        self.opt_update = None
        self.EnergyClass = EnergyClass
        self.NoiseDistrClass = NoiseClass
        self.model = model
        self.problem_name = self.config["problem_name"]
        self.dataset_name = self.config["dataset_name"]
        self.eval_step_factor = self.config["eval_step_factor"]
        self.n_sampling_rounds = self.config["n_sampling_rounds"]
        self.sampling_temp = self.config["sampling_temp"]
        print("EVAL STEP FACTOR is", self.eval_step_factor)

        self.vmapped_make_one_step = jax.vmap(self.model.make_one_step, in_axes=(None, None, 1, None, 0),
                                              out_axes=(1, 0))
        self.vmapped_sample_with_temp = jax.vmap(self.sample_with_temp, in_axes=(None, 1, None, 0), out_axes=(1, 0))

        self.NoiseDistrClass = NoiseClass
        self.Noise_func = self.NoiseDistrClass.calc_noise_loss
        self.beta_arr = self.NoiseDistrClass.beta_arr
        self.calc_loss = self.NoiseDistrClass.combine_losses

        self.loss_grad = jax.jit(jax.value_and_grad(self.get_loss, has_aux=True))
        self.pmap_sample = jax.pmap(self.sample, in_axes=(0, 0, 0, None, 0))
        self.pmap_sample_for_estimate = jax.pmap(self._environment_steps_scan_estimate, in_axes=(0, 0, 0, None, None, 0))
        self.pmap_sample_for_estimate_v2 = jax.pmap(self.sample_for_estimate_v2, in_axes=(0, 0, 0, None, None, 0))
        #self.pmap_sample_MCMC = jax.pmap(self.sample_MCMC, in_axes=(0, 0, 0, None, 0))
        self.pmap_update = jax.pmap(self.__update_params, in_axes=(0, 0, 0))
        self.pmap_loss_backward = jax.pmap(self.loss_backward, in_axes=(0, 0, 0, 0, None, 0), axis_name="device")

        self.vmapped_sample_forward_diff_process = jax.vmap(self.NoiseDistrClass.sample_forward_diff_process, in_axes=(1, None, 0), out_axes=(1,1, 0))

        self.relaxed_energy = EnergyClass.calculate_Energy
        self.vmapped_relaxed_energy = jax.vmap(self.relaxed_energy, in_axes=(None, 1, None), out_axes=(1, 1, 1))

        self.energy_CE = EnergyClass.calculate_Energy_CE
        self.vmapped_energy_CE = jax.vmap(self.energy_CE, in_axes=(None, 1, None), out_axes=(1))

        self.calculate_Energy_CE_p_values = EnergyClass.calculate_Energy_CE_p_values
        self.vmapped_calculate_Energy_CE_p_values = jax.vmap(self.calculate_Energy_CE_p_values, in_axes=(None,1), out_axes=(1))

        self.energy_feasible = EnergyClass.calculate_Energy_feasible
        self.vmapped_energy_feasible = jax.vmap(self.energy_feasible, in_axes=(None, 1), out_axes=(1, 0, 1))

        self.relaxed_Energy_for_Loss = EnergyClass.calculate_Energy_loss
        self.vmapped_relaxed_energy_for_Loss = jax.vmap(self.relaxed_Energy_for_Loss, in_axes=(None, 1, None),
                                                        out_axes=(1))

        self.pmap_apply_CE_on_p = jax.pmap(self.apply_CE_on_p, in_axes=(0, 0))

        self.n_diffusion_steps = self.config["n_diffusion_steps"]
        self.N_basis_states = self.config["N_basis_states"]
        self.batch_size = self.config["batch_size"]

        if self.problem_name == "TSP":
            self.config["edge_updates"] = True
            if "20" in self.dataset_name:
                self.n_bernoulli_features = 20
            elif "100" in self.dataset_name:
                self.n_bernoulli_features = 100
        else:
            self.n_bernoulli_features = 2

        self.unbiased_list = ["IsingModel", "SpinGlass"]
        if self.config["problem_name"] in self.unbiased_list:
            self.T_target = self.config["T_target"]
            self.dataset_name = self.config["dataset_name"]
            self.beta = 1 / self.T_target
            print(self.dataset_name)

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
                raise NotImplementedError(
                    'Dataset name must contain either "4", "8", "16", "24", "32" to infer the number of nodes')
            
            print(f"beta: {self.beta}")
            self.free_energy = self.calculate_ising_free_energy(self.beta, self.size)
            self.internal_energy = self.calculate_ising_internal_energy(self.beta, self.size)
            self.entropy = self.calculate_ising_entropy(self.beta, self.internal_energy, self.free_energy)
            print(f"Free Energy: {self.free_energy}")
            print(f"Internal Energy: {self.internal_energy}")
            print(f"Entropy: {self.entropy}")

    # print(f"Internal Energy: {self.internal_energy}")
    # print(f"Entropy: {self.entropy}")

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    def _reverse_KL_loss_value(self, log_q_0_T, log_p_0_T, diff_step_axis = 0):

        loss = jnp.mean(jnp.sum(log_q_0_T, axis = diff_step_axis) - jnp.sum(log_p_0_T , axis = diff_step_axis))

        return loss

    def _forward_KL_loss_value(self, log_q_0_T, log_p_0_T):
        weights = self._compute_importance_weights_(log_q_0_T, log_p_0_T)
        forward_KL_per_graph = -jnp.sum(weights * jnp.sum(log_q_0_T, axis=0), axis=-1)
        forward_KL = jnp.mean(forward_KL_per_graph)
        return forward_KL

    def _compute_importance_weights_(self, log_q_0_T, log_p_0_T):
        weights = jax.nn.softmax(jnp.sum(log_p_0_T - log_q_0_T, axis=0), axis=-1)
        return weights

    def _apply_CE(self):
        pass

    def train_step(self, params, opt_state, graphs, energy_graph_batch, T, key):

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))

        (loss, (log_dict, _)), params, opt_state = self.pmap_loss_backward_step(params, opt_state, graphs, energy_graph_batch, T, batched_key)
        return params, opt_state, loss, (log_dict, energy_graph_batch, key)


    def evaluation_step(self, params, graph_batch, energy_graph_batch, T, batched_key, mode="eval", key=None, n_sampling_rounds=None, sampling_temp=None, sampling_mode = "temps", epoch = None, epochs = None):
        start_forw_pass_time = time.time()
        loss, (log_dict, _) = self.pmap_sample(params, graph_batch, energy_graph_batch, T, batched_key)
        end_forw_pass_time = time.time()

        log_dict["time"] = {}
        log_dict["time"]["forward_pass"] = end_forw_pass_time - start_forw_pass_time

        if(mode == "test"):
            # print("testing eval step factor is", self.eval_step_factor)
            if(self.config["problem_name"] != "TSP"):
                p_0 = jnp.exp(log_dict["log_p_0"][...,1])
                start_CE_time = time.time()
                X_0_CE, energies_CE, Hb_per_node = self.pmap_apply_CE_on_p(energy_graph_batch, p_0)
                end_CE_time = time.time()
                ####remove padded energies
                last_node_idx = jnp.sum(energy_graph_batch.n_node[0, 0:-1])
                log_dict["metrics"]["X_0_CE"], log_dict["metrics"]["energies_CE"] = X_0_CE[:, :last_node_idx, ...], energies_CE[:,:-1]
                log_dict["time"]["CE"] = end_CE_time - start_CE_time
            else:
                log_dict["metrics"]["X_0_CE"] = log_dict["metrics"]["X_0"]
                log_dict["metrics"]["energies_CE"] =  log_dict["metrics"]["energies"]
                log_dict["time"]["CE"] = 0.
        else:
            log_dict["time"]["CE"] = 0.


        if (self.config["problem_name"] in self.unbiased_list  and epoch % int(epochs/50) == 0 and self.n_sampling_rounds > 0):
            jax.config.update("jax_enable_x64", True)
            if key is None:
                key = jax.random.PRNGKey(0)

            if n_sampling_rounds is None:
                n_sampling_rounds = self.n_sampling_rounds
            if sampling_temp is not None:
                self.sampling_temp = sampling_temp

            self.sampling_mode = sampling_mode

            energies = []
            log_p = []
            log_q = []
            for i in tqdm(range(n_sampling_rounds)):
                subkey = jax.random.fold_in(key, i)
                batched_key = jax.random.split(subkey, num=len(jax.devices()))

                result_dict = self.pmap_sample_for_estimate(params, graph_batch, energy_graph_batch, self.T_target, self.sampling_temp, batched_key)

                energies.append(result_dict["energies"])
                log_p.append(result_dict["log_p"])
                log_q.append(result_dict["log_q"])

            log_p = jnp.concatenate(log_p, axis=-1)
            log_q = jnp.concatenate(log_q, axis=-1)
            energies = jnp.concatenate(energies, axis=-1)

            esimate_dict = self.unbiased_estimates(log_p, log_q, energies)

            X_0 = result_dict["X_0"]
            X_sequences = result_dict["X_sequences"]
            log_dict["energies"]["unbiased_free_energy"] = esimate_dict["free_energies"][-1]
            log_dict["energies"]["unbiased_internal_energy"] = esimate_dict["internal_energies"][-1]
            log_dict["energies"]["unbiased_entropy"] = esimate_dict["entropies"][-1]
            # log_dict["energies"]["energies_sampled"] = esimate_dict["free_energies"]
            # log_dict["energies"]["n_states"] = esimate_dict["n_states"]
            log_dict["energies"]["gt_unbiased_free_energy"] = self.free_energy
            log_dict["energies"]["gt_unbiased_internal_energy"] = self.internal_energy
            log_dict["energies"]["absolute_error_free_energy"] = np.abs(self.free_energy - esimate_dict["free_energies"][-1])
            log_dict["energies"]["absolute_error_internal_energy"] = np.abs(self.internal_energy - esimate_dict["internal_energies"][-1])
            ### TODO add new key matrices?
            log_dict["energies"]["log_p"] = log_p
            log_dict["energies"]["log_q"] = log_q
            N = self.size * self.size

            States = []
            for i in range(4):
                state = X_0[0, :N, i, 0]
                state = jnp.reshape(state, (self.size, self.size))
                States.append(state)

            Magnetization_dict = self._calculate_Magnetisations(X_0)

            for mag_key in Magnetization_dict:
                log_dict["energies"][mag_key] = Magnetization_dict[mag_key]

            log_dict["figures"]["Ising_states"] = {"X_0": States, "type": "Ising", "X_sequences": X_sequences}
            log_dict["figures"]["free_energies"] = {"type": "Ising", "y_axis": esimate_dict["free_energies"], "x_axis": esimate_dict["n_states"] , "baseline": self.free_energy}
            log_dict["figures"]["internal_energies"] = {"type": "Ising", "y_axis": esimate_dict["internal_energies"], "x_axis": esimate_dict["n_states"],  "baseline": self.internal_energy}
            log_dict["figures"]["entropies"] = {"type": "Ising", "y_axis": esimate_dict["entropies"], "x_axis": esimate_dict["n_states"],  "baseline": self.entropy}
            log_dict["figures"]["eff_sample_size"] = {"type": "Ising", "y_axis": esimate_dict["effective_sample_size"], "x_axis": esimate_dict["n_states"]}

            free_energy_err = np.abs(self.free_energy - np.array(esimate_dict["free_energies"]))
            internal_energy_err = np.abs(self.internal_energy - np.array(esimate_dict["internal_energies"]))
            log_dict["figures"]["free_energies_err"] = {"type": "Ising", "y_axis": free_energy_err, "x_axis": esimate_dict["n_states"]}
            log_dict["figures"]["internal_energies_err"] = {"type": "Ising", "y_axis": internal_energy_err, "x_axis": esimate_dict["n_states"]}

            self.n_sampling_rounds = n_sampling_rounds
            print("Start MCMC")
            MCMC_dict = self.sample_MCMC(params, graph_batch, energy_graph_batch, self.sampling_temp, key)
            log_dict["samples"] = {}
            log_dict["samples"]["unbiased_X_sequences"] = MCMC_dict["unbiased_X_sequences"]
            log_dict["samples"]["biased_X_sequences"] = MCMC_dict["biased_X_sequences"]
            unbiased_internal_energy_MCMC = self._calculate_Ising_energy(MCMC_dict["energies_list"][-1])
            print("MCMC interneal energy", unbiased_internal_energy_MCMC)
            log_dict["energies"]["unbiased_internal_energy_MCMC"] = unbiased_internal_energy_MCMC

            log_dict["figures"]["internal_energies_MCMC"] = {"type": "Ising", "y_axis": [self._calculate_Ising_energy(el) for el in MCMC_dict["energies_list"]], "x_axis": np.arange(0, self.n_sampling_rounds + 1),  "baseline": self.internal_energy}

            jax.config.update("jax_enable_x64", False)
        return loss, (log_dict, _)

    def _calculate_Ising_energy(self, Energies):
        return jnp.mean(Energies)/self.size**2

    def _calculate_Magnetisations(self, X_0):
        N = self.size * self.size
        sigma_0 = 2*X_0[0,:N,:,0] -1

        Magnetisation_per_state = jnp.sum(sigma_0, axis = 0)
        abs_mean_Magnetization = jnp.mean(jnp.abs(Magnetisation_per_state))
        mean_Magnetization = jnp.mean(Magnetisation_per_state)

        Magnetization_dict = {"abs_mean_Magnetization": abs_mean_Magnetization, "mean_Magnetization": mean_Magnetization}
        return Magnetization_dict

    def unbiased_estimates(self, log_p, log_q, energies):
        log_p_np = np.asarray(log_p, dtype=np.float64)[0]
        log_q_np = np.asarray(log_q, dtype=np.float64)[0]
        energies_np = np.asarray(energies, dtype=np.float64)[0]

        estimate_dict = {}

        estimate_dict["free_energies"] = []
        estimate_dict["free_energies_abs_err"] = []
        estimate_dict["effective_sample_size"] = []
        estimate_dict["internal_energies"] = []
        estimate_dict["internal_energies_abs_err"] = []
        estimate_dict["entropies"] = []
        ### TODO add entropy here

        # set range of states to evaluate
        num_points = 300
        n_states = np.array(log_p_np.shape[-1]* np.linspace(1/num_points,1, num_points, endpoint= True), np.int32)
        estimate_dict["n_states"] = n_states

        print("Number of Samples: ", log_p_np.shape[-1])
        for n in n_states:
            # Randomly sample n indices from the range of log_p_0_t_np.shape[2]
            random_indices = np.arange(0,n)#np.random.choice(log_p_0_t_np.shape[2], n, replace=False)
            log_p_ = log_p_np[..., random_indices]
            log_q_ = log_q_np[..., random_indices]

            energies_ = energies_np[..., random_indices]

            # stable testing:
            free_energy_ = self._calc_free_energy_from_im_weights(log_p_, log_q_)
            estimate_dict["free_energies"].append(free_energy_)
            estimate_dict["free_energies_abs_err"].append(np.abs(free_energy_ - self.free_energy))

            internal_energy_ = self._calc_internal_energy_from_im_weights(log_p_, log_q_, energies_)
            estimate_dict["internal_energies"].append(internal_energy_)
            estimate_dict["internal_energies_abs_err"].append(np.abs(internal_energy_ - self.internal_energy))

            estimate_dict["entropies"].append(self.calculate_ising_entropy(self.beta, internal_energy_, free_energy_))

            importance_weights = scipy.special.softmax(log_p_- log_q_, axis = -1)
            effective_sample_size = jnp.sum(importance_weights)**2/ jnp.sum(importance_weights**2)
            estimate_dict["effective_sample_size"].append(effective_sample_size)

        print(f"free_energy_ is {free_energy_}", self.free_energy)
        print(f"internal_energy_ is {internal_energy_}", self.internal_energy)
        print(f"entropy is {self.calculate_ising_entropy(self.beta, internal_energy_, free_energy_)}", self.entropy )
        return estimate_dict

    def _calc_free_energy_from_im_weights(self,log_p_, log_q_):
        np_log_Z = sp.special.logsumexp(log_p_ - log_q_) - jnp.log(log_p_.shape[-1])
        free_energy_ = - 1 / self.beta * np_log_Z / self.size ** 2

        return free_energy_

    def _calc_internal_energy_from_im_weights(self,log_p_, log_q_, energies):
        # print(energies.shape, scipy.special.softmax(log_p_- log_q_, axis = -1).shape)
        # print(scipy.special.softmax(log_p_- log_q_, axis = -1), energies)
        internal_energy = jnp.sum(scipy.special.softmax(log_p_- log_q_, axis = -1)*energies)
        return internal_energy/ self.size ** 2

    @partial(jax.jit, static_argnums=(0,))
    def sample_for_estimate(self, params, graphs, energy_graph_batch, T, key):
        ### TODO sample X_0_tilde with prob eps and simulate forward diffusion process X_1_T_tilde
        ### TODO generate reverse diffusion process X_0:T_hat while keeping X_1_T_tilde constant

        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, log_q_T, one_hot_state, log_p_uniform, key = self.model.sample_prior_w_probs(energy_graph_batch,
                                                                                             self.N_test_basis_states,
                                                                                             key)
        n_graphs = energy_graph_batch.n_node.shape[0]

        Xs_over_different_steps = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], 1))
        log_q_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))
        log_p_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))

        log_q_0_T = log_q_0_T.at[0].set(log_q_T)
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)

        node_gr_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)
        node_graph_idx, n_graph, n_node = self.get_graph_info(graphs)

        for i in range(overall_diffusion_steps):
            model_step_idx = jnp.array([i / self.eval_step_factor], dtype=jnp.int16)
            model_step_idx_per_node = model_step_idx[0] * jnp.ones((energy_graph_batch.nodes.shape[0], 1),
                                                                   dtype=jnp.int16)
            key, subkey = jax.random.split(key)
            batched_key = jax.random.split(subkey, num=self.N_test_basis_states)

            out_dict, _ = self.vmapped_make_one_step(params, graphs, X_prev, model_step_idx_per_node, batched_key)

            spin_logits = out_dict["spin_logits"]
            spin_logits = jnp.array(spin_logits, dtype=jnp.float64)

            sampling_temp_at_t = (self.sampling_temp) * ((i + 1)/ overall_diffusion_steps)**2
            output, _ = self.vmapped_sample_with_temp(spin_logits, sampling_temp_at_t,
                                                      batched_key)

            X_next = output["X_next"]
            spin_log_probs = output["spin_log_probs"]
            state_log_probs = output["state_log_probs"]

            log_q_t = state_log_probs

            X_prev = jnp.array(X_prev, dtype=jnp.float64)
            X_next = jnp.array(X_next, dtype=jnp.float64)
            log_p_t = self.NoiseDistrClass.get_log_p_T_0(energy_graph_batch, X_prev, X_next, model_step_idx, T)
            X_prev = X_next
            log_q_0_T = log_q_0_T.at[i + 1].set(log_q_t)
            log_p_0_T = log_p_0_T.at[i].set(log_p_t)

        X_0 = X_next
        X_0 = jnp.array(X_0, dtype=jnp.float64)
        energies, _, _ = self.vmapped_relaxed_energy(energy_graph_batch, X_0, node_gr_idx)
        log_p_0 = self.EnergyClass.get_log_p_0_from_energy(energies, T)
        log_p_0_T = log_p_0_T.at[i + 1].set(log_p_0)

        log_p_0_T = log_p_0_T[:, :-1, :]
        log_q_0_T = log_q_0_T[:, :-1, :]

        result_dict = {
            "X_0": X_0,
            "X_sequences": Xs_over_different_steps,
            "energies": energies[:-1,:,0],
            "log_p": jnp.sum(log_p_0_T, axis = 0),
            "log_q": jnp.sum(log_q_0_T, axis = 0),
        }
        return result_dict

    @partial(jax.jit, static_argnums=(0,-1))
    def _environment_steps_scan_estimate(self, params, graphs, energy_graph_batch, T, eps, key):
        ### TDOD cahnge rewards to non exact expectation rewards
        print("scan function is being jitted")
        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        X_prev, log_q_T, one_hot_state, log_p_uniform, key = self.model.sample_prior_w_probs(energy_graph_batch,
                                                                                             self.N_test_basis_states,
                                                                                             key)
        n_graphs = energy_graph_batch.n_node.shape[0]

        Xs_over_different_steps = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], 1))
        log_q_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))
        log_p_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))

        log_q_0_T = log_q_0_T.at[0].set(log_q_T)
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)



        node_gr_idx, _, total_num_nodes = self._compute_aggr_utils(energy_graph_batch)
        scan_dict = {"log_q_0_T": log_q_0_T, "log_p_0_T": log_p_0_T, "Xs_over_different_steps": Xs_over_different_steps,
                     "step": 0, "node_gr_idx": node_gr_idx, "params": params, "key": key, "X_prev": X_prev,
                     "graphs": graphs, "energy_graph_batch": energy_graph_batch, "T": T, "eps": eps}

        scan_dict, out_dict_list = jax.lax.scan(self.scan_body_for_estimate, scan_dict, None, length=overall_diffusion_steps)

        key = scan_dict["key"]

        log_p_0_T = scan_dict["log_p_0_T"]
        log_q_0_T = scan_dict["log_q_0_T"]
        X_next = scan_dict["X_prev"]  #

        X_0 = X_next
        Xs_over_different_steps = scan_dict["Xs_over_different_steps"]

        energies, _, _ = self.vmapped_relaxed_energy(energy_graph_batch, X_0, node_gr_idx)
        log_p_0 = self.EnergyClass.get_log_p_0_from_energy(energies, T)
        log_p_0_T = log_p_0_T.at[-1].set(log_p_0)

        log_p_0_T = log_p_0_T[:, :-1, :]
        log_q_0_T = log_q_0_T[:, :-1, :]

        result_dict = {
            "X_0": X_0,
            "X_sequences": Xs_over_different_steps,
            "energies": energies[:-1,:,0],
            "log_p": jnp.sum(log_p_0_T, axis = 0),
            "log_q": jnp.sum(log_q_0_T, axis = 0),
        }
        return result_dict

    @partial(jax.jit, static_argnums=(0,))
    def scan_body_for_estimate(self, scan_dict, y):
        i = scan_dict["step"]
        T = scan_dict["T"]
        params = scan_dict["params"]
        X_prev = scan_dict["X_prev"]
        graphs = scan_dict["graphs"]
        energy_graph_batch = scan_dict["energy_graph_batch"]
        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor

        model_step_idx = jnp.array([i / self.eval_step_factor], dtype=jnp.int16)
        model_step_idx_per_node = model_step_idx[0] * jnp.ones((energy_graph_batch.nodes.shape[0], 1), dtype=jnp.int16)

        key = scan_dict["key"]
        subkey = jax.random.fold_in(key, i)
        batched_key = jax.random.split(subkey, num=self.N_test_basis_states)

        out_dict, _ = self.vmapped_make_one_step(params, graphs, X_prev, model_step_idx_per_node,
                                                 batched_key)

        spin_logits = out_dict["spin_logits"]
        spin_logits = jnp.array(spin_logits, dtype=jnp.float64)

        subkey = jax.random.fold_in(key, i)
        batched_key = jax.random.split(subkey, num=self.N_test_basis_states)
        sampling_temp_at_t = (scan_dict["eps"]) * ((i + 1) / overall_diffusion_steps) ** 2
        output, _ = self.vmapped_sample_with_temp(graphs, spin_logits, sampling_temp_at_t, batched_key)

        #output = out_dict

        X_next = output["X_next"]
        spin_logits_next = output["spin_logits"]
        state_log_probs = output["state_log_probs"]

        # print('eps: ', sampling_temp_at_t, spin_logits.shape, jnp.exp(spin_logits)[0].shape)
        # print("before probs", jnp.exp(spin_logits)[0, 0])
        # print("probs", jnp.exp(spin_logits_next)[0, 0])

        log_q_t = state_log_probs
        log_p_t = self.NoiseDistrClass.get_log_p_T_0(energy_graph_batch, X_prev, X_next, model_step_idx, T)

        scan_dict["key"] = key
        scan_dict["log_q_0_T"] = scan_dict["log_q_0_T"].at[i + 1].set(log_q_t)
        scan_dict["log_p_0_T"] = scan_dict["log_p_0_T"].at[i].set(log_p_t)

        X_prev = X_next
        scan_dict["Xs_over_different_steps"] = scan_dict["Xs_over_different_steps"].at[i + 1].set(X_next)

        scan_dict["X_prev"] = X_prev
        scan_dict["step"] += 1

        out_dict = {}
        out_dict["spin_log_probs"] = state_log_probs
        out_dict["spin_logits_next"] = spin_logits_next
        return scan_dict, out_dict


    @partial(jax.jit, static_argnums=(0,))
    def sample_for_estimate_v2(self, params, graphs, energy_graph_batch, T, eps, key):
        overall_diffusion_steps = self.n_diffusion_steps * self.eval_step_factor
        node_gr_idx, n_graph, n_node = self.get_graph_info(graphs)

        X_prev, log_q_T, one_hot_state, log_p_uniform, key = self.model.sample_prior_w_probs(energy_graph_batch,
                                                                                             self.N_test_basis_states,
                                                                                             key)
        ### TODO sample X_0_tilde with prob eps and simulate forward diffusion process X_1_T_tilde
        ### TODO generate reverse diffusion process X_0:T_hat while keeping X_1_T_tilde constant
        shape = (energy_graph_batch.nodes.shape[0], self.N_test_basis_states,1)
        X_tilde_dict, key = self.sample_X_0_tilde( eps, shape,node_gr_idx, n_graph, key)
        ### TODO calculate p(X_0_tilde)
        mask = X_tilde_dict["mask"]
        X_0_tilde = X_tilde_dict["X_0_tilde"]

        X_curr_tilde = X_0_tilde

        X_sequence = jnp.zeros((overall_diffusion_steps + 1, energy_graph_batch.nodes.shape[0], self.N_test_basis_states,1))

        X_sequence = X_sequence.at[-1].set(X_curr_tilde)
        spin_log_probs_per_node_sequence = jnp.zeros((overall_diffusion_steps, energy_graph_batch.nodes.shape[0], self.N_test_basis_states,1))

        for i in range(overall_diffusion_steps):
            subkey = jax.random.fold_in(key, i)
            batched_key = jax.random.split(subkey, num=self.N_test_basis_states)
            X_next_tilde, spin_log_probs_per_node, _ = self.vmapped_sample_forward_diff_process(X_curr_tilde, overall_diffusion_steps - i - 1, batched_key)
            X_next_tilde = jnp.where(mask == 1, X_next_tilde, X_curr_tilde)

            X_curr_tilde = X_next_tilde
            X_sequence = X_sequence.at[- i - 2].set(X_next_tilde)
            spin_log_probs_per_node_sequence = spin_log_probs_per_node_sequence.at[overall_diffusion_steps - i - 1].set(spin_log_probs_per_node)

        n_graphs = energy_graph_batch.n_node.shape[0]

        Xs_over_different_steps = jnp.zeros(
            (overall_diffusion_steps + 1, X_prev.shape[0], X_prev.shape[1], 1))
        log_q_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))
        log_p_0_T = jnp.zeros((overall_diffusion_steps + 1, n_graphs, X_prev.shape[1]))

        log_q_T_per_node = jnp.log(0.5)*jnp.ones((energy_graph_batch.nodes.shape[0], self.N_test_basis_states))
        masked_log_q_T_per_graph = jax.ops.segment_sum(log_q_T_per_node, node_gr_idx, n_graph)

        log_q_0_T = log_q_0_T.at[0].set(masked_log_q_T_per_graph)
        X_prev = (1-mask)*X_prev + mask * X_sequence[0]
        Xs_over_different_steps = Xs_over_different_steps.at[0].set(X_prev)

        for i in range(overall_diffusion_steps):
            model_step_idx = jnp.array([i / self.eval_step_factor], dtype=jnp.int16)
            model_step_idx_per_node = model_step_idx[0] * jnp.ones((energy_graph_batch.nodes.shape[0], 1),
                                                                   dtype=jnp.int16)
            key, subkey = jax.random.split(key)
            batched_key = jax.random.split(subkey, num=self.N_test_basis_states)

            out_dict, _ = self.vmapped_make_one_step(params, graphs, X_prev, model_step_idx_per_node, batched_key)

            X_next = out_dict["X_next"]
            X_next = (1 - mask) * X_next + mask * X_sequence[i + 1]

            spin_logits = out_dict["spin_logits"]

            one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
            # X_next = jnp.expand_dims(X_next, axis = -1)
            spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
            # print(X_next.shape, X_next, jnp.exp(spin_log_probs))
            X_next_log_prob = jax.ops.segment_sum(spin_log_probs[..., 0], node_gr_idx, n_graph)

            X_prev = jnp.array(X_prev, dtype=jnp.float64)
            X_next = jnp.array(X_next, dtype=jnp.float64)

            log_p_t_per_node = self.NoiseDistrClass.get_log_p_T_0_per_node( X_prev, X_next, model_step_idx)
            masked_log_p_t_per_graph = jax.ops.segment_sum(log_p_t_per_node, node_gr_idx, n_graph)

            X_prev = X_next
            Xs_over_different_steps = Xs_over_different_steps.at[i + 1].set(X_prev)
            log_q_0_T = log_q_0_T.at[i + 1].set(X_next_log_prob)
            log_p_0_T = log_p_0_T.at[i].set(masked_log_p_t_per_graph)

        X_0 = X_next
        X_0 = jnp.array(X_0, dtype=jnp.float64)
        energies, _, _ = self.vmapped_relaxed_energy(energy_graph_batch, X_0, node_gr_idx)
        log_p_0 = self.EnergyClass.get_log_p_0_from_energy(energies, T)
        log_p_0_T = log_p_0_T.at[i + 1].set(log_p_0)

        log_p_0_T = log_p_0_T[:, :-1, :]
        log_q_0_T = log_q_0_T[:, :-1, :]

        log_p_1_T = jnp.sum(log_p_0_T[:-1], axis = 0)

        log_prob_q = jnp.array([jnp.sum(log_q_0_T, axis = 0) + jnp.log(1-eps), jnp.log(eps) + log_p_1_T + jnp.log(0.5)*jnp.ones_like(log_p_1_T)])
        log_tilde_q = jax.scipy.special.logsumexp(log_prob_q, axis = 0)
        # print("log_tilde_q", log_tilde_q.shape)
        # print("compare", eps, jnp.mean(log_tilde_q), jnp.mean(jnp.sum(log_q_0_T, axis = 0)))
        #print("log_tilde_q", jnp.mean(log_tilde_q), jnp.min(log_tilde_q), jnp.max(log_tilde_q))
        #log_tilde_q= jnp.log((1-eps)*prob_q + eps*prob_p_rand)

        result_dict = {
            "X_0": X_0,
            "X_sequences": Xs_over_different_steps,
            "energies": energies[:-1,:,0],
            "log_p": jnp.sum(log_p_0_T, axis = 0),
            "epsilon": T,
            "log_q": log_tilde_q,
        }
        return result_dict

    def sample_MCMC(self, params, graphs, energy_graph_batch, eps, key):
        n_MCMC_steps = self.n_sampling_rounds
        log_p_list = []
        log_q_list = []
        energies_list = []

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))

        result_dict = self.pmap_sample_for_estimate(params, graphs, energy_graph_batch, self.T_target,eps,
                                                        batched_key)
        log_p = result_dict["log_p"]
        log_q = result_dict["log_q"]
        energies = result_dict["energies"]
        X_sequences_init = result_dict["X_sequences"]
        X_sequences = X_sequences_init

        log_p_list.append(log_p)
        log_q_list.append(log_q)
        energies_list.append(energies)

        for i in tqdm(range(n_MCMC_steps)):
            subkey = jax.random.fold_in(key, i)
            batched_key = jax.random.split(subkey, num=len(jax.devices()))
            # if (self.sampling_mode == "eps"):
            #     result_dict = self.pmap_sample_for_estimate_v2(params, graphs, energy_graph_batch, self.T_target,eps,
            #                                                    batched_key)
            # else:
            result_dict = self.pmap_sample_for_estimate(params, graphs, energy_graph_batch, self.T_target, eps,
                                                            batched_key)
            log_p_prime = result_dict["log_p"]
            log_q_prime = result_dict["log_q"]
            energies_prime = result_dict["energies"]
            X_sequences_prime = result_dict["X_sequences"]

            _log_p = log_p
            _log_p_prime = log_p_prime
            _log_q = log_q
            _log_q_prime = log_q_prime

            ratio = _log_p_prime - _log_p + _log_q - _log_q_prime

            acceptance_prob = jnp.minimum(1, jnp.exp(ratio))

            key, subkey = jax.random.split(key)
            uniform_random = jax.random.uniform(subkey, shape=acceptance_prob.shape)
            accept = uniform_random <= acceptance_prob
            log_p = jnp.where(accept, log_p_prime, log_p)
            log_q = jnp.where(accept, log_q_prime, log_q)

            energies = jnp.where(accept, energies_prime, energies)

            X_accept = uniform_random[:, :, None, : , None] <= acceptance_prob[:, :, None, : , None]
            X_sequences = jnp.where(X_accept, X_sequences_prime, X_sequences)

            log_p_list.append(log_p)
            log_q_list.append(log_q)
            energies_list.append(energies)

        # log_p_list = jnp.concatenate(log_p_list, axis=-1)
        # log_q_list = jnp.concatenate(log_q_list, axis=-1)
        # energies_list = jnp.concatenate(energies_list, axis=-1)

        result = {
            "log_p_list": log_p_list,
            "log_q_list": log_q_list,
            "energies_list": energies_list,
            "X_0": result_dict["X_0"],
            "unbiased_X_sequences": X_sequences,
            "biased_X_sequences": X_sequences_init
        }
        return result

    def sample_X_0_tilde(self, eps, shape, node_gr_idx, n_graph, key):
        key, subkey = jax.random.split(key)
        sampled_ps = jax.random.uniform(subkey, shape)

        key, subkey = jax.random.split(key)
        X_0_proposal = jax.random.randint(subkey, shape, minval=0, maxval=2)
        X_0_hat = jnp.where(sampled_ps <= eps, X_0_proposal, -1 * jnp.ones_like(X_0_proposal))

        mask = jnp.where(X_0_hat != -1, 1, 0)
        #log_p_X_0_tilde_per_node = jnp.where(mask == 0, jnp.log(1 - eps), jnp.log(0.5 * eps))
        #log_p_X_0_tilde_per_graph = jax.ops.segment_sum(log_p_X_0_tilde_per_node, node_gr_idx, n_graph)[...,0]

        X_0_tilde_dict = {}
        X_0_tilde_dict["mask"] = mask
        X_0_tilde_dict["X_0_tilde"] = X_0_hat
        #X_0_tilde_dict["log_p_X_0_tilde_per_graph"] = log_p_X_0_tilde_per_graph
        return X_0_tilde_dict, key

    @partial(jax.jit, static_argnums=0)
    def sample_with_temp(self, graph, spin_logits, sample_eps, key):
        # key, subkey = jax.random.split(key)
        print('eps: ', sample_eps)
        ones_mat = jnp.ones_like(spin_logits)
        new_logits = jnp.array([jnp.log(1-sample_eps)*ones_mat + jax.nn.log_softmax(spin_logits ), jnp.log(sample_eps)*ones_mat +  jnp.log(0.5)*ones_mat])
        spin_logits_ = jax.scipy.special.logsumexp(new_logits, axis = 0)

        X_next = jax.random.categorical(key=key,
                                        logits=spin_logits_,
                                        axis=-1,
                                        shape=spin_logits_.shape[:-1])

        one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
        spin_log_probs = jnp.sum(spin_logits_ * one_hot_state, axis=-1)

        node_graph_idx, n_graph, n_node = self.get_graph_info(graph)

        state_log_probs = jraph.segment_sum(spin_log_probs[..., 0], node_graph_idx, n_graph)


        print(f"spin_log_probs.dtype: {spin_log_probs.dtype}")
        output = {
            "X_next": X_next,
            "spin_log_probs": spin_log_probs,
            "state_log_probs": state_log_probs,
            "spin_logits": spin_logits_
        }
        return output, key

    def get_graph_info(self, jraph_graph_list):
        first_graph = jraph_graph_list["graphs"][0]
        nodes = first_graph.nodes
        n_node = first_graph.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
        return node_graph_idx, n_graph, n_node

    @partial(jax.jit, static_argnums=(0,))
    def apply_CE_on_p(self, energy_graph_batch, p_0):
        X_0_CE, energies_CE, Hb_per_node = self.vmapped_calculate_Energy_CE_p_values(energy_graph_batch, p_0)

        return X_0_CE, energies_CE, Hb_per_node

    # @abstractmethod
    # def sample(self):
    #     pass

    @partial(jax.jit, static_argnums=(0,))
    def __update_params(self, params, grads, opt_state):
        grad_update, opt_state = self.opt_update(grads, opt_state, params)

        params = optax.apply_updates(params, grad_update)
        return params, opt_state

    def pmap_loss_backward_step(self, params, opt_state, graphs, energy_graph_batch, T, key):
        (loss, (log_dict, key)), params, opt_state = self.pmap_loss_backward(params, opt_state, graphs,
                                                                             energy_graph_batch, T,
                                                                             key)

        return (loss, (log_dict, key)), params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def loss_backward(self, params, opt_state, graphs, energy_graph_batch, T, key):
        (loss, (log_dict, key)), grad = self.loss_grad(params, graphs, energy_graph_batch, T, key)

        grad = jax.lax.pmean(grad, axis_name='device')
        params, opt_state = self.__update_params(params, grad, opt_state)
        return (loss, (log_dict, key)), params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_aggr_utils(self, jraph_graph):
        nodes = jraph_graph.nodes
        n_node = jraph_graph.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
        return node_graph_idx, n_graph, total_num_nodes


    def calculate_ising_entropy(self, beta, internal_energy, free_energy):
        """
        Calculate the entropy of the NxN lattice
        """
        return (internal_energy - free_energy) / (1 / beta)

    def calculate_ising_internal_energy(self, beta, L):
        """
        Calculate the internal energy per spin of the NxN lattice

        https://journals.aps.org/pr/pdf/10.1103/PhysRev.185.832
        """
        Z, (Z_1, Z_2, Z_3, Z_4) = self.calculate_ising_partition_function(beta, L)

        Z_1_prime = 0
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r + 1, beta, L)
            gamma_prime = self.calculate_gamma_prime(2 * r + 1, beta, L)
            Z_1_prime += gamma_prime * np.tanh(1 / 2 * L * gamma, dtype=np.float64)
        Z_1_prime = 1 / 2 * L * Z_1_prime * Z_1

        Z_2_prime = 0
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r + 1, beta, L)
            gamma_prime = self.calculate_gamma_prime(2 * r + 1, beta, L)
            Z_2_prime += gamma_prime * self.coth(1 / 2 * L * gamma)
        Z_2_prime = 1 / 2 * L * Z_2_prime * Z_2

        Z_3_prime = 0
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r, beta, L)
            gamma_prime = self.calculate_gamma_prime(2 * r, beta, L)
            Z_3_prime += gamma_prime * np.tanh(1 / 2 * L * gamma, dtype=np.float64)
        Z_3_prime = 1 / 2 * L * Z_3_prime * Z_3

        Z_4_prime = 0
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r, beta, L)
            gamma_prime = self.calculate_gamma_prime(2 * r, beta, L)
            Z_4_prime += gamma_prime * self.coth(1 / 2 * L * gamma)
        Z_4_prime = 1 / 2 * L * Z_4_prime * Z_4

        return -self.coth(2 * beta) - 1 / L ** 2 * (
                np.sum([Z_1_prime, Z_2_prime, Z_3_prime, Z_4_prime]) / np.sum([Z_1, Z_2, Z_3, Z_4]))

    def calculate_ising_free_energy(self, beta, L):
        """
        Calculate the free energy per spin of the NxN lattice
        """
        Z, _ = self.calculate_ising_partition_function(beta, L)
        return -1 / beta * np.log(Z) / self.size ** 2

    def calculate_ising_partition_function(self, beta, L):
        """
        Calculate the partition function of the NxN lattice

        https://journals.aps.org/pr/pdf/10.1103/PhysRev.185.832
        """
        Z_1 = 1
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r + 1, beta, L)
            Z_1 *= 2 * np.cosh(1 / 2 * L * gamma, dtype=np.float64)

        Z_2 = 1
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r + 1, beta, L)
            Z_2 *= 2 * np.sinh(1 / 2 * L * gamma, dtype=np.float64)

        Z_3 = 1
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r, beta, L)
            Z_3 *= 2 * np.cosh(1 / 2 * L * gamma, dtype=np.float64)

        Z_4 = 1
        for r in range(0, L):
            gamma = self.calculate_gamma(2 * r, beta, L)
            Z_4 *= 2 * np.sinh(1 / 2 * L * gamma, dtype=np.float64)

        return 1 / 2 * (2 * np.sinh(2 * beta, dtype=np.float64)) ** (L ** 2 / 2) * (Z_1 + Z_2 + Z_3 + Z_4), (Z_1, Z_2, Z_3, Z_4)

    def calculate_gamma(self, r, beta, L):
        if r == 0:
            gamma = 2 * beta + np.log(np.tanh(beta, dtype=np.float64))
        else:
            c_r = np.cosh(2 * beta, dtype=np.float64) * self.coth(2 * beta) - np.cos(np.pi * r / L)
            gamma = np.log(c_r + np.sqrt(c_r ** 2 - 1))
        return gamma

    def calculate_gamma_prime(self, r, beta, L):

        if r == 0:
            gamma_prime = self.csch(beta) / np.cosh(beta) + 2
        else:
            cl = np.cosh(2 * beta, dtype=np.float64) * self.coth(2 * beta) - np.cos(np.pi * r / L)

            nenner = 2 * (np.cosh(2 * beta) - self.coth(2 * beta) * self.csch(2 * beta))
            zaehler = np.sqrt((cl) ** 2 - 1)
            gamma_prime = nenner / zaehler
        return gamma_prime

    def coth(self, x):
        """
        Coth function
        """
        return np.cosh(x, dtype=np.float64) / np.sinh(x, dtype=np.float64)

    def csch(self, x):
        """
        Csch function
        """
        return 1 / np.sinh(x, dtype=np.float64)

@partial(jax.jit, static_argnums=())
def repeat_along_nodes(nodes, n_node, target_per_graph):
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    target_per_node = jnp.repeat(target_per_graph, n_node, axis=0,
                                          total_repeat_length=total_nodes)

    return target_per_node

