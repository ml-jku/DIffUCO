import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
from tqdm import tqdm

class MCMCSamplerClass():
    def __init__(self, model, sample_func, EnergyClass, NoiseClass):
        ### TODO implement asymptotically unbiased samples here
        ### TODO there is a numerical problem when T = 0 in energy log_p, maybe always log log_probs as t* log p and then in the division it does not play a role anymore
        self.model = model
        self.sample = sample_func
        self.EnergyClass = EnergyClass
        self.NoiseClass = NoiseClass

        self._vmap_and_pmap_functions()
        self._reset_MCMC_Energy_list()

    def _reset_MCMC_Energy_list(self):
        self.MCMC_Energ_list = []

    def _vmap_and_pmap_functions(self):
        self.vmapped_log_q = jax.vmap(self.model.calc_log_q, in_axes=(None, None, -2, -2, -1, 0))
        self.vmapped_vmapped_log_q = jax.vmap(self.vmapped_log_q, in_axes=(None, None, 0, 0, 0, 0))
        self.pmap_make_MCMC_dict = jax.pmap(self.make_MCMC_dict, in_axes=(0, 0, 0, 0, None,0))
        self.pmap_update_sample = jax.pmap(self.update_sample, in_axes=(0, 0, 0, 0))
        self.pmap_sample_MCMC_dict = jax.pmap(self.sample_MCMC_dict, in_axes=(0, 0, 0, None, 0))
        self.vmapped_get_log_p_T_0 = jax.vmap(self.NoiseClass.get_log_p_T_0 , in_axes=(None, 0, 0 ,0, None))
        self.vmapped_get_log_p_0 = jax.vmap(self.EnergyClass.get_log_p_0, in_axes = (None, -2, None))

    def evaluate_log_q(self, params, GNN_graphs, batched_jraphs, bin_sequence, key):
        X_T = bin_sequence[0]
        X_prev = bin_sequence[0:-1]
        X_next = bin_sequence[1:]
        ### TODO vmap key? TODO test shapes
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=(X_prev.shape[0], X_prev.shape[-2]))

        t_idx = jnp.arange(0, X_prev.shape[0])[:, None, None] + jnp.zeros(X_prev.shape[:-1])
        log_q_T_0 = self.vmapped_vmapped_log_q(params, GNN_graphs, X_prev, X_next, t_idx, batched_key)
        log_q_T_0 = jnp.swapaxes(log_q_T_0, -2, -1)

        log_q_T = self.model.calc_log_q_T(batched_jraphs, X_T)

        # print("log q t", jnp.mean(jnp.exp(log_q_T)))
        # print("log q t 0 ", jnp.mean(jnp.exp(log_q_T_0)))

        log_q_over_steps = jnp.concatenate([log_q_T[None, ...], log_q_T_0], axis = 0)
        return log_q_over_steps, key

    def evaluate_log_p(self, batched_jraphs, bin_sequence, T):
        X_0 = bin_sequence[-1]
        X_prev = bin_sequence[0:-1]
        X_next = bin_sequence[1:]
        log_p_0 = self.vmapped_get_log_p_0(batched_jraphs, X_0, T)
        log_p_0 = jnp.swapaxes(log_p_0, -1, -2)

        t_idx = jnp.arange(0, X_prev.shape[0])
        log_p_T_0 = self.vmapped_get_log_p_T_0(batched_jraphs, X_prev, X_next, t_idx, T)

        # print("log p t", jnp.mean(jnp.exp(log_p_T_0)))
        # print("log p t 0 ", jnp.mean(jnp.exp(log_p_0)), jnp.mean(log_p_0))

        log_p_over_steps = jnp.concatenate([ log_p_T_0, log_p_0[None, ...]], axis = 0)
        return log_p_over_steps

    ### TODO pmap this
    def make_MCMC_dict(self, params, GNN_graphs, batched_jraphs, bin_sequence, T, key):
        log_q, key = self.evaluate_log_q(params, GNN_graphs, batched_jraphs, bin_sequence, key)
        log_p = self.evaluate_log_p(batched_jraphs, bin_sequence, T)

        MCMC_dict = {"bin_sequence": bin_sequence, "log_q": log_q, "log_p": log_p}
        return MCMC_dict, key

    ### TODO pmap this
    def sample_MCMC_dict(self, params, GNN_graphs, energy_graph_batch, T, key):
        ### TODO make sure the same key is used for rand node features in sampling and in evaluation
        _, (log_dict, key) = self.sample( params, GNN_graphs, energy_graph_batch, T, key)
        bin_sequence = log_dict["bin_sequence"]
        log_q, key = self.evaluate_log_q(params, GNN_graphs, energy_graph_batch, bin_sequence, key)
        log_p = self.evaluate_log_p(energy_graph_batch, bin_sequence, T)

        MCMC_dict = {"bin_sequence": bin_sequence, "log_q": log_q, "log_p": log_p}
        return MCMC_dict, key

    def update_buffer(self, params, GNN_graphs, energy_graph_batch, bin_sequence, key, T, n_steps = 2):
        '''

        :param bin_sequence: shape = (n_diff_steps, batched_jraph_nodes, basis_states, 1)
        :return:
        '''
        #### TODO make this compatible with pmap
        best_MCMC_dict, key = self.pmap_make_MCMC_dict(params, GNN_graphs, energy_graph_batch, bin_sequence, T, key)

        acceptance_list = []
        for i in tqdm(np.arange(0,n_steps)):
            # print("prev", "log p", "log q")
            # print(jnp.mean(jnp.exp(best_MCMC_dict["log_p"])), jnp.mean(jnp.exp(best_MCMC_dict["log_q"])))
            new_MCMC_dict, key = self.pmap_sample_MCMC_dict(params, GNN_graphs, energy_graph_batch,T, key)
            # print("new", "log p", "log q")
            # print(jnp.mean(jnp.exp(new_MCMC_dict["log_p"])), jnp.mean(jnp.exp(new_MCMC_dict["log_q"])))
            # print(jnp.max(jnp.exp(new_MCMC_dict["log_p"])), jnp.max(jnp.exp(new_MCMC_dict["log_q"])))
            # print(jnp.min(jnp.exp(new_MCMC_dict["log_p"])), jnp.min(jnp.exp(new_MCMC_dict["log_q"])))
            best_MCMC_dict, key, A_mat = self.pmap_update_sample(energy_graph_batch, best_MCMC_dict, new_MCMC_dict, key)
            acceptance_list.append(jnp.mean(A_mat))

        MCMC_Energy = jnp.mean(-T*best_MCMC_dict["log_p"][:,-1,0:-1])
        self.MCMC_Energ_list.append(MCMC_Energy)
        print("acceptance_list ", acceptance_list)
        print("mean Energy", np.mean(self.MCMC_Energ_list))
        return best_MCMC_dict, MCMC_Energy, key

    @partial(jax.jit, static_argnames=["self",])
    def update_sample(self, j_graphs, best_MCMC_dict, new_MCMC_dict, key):
        ### TODO pmap this
        X_best = best_MCMC_dict["bin_sequence"]
        X_new = new_MCMC_dict["bin_sequence"]
        log_forward_best = best_MCMC_dict["log_p"]
        log_backward_best = best_MCMC_dict["log_q"]
        log_forward_new = new_MCMC_dict["log_p"]
        log_backward_new = new_MCMC_dict["log_q"]

        proposal_log_probs = log_forward_new - log_backward_new  + log_backward_best - log_forward_best
        proposal_log_probs = proposal_log_probs[...,None]
        print("proposal", jnp.mean(proposal_log_probs))
        #proposal_log_probs = self.__aggr_log_probs(j_graphs, all_logs)

        proposol_mat = jnp.concatenate([jnp.zeros_like(proposal_log_probs), proposal_log_probs], axis = -1)
        A = jnp.min(proposol_mat, axis = -1, keepdims=True)

        key, subkey = jax.random.split(key)
        drawn_p = jax.random.uniform(subkey, shape =  proposal_log_probs.shape)

        graph_A, graph_drawn_p = self.__repeat_along_graphs(j_graphs, A, drawn_p, axis = 1)

        X_sequence_old = jnp.where((jnp.log(graph_drawn_p) <= graph_A), X_new, X_best  )
        log_forward_best = jnp.where((jnp.log(drawn_p) <= A)[...,0], log_forward_new, log_forward_best  )
        log_backward_best = jnp.where((jnp.log(drawn_p) <= A)[...,0], log_backward_new, log_backward_best  )

        ### TODO check correctness of shapes

        updated_MCMC_dict = {"bin_sequence": X_sequence_old,  "log_q": log_backward_best, "log_p": log_forward_best}

        return updated_MCMC_dict, key, 1*(jnp.log(drawn_p) <= A)

    def __aggr_log_probs(self, j_graph, spin_log_probs, axis = 0):
        spin_log_probs = jnp.swapaxes(spin_log_probs, 0, 1)

        nodes = j_graph.nodes
        n_node = j_graph.n_node
        n_graph = j_graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[axis]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=axis, total_repeat_length=total_nodes)

        aggr_log_probs = jax.ops.segment_sum(spin_log_probs, node_graph_idx, n_graph)
        aggr_log_probs = jnp.swapaxes(aggr_log_probs, 0, 1)
        return aggr_log_probs

    def __repeat_along_graphs(self, j_graphs, A, drawn_p, axis = 0):
        nodes = j_graphs.nodes
        n_node = j_graphs.n_node
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[axis]
        graph_drawn_p = jnp.repeat(drawn_p, n_node, axis=axis, total_repeat_length=total_nodes)
        graph_A = jnp.repeat(A, n_node, axis=axis, total_repeat_length=total_nodes)
        return graph_A, graph_drawn_p



