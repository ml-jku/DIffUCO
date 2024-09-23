from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax
from functools import partial
import time

class BaseEnergyClass(ABC):

    def __init__(self, config):
        self.n_bernoulli_features = config["n_bernoulli_features"]
        self.vmap_calculate_Energy_per_node = jax.vmap(self.calculate_Energy_per_node, in_axes=(None, 0, None))
        pass

    @abstractmethod
    def calculate_Energy(self):
        raise ValueError("not implemented")

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_per_node(self, H_graph, bins, node_graph_idx):
        Energy, _, HB_per_graph = self.calculate_Energy(H_graph, bins, node_graph_idx)
        ### TODO move this inside calculate energy?
        #Energy_per_node = jnp.repeat(Energy, n_node, axis=0, total_repeat_length=total_nodes)
        return Energy, HB_per_graph

    @abstractmethod
    def calculate_relaxed_Energy(self):
        pass

    @abstractmethod
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, Energy_func):
        p = jnp.exp(logits[...,1])
        return None

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_CE(self, graphs, X_0, node_gr_idx):
        n_node = graphs.n_node
        cum_sum = jnp.concatenate([jnp.array([0]), jax.lax.cumsum(n_node)[:-1]], axis=0)
        cum_max_sum = jax.lax.cumsum(n_node)
        max_steps = jnp.max(graphs.n_node)

        Energy, violations_per_node, HB_per_graph = self.calculate_Energy(graphs, X_0, node_gr_idx)
        p_idxs = sort_ps(graphs,  violations_per_node/(jnp.max(violations_per_node) + 1), node_gr_idx)


        bern_feat = jnp.arange(0, self.n_bernoulli_features)

        sorted_cum_sum_idxs = p_idxs[cum_sum]
        n_bern_X_0 = jnp.repeat(X_0[None, ...], self.n_bernoulli_features, axis=0)
        n_bern_X_0 = n_bern_X_0.at[bern_feat[ :, None],sorted_cum_sum_idxs[None, :], 0].set(bern_feat[ :, None] * jnp.ones_like(cum_sum)[None, :])

        #p_idxs = sort_ps(graphs, X_0/(self.n_bernoulli_features+1))
        #_, down_Energy, HB_per_graph = self.calculate_Energy_per_node(graphs, X_0)
        Hb = jnp.mean(HB_per_graph[:-1])
        vmap_calculate_Energy_per_node = jax.vmap(lambda x: self.calculate_Energy_per_node(graphs,x,node_gr_idx), in_axes=(0))

        # Energies, HB_per_graph = vmap_calculate_Energy_per_node(n_bern_X_0)
        # ### TODO do not make so many copies for each node, only overwrite the specific index
        #
        # min_idx = jnp.argmin(Energies, axis=0)
        # min_idx = jnp.squeeze(min_idx, axis=-1)
        # best_X_0 = X_0.at[sorted_cum_sum_idxs, 0].set(min_idx)
        #
        # print("here")
        # print(min_idx)
        # print(bern_feat)
        # print(Energies)
        # print("best, best_Energy",jnp.min(Energies, axis = 0))
        # print(Energies[min_idx[...,0], :])
        @jax.jit
        def cond(arg):
            step = arg["step"]
            max_steps = arg["max_steps"]
            return step < max_steps#jax.lax.bitwise_and(Hb > 0., step < max_steps)

        @jax.jit
        def body(arg):
            #step, n_bern_X_0, X_0, p_idxs, cum_sum, cum_max_sum, graphs, _, _, _, node_graph_idx = arg
            scan_dict = arg
            step = scan_dict["step"]
            n_bern_X_0 = scan_dict["n_bern_X_0"]
            X_0 = scan_dict["X_0"]
            p_idxs = scan_dict["p_idxs"]
            cum_sum = scan_dict["cum_sum"]
            cum_max_sum = scan_dict["cum_max_sum"]

            sorted_cum_sum_idxs = p_idxs[cum_sum]
            #sorted_cum_sum_idxs = cum_sum

            bern_feat = jnp.arange(0, self.n_bernoulli_features)
            n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(bern_feat[:, None] * jnp.ones_like(cum_sum)[None, :])
            Energies, HB_per_graph = vmap_calculate_Energy_per_node(n_bern_X_0)
            ### TODO do not make so many copies for each node, only overwrite the specific index

            min_idx = jnp.argmin(Energies, axis = 0)
            min_idx = jnp.squeeze(min_idx, axis = -1)
            best_X_0 = X_0.at[sorted_cum_sum_idxs,0].set(min_idx)

            min_idx_rep = jnp.repeat(min_idx[None, ...], self.n_bernoulli_features, axis=0)
            n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(min_idx_rep * jnp.ones_like(cum_sum)[None, :])

            # min_idx = jnp.argmin(n_bern_Energies, axis = 0)
            # ### TODO the next line is also unneccesarily compicated
            # best_X_0 = n_bern_X_0[min_idx[None, ...], jnp.arange(n_bern_X_0.shape[1])[None, :, None], jnp.arange(n_bern_X_0.shape[2])[None, None, :]][0, ...]

            best_Energy = jnp.min(Energies, axis = 0)

            HB_per_graph_new = HB_per_graph[min_idx, jnp.arange(0, HB_per_graph.shape[1])]
            Hb = jnp.mean(HB_per_graph_new[:-1])
            cum_sum_p_1 = cum_sum + 1
            cum_sum = jnp.where(cum_sum_p_1 < cum_max_sum, cum_sum_p_1, cum_max_sum - 1)

            scan_dict["step"] = step + 1
            scan_dict["n_bern_X_0"] = n_bern_X_0
            scan_dict["X_0"] = best_X_0
            scan_dict["cum_sum"] = cum_sum
            scan_dict["Hb"] = Hb
            scan_dict["Energy"] = best_Energy
            scan_dict["HB_per_graph"] = HB_per_graph_new
            return scan_dict

        scan_dict = {}
        scan_dict["step"] = 0
        scan_dict["n_bern_X_0"] = n_bern_X_0
        scan_dict["X_0"] = X_0
        scan_dict["p_idxs"] = p_idxs
        scan_dict["cum_sum"] = cum_sum
        scan_dict["cum_max_sum"] = cum_max_sum
        scan_dict["max_steps"] = max_steps
        #scan_dict["graphs"] = graphs
        scan_dict["Hb"] = Hb
        scan_dict["Energy"] = Energy
        scan_dict["HB_per_graph"] = HB_per_graph
        #scan_dict["node_gr_idx"] = node_gr_idx

        out_scan_dict = jax.lax.while_loop(cond,body,scan_dict)

        return out_scan_dict["X_0"], out_scan_dict["Energy"], out_scan_dict["HB_per_graph"]

    def calculate_CE_debug(self, graphs, X_0, node_gr_idx):
        n_node = graphs.n_node
        cum_sum = jnp.concatenate([jnp.array([0]), jax.lax.cumsum(n_node)[:-1]], axis=0)
        cum_max_sum = jax.lax.cumsum(n_node)
        max_steps = jnp.max(graphs.n_node)

        Energy, violations_per_node, HB_per_graph = self.calculate_Energy(graphs, X_0, node_gr_idx)
        p_idxs = sort_ps(graphs, violations_per_node / (jnp.max(violations_per_node) + 1), node_gr_idx)

        bern_feat = jnp.arange(0, self.n_bernoulli_features)

        sorted_cum_sum_idxs = p_idxs[cum_sum]
        n_bern_X_0 = jnp.repeat(X_0[None, ...], self.n_bernoulli_features, axis=0)
        n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(
            bern_feat[:, None] * jnp.ones_like(cum_sum)[None, :])

        # p_idxs = sort_ps(graphs, X_0/(self.n_bernoulli_features+1))
        # _, down_Energy, HB_per_graph = self.calculate_Energy_per_node(graphs, X_0)
        Hb = jnp.mean(HB_per_graph[:-1])
        vmap_calculate_Energy_per_node = jax.vmap(lambda x: self.calculate_Energy_per_node(graphs, x, node_gr_idx),
                                                  in_axes=(0))

        # Energies, HB_per_graph = vmap_calculate_Energy_per_node(n_bern_X_0)
        # ### TODO do not make so many copies for each node, only overwrite the specific index
        #
        # min_idx = jnp.argmin(Energies, axis=0)
        # min_idx = jnp.squeeze(min_idx, axis=-1)
        # best_X_0 = X_0.at[sorted_cum_sum_idxs, 0].set(min_idx)
        #
        # print("here")
        # print(min_idx)
        # print(bern_feat)
        # print(Energies)
        # print("best, best_Energy",jnp.min(Energies, axis = 0))
        # print(Energies[min_idx[...,0], :])

        #@jax.jit
        def cond(arg):
            Hb = arg["Hb"]
            step = arg["step"]
            max_steps = arg["max_steps"]
            return step < max_steps  # jax.lax.bitwise_and(Hb > 0., step < max_steps)

        #@jax.jit
        def body(arg):
            # step, n_bern_X_0, X_0, p_idxs, cum_sum, cum_max_sum, graphs, _, _, _, node_graph_idx = arg
            scan_dict = arg
            step = scan_dict["step"]
            n_bern_X_0 = scan_dict["n_bern_X_0"]
            X_0 = scan_dict["X_0"]
            p_idxs = scan_dict["p_idxs"]
            cum_sum = scan_dict["cum_sum"]
            cum_max_sum = scan_dict["cum_max_sum"]

            sorted_cum_sum_idxs = p_idxs[cum_sum]
            # sorted_cum_sum_idxs = cum_sum

            bern_feat = jnp.arange(0, self.n_bernoulli_features)
            n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(
                bern_feat[:, None] * jnp.ones_like(cum_sum)[None, :])
            Energies, HB_per_graph = vmap_calculate_Energy_per_node(n_bern_X_0)
            ### TODO do not make so many copies for each node, only overwrite the specific index

            min_idx = jnp.argmin(Energies, axis=0)
            min_idx = jnp.squeeze(min_idx, axis=-1)
            best_X_0 = X_0.at[sorted_cum_sum_idxs, 0].set(min_idx)

            min_idx_rep = jnp.repeat(min_idx[None, ...], self.n_bernoulli_features, axis=0)
            n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(
                min_idx_rep * jnp.ones_like(cum_sum)[None, :])

            # min_idx = jnp.argmin(n_bern_Energies, axis = 0)
            # ### TODO the next line is also unneccesarily compicated
            # best_X_0 = n_bern_X_0[min_idx[None, ...], jnp.arange(n_bern_X_0.shape[1])[None, :, None], jnp.arange(n_bern_X_0.shape[2])[None, None, :]][0, ...]

            best_Energy = jnp.min(Energies, axis=0)

            HB_per_graph_new = HB_per_graph[min_idx[None, ...], jnp.arange(0, HB_per_graph.shape[1])[None, :]]

            Hb = jnp.mean(HB_per_graph_new[:-1])
            cum_sum_p_1 = cum_sum + 1
            cum_sum = jnp.where(cum_sum_p_1 < cum_max_sum, cum_sum_p_1, cum_max_sum - 1)

            scan_dict["step"] = step + 1
            scan_dict["n_bern_X_0"] = n_bern_X_0
            scan_dict["X_0"] = best_X_0
            scan_dict["cum_sum"] = cum_sum
            scan_dict["Hb"] = Hb
            scan_dict["Energy"] = best_Energy
            scan_dict["HB_per_graph"] = HB_per_graph_new
            return scan_dict

        scan_dict = {}
        print("inpout", HB_per_graph.shape)
        scan_dict["step"] = 0
        scan_dict["n_bern_X_0"] = n_bern_X_0
        scan_dict["X_0"] = X_0
        scan_dict["p_idxs"] = p_idxs
        scan_dict["cum_sum"] = cum_sum
        scan_dict["cum_max_sum"] = cum_max_sum
        scan_dict["max_steps"] = max_steps
        scan_dict["Hb"] = Hb
        scan_dict["Energy"] = Energy
        scan_dict["HB_per_graph"] = HB_per_graph
        # scan_dict["node_gr_idx"] = node_gr_idx

        while(cond(scan_dict)):
            start_time = time.time()
            scan_dict = body(scan_dict)
            end_time = time.time()
            print("time", end_time-start_time)

        out_scan_dict = scan_dict

        res = self.calculate_Energy(graphs, out_scan_dict["X_0"], node_gr_idx)

        print("output", out_scan_dict["HB_per_graph"].shape)
        return out_scan_dict["X_0"], out_scan_dict["Energy"], res[2], out_scan_dict["HB_per_graph"]

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_CE_p_values(self, graphs, ps):
        n_node = graphs.n_node
        cum_sum = jnp.concatenate([jnp.array([0]), jax.lax.cumsum(n_node)[:-1]], axis=0)
        max_steps = jnp.max(graphs.n_node)
        cum_max_sum = jax.lax.cumsum(n_node)

        nodes = graphs.nodes
        n_node = graphs.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
        Energy, Hb_per_node, _ = self.calculate_Energy(graphs, ps, node_graph_idx)

        p_idxs = sort_ps(graphs, ps, node_graph_idx)

        @jax.jit
        def cond_feas(arg):
            step, _, _, _, cum_sum, graphs, _, max_steps, _, _ = arg
            return step < max_steps

        @jax.jit
        def body_feas(arg):
            step, X_0, _, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps, cum_max_sum, p_idxs = arg

            sorted_cum_sum_idxs = p_idxs[cum_sum]

            X_up = X_0.at[sorted_cum_sum_idxs,0].set(1*jnp.ones_like(sorted_cum_sum_idxs))
            X_down = X_0.at[sorted_cum_sum_idxs,0].set(0*jnp.ones_like(sorted_cum_sum_idxs))

            Energy_up, Hb_per_node, _ = self.calculate_Energy(graphs, X_up, node_graph_idx)
            Energy_down, Hb_per_node, _ = self.calculate_Energy(graphs, X_down, node_graph_idx)

            best_bin_values = jnp.where(Energy_up < Energy_down, 1., 0.)
            X_0 = X_0.at[sorted_cum_sum_idxs].set(best_bin_values)

            cum_sum_p_1 = cum_sum + 1
            cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
            return (step + 1, X_0, Energy, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps, cum_max_sum, p_idxs)

        return_tuple = jax.lax.while_loop(cond_feas,
             body_feas, (0, ps, Energy, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps, cum_max_sum, p_idxs))

        best_X_0 = return_tuple[1]
        Energy_best, Hb_per_node, _ = self.calculate_Energy(graphs, best_X_0, node_graph_idx)
        return best_X_0, Energy_best, Hb_per_node

    def calculate_Energy_CE_p_values_debug(self, graphs, ps):
        n_node = graphs.n_node
        cum_sum = jnp.concatenate([jnp.array([0]), jax.lax.cumsum(n_node)[:-1]], axis=0)
        max_steps = jnp.max(graphs.n_node)
        cum_max_sum = jax.lax.cumsum(n_node)

        nodes = graphs.nodes
        n_node = graphs.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
        Energy, Hb_per_node, _ = self.calculate_Energy(graphs, ps, node_graph_idx)

        p_idxs = sort_ps(graphs, ps, node_graph_idx)

        #@jax.jit
        def cond_feas(arg):
            step, _, _, _, cum_sum, graphs, _, max_steps, _, _ = arg
            return step < max_steps

        #@jax.jit
        def body_feas(arg):
            step, X_0, _, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps, cum_max_sum, p_idxs = arg

            sorted_cum_sum_idxs = p_idxs[cum_sum]

            X_up = X_0.at[sorted_cum_sum_idxs,0].set(1*jnp.ones_like(sorted_cum_sum_idxs))
            X_down = X_0.at[sorted_cum_sum_idxs,0].set(0*jnp.ones_like(sorted_cum_sum_idxs))

            Energy_up, Hb_per_node, _ = self.calculate_Energy(graphs, X_up, node_graph_idx)
            Energy_down, Hb_per_node, _ = self.calculate_Energy(graphs, X_down, node_graph_idx)

            best_bin_values = jnp.where(Energy_up < Energy_down, 1., 0.)
            print("here", best_bin_values.shape, sorted_cum_sum_idxs.shape)
            X_0 = X_0.at[sorted_cum_sum_idxs].set(best_bin_values)

            print("Energy up", Energy_up[0])
            print("Energy down", Energy_down[0])
            print("new energy", self.calculate_Energy(graphs, X_0, node_graph_idx)[0][0])

            cum_sum_p_1 = cum_sum + 1
            cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
            return (step + 1, X_0, Energy, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps, cum_max_sum, p_idxs)

        # return_tuple = jax.lax.while_loop(cond_feas,
        #     body_feas, (0, ps, Energy, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps, cum_max_sum, p_idxs))

        return_tuple = (0, ps, Energy, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps, cum_max_sum, p_idxs)
        while(cond_feas(return_tuple)):
            return_tuple = body_feas(return_tuple)


        best_X_0 = return_tuple[1]
        Energy_best, Hb_per_node, _ = self.calculate_Energy(graphs, best_X_0, node_graph_idx)
        return best_X_0, Energy_best, Hb_per_node

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_CE_p_values_buggy(self, graphs, ps):
        n_node = graphs.n_node
        cum_sum = jnp.concatenate([jnp.array([0]), jax.lax.cumsum(n_node)[:-1]], axis=0)
        cum_max_sum = jax.lax.cumsum(n_node)
        max_steps = jnp.max(graphs.n_node)

        nodes = graphs.nodes
        n_node = graphs.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        Energy, violations_per_node, HB_per_graph = self.calculate_Energy(graphs, ps, node_graph_idx)
        p_idxs = sort_ps(graphs, ps, node_graph_idx)

        bern_feat = jnp.arange(0, self.n_bernoulli_features) # jnp.arange(0, self.n_bernoulli_features)

        sorted_cum_sum_idxs = p_idxs[cum_sum]
        n_bern_X_0 = jnp.repeat(ps[None, ...], self.n_bernoulli_features, axis=0)
        n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(
            bern_feat[:, None] * jnp.ones_like(cum_sum)[None, :])

        Hb = jnp.mean(HB_per_graph[:-1])
        vmap_calculate_Energy_per_node = jax.vmap(lambda x: self.calculate_Energy_per_node(graphs, x, node_graph_idx),
                                                  in_axes=(0))

        @jax.jit
        def cond(arg):
            step = arg["step"]
            max_steps = arg["max_steps"]
            return step < max_steps  # jax.lax.bitwise_and(Hb > 0., step < max_steps)

        @jax.jit
        def body(arg):
            # step, n_bern_X_0, X_0, p_idxs, cum_sum, cum_max_sum, graphs, _, _, _, node_graph_idx = arg
            scan_dict = arg
            step = scan_dict["step"]
            n_bern_X_0 = scan_dict["n_bern_X_0"]
            X_0 = scan_dict["X_0"]
            p_idxs = scan_dict["p_idxs"]
            cum_sum = scan_dict["cum_sum"]
            cum_max_sum = scan_dict["cum_max_sum"]

            sorted_cum_sum_idxs = p_idxs[cum_sum]
            # sorted_cum_sum_idxs = cum_sum

            bern_feat = jnp.arange(0, self.n_bernoulli_features)
            n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(
                bern_feat[:, None] * jnp.ones_like(cum_sum)[None, :])
            Energies, HB_per_graph = vmap_calculate_Energy_per_node(n_bern_X_0)
            ### TODO do not make so many copies for each node, only overwrite the specific index

            min_idx = jnp.argmin(Energies, axis=0)
            min_idx = jnp.squeeze(min_idx, axis=-1)
            best_X_0 = X_0.at[sorted_cum_sum_idxs, 0].set(min_idx)

            min_idx_rep = jnp.repeat(min_idx[None, ...], self.n_bernoulli_features, axis=0)
            n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(
                min_idx_rep * jnp.ones_like(cum_sum)[None, :])

            # min_idx = jnp.argmin(n_bern_Energies, axis = 0)
            # ### TODO the next line is also unneccesarily compicated
            # best_X_0 = n_bern_X_0[min_idx[None, ...], jnp.arange(n_bern_X_0.shape[1])[None, :, None], jnp.arange(n_bern_X_0.shape[2])[None, None, :]][0, ...]

            best_Energy = jnp.min(Energies, axis=0)

            HB_per_graph = HB_per_graph[min_idx[..., 0], :]
            Hb = jnp.mean(HB_per_graph[:-1])


            cum_sum_p_1 = cum_sum + 1
            cum_sum = jnp.where(cum_sum_p_1 < cum_max_sum, cum_sum_p_1, cum_max_sum - 1)

            scan_dict["step"] = step + 1
            scan_dict["n_bern_X_0"] = n_bern_X_0
            scan_dict["X_0"] = best_X_0
            scan_dict["cum_sum"] = cum_sum
            scan_dict["Hb"] = Hb
            scan_dict["Energy"] = best_Energy
            scan_dict["HB_per_graph"] = HB_per_graph
            return scan_dict

        scan_dict = {}
        scan_dict["step"] = 0
        scan_dict["n_bern_X_0"] = n_bern_X_0
        scan_dict["X_0"] = ps
        scan_dict["p_idxs"] = p_idxs
        scan_dict["cum_sum"] = cum_sum
        scan_dict["cum_max_sum"] = cum_max_sum
        scan_dict["max_steps"] = max_steps
        # scan_dict["graphs"] = graphs
        scan_dict["Hb"] = Hb
        scan_dict["Energy"] = Energy
        scan_dict["HB_per_graph"] = HB_per_graph
        # scan_dict["node_gr_idx"] = node_gr_idx

        out_scan_dict = jax.lax.while_loop(cond, body, scan_dict)

        # print("check")
        # print(out_scan_dict["Energy"], out_scan_dict["HB_per_graph"])
        ### TODO find out why this line is neccesary
        res = self.calculate_Energy(graphs, out_scan_dict["X_0"], node_graph_idx)

        # print(res[0], res[2])
        return out_scan_dict["X_0"], res[0], res[2]

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_feasible(self, graphs, X_0):
        ### TODo update energy function in every energy class
        n_node = graphs.n_node
        cum_sum = jnp.concatenate([jnp.array([0]), jax.lax.cumsum(n_node)[:-1]], axis=0)
        max_steps = jnp.max(graphs.n_node)

        nodes = graphs.nodes
        n_node = graphs.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
        Energy, Hb_per_node, _ = self.calculate_Energy(graphs, X_0, node_graph_idx)
        Hb = jnp.sum(jnp.abs(Hb_per_node))

        @jax.jit
        def cond_feas(arg):
            step, _, Hb, _, _, cum_sum, graphs, _, max_steps = arg
            return jax.lax.bitwise_and(Hb > 0., step < max_steps)

        @jax.jit
        def flip_value_feas(bins, sign):
            spins = 2 * bins - 1
            flipped_spins = sign * spins
            flipped_bins = (flipped_spins + 1) / 2
            return flipped_bins

        @jax.jit
        def body_feas(arg):
            step, X_0, _, _, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps = arg

            Hb_idxs = sort_violations(graphs, Hb_per_node)
            sorted_cum_sum_idxs = Hb_idxs[cum_sum]

            flip_per_node = jnp.where(Hb_per_node[sorted_cum_sum_idxs, 0] > 0, jnp.ones_like(sorted_cum_sum_idxs),
                                      jnp.zeros_like(sorted_cum_sum_idxs))
            flip_sign = jnp.where(flip_per_node, -jnp.ones_like(sorted_cum_sum_idxs),
                                  jnp.ones_like(sorted_cum_sum_idxs))

            flipped_X_0 = flip_value_feas(X_0[sorted_cum_sum_idxs, 0], flip_sign)

            X_0 = X_0.at[sorted_cum_sum_idxs, 0].set(flipped_X_0)

            Energy, Hb_per_node, _ = self.calculate_Energy(graphs, X_0, node_graph_idx)
            Hb = jnp.sum(jnp.abs(Hb_per_node))
            # cum_sum_p_1 = cum_sum + 1
            # cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
            return (step + 1, X_0, Hb, Energy, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps)

        return_tuple = jax.lax.while_loop(cond_feas,
            body_feas, (0, X_0, Hb, Energy, Hb_per_node, cum_sum, graphs, node_graph_idx, max_steps))
        return return_tuple[1], jnp.array([return_tuple[2]]), return_tuple[3]



    def get_log_p_0(self, batched_jraphs, X_0, T):
        T = jnp.max(jnp.array([T, 10**-6]))
        nodes = batched_jraphs.nodes
        n_node = batched_jraphs.n_node
        n_graph = batched_jraphs.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        Energy, _, _ = self.calculate_Energy(batched_jraphs, X_0, node_gr_idx)
        log_p_0 = self.get_log_p_0_from_energy(Energy, T)
        print("BaseEnergy", jnp.mean(log_p_0), T)
        return log_p_0

    def get_log_p_0_from_energy(self, Energy, T):
        T = jnp.max(jnp.array([T, 10**-6]))

        log_p_0 = -1/T*Energy[...,0]
        return log_p_0

@jax.jit
def sort_ps(graphs, ps, node_graph_idx):

    shifted_ps = -ps + node_graph_idx[:,None]
    #sorted_ps = jax.lax.sort(shifted_ps, dimension = 0)
    p_idxs = jax.numpy.argsort(shifted_ps, axis = 0, stable = False)

    return p_idxs[:,0]

@jax.jit
def sort_violations(graphs, HB_per_node):
    nodes = graphs.nodes
    n_node = graphs.n_node
    n_graph = graphs.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

    shifted_HB_per_node = -HB_per_node/(jnp.max(HB_per_node)+1.) + node_graph_idx[:,None]
    #sorted_ps = jax.lax.sort(shifted_ps, dimension = 0)
    HB_per_node_idxs = jax.numpy.argsort(shifted_HB_per_node, axis = 0)

    return HB_per_node_idxs[:,0]