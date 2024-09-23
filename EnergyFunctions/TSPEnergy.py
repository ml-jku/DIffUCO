from .BaseEnergy import BaseEnergyClass
from functools import partial
import jax
import jax.numpy as jnp
import jraph

class TSPEnergyClass(BaseEnergyClass):

    def __init__(self, config):
        super().__init__(config)

        self.cycl_perm_mat = jnp.diag(jnp.ones((self.n_bernoulli_features - 1, )), k = -1)
        self.cycl_perm_mat = self.cycl_perm_mat.at[0,-1].set(1) ###

        self.vmap_calculate_Energy_per_instance = jax.vmap(self.calculate_Energy_per_instance, in_axes = (0,0))
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_old(self, H_graph, X_0_classes, node_gr_idx, A = 1.45):
        '''
        :param H_graph:
        :param bins:
        :param node_gr_idx:
        :param A:
        :param B:
        :return:
        '''

        x_mat = jax.nn.one_hot(X_0_classes[:,0], num_classes=self.n_bernoulli_features, dtype=jnp.float32)

        n_node = H_graph.n_node
        n_graph = n_node.shape[0]
        sum_n_node = H_graph.nodes.shape[0]

        idxs = jnp.arange(0, x_mat.shape[-1])
        idxs_p1 = (idxs + 1) % (x_mat.shape[-1])

        receivers = H_graph.receivers
        senders = H_graph.senders

        Obj1_per_graph_per_feature = (1-jraph.segment_sum(x_mat, node_gr_idx, n_graph))**2

        X_senders = X_0_classes[senders]
        X_receivers = X_0_classes[receivers]

        n_copy_senders = jnp.where(X_receivers == X_senders, 1., 0.)

        Obj1_per_node = jraph.segment_sum(n_copy_senders, H_graph.receivers, sum_n_node)
        Obj1_per_graph = jnp.sum(Obj1_per_graph_per_feature, axis=-1, keepdims=True)

        # diagonal_matrix = jnp.eye(x_mat.shape[-1])
        # ones_matrix = jnp.ones((x_mat.shape[0], x_mat.shape[-1], x_mat.shape[-1]))

        #result_matrix = ones_matrix - diagonal_matrix

        # Obj2_per_node = 1 - jnp.sum(x_mat, axis=-1, keepdims=True) + jnp.sum(
        #     jnp.sum(result_matrix * x_mat[:, :, jnp.newaxis] * x_mat[:, jnp.newaxis, :], axis=-1), axis=-1, keepdims=True)
        # Obj2_per_graph = jraph.segment_sum(Obj2_per_node, node_gr_idx, n_graph)
        Obj2_per_graph = 0.

        edge_features = H_graph.edges
        tour_messages = jnp.sum(x_mat[senders[:, jnp.newaxis], idxs[jnp.newaxis, :]] * x_mat[
            receivers[:, jnp.newaxis], idxs_p1[jnp.newaxis, :]], axis=-1)
        tour_messages = jnp.expand_dims(tour_messages, axis=-1)

        Obj3_per_node = jraph.segment_sum(edge_features * tour_messages, H_graph.receivers, sum_n_node)
        Obj3_per_graph = jraph.segment_sum(Obj3_per_node, node_gr_idx, n_graph)

        HA_per_graph = Obj3_per_graph
        HB_per_graph = A * Obj1_per_graph + A * Obj2_per_graph

        Energy = HA_per_graph + HB_per_graph
        return Energy, Obj1_per_node, HB_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy(self, graphs, X_0_classes, node_gr_idx):
        print("function is jitted")
        positions = graphs.nodes
        n_node = graphs.n_node
        positions = jnp.reshape(positions, (n_node.shape[0], self.n_bernoulli_features, 2))
        X_0_classes = jnp.reshape(X_0_classes, (n_node.shape[0], self.n_bernoulli_features, 1))

        Energy, X_0_classe_violation_per_node, HB_per_graph = self.vmap_calculate_Energy_per_instance(positions, X_0_classes)
        X_0_classe_violation_per_node = jnp.reshape(X_0_classe_violation_per_node,(n_node.shape[0]* self.n_bernoulli_features, 1))
        return Energy, X_0_classe_violation_per_node, HB_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_per_instance(self, positions, X_0_classes, A = 1.45):
        x_mat = jax.nn.one_hot(X_0_classes[:,0], num_classes=self.n_bernoulli_features)
        distance_matrix = jnp.sqrt(jnp.sum((positions[:, None] - positions[None, :])**2, axis = -1))

        cycl_perm_mat = self.cycl_perm_mat
        x_mat_cycl = jnp.tensordot(x_mat, cycl_perm_mat, axes=[[-1], [0]])
        H_mat = jnp.tensordot(x_mat_cycl, x_mat, axes=[[-1], [-1]])

        X_0_classe_violation = jnp.where(X_0_classes[None, :] == X_0_classes[:, None], 1., 0.)
        X_0_classe_violation_per_node = jnp.sum(X_0_classe_violation, axis = -2) - 1

        Obj1_per_graph = (jnp.sum(x_mat, axis = 0) -1)**2
        Obj2_per_graph = (jnp.sum(x_mat, axis=1) - 1) ** 2

        HB_per_graph =  A* jnp.sum(Obj1_per_graph)
        H2 = A* jnp.sum(Obj2_per_graph)
        H3 = jnp.sum(H_mat* distance_matrix)

        Energy = HB_per_graph + H2 + H3

        return Energy[...,None], X_0_classe_violation_per_node, HB_per_graph

    @partial(jax.jit, static_argnums=(0,))
    def calculate_relaxed_Energy(self, H_graph, bins, node_gr_idx):
        self.calculate_Energy(H_graph, bins, node_gr_idx)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_Energy_loss(self, H_graph, logits, node_gr_idx, Energy_func):
        p = jnp.exp(logits[...,1])
        return self.calculate_Energy(H_graph, p, node_gr_idx)

    # #@partial(jax.jit, static_argnums=(0,))
    # def TSP_Energy(self, H_graph, bins, B=1.42):
    #     ### TODO this function ins probably stale
    #     nodes = H_graph.nodes
    #     n_node = H_graph.n_node
    #     n_graph = n_node.shape[0]
    #     graph_idx = jnp.arange(n_graph)
    #     sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    #     node_gr_idx = jnp.repeat(
    #         graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    #
    #     x_mat = bins
    #
    #     n_node = H_graph.n_node
    #     n_graph = n_node.shape[0]
    #     sum_n_node = H_graph.nodes.shape[0]
    #
    #     idxs = jnp.arange(0, x_mat.shape[-1])
    #     idxs_p1 = (idxs + 1) % (x_mat.shape[-1])
    #
    #     receivers = H_graph.receivers
    #     senders = H_graph.senders
    #
    #     Obj1_per_graph_per_feature = 1 - jraph.segment_sum(x_mat, node_gr_idx, n_graph) +  jraph.segment_sum(jraph.segment_sum(x_mat[senders]*x_mat[receivers], H_graph.receivers, sum_n_node), node_gr_idx, n_graph)
    #     Obj1_per_graph = jnp.sum(Obj1_per_graph_per_feature, axis = -1, keepdims=True)
    #
    #     diagonal_matrix = jnp.eye(x_mat.shape[-1])
    #     ones_matrix = jnp.ones((x_mat.shape[0],x_mat.shape[-1], x_mat.shape[-1]))
    #     result_matrix = ones_matrix - diagonal_matrix
    #
    #     Obj2_per_node = 1 - jnp.sum(x_mat, axis=-1, keepdims=True) + jnp.sum(jnp.sum(result_matrix*x_mat[:,:,np.newaxis]*x_mat[:,np.newaxis,:], axis = -1), axis = -1, keepdims=True)
    #     Obj2_per_graph = jraph.segment_sum(Obj2_per_node, node_gr_idx, n_graph)
    #
    #     edge_features = H_graph.edges
    #     tour_messages = jnp.sum(x_mat[senders[:, jnp.newaxis], idxs[jnp.newaxis, :]] * x_mat[
    #         receivers[:, jnp.newaxis], idxs_p1[jnp.newaxis, :]], axis=-1)
    #     tour_messages = jnp.expand_dims(tour_messages, axis=-1)
    #
    #     Obj3_per_node = jraph.segment_sum(edge_features * tour_messages, H_graph.receivers, sum_n_node)
    #     Obj3_per_graph = jraph.segment_sum(Obj3_per_node, node_gr_idx, n_graph)
    #
    #     HA_per_graph = Obj3_per_graph
    #     HB_per_graph = B * Obj1_per_graph + B * Obj2_per_graph
    #     ### TODO find out position of violation: i.e node and idx
    #     Energy = HA_per_graph + HB_per_graph
    #     print(HA_per_graph, Obj1_per_graph, Obj2_per_graph)
    #     print(jnp.sum((1-jnp.sum(x_mat, axis = -1))**2), jnp.sum((1-jnp.sum(x_mat, axis = 0))**2))
    #     return Energy