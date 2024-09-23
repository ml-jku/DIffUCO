import jax
import jax.numpy as jnp
import jax.tree_util as tree

def calcEnergy_sparse(H_graph, spins, B = 1.):
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

    adjacency = H_graph.edges

    raveled_bins = jnp.reshape(spins, (spins.shape[0] * spins.shape[1], 1))[:(spins.shape[0] - 1) * spins.shape[1] + 1]
    Energy_messages = adjacency * raveled_bins[H_graph.senders] * raveled_bins[H_graph.receivers]
    Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

    Hb = B * jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)[:-1, 0]

    Energy = Hb

    return Energy

def calcEnergy(H_graph, spins, B = 1.):
    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(
        graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

    adjacency = H_graph.edges

    raveled_bins = jnp.reshape(spins, (spins.shape[0] * spins.shape[1], 1))
    Energy_messages = adjacency * raveled_bins[H_graph.senders] * raveled_bins[H_graph.receivers]
    Energy_per_node = 0.5 * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes)

    Hb = B * jax.ops.segment_sum(Energy_per_node, node_gr_idx, n_graph)

    Energy = Hb

    return Energy

def calcCutValue():
    pass