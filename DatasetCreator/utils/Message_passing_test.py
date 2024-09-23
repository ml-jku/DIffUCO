
import jax
import jax.numpy as jnp

def MessagePassing(H_graph, nodes):
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = H_graph.edges * nodes[H_graph.senders] * nodes[H_graph.receivers]
    Energy_per_node =  jax.ops.segment_sum(Energy_messages, H_graph.receivers, sum_n_node) + nodes*H_graph.nodes
    Energy_per_node = (Energy_per_node - jnp.mean(Energy_per_node))/(jnp.std(Energy_per_node)+ 10**-6)
    return Energy_per_node

def MeanAggr(H_graph, nodes):
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = H_graph.edges * nodes[H_graph.senders] * nodes[H_graph.receivers]
    mean_messages = jnp.ones_like(H_graph.edges) * nodes[H_graph.senders] * nodes[H_graph.receivers]
    mean_per_node = jax.ops.segment_sum(mean_messages, H_graph.receivers, sum_n_node)
    mean_per_node = jnp.where(mean_per_node == 0, jnp.ones_like(mean_per_node), mean_per_node)

    Energy_per_node =  jax.ops.segment_max(Energy_messages, H_graph.receivers, sum_n_node)

    return Energy_per_node

def MaxAggr(H_graph, nodes):
    n_node = H_graph.n_node
    n_graph = n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    sum_n_node = H_graph.nodes.shape[0]
    node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0)

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = H_graph.edges * nodes[H_graph.senders] * nodes[H_graph.receivers]

    Energy_per_node =  jax.ops.segment_max(Energy_messages, H_graph.receivers, sum_n_node)


    return Energy_per_node

def MaxPool(max_aggr_graph, pool_graph, nodes):
    updated_nodes = MaxAggr(max_aggr_graph, nodes)
    updated_nodes = updated_nodes[pool_graph.nodes[..., 0]]
    return updated_nodes
