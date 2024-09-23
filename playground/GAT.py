
import jax
import jax.numpy as jnp
from flax import linen as nn
import jraph

def add_self_edges_fn(jraph):
    r"""Adds self edges. Assumes self edges are not in the graph yet."""
    total_num_nodes = jraph.nodes.shape[0]
    receivers = jnp.concatenate((jraph.receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((jraph.senders, jnp.arange(total_num_nodes)), axis=0)
    edges = jnp.concatenate((jraph.edges, jnp.zeros((total_num_nodes, jraph.edges.shape[-1]))), axis=0)
    jraph = jraph._replace(receivers = receivers, senders = senders, edges = edges)
    return jraph

def _node_update_fn(node_features):
    print(node_features.shape)
    return node_features


class AttentionQueryModule(nn.Module):
    proj_features: int
    def setup(self):
        self.proj_layer = nn.Dense(features=self.proj_features, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)

    def __call__(self, senders_attr):
        head_query = self.proj_layer(senders_attr)

        return head_query
class AttentionlogitModule(nn.Module):
    features: int
    def setup(self):
        self.n_features_list = [self.features, self.features, 1]
        layers = []
        # add hidden layers
        for n_features in self.n_features_list[:-1]:
            layers.append(nn.Dense(features=n_features, kernel_init=nn.initializers.he_normal(),
                                   bias_init=nn.initializers.zeros))
            layers.append(jax.nn.relu)
            layers.append(nn.LayerNorm())
        # add output layer
        layers.append(nn.Dense(features=self.n_features_list[-1], kernel_init=nn.initializers.he_normal(),
                               bias_init=nn.initializers.zeros))
        layers.append(jax.nn.relu)
        layers.append(nn.LayerNorm())

        self.mlp = nn.Sequential(layers)

    def __call__(self, sender_attr: jnp.ndarray, receiver_attr: jnp.ndarray,edges: jnp.ndarray):
        x = jnp.concatenate((sender_attr, receiver_attr, edges), axis=-1)
        return self.mlp(x)[...,None]

class GraphAttentionNetwork(nn.Module):
    proj_features: int
    features: int

    def setup(self):
        self.att_q_func = AttentionQueryModule(proj_features=self.proj_features)

        self.att_logit_func = AttentionlogitModule(features=self.features)

        self.GAT_layer = jraph.GAT(attention_query_fn = self.att_q_func, attention_logit_fn=self.att_logit_func)

    def __call__(self, jraph, copy_arr):
        # Define the GAT layer

        # Apply GAT layer to inputs with adjacency matrix
        jraph = add_self_edges_fn(jraph)
        outputs = self.GAT_layer(jraph)

        return outputs.nodes

class MultiheadGraphAttentionNetwork(nn.Module):
    proj_features: int
    features: int
    n_heads: int

    @nn.compact
    def __call__(self, jraph):
        copy_arr = jnp.ones((self.n_heads, 1))
        VmapMLP = nn.vmap(GraphAttentionNetwork, variable_axes={'params': 0}, split_rngs={'params': True}, in_axes=(None, 0), out_axes=(-1))
        MultiheadGAT_output = VmapMLP(self.proj_features, self.features, name='mlp')(jraph, copy_arr)
        print("here",MultiheadGAT_output.shape)
        GAT_output = jnp.sum(MultiheadGAT_output, axis = -1)
        print(GAT_output.shape)
        return GAT_output




if(__name__ == "__main__"):
    # Example usage
    node_features = jnp.array([[0.], [1.], [2.]])

    # We will construct a graph for which there is a directed edge between each node
    # and its successor. We define this with `senders` (source nodes) and `receivers`
    # (destination nodes).
    senders = jnp.array([0, 1, 2])
    receivers = jnp.array([1, 2, 0])

    # You can optionally add edge attributes.
    edges = jnp.array([[5.], [6.], [7.]])

    # We then save the number of nodes and the number of edges.
    # This information is used to make running GNNs over multiple graphs
    # in a GraphsTuple possible.
    n_node = jnp.array([3])
    n_edge = jnp.array([3])

    # Optionally you can add `global` information, such as a graph label.

    global_context = jnp.array([[1]])
    graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
                              edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)

    # Create an instance of GraphAttentionNetwork
    gat = MultiheadGraphAttentionNetwork(
                                features=64, proj_features=64, n_heads=8,)

    # Initialize parameters and apply the network
    rng = jax.random.PRNGKey(0)
    params = gat.init(rng, graph)
    print(jax.tree_util.tree_map(lambda x: x.shape, params))
    output = gat.apply(params, graph)
