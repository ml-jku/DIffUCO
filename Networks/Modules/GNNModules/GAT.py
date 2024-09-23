import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
import jraph
from Networks.Modules.MLPModules.MLPs import ReluMLP
from Networks.Modules.GNNModules.GraphNorm import GraphNorm

def add_self_edges_fn(jraph):
    r"""Adds self edges. Assumes self edges are not in the graph yet."""
    total_num_nodes = jraph.nodes.shape[0]
    receivers = jnp.concatenate((jraph.receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((jraph.senders, jnp.arange(total_num_nodes)), axis=0)
    edges = jnp.concatenate((jraph.edges, jnp.zeros((total_num_nodes, jraph.edges.shape[-1]))), axis=0)
    jraph = jraph._replace(receivers = receivers, senders = senders, edges = edges)
    return jraph


class AttentionQueryModule(nn.Module):
    proj_features: int
    def setup(self):
        self.proj_layer = nn.Dense(features=self.proj_features, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)

    @flax.linen.jit
    def __call__(self, sender_attr, receiver_attr, edges):
        x = jnp.concatenate((sender_attr, receiver_attr, edges), axis=-1)
        head_query = self.proj_layer(x)

        return head_query
class AttentionlogitModule(nn.Module):
    feature_list: jnp.array
    def setup(self):
        self.n_features_list = self.feature_list
        layers = []
        # add hidden layers
        for n_features in self.n_features_list:
            layers.append(nn.Dense(features=n_features, kernel_init=nn.initializers.he_normal(),
                                   bias_init=nn.initializers.zeros))
            layers.append(jax.nn.relu)
            layers.append(nn.LayerNorm())
        # add output layer
        layers.append(nn.Dense(features=1, kernel_init=nn.initializers.he_normal(),
                               bias_init=nn.initializers.zeros))
        layers.append(jax.nn.relu)

        self.mlp = nn.Sequential(layers)

    @flax.linen.jit
    def __call__(self, sender_attr: jnp.ndarray, receiver_attr: jnp.ndarray,edges: jnp.ndarray):
        x = jnp.concatenate((sender_attr, receiver_attr, edges), axis=-1)
        return self.mlp(x)

class GraphAttentionNetwork(nn.Module):
    feature_list: jnp.array

    def setup(self):
        self.att_q_func = AttentionQueryModule(proj_features=self.feature_list[0])

        self.att_logit_func = AttentionlogitModule(feature_list=self.feature_list)

        self.GAT_layer = ownGAT(attention_query_fn = self.att_q_func, attention_logit_fn=self.att_logit_func)

    @flax.linen.jit
    def __call__(self, jraph, copy_arr):
        # Define the GAT layer

        # Apply GAT layer to inputs with adjacency matrix
        jraph_self_edges = add_self_edges_fn(jraph)
        out_jraph = self.GAT_layer(jraph_self_edges)

        return out_jraph.nodes

class MultiheadGraphAttentionNetwork(nn.Module):
    n_features_list_nodes: jnp.asarray
    n_features_list_messages: jnp.asarray
    n_heads: int = 6
    graph_norm: bool = False

    def setup(self):
        VmapMLP = nn.vmap(GraphAttentionNetwork, variable_axes={'params': 0}, split_rngs={'params': True},
                          in_axes=(None, 0), out_axes=(-1))
        self.MultiheadGAT = VmapMLP(self.n_features_list_messages, name='GAT')
        self.lin_proj = nn.Dense(features=self.n_features_list_messages[-1], kernel_init=nn.initializers.he_normal(),
                 bias_init=nn.initializers.zeros)
        self.ln = nn.LayerNorm()
        self.GraphNorm = GraphNorm()

    @nn.compact
    def __call__(self, jraph):
        copy_arr = jnp.ones((self.n_heads, 1))
        MultiheadGAT_output = self.MultiheadGAT(jraph, copy_arr)

        GAT_output = jnp.sum(MultiheadGAT_output, axis = -1)

        if (self.graph_norm):
            GAT_output = self.GraphNorm(jraph, GAT_output)

        GAT_output = ReluMLP(n_features_list=self.n_features_list_nodes)(GAT_output)

        GAT_output = self.ln(GAT_output + self.lin_proj(jraph.nodes))

        return jraph._replace(nodes = GAT_output)


from jraph._src import utils
def ownGAT(attention_query_fn,
        attention_logit_fn,node_update_fn = None):
  """Returns a method that applies a Graph Attention Network layer.

  Graph Attention message passing as described in
  https://arxiv.org/abs/1710.10903. This model expects node features as a
  jnp.array, may use edge features for computing attention weights, and
  ignore global features. It does not support nests.

  NOTE: this implementation assumes that the input graph has self edges. To
  recover the behavior of the referenced paper, please add self edges.

  Args:
    attention_query_fn: function that generates attention queries
      from sender node features.
    attention_logit_fn: function that converts attention queries into logits for
      softmax attention.
    node_update_fn: function that updates the aggregated messages. If None,
      will apply leaky relu and concatenate (if using multi-head attention).

  Returns:
    A function that applies a Graph Attention layer.
  """
  # pylint: disable=g-long-lambda
  if node_update_fn is None:
    # By default, apply the leaky relu and then concatenate the heads on the
    # feature axis.
    node_update_fn = lambda x: jnp.reshape(
        jax.nn.leaky_relu(x), (x.shape[0], -1))
  def _ApplyGAT(graph):
    """Applies a Graph Attention layer."""
    nodes, edges, receivers, senders, _, _, _ = graph
    # Equivalent to the sum of n_node, but statically known.
    try:
      sum_n_node = nodes.shape[0]
    except IndexError:
      raise IndexError('GAT requires node features')  # pylint: disable=raise-missing-from

    # First pass nodes through the node updater.
    # pylint: disable=g-long-lambda
    # We compute the softmax logits using a function that takes the
    # embedded sender and receiver attributes.
    sent_attributes = nodes[senders]
    received_attributes = nodes[receivers]
    concat_attributes = attention_query_fn(sent_attributes, received_attributes, edges)
    softmax_logits = attention_logit_fn(
        sent_attributes, received_attributes, edges)

    # Compute the softmax weights on the entire tree.
    weights = utils.segment_softmax(softmax_logits, segment_ids=receivers,
                                    num_segments=sum_n_node)
    # Apply weights
    messages = concat_attributes * weights
    # Aggregate messages to nodes.
    nodes = utils.segment_sum(messages, receivers, num_segments=sum_n_node)

    # Apply an update function to the aggregated messages.
    nodes = node_update_fn(nodes)
    return graph._replace(nodes=nodes)
  # pylint: enable=g-long-lambda
  return _ApplyGAT

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
