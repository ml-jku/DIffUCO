from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
from flax.linen.dtypes import canonicalize_dtype

from flax.linen.module import Module, compact, merge_param  # pylint: disable=g-multiple-import
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp
import jax

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?

Axes = Union[int, Any]


def _compute_stats():
    pass

class GraphNorm(Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).

    LayerNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over
      the examples on the first two and last two devices. See `jax.lax.psum`
      for more details.
    """
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    reduction_axes: Axes = -1
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @compact
    def __call__(self, graphs, x):
        """Applies layer normalization on the input.

        Args:
          x: the inputs

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        return _normalize(self, graphs, x, self.reduction_axes, self.feature_axes, self.dtype, self.param_dtype, self.epsilon,
            self.use_bias, self.use_scale,
            self.bias_init, self.scale_init)

def _normalize(mdl: Module, graphs, x: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
               scale_init: Callable[[PRNGKey, Shape, Dtype], Array]):
    """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

    Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
      in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    param_dtype: The dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.

    Returns:
    The normalized input.
    """
    mean, _ = calc_mean(graphs, x)

    if use_scale:
        mean_scale = mdl.param('mean_scale', scale_init, (x.shape[-1]),param_dtype)
        mean *= mean_scale

    y = x - mean
    var, _ = calc_var(graphs, x, mean)
    mul = lax.rsqrt(var + epsilon)

    if use_scale:
        scale = mdl.param('scale', scale_init, (x.shape[-1],), param_dtype)
        mul *= scale

        y *= mul
    if use_bias:
        bias = mdl.param('bias', bias_init, (x.shape[-1]), param_dtype)
        y += bias

    return jnp.asarray(y, dtype)

def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))

@jax.jit
def calc_mean(H_graph, features):
    n_node = H_graph.n_node
    n_graph = H_graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(features)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

    sum_per_graph = jax.ops.segment_sum(features, node_graph_idx, n_graph)
    nodes_per_graph = jax.ops.segment_sum(jnp.ones(features.shape[:-1] + (1,)), node_graph_idx, n_graph)
    mean_per_graph = sum_per_graph/nodes_per_graph
    mean_per_node = jnp.repeat(mean_per_graph, n_node, axis=0, total_repeat_length=total_nodes)
    return mean_per_node, mean_per_graph


@jax.jit
def calc_var(H_graph, features, mean_per_node):
    n_node = H_graph.n_node
    n_graph = H_graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(features)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

    var_sum_per_graph = jax.ops.segment_sum((features - mean_per_node)**2, node_graph_idx, n_graph)
    nodes_per_graph = jax.ops.segment_sum(jnp.ones(features.shape[:-1] + (1,)), node_graph_idx, n_graph)
    var_per_graph = var_sum_per_graph / nodes_per_graph
    var_per_node = jnp.repeat(var_per_graph, n_node, axis=0, total_repeat_length=total_nodes)
    return var_per_node, var_per_graph



if(__name__ == "__main__"):
    import jraph
    import networkx as nx

    jax.config.update('jax_platform_name', 'cpu')
    def create_graph(n_nodes):
        # Define a three node graph, each node has an integer as its feature.
        gnx = nx.barabasi_albert_graph(n=n_nodes, m=4)
        edges = list(gnx.edges)

        for el in edges:
            if (el[0] == el[1]):
                raise ValueError("Self loops included")

        node_features = jnp.zeros((n_nodes, 1))

        # We will construct a graph for which there is a directed edge between each node
        # and its successor. We define this with `senders` (source nodes) and `receivers`
        # (destination nodes).
        senders = jnp.array([e[0] for e in edges])
        receivers = jnp.array([e[1] for e in edges])

        # You can optionally add edge attributes.

        edges = jnp.ones_like(senders)[:, None]

        # We then save the number of nodes and the number of edges.
        # This information is used to make running GNNs over multiple graphs
        # in a GraphsTuple possible.
        n_node = jnp.array([node_features.shape[0]])
        n_edge = jnp.array([3])

        # Optionally you can add `global` information, such as a graph label.

        global_context = jnp.array([[1]])
        graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
                                  edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)
        return graph

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    node_list = [10, 12, 8, 6, 30]
    graph_list = [create_graph(n) for n in node_list]
    batched_graph = jraph.batch(graph_list)

    X_prev = 5*jax.random.uniform(key, (batched_graph.nodes.shape[0], batched_graph.nodes.shape[0])) + 10

    GN = GraphNorm()
    params = GN.init({"params": subkey}, batched_graph, X_prev)


    out = GN.apply(params, batched_graph, X_prev)

    mean_per_node, mean_per_graph = calc_mean(batched_graph, out)
    _, std_per_graph = calc_var(batched_graph, out, mean_per_node)
    print(mean_per_graph)
    print(std_per_graph)
    pass
