import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from Networks.Modules.MLPModules.MLPs import ProbMLP, ValueMLP
from functools import partial
import flax


def get_graph_info(jraph_graph_list):
    first_graph = jraph_graph_list["graphs"][0]
    nodes = first_graph.nodes
    n_node = first_graph.n_node
    n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
    return node_graph_idx, n_graph, n_node


def global_graph_aggr(feature, node_graph_idx, n_graph):
    aggr_feature = jax.ops.segment_sum(feature, node_graph_idx, n_graph)
    return aggr_feature

class RLHeadModule_agg_before(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray
    dtype: any

    def setup(self):
        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob, dtype = self.dtype)
        value_feature_list = [120, 64, 1]
        self.ValueMLP = ValueMLP(n_features_list=value_feature_list, dtype = self.dtype)

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self,jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        spin_logits = self.probMLP(x)

        node_graph_idx, n_graph, n_node = get_graph_info(jraph_graph_list)
        Value_embeddings = global_graph_aggr(x, node_graph_idx, n_graph) / jnp.sqrt(n_node[..., None, None])
        Values = self.ValueMLP(Value_embeddings)[..., 0, 0]
        out_dict["spin_logits"] = spin_logits
        out_dict["Values"] = Values
        return out_dict


class RLHeadModule_agg_after(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray
    dtype: any

    def setup(self):
        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob)
        value_feature_list = [120, 64, 1]
        self.ValueMLP = ValueMLP(n_features_list=value_feature_list)

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self,jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        spin_logits = self.probMLP(x)

        node_graph_idx, n_graph, n_node = get_graph_info(jraph_graph_list)
        Values = self.ValueMLP(x)
        aggr_Values = global_graph_aggr(Values, node_graph_idx, n_graph)[..., 0, 0]
        out_dict["spin_logits"] = spin_logits
        out_dict["Values"] = aggr_Values
        return out_dict

    def get_graph_info(self, jraph_graph_list):
        first_graph = jraph_graph_list["graphs"][0]
        nodes = first_graph.nodes
        n_node = first_graph.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
        return node_graph_idx, n_graph, n_node


class RLHeadModuleTSP(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray
    dtype: any

    def setup(self):
        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob, dtype= self.dtype)
        value_feature_list = [120, 120, 1]
        self.ValueMLP = ValueMLP(n_features_list=value_feature_list, dtype= self.dtype)

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self,jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array (batch_size, 1, n_cities, n_features)
        """
        x_aggr = jnp.mean(x, axis = -2, keepdims=True)
        rep_x_aggr = jnp.repeat(x_aggr, x.shape[-2], axis = -2)
        x = jnp.concatenate([x, rep_x_aggr], axis = -1)
        
        spin_logits = self.probMLP(x)
        spin_logits = spin_logits[:,0,...]

        Values = self.ValueMLP(x_aggr)
        Values = Values[..., 0, 0]

        padded_spin_logits = jnp.concatenate([spin_logits, jnp.zeros((1, *spin_logits.shape[1:]))], axis = 0)
        padded_values = jnp.concatenate([Values, jnp.zeros((1, *Values.shape[1:]))], axis = 0)

        spin_logits = jnp.reshape(padded_spin_logits, (padded_spin_logits.shape[0]*padded_spin_logits.shape[1],) + (1,padded_spin_logits.shape[-1]))
        Values = jnp.reshape(padded_values, (padded_values.shape[0]*padded_values.shape[1],) + padded_values.shape[2:])

        out_dict["spin_logits"] = spin_logits
        out_dict["Values"] = Values
        return out_dict