import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from Networks.Modules.MLPModules.MLPs import ProbMLP
from functools import partial
import flax
class NormalHeadModule(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray
    dtype: any
    def setup(self):
        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob, dtype = self.dtype)

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self, jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        spin_logits = self.probMLP(x)
        out_dict["spin_logits"] = spin_logits

        return out_dict

class TransformerHead(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray
    dtype: any

    def setup(self):
        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob, dtype= self.dtype)

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self,jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        x_aggr = jnp.mean(x, axis = -2, keepdims=True)
        rep_x_aggr = jnp.repeat(x_aggr, x.shape[-2], axis = -2)
        x = jnp.concatenate([x, rep_x_aggr], axis = -1)
        spin_logits = self.probMLP(x)[:,0,...]


        spin_logits = jnp.reshape(spin_logits, (spin_logits.shape[0]*spin_logits.shape[1],) + (1,spin_logits.shape[-1]))

        out_dict["spin_logits"] = spin_logits
        return out_dict