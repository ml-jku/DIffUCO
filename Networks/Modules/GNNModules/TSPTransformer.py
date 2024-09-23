import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
from Networks.Modules.MLPModules.MLPs import ReluMLP, ReluMLP_skip


class TSPTransformer(nn.Module):
    """
    EncodeProcessDecode Architecture

    @params n_features_list_nodes: feature list for node MLP in message passing layer
    @params n_features_list_edges: feature list for edge MLP in message passing layer
    @params n_features_list_messages: feature list for message MLP in message passing layer
    @params n_features_list_encode: feature list for encoders
    @params n_features_list_encode: feature list for decoders
    @params n_message_passes: number of message passing steps in process block
    @params weight_tied: the weights in the process block are tied (i.e. the same message passing layer is used over all n messages passing steps)
    """
    n_features_list_nodes: np.ndarray
    n_features_list_edges: np.ndarray
    n_features_list_messages: np.ndarray

    n_features_list_encode: np.ndarray
    n_features_list_decode: np.ndarray
    dtype: any

    edge_updates: bool
    linear_message_passing: bool = True

    n_message_passes: int = 5
    weight_tied: bool = True
    mean_aggr: bool = False
    graph_norm: bool = False
    num_heads: int = 8
    qkv_features: int = 64


    def setup(self):
        encoder_list = [ 2*el for el in self.n_features_list_encode]
        encoder_list.append(self.n_features_list_encode[-1])
        self.node_encoder = ReluMLP(n_features_list= encoder_list , dtype = self.dtype)
        self.node_decoder = ReluMLP_skip(n_features_list=self.n_features_list_decode, dtype = self.dtype)

        qkv_features = self.n_features_list_encode[0]

        process_block = []
        MLP_layer = []
        layer_norms1 = []
        layer_norms2 = []

        for _ in range(self.n_message_passes):
            layer = nn.MultiHeadDotProductAttention(num_heads= self.num_heads, qkv_features=qkv_features*self.num_heads)
            MLP_layer.append(ReluMLP(n_features_list=self.n_features_list_encode, dtype = self.dtype))
            layer_norms1.append(nn.LayerNorm())
            layer_norms2.append(nn.LayerNorm())
            process_block.append(layer)

        self.TransformerLayer = process_block
        self.layer_norms1 = layer_norms1
        self.layer_norms2 = layer_norms2
        self.MLP_layer = MLP_layer

    @flax.linen.jit
    def __call__(self, input_dict, X_prev: jnp.ndarray) -> jnp.ndarray:
        """
        @params jraph_graph: graph of type jraph.GraphsTuple

        @returns: decoded nodes after encode-process-decode procedure
        """

        j_graph = input_dict["graphs"][0]
        X_states = X_prev
        X_pos_encoding = input_dict["graphs"][0].nodes

        #original_shape = X_states.shape[:-1]
        X_states = jnp.reshape(X_states, (j_graph.n_node.shape[0], -1 , X_states.shape[-1]))
        X_pos_encoding = jnp.reshape(X_pos_encoding, (j_graph.n_node.shape[0], -1 , X_pos_encoding.shape[-1]))


        overall_input = jnp.concatenate([X_pos_encoding, X_states], axis = -1)
        overall_input = self.node_encoder(overall_input)

        for transformer, ln1, ln2, MLP_layer in zip(self.TransformerLayer, self.layer_norms1, self.layer_norms2, self.MLP_layer):
            transformed_input = transformer(overall_input)
            intermediate_input = ln1(transformed_input + overall_input)

            overall_input = ln1(intermediate_input + MLP_layer(intermediate_input))

        #overall_input = self.node_decoder(overall_input)
        #overall_input = jnp.reshape(overall_input, original_shape + (overall_input.shape[-1],))
        return overall_input
