import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
from Networks.Modules.MLPModules.MLPs import ReluMLP
from Networks.Modules.GNNModules.GAT import MultiheadGraphAttentionNetwork


class TSPModel(nn.Module):
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

	edge_updates: bool
	linear_message_passing: bool = True

	n_message_passes: int = 5
	weight_tied: bool = True
	mean_aggr: bool = False
	graph_norm: bool = False

	def setup(self):
		self.node_encoder = ReluMLP(n_features_list=self.n_features_list_encode)

		self.node_decoder = ReluMLP(n_features_list=self.n_features_list_decode)

		process_block = []

		for _ in range(self.n_message_passes):
			message_passing_layer = MultiheadGraphAttentionNetwork(n_features_list_nodes=self.n_features_list_nodes,
																  n_features_list_messages=self.n_features_list_messages, graph_norm = self.graph_norm)


			process_block.append(message_passing_layer)

		self.process_block = process_block

	@flax.linen.jit
	def __call__(self, jraph_graph_list, X_prev: jnp.ndarray) -> jnp.ndarray:
		"""
		@params jraph_graph: graph of type jraph.GraphsTuple

		@returns: decoded nodes after encode-process-decode procedure
		"""
		nodes = X_prev
		jraph_graph = jraph_graph_list["graphs"][0]
		nodes_encoded = self.node_encoder(nodes)
		jraph_graph = jraph_graph._replace(nodes=nodes_encoded)


		for message_pass in self.process_block:
			jraph_graph = message_pass(jraph_graph)

		decoded_nodes = self.node_decoder(jraph_graph.nodes)
		return decoded_nodes

