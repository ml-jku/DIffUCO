import jax
import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
import jraph
from Networks.Modules.MLPModules.MLPs import ReluMLP, ReluMLP_skip
from Networks.Modules.GNNModules.GraphNorm import GraphNorm

class NonLinearMessagePassingLayer(nn.Module):
	"""
	Non Linear Message Passing

	@param n_features_list_nodes: list of the number of features in the layers (number of nodes) for the node MLP
	@param n_features_list_edges: list of the number of features in the layers (number of nodes) for the edge MLP
	@param n_features_list_messages: list of the number of features in the layers (number of nodes) for the message MLP

	Example for n_features_list_...: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
	"""
	n_features_list_nodes: np.ndarray
	n_features_list_edges: np.ndarray
	n_features_list_messages: np.ndarray
	edge_updates: bool = False
	mean_aggr: bool = False
	graph_norm: bool = False

	def setup(self):
		self.LayerNorm = nn.LayerNorm()
		self.GraphNorm = GraphNorm()

		self.W_node = nn.Dense(features=self.n_features_list_nodes[-1], use_bias=False, kernel_init=nn.initializers.he_normal())
		self.W_edge = nn.Dense(features=self.n_features_list_edges[-1], use_bias=False, kernel_init=nn.initializers.he_normal())
		self.W_message = nn.Dense(features=self.n_features_list_messages[-1], use_bias=False, kernel_init=nn.initializers.he_normal())

		self.NodeMLP = ReluMLP(n_features_list=self.n_features_list_nodes)
		self.EdgeMLP = ReluMLP(n_features_list=self.n_features_list_edges)
		self.MessageMLP = ReluMLP(n_features_list=self.n_features_list_messages)

	@flax.linen.jit
	def __call__(self, jraph_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
		"""
		@params jraph_graph: graph of typpe jraph.GraphsTuple

		@returns: updated jraph graph after message passing step
		"""
		nodes, edges, receivers, senders, _, n_node, n_edges = jraph_graph

		# jitable version to get total number of nodes
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

		sender_features = nodes[senders]
		receiver_features = nodes[receivers]

		messageMLP_input = jnp.concatenate([sender_features, receiver_features, edges], axis=-1)
		messages_out = self.MessageMLP(messageMLP_input)
		aggregated_messages = jax.ops.segment_sum(data=messages_out, segment_ids=receivers, num_segments=total_nodes)

		if(self.mean_aggr):
			norm = jax.ops.segment_sum(data=jnp.ones((messages_out.shape[0],1)), segment_ids=receivers,
													  num_segments=total_nodes)
			norm = jnp.where(norm == 0, jnp.ones_like(norm), norm)
			aggregated_messages = aggregated_messages/(jnp.sqrt(norm))
		if(self.GraphNorm):
			aggregated_messages = self.GraphNorm(jraph_graph, aggregated_messages)

		nodeMLP_input = jnp.concatenate([nodes, aggregated_messages], axis=-1)
		nodes_out = self.NodeMLP(nodeMLP_input)
		nodes_new = self.LayerNorm(self.W_node(nodes) + nodes_out)

		# if(self.edge_updates):
		# 	edges_out = self.EdgeMLP(messageMLP_input)
		# 	edges_new = self.LayerNorm(self.W_edge(edges) + edges_out)
		# 	jraph_graph = jraph_graph._replace(edges=edges_new)

		return jraph_graph._replace(nodes=nodes_new)


class LinearMessagePassingLayer_Residual(nn.Module):
	"""
	Linear Message Passing

	@param n_features_list_nodes: list of the number of features in the layers (number of nodes) for the node MLP
	@param n_features_list_messages: list of the number of features in the layers (number of nodes) for the message MLP

	Example for n_features_list_...: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
	"""
	n_features_list_nodes: np.ndarray
	n_features_list_messages: np.ndarray
	mean_aggr: bool = False
	graph_norm: bool = False

	def setup(self):
		self.LayerNorm = nn.LayerNorm()
		if(self.graph_norm):
			self.GraphNorm = GraphNorm()

		self.W_message = nn.Dense(features=self.n_features_list_messages[-1], use_bias=False, kernel_init=nn.initializers.glorot_normal())
		self.NodeMLP = ReluMLP_skip(n_features_list=self.n_features_list_nodes)
		self.ln_after_aggr = nn.LayerNorm()

	@flax.linen.jit
	def __call__(self, jraph_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
		"""
		@params jraph_graph: graph of typpe jraph.GraphsTuple

		@returns: updated jraph graph after message passing step
		"""
		nodes, edges, receivers, senders, _, n_node, n_edges = jraph_graph

		# jitable version to get total number of nodes
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

		sender_features = nodes[senders]

		messageMLP_input = jnp.concatenate([sender_features, edges], axis=-1)
		messages_out = self.W_message(messageMLP_input)

		aggregated_messages = jax.ops.segment_sum(data=messages_out, segment_ids=receivers, num_segments=total_nodes)

		if(self.mean_aggr):
			norm = jax.ops.segment_sum(data=jnp.ones((messages_out.shape[0],1)), segment_ids=receivers,
													  num_segments=total_nodes)
			norm = jnp.where(norm == 0, jnp.ones_like(norm), norm)
			aggregated_messages = aggregated_messages/(jnp.sqrt(norm))
		if (self.graph_norm):
			aggregated_messages = self.GraphNorm(jraph_graph, aggregated_messages)

		aggregated_messages = self.ln_after_aggr(jax.nn.relu(aggregated_messages) + nodes)
		nodeMLP_input = aggregated_messages
		nodes_out = self.NodeMLP(nodeMLP_input)
		nodes_new = self.LayerNorm(nodes + nodes_out)

		return jraph_graph._replace(nodes=nodes_new)

class LinearMessagePassingLayer(nn.Module):
	"""
	Linear Message Passing

	@param n_features_list_nodes: list of the number of features in the layers (number of nodes) for the node MLP
	@param n_features_list_messages: list of the number of features in the layers (number of nodes) for the message MLP

	Example for n_features_list_...: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
	"""
	n_features_list_nodes: np.ndarray
	n_features_list_messages: np.ndarray
	dtype: any
	mean_aggr: bool = False
	graph_norm: bool = False

	def setup(self):
		self.LayerNorm = nn.LayerNorm(dtype = self.dtype)
		if(self.graph_norm):
			self.GraphNorm = GraphNorm(dtype = self.dtype)

		self.W_node = nn.Dense(features=self.n_features_list_nodes[-1], use_bias=False, kernel_init=nn.initializers.glorot_normal(), dtype = self.dtype)
		self.W_message = nn.Dense(features=self.n_features_list_messages[-1], use_bias=False, kernel_init=nn.initializers.glorot_normal(), dtype = self.dtype)
		self.NodeMLP = ReluMLP(n_features_list=self.n_features_list_nodes, dtype = self.dtype)

	@flax.linen.jit
	def __call__(self, jraph_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
		"""
		@params jraph_graph: graph of typpe jraph.GraphsTuple

		@returns: updated jraph graph after message passing step
		"""
		nodes, edges, receivers, senders, _, n_node, n_edges = jraph_graph

		# jitable version to get total number of nodes
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

		sender_features = nodes[senders]

		messageMLP_input = jnp.concatenate([sender_features, edges], axis=-1)
		messages_out = self.W_message(messageMLP_input)

		aggregated_messages = jax.ops.segment_sum(data=messages_out, segment_ids=receivers, num_segments=total_nodes)
		if(self.mean_aggr):
			norm = jax.ops.segment_sum(data=jnp.ones((messages_out.shape[0],1), dtype = aggregated_messages.dtype), segment_ids=receivers,
													  num_segments=total_nodes)
			norm = jnp.where(norm == 0, jnp.ones_like(norm, dtype = aggregated_messages.dtype), norm)
			aggregated_messages = aggregated_messages/(jnp.sqrt(norm))
		if (self.graph_norm):
			aggregated_messages = self.GraphNorm(jraph_graph, aggregated_messages)

		nodeMLP_input = jnp.concatenate([nodes, aggregated_messages], axis=-1)
		nodes_out = self.NodeMLP(nodeMLP_input)
		nodes_new = self.LayerNorm(self.W_node(nodes) + nodes_out)

		return jraph_graph._replace(nodes=nodes_new)


class EncodeProcessDecode(nn.Module):
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

	def setup(self):
		self.node_encoder = ReluMLP(n_features_list=self.n_features_list_encode, dtype = self.dtype)
		self.edge_encoder = ReluMLP(n_features_list=self.n_features_list_encode, dtype = self.dtype)

		self.node_decoder = ReluMLP(n_features_list=self.n_features_list_decode, dtype = self.dtype)

		process_block = []

		for _ in range(self.n_message_passes):
			if self.linear_message_passing:
				message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=self.n_features_list_nodes,
																  n_features_list_messages=self.n_features_list_messages,
																  mean_aggr = self.mean_aggr, graph_norm = self.graph_norm, dtype = self.dtype)

			else:
				message_passing_layer = NonLinearMessagePassingLayer(
					n_features_list_nodes=self.n_features_list_nodes,
					n_features_list_edges=self.n_features_list_edges,
					n_features_list_messages=self.n_features_list_messages,
					edge_updates = self.edge_updates,
				 mean_aggr = self.mean_aggr, graph_norm = self.graph_norm)

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

		if(self.edge_updates):
			edges = jraph_graph.edges
			edges_encoded = self.edge_encoder(edges)
			jraph_graph = jraph_graph._replace(edges=edges_encoded)

		for message_pass in self.process_block:
			jraph_graph = message_pass(jraph_graph)

		decoded_nodes = self.node_decoder(jraph_graph.nodes)
		return decoded_nodes

class GINConv(nn.Module):
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

	linear_message_passing: bool = True

	n_message_passes: int = 5
	weight_tied: bool = True

	def setup(self):
		self.node_encoder = ReluMLP(n_features_list=self.n_features_list_encode)
		self.edge_encoder = ReluMLP(n_features_list=self.n_features_list_encode)

		self.node_decoder = ReluMLP(n_features_list=self.n_features_list_decode)

		process_block = []
		if self.weight_tied:
			if self.linear_message_passing:
				message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=self.n_features_list_nodes,
																  n_features_list_messages=self.n_features_list_messages)

			else:
				message_passing_layer = NonLinearMessagePassingLayer(n_features_list_nodes=self.n_features_list_nodes,
																	 n_features_list_edges=self.n_features_list_edges,
																	 n_features_list_messages=self.n_features_list_messages)
			for _ in range(self.n_message_passes):
				process_block.append(message_passing_layer)
			self.process_block = process_block

		else:
			for _ in range(self.n_message_passes):
				if self.linear_message_passing:
					message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=self.n_features_list_nodes,
																	  n_features_list_messages=self.n_features_list_messages)

				else:
					message_passing_layer = NonLinearMessagePassingLayer(
						n_features_list_nodes=self.n_features_list_nodes,
						n_features_list_edges=self.n_features_list_edges,
						n_features_list_messages=self.n_features_list_messages)

				process_block.append(message_passing_layer)
			self.process_block = process_block

	@flax.linen.jit
	def __call__(self, jraph_graph: jraph.GraphsTuple) -> jnp.ndarray:
		"""
		@params jraph_graph: graph of type jraph.GraphsTuple

		@returns: decoded nodes after encode-process-decode procedure
		"""
		nodes = jraph_graph.nodes
		edges = jraph_graph.edges

		nodes_encoded = self.node_encoder(nodes)
		edges_encoded = self.edge_encoder(edges)

		jraph_graph = jraph_graph._replace(nodes=nodes_encoded, edges=edges_encoded)

		for message_pass in self.process_block:
			jraph_graph = message_pass(jraph_graph)

		decoded_nodes = self.node_decoder(jraph_graph.nodes)
		return decoded_nodes