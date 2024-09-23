import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from Networks.Modules.EncodeProcessDecode import LinearMessagePassingLayer, NonLinearMessagePassingLayer
from Networks.Modules.MLPs import ReluMLP


class GraphUNet(nn.Module):
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
    n_reps: int = 2
    weight_tied: bool = True
    mean_aggr: bool = False
    growth_factor: float = 1.3

    def setup(self):
        self.node_encoder = ReluMLP(n_features_list=self.n_features_list_encode)
        self.node_decoder = ReluMLP(n_features_list=self.n_features_list_decode)

        process_block = []
        encode_layer_norms = []
        linear_message_passing = self.linear_message_passing
        messages_til_bottleneck = 1#int(self.n_message_passes/2)
        for idx in range(messages_til_bottleneck):
            reps_list = []
            for n_rep in range(self.n_reps):
                growth_factor = self.growth_factor**idx
                if (linear_message_passing or idx == 0):
                    node_features = [int(growth_factor * nh) for nh in self.n_features_list_nodes]
                    message_passing_features = [int(growth_factor * nh) for nh in self.n_features_list_messages]
                    message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=node_features,
                                                                      n_features_list_messages=message_passing_features, mean_aggr = self.mean_aggr)

                else:
                    node_features = self.n_features_list_nodes
                    message_passing_features = self.n_features_list_messages
                    message_passing_layer = NonLinearMessagePassingLayer(
                        n_features_list_nodes=node_features,
                        n_features_list_edges=self.n_features_list_edges,
                        n_features_list_messages=message_passing_features,
                        edge_updates = self.edge_updates,
                     mean_aggr = self.mean_aggr)

                reps_list.append(message_passing_layer)


                message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=node_features,
                                                                  n_features_list_messages=message_passing_features, mean_aggr = self.mean_aggr)

                encode_layer_norms.append(message_passing_layer)

            process_block.append(reps_list)
        self.encode_block = process_block
        self.encode_layer_norms = encode_layer_norms

        process_block = []
        decode_layer_norms = []
        for idx in range(messages_til_bottleneck):
            reps_list = []
            for n_rep in range(self.n_reps):
                growth_factor = self.growth_factor**idx

                if (linear_message_passing or idx == 0):
                    node_features = [int(growth_factor * nh) for nh in self.n_features_list_nodes]
                    message_passing_features = [int(growth_factor * nh) for nh in self.n_features_list_messages]
                    message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=node_features,
                                                                      n_features_list_messages=message_passing_features, mean_aggr = self.mean_aggr)

                else:
                    node_features = self.n_features_list_nodes
                    message_passing_features = self.n_features_list_messages
                    message_passing_layer = NonLinearMessagePassingLayer(
                        n_features_list_nodes= node_features,
                        n_features_list_edges=self.n_features_list_edges,
                        n_features_list_messages=message_passing_features,
                        edge_updates = self.edge_updates,
                     mean_aggr = self.mean_aggr)
                reps_list.append(message_passing_layer)

                message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=node_features,
                                                                  n_features_list_messages=message_passing_features, mean_aggr = self.mean_aggr)


                decode_layer_norms.append(message_passing_layer)

            process_block.append(reps_list)
        self.decode_block = process_block
        self.decode_layer_norms = decode_layer_norms

        bottleneck_layer_list = []
        for n_rep in range(self.n_reps):
            growth_factor = self.growth_factor ** (messages_til_bottleneck-1)
            if linear_message_passing:
                node_features = [int(growth_factor * nh) for nh in self.n_features_list_nodes]
                message_passing_features = [int(growth_factor * nh) for nh in self.n_features_list_messages]
                message_passing_layer = LinearMessagePassingLayer(n_features_list_nodes=node_features,
                                                                  n_features_list_messages=message_passing_features,
                                                                  mean_aggr=self.mean_aggr)

            else:
                message_passing_layer = NonLinearMessagePassingLayer(
                    n_features_list_nodes=node_features,
                    n_features_list_edges=self.n_features_list_edges,
                    n_features_list_messages=message_passing_features,
                    edge_updates=self.edge_updates,
                    mean_aggr=self.mean_aggr)

            bottleneck_layer_list.append(message_passing_layer)
        self.bottleneck_layer_list = bottleneck_layer_list


    def __call__(self, jraph_graph_dict, input_nodes) -> jnp.ndarray:
        """
        @params jraph_graph: graph of type jraph.GraphsTuple

        @returns: decoded nodes after encode-process-decode procedure
        """

        ### TODO add option to make multiple message passing steps after each pooling operation
        jraph_list = jraph_graph_dict["graphs"][0:len(self.encode_block)]
        max_aggr_graph_list = jraph_graph_dict["downsampling_graph"][0:len(self.encode_block)]
        downpooling_graph_list = jraph_graph_dict["downpooling_graph"][0:len(self.encode_block)]
        uppooling_graph_list = jraph_graph_dict["uppooling_graph"][0:len(self.encode_block)]
        upsample_graph_list = jraph_graph_dict["upsampling_graph"][0:len(self.encode_block)]
        bottleneck_graph = jraph_graph_dict["bottleneck_graph"]

        # for key in jraph_graph_dict.keys():
        #     if(key != "bottleneck_graph"):
        #         for graph in jraph_graph_dict[key]:
        #             print(key, type(graph))
        #             print(graph.nodes.shape, graph.edges.shape)
        #print("nodes", input_nodes.shape)
        nodes = input_nodes
        node_features = self.node_encoder(nodes)

        intermediate_downpool_features = []

        for graph, Downsample_graph, downpool_graph, uppool_graph, encode_block, encode_aggr_layer in zip(jraph_list, max_aggr_graph_list, downpooling_graph_list, uppooling_graph_list, self.encode_block, self.encode_layer_norms):
            for GGN_layer in encode_block:
                graph = graph._replace(nodes=node_features)
                node_features = GGN_layer(graph).nodes
            intermediate_downpool_features.append(node_features)

            node_features = Downsample(Downsample_graph, downpool_graph, uppool_graph, node_features, encode_aggr_layer)

        for bottleneck_layer in self.bottleneck_layer_list:
            bottleneck_graph = bottleneck_graph._replace(nodes = node_features)
            node_features = bottleneck_layer(bottleneck_graph).nodes

        for graph, uppool_graph, downpool_graph, upsample_graph, intermediate_features, decode_block, decode_aggr_layer in zip(reversed(jraph_list),
                                                                                     reversed(uppooling_graph_list), reversed(downpooling_graph_list),reversed(upsample_graph_list),
                                                                                    reversed(intermediate_downpool_features), reversed(self.decode_block), reversed(self.decode_layer_norms)):

            node_features = Upsample(upsample_graph, uppool_graph, downpool_graph, node_features, decode_aggr_layer)
            for GGN_layer in decode_block:
                graph = graph._replace(nodes=node_features)
                node_features = GGN_layer(graph).nodes
                node_features = node_features + intermediate_features

        decoded_nodes = self.node_decoder(node_features)
        return decoded_nodes


def Upsampling(H_graph, aggr = "max"):
    nodes = H_graph.nodes
    sum_n_node = H_graph.nodes.shape[0]

    #print("nodes", nodes.shape, H_graph.edges.shape, spins.shape)
    Energy_messages = nodes[H_graph.senders] * nodes[H_graph.receivers]
    #print("energy messages", Energy_messages.shape, jnp.mean(nodes), jnp.mean(Energy_messages), H_graph.receivers.shape)
    if(aggr == "mean"):
        mean_messages = jnp.ones_like(H_graph.edges) * nodes[H_graph.senders] * nodes[H_graph.receivers]
        mean_per_node = jax.ops.segment_sum(mean_messages, H_graph.receivers, sum_n_node)
        mean_per_node = jnp.where(mean_per_node == 0, jnp.ones_like(mean_per_node), mean_per_node)
        Energy_per_node =  jax.ops.segment_sum(Energy_messages, H_graph.receivers, sum_n_node)/ mean_per_node
    elif(aggr == "max"):
        Energy_per_node = jax.ops.segment_max(Energy_messages, H_graph.receivers, sum_n_node)

    return Energy_per_node

def MaxAggr(H_graph, nodes):
    sum_n_node = H_graph.nodes.shape[0]
    Energy_messages = nodes[H_graph.senders] * nodes[H_graph.receivers]
    Energy_per_node =  jax.ops.segment_max(Energy_messages, H_graph.receivers, sum_n_node)

    return Energy_per_node

def Downsample(downsampling_graph, downpooling_graph, uppooling_graph, nodes, aggr_func):

    ### TODO overwrite upsampling nodes
    downsampling_indices = downpooling_graph.nodes
    input_nodes = nodes[uppooling_graph.nodes[...,0]]

    max_aggr_graph = downsampling_graph._replace(nodes = input_nodes)
    updated_graph = aggr_func(max_aggr_graph)
    updated_nodes = updated_graph.nodes

    ### TODO select downsampling nodes
    updated_nodes = updated_nodes[downsampling_indices[..., 0]]
    return updated_nodes

def Upsample(upsampling_graph, uppooling_graph, downpooling_graph, nodes, aggr_func):

    updated_nodes = Downsample(upsampling_graph, uppooling_graph, downpooling_graph, nodes, aggr_func)
    return updated_nodes

