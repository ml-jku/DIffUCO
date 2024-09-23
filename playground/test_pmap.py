
import functools
import logging
import pathlib
import pickle
from typing import Iterator
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import jraph
from jraph.ogb_examples import data_utils
import optax
import os
from jraph_utils import __nearest_bigger_power_of_k
import jraph_utils



if(__name__ == "__main__"):
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,3"
    #jax.config.update('jax_platform_name', 'cpu')
    def device_batch(
            graph_generator) -> Iterator[jraph.GraphsTuple]:
        """Batches a set of graphs the size of the number of devices."""
        num_devices = jax.local_device_count()
        batch = []
        for idx, graph in enumerate(graph_generator):
            if idx % num_devices == num_devices - 1:
                batch.append(graph)
                yield jax.tree_map(lambda *x: jnp.stack(x, axis=0), *batch)
                batch = []
            else:
                batch.append(graph)


    def compute_mean(graph):

        return jnp.mean(graph.nodes)


    list = [1,2,3,4,5,6,7,8,9]
    list[1:3]

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
    graph1 = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
                              edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)

    node_features = jnp.array([[0.], [1.], [2.], [3.]])
    devices = jax.devices()
    print("Available devices:", devices)
    device = jax.device_get(node_features)
    print("device",device)
    array = jnp.arange(4)
    device = jax.device_get(array)
    print(device)
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
    n_node = jnp.array([4])
    n_edge = jnp.array([3])

    # Optionally you can add `global` information, such as a graph label.

    global_context = jnp.array([[1]])
    graph2 = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
                              edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)


    graph1 = jax.device_put(graph1, device = devices[0])
    graph_list = [graph1, graph2]*5

    padded_graph_list = jraph_utils.pad_graphs_to_same_size(graph_list)

    res = device_batch(padded_graph_list)

    print(jax.local_device_count())
    bg = next(res)
    print(bg.nodes.shape)

    res = jax.pmap(compute_mean)(bg)

    print(res)
    ### IDEA pad both graph batches so tha tboth have same amount of ndoes etc.
