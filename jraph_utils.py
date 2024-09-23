import jraph
import jax.numpy as jnp
import numpy as np
import jax
import time


def pad_with_graphs(graph: jraph.GraphsTuple,
                    n_node: int,
                    n_edge: int,
                    n_graph: int = 2) -> jraph.GraphsTuple:
    """Pads a ``GraphsTuple`` to size by adding computation preserving graphs.

    The ``GraphsTuple`` is padded by first adding a dummy graph which contains the
    padding nodes and edges, and then empty graphs without nodes or edges.

    The empty graphs and the dummy graph do not interfer with the graphnet
    calculations on the original graph, and so are computation preserving.

    The padding graph requires at least one node and one graph.

    This function does not support jax.jit, because the shape of the output
    is data-dependent.

    Args:
    graph: ``GraphsTuple`` padded with dummy graph and empty graphs.
    n_node: the number of nodes in the padded ``GraphsTuple``.
    n_edge: the number of edges in the padded ``GraphsTuple``.
    n_graph: the number of graphs in the padded ``GraphsTuple``. Default is 2,
      which is the lowest possible value, because we always have at least one
      graph in the original ``GraphsTuple`` and we need one dummy graph for the
      padding.

    Raises:
    ValueError: if the passed ``n_graph`` is smaller than 2.
    RuntimeError: if the given ``GraphsTuple`` is too large for the given
      padding.

    Returns:
    A padded ``GraphsTuple``.
    """
    if n_graph < 2:
        raise ValueError(f'n_graph is {n_graph}, which is smaller than minimum value of 2.')
    pad_n_node = int(n_node - np.sum(graph.n_node))
    pad_n_edge = int(n_edge - np.sum(graph.n_edge))
    pad_n_graph = int(n_graph - graph.n_node.shape[0])
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            'Given graph is too large for the given padding. difference: '
            f'n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}')

    pad_n_empty_graph = pad_n_graph - 1

    tree_nodes_pad = (
        lambda leaf: np.zeros((pad_n_node,) + leaf.shape[1:], dtype=leaf.dtype))
    tree_edges_pad = (
        lambda leaf: np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype))
    tree_globs_pad = (
        lambda leaf: np.zeros((pad_n_graph,) + leaf.shape[1:], dtype=leaf.dtype))

    padding_graph = jraph.GraphsTuple(
        n_node=np.concatenate(
            [np.array([pad_n_node], dtype=np.int32),
             np.zeros(pad_n_empty_graph, dtype=np.int32)]),
        n_edge=np.concatenate(
            [np.array([pad_n_edge], dtype=np.int32),
             np.zeros(pad_n_empty_graph, dtype=np.int32)]),
        nodes=tree.tree_map(tree_nodes_pad, graph.nodes),
        edges=tree.tree_map(tree_edges_pad, graph.edges),
        globals=tree.tree_map(tree_globs_pad, graph.globals),
        senders=np.zeros(pad_n_edge, dtype=np.int32),
        receivers=np.zeros(pad_n_edge, dtype=np.int32),
    )
    return jraph.batch_np([graph, padding_graph])

def __nearest_bigger_power_of_k(x: int, k: float) -> int:
    """Computes the nearest power of two greater than x for padding."""
    if x == 0:
        return 0
    exponent = np.log(x) / np.log(k)
    return int(k**(int(exponent) + 1))


def pad_graph_to_nearest_power_of_k(graphs_tuple: jraph.GraphsTuple, k = 1.1, np_ = np, pad_func = pad_with_graphs, return_size = False) -> jraph.GraphsTuple:
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = __nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_node), k) + 1
    pad_edges_to = __nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_edge), k) + 1
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1

    if(not return_size):
        return pad_func(graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)
    else:
        return pad_func(graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to), pad_nodes_to, pad_edges_to


def pad_graph_to_size(graphs_tuple: jraph.GraphsTuple, pad_nodes_to, pad_edges_to, np_=jnp, pad_func= pad_with_graphs) -> jraph.GraphsTuple:
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1

    return pad_func(graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)

def add_random_node_features(jraph_graph, n_random_node_features, seed):
    np.random.seed(seed)
    external_fields = jraph_graph.nodes
    random_bin_state = np.random.randint(0, 2, size=(len(external_fields), n_random_node_features))
    #random_bin_state = np.expand_dims(random_bin_state, axis=-1)

    jraph_nodes = np.concatenate((external_fields, random_bin_state), axis=1)
    return jraph_graph._replace(nodes=jraph_nodes)


def calc_pad_number(graphs_tuple,k, np_ = jnp, double_edges = True):
    if(not double_edges):
        p = 2
    else:
        p = 1

    if(k == 1.):
        pad_nodes_to = np_.sum(graphs_tuple.n_node) + 1
        pad_edges_to = np_.sum(graphs_tuple.n_edge) + 1
    else:
        pad_nodes_to = __nearest_bigger_power_of_k(np_.sum(graphs_tuple.n_node), k) + 1
        pad_edges_to = __nearest_bigger_power_of_k(p*np_.sum(graphs_tuple.n_edge), k)
    return pad_nodes_to, pad_edges_to

def calc_pad_number_from_statistics(graphs_tuple, dataset_statistics_dict, np_ = jnp):

    min_edges = dataset_statistics_dict["min_edges"]
    max_edges = dataset_statistics_dict["max_edges"]

    min_nodes = dataset_statistics_dict["min_nodes"]
    max_nodes = dataset_statistics_dict["max_nodes"]

    grid_num = dataset_statistics_dict["grid_num"]

    pad_nodes_to = _nearest_number_of_min_max(min_nodes, max_nodes, np_.sum(graphs_tuple.n_node), grid_num, len(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_number_of_min_max(min_edges, max_edges, np_.sum(graphs_tuple.n_edge), grid_num, len(graphs_tuple.n_node)) + 1
    return pad_nodes_to, pad_edges_to

def _nearest_number_of_min_max(min_value, max_value, graph_value, grid_num, n_graphs):
    if(min_value == max_value):
        return max_value
    elif(n_graphs == 1):
        return max_value
    else:
        grid_candidates = np.linspace(min_value, max_value, grid_num, endpoint=True)
        return int(find_smallest_greater_value(grid_candidates, graph_value))

def find_smallest_greater_value(grid_candidates, value):
    mask = grid_candidates >= value
    if np.any(mask):
        return np.min(grid_candidates[mask])
    else:
        return None

### this can be done in collate function
def pad_graphs_to_same_size(graphs_list, k = 1.1, pad_func = pad_with_graphs, double_edges = True):
    max_pad_nodes_to = 1
    max_pad_edges_to = 1
    for graph in graphs_list:
        pad_nodes_to, pad_edges_to = calc_pad_number(graph, k, double_edges = double_edges)
        max_pad_nodes_to = max([max_pad_nodes_to, pad_nodes_to])
        max_pad_edges_to = max([max_pad_edges_to, pad_edges_to])

    max_pad_nodes_to = np.max([max_pad_nodes_to, 1])
    # print("here")
    # for graph in graphs_list:
    #     print(len(graphs_list), )
    #     print((graph.nodes.shape, graph.edges.shape, pad_nodes_to, pad_edges_to, graph.n_node.shape[0] + 1))
    #     print(jraph.pad_with_graphs(graph, pad_nodes_to, pad_edges_to, graph.n_node.shape[0] + 1))
    #     print("finished")
    padded_graph_list = [pad_func(graph, max_pad_nodes_to, max_pad_edges_to, graph.n_node.shape[0] + 1) for graph in graphs_list]
    return padded_graph_list, max_pad_nodes_to, max_pad_edges_to

def pad_graphs_to_same_size_from_statistics(graphs_list, dataset_statistics_dict, pad_func = pad_with_graphs):
    max_pad_nodes_to = 1
    max_pad_edges_to = 1
    for graph in graphs_list:
        pad_nodes_to, pad_edges_to = calc_pad_number_from_statistics(graph, dataset_statistics_dict)
        max_pad_nodes_to = max([max_pad_nodes_to, pad_nodes_to])
        max_pad_edges_to = max([max_pad_edges_to, pad_edges_to])

    max_pad_nodes_to = np.max([max_pad_nodes_to, 1])
    # print("here")
    # for graph in graphs_list:
    #     print(len(graphs_list), )
    #     print((graph.nodes.shape, graph.edges.shape, pad_nodes_to, pad_edges_to, graph.n_node.shape[0] + 1))
    #     print(jraph.pad_with_graphs(graph, pad_nodes_to, pad_edges_to, graph.n_node.shape[0] + 1))
    #     print("finished")
    padded_graph_list = [pad_func(graph, max_pad_nodes_to, max_pad_edges_to, graph.n_node.shape[0] + 1) for graph in graphs_list]
    return padded_graph_list, max_pad_nodes_to, max_pad_edges_to

def pad_graphs_to(graphs_list, max_pad_nodes_to, max_pad_edges_to, pad_func = pad_with_graphs):
    padded_graph_list = [pad_func(graph, max_pad_nodes_to, max_pad_edges_to, graph.n_node.shape[0] + 1) for graph in graphs_list]
    return padded_graph_list, max_pad_nodes_to, max_pad_edges_to


def device_batch(graph_generator, np_ = np):
    """Batches a set of graphs the size of the number of devices."""
    num_devices = jax.local_device_count()
    batch = []
    for idx, graph in enumerate(graph_generator):
        if idx % num_devices == num_devices - 1:
            batch.append(graph)
            yield jax.tree_map(lambda *x: np_.stack(x, axis=0), *batch) ### TODO text wheter numpy or jnp is better here
            batch = []
        else:
            batch.append(graph)


def pmap_transformer_list(jraph_graph_list, k = 1.2, pad_func = pad_with_graphs, return_size = False, double_edges = True):
    n_devices = jax.local_device_count()
    n_graphs_per_device = int(len(jraph_graph_list) / n_devices)
    # if (len(jraph_graph_list) % n_devices != 0):
    #     print("batchsize", len(jraph_graph_list))
    #     print("n_devices", n_devices)
    #     raise ValueError("batchisze must be devisible by number of devices")
    device_batched_graphs = [jraph.batch_np(jraph_graph_list[idx * n_graphs_per_device: (idx + 1) * n_graphs_per_device] )
                             for idx in range(n_devices)] ### TODO move this to collate function

    device_batched_graphs = [jraph.batch_np([graph, jraph_graph_list[0]] )
                             for graph in device_batched_graphs]

    device_batched_graphs = next(device_batch(device_batched_graphs))
    # print("make list", step2-step1)
    # print("pad graphs", step3-step2)
    # print("next generator", step4-step3)
    return device_batched_graphs

def pmap_graph_list(jraph_graph_list, k = 1.2, pad_func = pad_with_graphs, return_size = False, double_edges = True):
    n_devices = jax.local_device_count()
    n_graphs_per_device = int(len(jraph_graph_list) / n_devices)
    # if (len(jraph_graph_list) % n_devices != 0):
    #     print("batchsize", len(jraph_graph_list))
    #     print("n_devices", n_devices)
    #     raise ValueError("batchisze must be devisible by number of devices")
    device_batched_graphs = [jraph.batch_np(jraph_graph_list[idx * n_graphs_per_device: (idx + 1) * n_graphs_per_device])
                             for idx in range(n_devices)] ### TODO move this to collate function


    padded_graph_list, max_pad_nodes_to, max_pad_edges_to = pad_graphs_to_same_size(device_batched_graphs, k = k, pad_func = pad_func, double_edges = double_edges)
    device_batched_graphs = next(device_batch(padded_graph_list))
    # print("make list", step2-step1)
    # print("pad graphs", step3-step2)
    # print("next generator", step4-step3)
    if(return_size):
        return device_batched_graphs, max_pad_nodes_to, max_pad_edges_to
    else:
        return device_batched_graphs

def pmap_graph_list_better(jraph_graph_list, dataset_statistics_dict, pad_func = pad_with_graphs, return_size = False):
    n_devices = jax.local_device_count()
    n_graphs_per_device = int(len(jraph_graph_list) / n_devices)
    # if (len(jraph_graph_list) % n_devices != 0):
    #     print("batchsize", len(jraph_graph_list))
    #     print("n_devices", n_devices)
    #     raise ValueError("batchisze must be devisible by number of devices")
    device_batched_graphs = [jraph.batch_np(jraph_graph_list[idx * n_graphs_per_device: (idx + 1) * n_graphs_per_device])
                             for idx in range(n_devices)] ### TODO move this to collate function


    padded_graph_list, max_pad_nodes_to, max_pad_edges_to = pad_graphs_to_same_size_from_statistics(device_batched_graphs, dataset_statistics_dict, pad_func = pad_func)
    device_batched_graphs = next(device_batch(padded_graph_list))
    # print("make list", step2-step1)
    # print("pad graphs", step3-step2)
    # print("next generator", step4-step3)
    if(return_size):
        return device_batched_graphs, max_pad_nodes_to, max_pad_edges_to
    else:
        return device_batched_graphs

def pmap_graph_list_to(jraph_graph_list, pad_nodes_to, pad_edges_to, pad_func = pad_with_graphs):
    n_devices = jax.local_device_count()
    n_graphs_per_device = int(len(jraph_graph_list) / n_devices)


    device_batched_graphs = [jraph.batch_np(jraph_graph_list[idx * n_graphs_per_device: (idx + 1) * n_graphs_per_device])
                             for idx in range(n_devices)] ### TODO move this to collate function

    padded_graph_list = pad_graphs_to(device_batched_graphs, pad_nodes_to, pad_edges_to, pad_func = pad_func)
    device_batched_graphs = next(device_batch(padded_graph_list))[0]


    return device_batched_graphs

def cast_Tuple_to(j_graph, np_ = jnp):

    j_graph = jraph.GraphsTuple(nodes= np_.asarray(j_graph.nodes), edges=np_.asarray(j_graph.edges), receivers=np_.asarray(j_graph.receivers),
                                          senders=np_.asarray(j_graph.senders),
                                          n_node= np_.asarray(j_graph.n_node),
                                          n_edge= np_.asarray(j_graph.n_edge), globals= np_.asarray(j_graph.globals) )

    return j_graph

def cast_Tuple_to_float32(j_graph, np_ = jnp):
    ### TODO check if replace is faster
    j_graph = jraph.GraphsTuple(nodes= np_.asarray(j_graph.nodes, dtype=np_.float32), edges=np_.asarray(j_graph.edges, dtype=np_.float32), receivers=np_.asarray(j_graph.receivers, np_.int32),
                                          senders=np_.asarray(j_graph.senders, np_.int32),
                                          n_node= np_.asarray(j_graph.n_node, np_.int32),
                                          n_edge= np_.asarray(j_graph.n_edge, np_.int32), globals= np_.asarray(j_graph.globals, np_.float32) )

    return j_graph

import jax.tree_util as tree
def shift_node_graph_index(pooling_graph, axis = 0, _np = np):
    #print(type(pooling_graph))
    nodes = pooling_graph.nodes
    n_node = pooling_graph.n_node
    sum_n_node = nodes.shape[axis]

    # for el in pooling_graph:
    #     print(type(el))
    #     if(not type(el).__module__ == np.__name__):
    #         raise ValueError("before not numpy")

    if(axis == 0):
        node_offset = _np.array([0] + list(_np.cumsum(pooling_graph.globals, axis = axis))[:-1])
        node_offset_per_node = _np.array(np.repeat(node_offset, n_node, axis=axis))
    else:
        # for i in range(nodes.shape[0]):
        #     nodes = nodes.at[i,nodes.shape[axis]- n_node[-1]:].set(jnp.arange(0, n_node[i, -1]))
        node_offset = _np.concatenate([jnp.zeros(pooling_graph.globals.shape[:-1] + (1,), dtype= _np.int32), _np.cumsum(pooling_graph.globals, axis=axis)[:,:-1]], axis = -1)

        #vectorized_repeat = np.vectorize(lambda x, y: np.repeat(x, y))

        # Apply the vectorized function to each pair of elements in arr and repeat_numbers
        node_offset_per_node = _np.array(jnp.repeat(node_offset, n_node, axis=axis))

    shifted_independent_indices = node_offset_per_node[..., None] + nodes
    pooling_graph = pooling_graph._replace(nodes = shifted_independent_indices)

    # for el in pooling_graph:
    #     print(type(el))
    #     if(not type(el).__module__ == np.__name__):
    #         raise ValueError("not numpy")

    return pooling_graph

#### TODO find out how to pmap this!
def batch_U_net_graph_dict(U_net_graph_dict_list, k = 1.3):
    keys = U_net_graph_dict_list[0].keys()
    batched_U_net_graph_dict = {key: [U_net_graph_dict_el[key] for U_net_graph_dict_el in U_net_graph_dict_list] for key in keys}


    #### TODO within each layer split list into n devices and batch and padd to nearest
    n_layers = len(batched_U_net_graph_dict["graphs"][0])
    batched_layer_dict = {"graphs": {}, "upsampling_graph": {}, "pooling_graph": {}, "max_aggr_graph": {}}
    for n_layer in range(n_layers):
        batched_layer_dict["graphs"][f"layer_{n_layer}"] = []
        batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = []
        batched_layer_dict["pooling_graph"][f"layer_{n_layer}"] = []
        batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"] = []

        for key in keys:
            if( "bottleneck_graph" != key):
                for el in batched_U_net_graph_dict[key]:
                    batched_layer_dict[key][f"layer_{n_layer}"].append(el[n_layer])


        k_layer = min([k + 1*n_layer, 4])
        k_layer_p_1 = min([k + 1*n_layer + 1, 4])
        if(n_layer == 0):
            batched_layer_dict["graphs"][f"layer_{n_layer}"] = pad_graph_to_nearest_power_of_k(jraph.batch_np(batched_layer_dict["graphs"][f"layer_{n_layer}"]), k = k_layer)

            batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = pad_graph_to_nearest_power_of_k(jraph.batch_np(batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"]), pad_func=pad_pool_graph , k = k_layer)
            #print([type(el) for el in batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]])
            #print([len(el.nodes) for el in batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]])
            pooled_graph, pad_pool_nodes_to, pad_pool_edges_to = pad_graph_to_nearest_power_of_k(jraph.batch_np(batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]), pad_func = pad_pool_graph,  k = k_layer_p_1, return_size=True)
            batched_layer_dict["pooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pooled_graph)
            batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"] = pad_graph_to_nearest_power_of_k(jraph.batch_np(batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"]), pad_func= pad_pool_graph, k = k_layer)
        elif(n_layer == 1):
            batched_layer_dict["graphs"][f"layer_{n_layer}"], pad_graph_nodes_to, pad_graph_edges_to = pad_graph_to_nearest_power_of_k(jraph.batch_np(batched_layer_dict["graphs"][f"layer_{n_layer}"]), k = k_layer, return_size=True)
            batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"], pad_up_nodes_to, pad_up_edges_to = pad_graph_to_nearest_power_of_k(jraph.batch_np(batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"]), pad_func=pad_pool_graph , k = k_layer, return_size=True)
            #print([type(el) for el in batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]])
            #print([len(el.nodes) for el in batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]])
            pooled_graph = pad_graph_to_size(jraph.batch_np(batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]), pad_pool_nodes_to, pad_pool_edges_to, pad_func = pad_pool_graph)
            batched_layer_dict["pooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pooled_graph)
            batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"], pad_max_nodes_to, pad_max_edges_to = pad_graph_to_nearest_power_of_k(jraph.batch_np(batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"]), pad_func= pad_pool_graph, k = k_layer, return_size=True)
        else:
            batched_layer_dict["graphs"][f"layer_{n_layer}"] = pad_graph_to_size(jraph.batch_np(batched_layer_dict["graphs"][f"layer_{n_layer}"]), pad_graph_nodes_to, pad_graph_edges_to)
            batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = pad_graph_to_size(jraph.batch_np(batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"]), pad_up_nodes_to, pad_up_edges_to, pad_func=pad_pool_graph )
            #print([type(el) for el in batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]])
            #print([len(el.nodes) for el in batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]])
            pooled_graph = pad_graph_to_size(jraph.batch_np(batched_layer_dict["pooling_graph"][f"layer_{n_layer}"]), pad_pool_nodes_to, pad_pool_edges_to, pad_func = pad_pool_graph)
            batched_layer_dict["pooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pooled_graph)
            batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"] = pad_graph_to_size(jraph.batch_np(batched_layer_dict["max_aggr_graph"][f"layer_{n_layer}"]), pad_max_nodes_to, pad_max_edges_to, pad_func= pad_pool_graph)

    batched_U_net_graph_dict["graphs"] = [batched_layer_dict["graphs"][key] for key in batched_layer_dict["graphs"]]
    batched_U_net_graph_dict["upsampling_graph"] = [batched_layer_dict["upsampling_graph"][key] for key in batched_layer_dict["upsampling_graph"]]
    batched_U_net_graph_dict["pooling_graph"] = [batched_layer_dict["pooling_graph"][key] for key in batched_layer_dict["pooling_graph"]]
    batched_U_net_graph_dict["max_aggr_graph"] = [batched_layer_dict["max_aggr_graph"][key] for key in batched_layer_dict["max_aggr_graph"]]
    batched_U_net_graph_dict["bottleneck_graph"] = [pad_graph_to_size(jraph.batch_np(batched_U_net_graph_dict["bottleneck_graph"]), pad_graph_nodes_to, pad_graph_edges_to)]

    return batched_U_net_graph_dict


def pmap_batch_U_net_graph_dict_and_pad(U_net_graph_dict_list, k = 1.2):
    ### TODO change this to new U_net graph
    keys = U_net_graph_dict_list[0].keys()
    batched_U_net_graph_dict = {key: [U_net_graph_dict_el[key] for U_net_graph_dict_el in U_net_graph_dict_list] for key
                                in keys}

    #### TODO within each layer split list into n devices and batch and padd to nearest
    n_layers = len(batched_U_net_graph_dict["graphs"][0])
    batched_layer_dict = {"graphs": {}, "upsampling_graph": {}, "uppooling_graph": {}, "downpooling_graph": {}, "downsampling_graph": {}}


    for n_layer in range(n_layers):
        batched_layer_dict["graphs"][f"layer_{n_layer}"] = []
        batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = []
        batched_layer_dict["downpooling_graph"][f"layer_{n_layer}"] = []
        batched_layer_dict["uppooling_graph"][f"layer_{n_layer}"] = []
        batched_layer_dict["downsampling_graph"][f"layer_{n_layer}"] = []

        for key in keys:
            if ("bottleneck_graph" != key):
                for el in batched_U_net_graph_dict[key]:
                    batched_layer_dict[key][f"layer_{n_layer}"].append(el[n_layer])

                    # for graph_key in el[n_layer]:
                    #     print(key)
                    #     print(type(graph_key), graph_key.shape)
                    #     if(not type(graph_key).__module__ == np.__name__):
                    #         raise ValueError("not numpy")

        k_layer = min([k + n_layer, 2])
        k_layer_p_1 = min([k + (n_layer + 1) , 2])
        #print("vefore", [el.nodes.shape for el in batched_layer_dict["graphs"][f"layer_{n_layer}"]], k_layer)
        if(True):#if(n_layer == 0):
            batched_layer_dict["graphs"][f"layer_{n_layer}"] = pmap_graph_list(batched_layer_dict["graphs"][f"layer_{n_layer}"], k = k_layer)
            ### TODO pad all other to 1/4 nodes and 1/4 edges of graphs
            #print("after", batched_layer_dict["graphs"][f"layer_{n_layer}"].nodes.shape)
            batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = pmap_graph_list(batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"], k = k_layer, double_edges = False, pad_func=pad_max_graph)
            pooling_graph, pad_pool_nodes_to, pad_pool_edges_to = pmap_graph_list(batched_layer_dict["downpooling_graph"][f"layer_{n_layer}"], k=k_layer_p_1, pad_func=pad_pool_graph, return_size=True)
            batched_layer_dict["downpooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pooling_graph, axis = 1)
            pooling_graph, pad_pool_nodes_to, pad_pool_edges_to = pmap_graph_list(batched_layer_dict["uppooling_graph"][f"layer_{n_layer}"], k=k_layer, pad_func=pad_pool_graph, return_size=True)
            batched_layer_dict["uppooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pooling_graph, axis = 1)
            batched_layer_dict["downsampling_graph"][f"layer_{n_layer}"] = pmap_graph_list(batched_layer_dict["downsampling_graph"][f"layer_{n_layer}"], k = k_layer, double_edges = False, pad_func=pad_max_graph)
        elif(n_layer == 1):
            batched_layer_dict["graphs"][f"layer_{n_layer}"], pad_graph_nodes_to, pad_graph_edges_to  = pmap_graph_list(batched_layer_dict["graphs"][f"layer_{n_layer}"], k = k_layer, return_size=True)
            #print("after", batched_layer_dict["graphs"][f"layer_{n_layer}"].nodes.shape)
            batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"], pad_up_nodes_to, pad_up_edges_to = pmap_graph_list(batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"], k = k_layer, double_edges = False, pad_func=pad_max_graph, return_size=True)
            pooled_graph = pmap_graph_list_to(batched_layer_dict["downpooling_graph"][f"layer_{n_layer}"], pad_pool_nodes_to, pad_pool_edges_to, pad_func=pad_pool_graph)
            batched_layer_dict["downpooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pooled_graph, axis = 1)
            pooled_graph = pmap_graph_list_to(batched_layer_dict["uppooling_graph"][f"layer_{n_layer}"], pad_pool_nodes_to, pad_pool_edges_to, pad_func=pad_pool_graph)
            batched_layer_dict["uppooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pooled_graph, axis = 1)
            batched_layer_dict["downsampling_graph"][f"layer_{n_layer}"], pad_max_nodes_to, pad_max_edges_to = pmap_graph_list(batched_layer_dict["downsampling_graph"][f"layer_{n_layer}"], k = k_layer, double_edges = False, pad_func=pad_max_graph, return_size=True)
        else:
            batched_layer_dict["graphs"][f"layer_{n_layer}"] = pmap_graph_list_to(batched_layer_dict["graphs"][f"layer_{n_layer}"], pad_graph_nodes_to, pad_graph_edges_to )
            #print("after", batched_layer_dict["graphs"][f"layer_{n_layer}"].nodes.shape)
            batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"] = pmap_graph_list_to(batched_layer_dict["upsampling_graph"][f"layer_{n_layer}"], pad_up_nodes_to, pad_up_edges_to, pad_func=pad_max_graph)
            batched_layer_dict["downpooling_graph"][f"layer_{n_layer}"] = shift_node_graph_index(pmap_graph_list_to(batched_layer_dict["downpooling_graph"][f"layer_{n_layer}"],  pad_pool_nodes_to, pad_pool_edges_to, pad_func=pad_pool_graph), axis = 1)
            batched_layer_dict["downsampling_graph"][f"layer_{n_layer}"] = pmap_graph_list_to(batched_layer_dict["downsampling_graph"][f"layer_{n_layer}"], pad_max_nodes_to, pad_max_edges_to, pad_func=pad_max_graph)
            batched_layer_dict["uppooling_graph"][f"layer_{n_layer}"] = pmap_graph_list_to(batched_layer_dict["uppooling_graph"][f"layer_{n_layer}"], pad_max_nodes_to, pad_max_edges_to, pad_func=pad_max_graph)

    batched_U_net_graph_dict["graphs"] = [batched_layer_dict["graphs"][key] for key in batched_layer_dict["graphs"]]
    batched_U_net_graph_dict["upsampling_graph"] = [batched_layer_dict["upsampling_graph"][key] for key in
                                                    batched_layer_dict["upsampling_graph"]]
    batched_U_net_graph_dict["downpooling_graph"] = [batched_layer_dict["downpooling_graph"][key] for key in
                                                 batched_layer_dict["downpooling_graph"]]
    batched_U_net_graph_dict["uppooling_graph"] = [batched_layer_dict["uppooling_graph"][key] for key in
                                                 batched_layer_dict["uppooling_graph"]]
    batched_U_net_graph_dict["downsampling_graph"] = [batched_layer_dict["downsampling_graph"][key] for key in
                                                  batched_layer_dict["downsampling_graph"]]
    batched_U_net_graph_dict["bottleneck_graph"] = pmap_graph_list(batched_U_net_graph_dict["bottleneck_graph"], k = k_layer)

    #batched_U_net_graph_dict["bottleneck_graph"] = pmap_graph_list_to(batched_U_net_graph_dict["bottleneck_graph"], pad_graph_nodes_to, pad_graph_edges_to )

    # print("here")
    # for key in batched_U_net_graph_dict.keys():
    #     if (key != "bottleneck_graph"):
    #         for graph in batched_U_net_graph_dict[key]:
    #             print(key, type(graph))
    #             print(graph.nodes.shape, graph.edges.shape)
    #     else:
    #         graph = batched_U_net_graph_dict[key]
    #         print(key, type(graph))
    #         print(graph.nodes.shape, graph.edges.shape)

    return batched_U_net_graph_dict

def pad_pool_graph(graph,
                    n_node: int,
                    n_edge: int,
                    n_graph: int = 2):
  """Pads a ``GraphsTuple`` to size by adding computation preserving graphs.

  The ``GraphsTuple`` is padded by first adding a dummy graph which contains the
  padding nodes and edges, and then empty graphs without nodes or edges.

  The empty graphs and the dummy graph do not interfer with the graphnet
  calculations on the original graph, and so are computation preserving.

  The padding graph requires at least one node and one graph.

  This function does not support jax.jit, because the shape of the output
  is data-dependent.

  Args:
    graph: ``GraphsTuple`` padded with dummy graph and empty graphs.
    n_node: the number of nodes in the padded ``GraphsTuple``.
    n_edge: the number of edges in the padded ``GraphsTuple``.
    n_graph: the number of graphs in the padded ``GraphsTuple``. Default is 2,
      which is the lowest possible value, because we always have at least one
      graph in the original ``GraphsTuple`` and we need one dummy graph for the
      padding.

  Raises:
    ValueError: if the passed ``n_graph`` is smaller than 2.
    RuntimeError: if the given ``GraphsTuple`` is too large for the given
      padding.

  Returns:
    A padded ``GraphsTuple``.
  """
  if n_graph < 2:
    raise ValueError(
        f'n_graph is {n_graph}, which is smaller than minimum value of 2.')
  #graph = jax.device_get(graph)
  pad_n_node = int(n_node - np.sum(graph.n_node))
  pad_n_edge = 0 #np.max([int(n_edge - np.sum(graph.n_edge)),pad_n_node])
  pad_n_graph = int(n_graph - graph.n_node.shape[0])
  if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
    raise RuntimeError(
        'Given graph is too large for the given padding. difference: '
        f'n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}')

  pad_n_empty_graph = pad_n_graph - 1

  tree_nodes_pad = (
      lambda leaf: np.arange(0, pad_n_node,  dtype=leaf.dtype)[:,None])
  tree_edges_pad = (
      lambda leaf: np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype))
  tree_globs_pad = (
      lambda leaf: np.zeros((pad_n_graph,) + leaf.shape[1:], dtype=leaf.dtype))

  padding_graph = jraph.GraphsTuple(
      n_node=np.concatenate(
          [np.array([pad_n_node], dtype=np.int32),
           np.arange(pad_n_empty_graph, dtype=np.int32)]),
      n_edge=np.concatenate(
          [np.array([pad_n_edge], dtype=np.int32),
           np.arange(pad_n_empty_graph, dtype=np.int32)]),
      nodes=tree.tree_map(tree_nodes_pad, graph.nodes),
      edges=tree.tree_map(tree_edges_pad, graph.edges),
      globals=tree.tree_map(tree_globs_pad, graph.globals),
      senders=np.clip(np.arange(pad_n_edge, dtype=np.int32), 0, pad_n_edge-1),
      receivers=np.clip(np.arange(pad_n_edge, dtype=np.int32), 0, pad_n_edge-1),
  )

  return jraph.batch_np([graph, padding_graph])

def pad_max_graph(graph,
                    n_node: int,
                    n_edge: int,
                    n_graph: int = 2):
  """Pads a ``GraphsTuple`` to size by adding computation preserving graphs.

  The ``GraphsTuple`` is padded by first adding a dummy graph which contains the
  padding nodes and edges, and then empty graphs without nodes or edges.

  The empty graphs and the dummy graph do not interfer with the graphnet
  calculations on the original graph, and so are computation preserving.

  The padding graph requires at least one node and one graph.

  This function does not support jax.jit, because the shape of the output
  is data-dependent.

  Args:
    graph: ``GraphsTuple`` padded with dummy graph and empty graphs.
    n_node: the number of nodes in the padded ``GraphsTuple``.
    n_edge: the number of edges in the padded ``GraphsTuple``.
    n_graph: the number of graphs in the padded ``GraphsTuple``. Default is 2,
      which is the lowest possible value, because we always have at least one
      graph in the original ``GraphsTuple`` and we need one dummy graph for the
      padding.

  Raises:
    ValueError: if the passed ``n_graph`` is smaller than 2.
    RuntimeError: if the given ``GraphsTuple`` is too large for the given
      padding.

  Returns:
    A padded ``GraphsTuple``.
  """
  if n_graph < 2:
    raise ValueError(
        f'n_graph is {n_graph}, which is smaller than minimum value of 2.')
  #graph = jax.device_get(graph)
  pad_n_node = int(n_node - np.sum(graph.n_node))
  pad_n_edge = np.max([int(n_edge - np.sum(graph.n_edge)),pad_n_node])
  pad_n_graph = int(n_graph - graph.n_node.shape[0])
  if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
    raise RuntimeError(
        'Given graph is too large for the given padding. difference: '
        f'n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}')

  pad_n_empty_graph = pad_n_graph - 1

  tree_nodes_pad = (
      lambda leaf: np.arange(0, pad_n_node,  dtype=leaf.dtype)[:,None])
  tree_edges_pad = (
      lambda leaf: np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype))
  tree_globs_pad = (
      lambda leaf: np.zeros((pad_n_graph,) + leaf.shape[1:], dtype=leaf.dtype))

  padding_graph = jraph.GraphsTuple(
      n_node=np.concatenate(
          [np.array([pad_n_node], dtype=np.int32),
           np.arange(pad_n_empty_graph, dtype=np.int32)]),
      n_edge=np.concatenate(
          [np.array([pad_n_edge], dtype=np.int32),
           np.arange(pad_n_empty_graph, dtype=np.int32)]),
      nodes=tree.tree_map(tree_nodes_pad, graph.nodes),
      edges=tree.tree_map(tree_edges_pad, graph.edges),
      globals=tree.tree_map(tree_globs_pad, graph.globals),
      senders=np.clip(np.arange(pad_n_edge, dtype=np.int32), 0, pad_n_edge-1),
      receivers=np.clip(np.arange(pad_n_edge, dtype=np.int32), 0, pad_n_edge-1),
  )

  return jraph.batch_np([graph, padding_graph])


def _map_split(nest, indices_or_sections):
    """Splits leaf nodes of nests and returns a list of nests."""
    if isinstance(indices_or_sections, int):
        n_lists = indices_or_sections
    else:
        n_lists = len(indices_or_sections) + 1
    concat = lambda field: np.split(field, indices_or_sections)
    nest_of_lists = jax.tree_util.tree_map(concat, nest)
    # pylint: disable=cell-var-from-loop
    list_of_nests = [
        jax.tree_util.tree_map(lambda _, x: x[i], nest, nest_of_lists)
        for i in range(n_lists)
    ]
    return list_of_nests

def unpmap_graph(n_node, nodes, idx):
    node_offsets = np.cumsum(n_node[idx, :-1])
    all_nodes = _map_split(nodes[idx, ...], node_offsets)[:-1]

    return all_nodes