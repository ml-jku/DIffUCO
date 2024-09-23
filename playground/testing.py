import jax
import jax.numpy as jnp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

def extract():
	data = """2632049 pts/4    00:00:02 python
			2632112 pts/4    00:00:01 python
			2632175 pts/4    00:00:01 python
			2632238 pts/4    00:00:01 python
			2632301 pts/4    00:00:01 python
			2632364 pts/4    00:00:01 python
			2632432 pts/4    00:00:01 python
			2632498 pts/4    00:00:01 python
			2663695 pts/3    00:00:06 python
			2663758 pts/3    00:00:05 python
			2663821 pts/3    00:00:06 python
			2663884 pts/3    00:00:06 python
			2663947 pts/3    00:00:06 python
			2664010 pts/3    00:00:06 python
			2664074 pts/3    00:00:06 python
			2664137 pts/3    00:00:05 python"""

	first_columns = [line.split()[0] for line in data.strip().split('\n')]

	# Print the result
	overall_str = ""
	for col in first_columns:
		overall_str += " " + col

	print("kill -KILL " + overall_str)
	print("asdasd")

import time
def batched_jraph_iteration(graphs):
	nodes = graphs.nodes
	n_node = graphs.n_node
	n_graph = graphs.n_node.shape[0]
	graph_idx = jnp.arange(n_graph)
	total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
	sum_n_node = graphs.nodes.shape[0]
	node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
	cum_sum = jnp.concatenate([jnp.array([0]),jax.lax.cumsum(n_node)[:-1]], axis = 0)
	cum_max_sum = jax.lax.cumsum(n_node)

	input_tuple = (graphs,0)
	max_val = int(jnp.max(graphs.n_node))

	nodes = nodes.at[cum_sum, jnp.zeros_like(cum_sum)].set(jnp.ones_like(cum_sum))

	ps = jnp.array(np.random.uniform(0, 1, (graphs.nodes.shape[0], 1)))
	ConditionalExpectation_test(graphs, ps, cum_sum, cum_max_sum)

	batch_size = 10
	ps = np.random.uniform(0, 1, (batch_size, graphs.nodes.shape[0], 1))
	print(ps.shape)

	#vmapped_CE = jax.vmap(lambda a,b,c,d: ConditionalExpectation_v2(a,b,c,d, max_steps= max_val), in_axes=(None, 0, None, None))
	vmapped_CE = jax.vmap(lambda a,b: ConditionalExpectation_v2(a,b), in_axes=(None, 0))
	vmapped_calc_energy = jax.vmap(calc_energy, in_axes=(None, 0))

	### TODO find out whether violations can occur
	reps = 5
	for rep in range(reps):
		### TODO vmap this
		start_time = time.time()
		(_, best_ps, _, _, _, _) = vmapped_CE(graphs, ps)
		end_time = time.time()
		print(ps.shape)
		print("time", end_time - start_time )
	#best_ps = ps
	Energy, Energy_per_node, HB_per_node = vmapped_calc_energy(graphs, best_ps)
	print("final Energies", Energy[0])
	# for i in range(graphs.nodes.shape[0]):
	# 	print(i, best_ps[0,i], Energy_per_node[0,i], HB_per_node[0,i])
	# print("here")
	pass

from functools import partial

@jax.jit
def calc_energy(H_graph, bins, A = 1., B = 1.1):
	nodes = H_graph.nodes
	n_node = H_graph.n_node
	n_graph = H_graph.n_node.shape[0]
	graph_idx = jnp.arange(n_graph)
	total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
	node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

	raveled_bins = bins
	Energy_messages = (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])

	# print("Energy_per_graph", Energy.shape)
	HA_per_node = - A * raveled_bins
	HB_per_node = B * jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_nodes)
	Energy = jax.ops.segment_sum(HA_per_node + HB_per_node, node_graph_idx, n_graph)
	Energy_per_node = jnp.repeat(Energy, n_node, axis=0, total_repeat_length=total_nodes)
	return Energy, Energy_per_node, HB_per_node

vmap_calc_energy = jax.vmap(calc_energy, in_axes=(None, 0))

@jax.jit
def ConditionalExpectation_v2(graphs, ps):
	n_node = graphs.n_node
	cum_sum = jnp.concatenate([jnp.array([0]),jax.lax.cumsum(n_node)[:-1]], axis = 0)
	cum_max_sum = jax.lax.cumsum(n_node)
	max_steps = jnp.max(graphs.n_node)

	p_idxs = sort_ps(graphs, ps)

	def cond(arg):
		step, _, _ , cum_sum, cum_max_sum, graphs = arg
		return (step < max_steps)

	@jax.jit
	def body(arg):
		step, ps, p_idxs, cum_sum, cum_max_sum, graphs = arg

		sorted_cum_sum_idxs = p_idxs[cum_sum]
		up_nodes = ps.at[sorted_cum_sum_idxs, 0].set(jnp.ones_like(cum_sum))
		down_nodes = ps.at[sorted_cum_sum_idxs, 0].set(jnp.zeros_like(cum_sum))

		_, up_Energy, _ = calc_energy(graphs, up_nodes)
		_, down_Energy, _ = calc_energy(graphs, down_nodes)

		best_ps = jnp.where(up_Energy < down_Energy , up_nodes , down_nodes)

		cum_sum_p_1 = cum_sum + 1
		cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
		return (step + 1, best_ps, p_idxs, cum_sum, cum_max_sum, graphs)

	return jax.lax.while_loop(
		cond,
		body,
		(0, ps, p_idxs, cum_sum, cum_max_sum, graphs)
	)

def ConditionalExpectation_test(graphs, ps, cum_sum, cum_max_sum, max_steps=5):
	max_steps = jnp.max(graphs.n_node)
	p_idxs = sort_ps(graphs, ps)

	def cond(arg):
		step, _ , _, cum_sum, cum_max_sum, graphs = arg
		return (step < max_steps)

	def body(arg):
		step, ps, p_idxs, cum_sum, cum_max_sum, graphs = arg

		sorted_cum_sum_idxs = p_idxs[cum_sum]

		n_bernoulli_features = 2
		n_bern_X_0 = jnp.repeat(ps[None, ...], n_bernoulli_features, axis=0)
		bern_feat = jnp.arange(0, n_bernoulli_features)
		n_bern_X_0 = n_bern_X_0.at[bern_feat[:, None], sorted_cum_sum_idxs[None, :], 0].set(bern_feat[:, None] * jnp.ones_like(cum_sum)[None, :])
		_, n_bern_Energies_per_node, _ = vmap_calc_energy(graphs, n_bern_X_0)

		print("shapes")
		print(n_bern_X_0)

		min_idx = jnp.argmin(n_bern_Energies_per_node, axis=0)
		best_ps = n_bern_X_0[min_idx[None, ...], jnp.arange(n_bern_X_0.shape[1])[None, :, None], jnp.arange(n_bern_X_0.shape[2])[None,None, :]][0,...]

		print(min_idx.shape,n_bern_X_0.shape,best_ps.shape )


		# up_nodes = ps.at[sorted_cum_sum_idxs, 0].set(jnp.ones_like(cum_sum))
		# down_nodes = ps.at[sorted_cum_sum_idxs, 0].set(jnp.zeros_like(cum_sum))

		# _, up_Energy, _ = calc_energy(graphs, up_nodes)
		# _, down_Energy, _ = calc_energy(graphs, down_nodes)
		# best_ps = jnp.where(up_Energy < down_Energy , up_nodes , down_nodes)

		cum_sum_p_1 = cum_sum + 1
		cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
		return (step + 1, best_ps, p_idxs, cum_sum, cum_max_sum, graphs)

	input_args = (0, ps, p_idxs, cum_sum, cum_max_sum, graphs)
	while(cond(input_args)):
		input_args = body(input_args)

	(_, ps, p_idxs, cum_sum, cum_max_sum, graphs) = input_args
	Energy, Energy_per_node, HB_per_node = calc_energy(graphs, ps)

	print("test Energy", Energy)
	if (jnp.sum(HB_per_node) != 0):
		raise ValueError("Test there is a violation")
	return input_args

@jax.jit
def ConditionalExpectation(graphs, ps, cum_sum, cum_max_sum, max_steps=5):
	def cond(arg):
		step, _, cum_sum, cum_max_sum, graphs = arg
		return (step < max_steps)

	@jax.jit
	def body(arg):
		step, ps, cum_sum, cum_max_sum, graphs = arg

		up_nodes = ps.at[cum_sum, 0].set(jnp.ones_like(cum_sum))
		down_nodes = ps.at[cum_sum, 0].set(jnp.zeros_like(cum_sum))

		_, up_Energy, _ = calc_energy(graphs, up_nodes)
		_, down_Energy, _ = calc_energy(graphs, down_nodes)
		best_ps = jnp.where(up_Energy < down_Energy, up_nodes, down_nodes)

		cum_sum_p_1 = cum_sum + 1
		cum_sum = jnp.where(cum_sum_p_1 <  cum_max_sum, cum_sum_p_1, cum_max_sum - 1)
		return (step + 1, best_ps, cum_sum, cum_max_sum, graphs)

	input_args = (0, ps, cum_sum, cum_max_sum, graphs)
	return jax.lax.while_loop(
		cond,
		body,
		input_args
	)


import networkx as nx
def create_graph(n_nodes):
	# Define a three node graph, each node has an integer as its feature.
	gnx = nx.barabasi_albert_graph(n=n_nodes, m=4)
	edges = list(gnx.edges)

	for el in edges:
		if(el[0] == el[1]):
			raise ValueError("Self loops included")

	node_features = jnp.zeros((n_nodes,1))

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


def for_loop_CE(graph_list):

	for graph in graph_list:
		nodes = graph.nodes
		ps = 0.5*jnp.ones_like(nodes)

		for node_idx in range(nodes.shape[0]):
			up_nodes = ps.at[node_idx,0].set(1)
			down_nodes = ps.at[node_idx,0].set(0)

			_, up_Energy, _ = calc_energy(graph, up_nodes)
			_, down_Energy, _ = calc_energy(graph, down_nodes)
			ps = jnp.where(up_Energy < down_Energy, up_nodes, down_nodes)

		Energy, Energy_per_node, HB_per_node = calc_energy(graph, ps)

		print("for loop Energy",Energy)
		if(jnp.sum(HB_per_node) != 0):
			raise ValueError("there is a violation")

#@jax.jit
def sort_ps(graphs, ps):

	#ps = np.random.uniform(0,1, (graphs.nodes.shape[0],1))
	nodes = graphs.nodes
	n_node = graphs.n_node
	n_graph = graphs.n_node.shape[0]
	graph_idx = jnp.arange(n_graph)
	total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
	node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

	shifted_ps = -ps + node_graph_idx[:,None]
	#sorted_ps = jax.lax.sort(shifted_ps, dimension = 0)
	p_idxs = jax.numpy.argsort(shifted_ps, axis = 0)

	return p_idxs[:,0]

if __name__ == '__main__':

	import jax
	import jax.numpy as jnp

	import re
	def extract_integer(input_string):
		# Use a regular expression to find digits at the end of the string
		match = re.search(r'\d+$', input_string)
		if match:
			return int(match.group())
		else:
			return None

	size = 13
	input_string = f"IsingModel{size}x{size}"
	A = extract_integer(input_string)
	print(A)
	raise ValueError("")
	def reshape_array(x):
		B, L_squared = x.shape
		L = jax.lax.rsqrt(L_squared).astype(int)
		return jnp.reshape(x, (B, L, L))


	# Example usage:
	B = 2
	L = 3
	x = jnp.arange(B * L * L).reshape(B, L * L)

	# JIT-compiled version
	reshape_array_jit = jax.jit(reshape_array)

	# Reshape the array
	reshaped_array = reshape_array_jit(x)
	print(reshaped_array)

	raise ValueError("")
	import igraph
	import numpy as np

	import jax
	import jax.numpy as jnp
	from jax import random

	t_idx_per_graph = jnp.repeat(jnp.arange(0,3)[:, None], 5,axis=-1)
	t_idx_per_graph = jnp.repeat(t_idx_per_graph[..., None, None], 4, axis = -2)
	n_node = jnp.array([3, 4, 2])
	total_nodes = jnp.sum(n_node)
	rand_diff_steps_per_node = jnp.repeat(t_idx_per_graph, n_node[:, None], axis=0,
										  total_repeat_length=total_nodes)

	key = random.PRNGKey(42)
	x = jnp.broadcast_to(jnp.arange(3), (2, 4, 3))

	print(random.shuffle(key, x, axis=1))
	print(jax.vmap(random.permutation)(random.split(key, x.shape[0]), x))
	print("here")
	print(random.permutation(key, x, axis = -1, independent=True))


	extract()
	# pmap example
	#https: // github.com / google - deepmind / jraph / blob / master / jraph / ogb_examples / train_pmap.py
	import jax
	jax.config.update('jax_platform_name', 'cpu')
	from jax import numpy as jnp

	arr = jnp.array([10,5,123,123123,4])
	idxs = jnp.array([0,1,2,3,4])

	res = jax.lax.sort(arr)
	idxs = jax.numpy.argsort(arr)
	print(res)
	print(idxs)

	diffusion_steps = 9
	mini_diff_steps = 3
	mini_N_b = 5
	n_basis_states = 10
	n_devices = 3
	arr = jnp.arange(0,diffusion_steps)
	arr = jnp.repeat(arr[None, ...], n_basis_states, axis = 0 )
	#arr = jnp.repeat(arr[None, ...], n_devices, axis = 0)

	key = jax.random.PRNGKey(0)

	key, subkey = jax.random.split(key)
	perm_array = jax.random.permutation(subkey, arr, axis = -1, independent=True)

	print(perm_array)
	split_array = jnp.split(perm_array, 3, axis=-1)

	for split_arr in split_array:
		split_arr_list = jnp.split(split_arr, n_basis_states//mini_N_b, axis = 0)
		print([el.shape for el in split_arr_list] , split_arr.shape)
		print([el for el in split_arr_list], split_arr.shape)

	from functools import partial
	from jax import random, grad, jit
	import jraph

	#node_list = [100, 120, 80, 40, 300]
	node_list = [10, 12, 8, 6, 30]
	graph_list = [create_graph(n) for n in node_list]
	for_loop_CE(graph_list)
	batched_graph = jraph.batch(graph_list)

	ps = np.random.uniform(0, 1, (batched_graph.nodes.shape[0], 1))
	sort_ps(batched_graph, ps)

	print(jax.devices())
	batched_jraph_iteration(batched_graph)
	for_loop_CE(graph_list)
	raise ValueError("")
	import jax
	import numpy as np
	import jax.numpy as jnp
	from jax.tree_util import tree_map, tree_flatten


	# Your function to compute the mean
	def compute_mean(arr):
		print(arr)
		return np.mean(arr[0])

	# Sample data
	K = 5
	D = 10
	a1 = jnp.array(np.random.rand(K))
	a2 = jnp.array(np.random.rand(D))

	# Create a Pytree structure
	data_pytree = {'a1': tree_flatten(graph)[0], 'a2': tree_flatten(graph)[0]}


	# Define a function to compute mean over a Pytree
	def compute_mean_pytree(pytree):
		return tree_map(compute_mean, pytree)


	# Compute means in parallel
	result_pytree = compute_mean_pytree(data_pytree)

	# Print the results
	print(result_pytree)

	# Function to be parallelized
	def square_and_add(x):
		return x ** 2 + x


	# Loss function
	def mean_squared_error(y_true, y_pred):
		return np.mean((y_true - y_pred) ** 2)


	# Generate random data
	key = random.PRNGKey(0)
	data = np.random.rand(8, 4)
	labels = np.random.rand(8, 1)

	# Get the number of devices (GPUs)
	num_devices = jax.local_device_count()

	# Split data and labels across devices
	data_per_device = np.array_split(data, num_devices)
	labels_per_device = np.array_split(labels, num_devices)


	# Function to be mapped
	def mapped_function(x, y_true):
		y_pred = jax.pmap(square_and_add)(x)
		loss = mean_squared_error(y_true, y_pred)
		return loss, y_pred


	# Compute gradients using pmap
	grad_function = jit(jax.grad(lambda x, y_true: np.mean(mapped_function(x, y_true)[0])))
	grads_per_device = jax.vmap(grad_function)(data_per_device, labels_per_device)

	# Combine gradients from different devices
	final_grads = sum(grads_per_device)

	print("Gradients:", final_grads)





