import jax
import jax.numpy as jnp
from .IsingModelEnergy import IsingModelEnergyClass

#@partial(jax.jit, static_argnums=(0,))
def _compute_aggr_utils(jraph_graph):
    nodes = jraph_graph.nodes
    n_node = jraph_graph.n_node
    n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
    return node_graph_idx, n_graph, total_num_nodes

def energy_registry(size, energy_func = "Ising"):
    ising = IsingModelEnergyClass({"n_bernoulli_features": 1, "N": size}, energy_func = energy_func)
    vmap_calculate_energy = jax.vmap(ising.calculate_Energy, in_axes=(None, 0, None))
    compute_energy = lambda a,b : hamiltonian(a,b, vmap_calculate_energy)
    return compute_energy

def hamiltonian(H_graph, spins, vmap_energy_func):
    node_grph_index = _compute_aggr_utils(H_graph)

    energy, bins = vmap_energy_func(H_graph, spins, node_grph_index[0])
    return energy[:, 0, 0]