#### see code for help https://github.com/jzhoulab/ddsm/blob/main/ddsm.py
import jax
import jax.numpy as jnp
import numpy as np
from torch.distributed import optim
import optax
import wandb
#jax.config.update('jax_platform_name', 'cpu')
from numbers import Real
from matplotlib import pyplot as plt
from functools import partial
import torch
from flax import linen as nn
import flax
from tensorflow_probability.substrates import jax as tfp
import jraph
import networkx as nx
from DatasetCreator.Gurobi.GurobiSolver import solveMIS_as_MIP
from Networks.Modules.GNNModules.EncodeProcessDecode import EncodeProcessDecode
from noise_distribution import log_jacobi_diffusion_density, jacobi_diffusion_density
def create_graph(n_nodes):
    # Define a three node graph, each node has an integer as its feature.
    gnx = nx.erdos_renyi_graph(n=n_nodes, p = 0.5)
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
    print("number of edges = ", len(edges), " max edges" , n_nodes*(n_node -1)/2)
    global_context = jnp.array([[1]])
    graph = jraph.GraphsTuple(nodes=node_features, senders=senders, receivers=receivers,
                              edges=edges, n_node=n_node, n_edge=n_edge, globals=global_context)
    return graph


def sample_beta( a,b, key, shape = (2000,)):
    key, subkey = jax.random.split(key)
    samples = jax.random.beta(subkey, a,b, shape)
    return samples, key

def beta_CDF(params, model, node_idx, x):
    out_dict = model.apply(params, node_idx)
    return jax.scipy.stats.beta.cdf(x, out_dict["a_values"], out_dict["b_values"], loc=0, scale=1)

def beta_CDF_per_node(params, model, node_idx, x):
    #print(ab)
    out_dict = model.apply(params, node_idx)
    return tfp.math.betainc(out_dict["a_values"], out_dict["b_values"],x)

def grad_beta_CDF(params, model, node_idx, x):
    return jax.grad(beta_CDF_per_node )( params, model, node_idx, x)

### TODO implement simple toy example

### TODO implemend gamma distribution that minimizes this
def calc_Energy(H_graph, bins):
    B = 1.1
    A = 1.

    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

    raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
    Energy_messages = (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])

    # print("Energy_per_graph", Energy.shape)
    HA_per_node = - A * raveled_bins
    HB_per_node = B * (jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes))

    Energy = jax.ops.segment_sum(HA_per_node + HB_per_node, node_gr_idx, n_graph)
    return jnp.squeeze(Energy)


@jax.custom_gradient
def sample_from_beta( a,b, subkey):
    samples = jax.random.beta(subkey, a,b, )
    pdf = jax.scipy.stats.beta.pdf(samples, a, b)
    return samples, lambda g: (-g*jax.grad(tfp.math.betainc, argnums=0)(a, b, samples)/pdf, -g*jax.grad(tfp.math.betainc, argnums=1)(a, b, samples)/pdf, None)

class GammaModel(nn.Module):
    n_features_list_nodes: np.ndarray
    n_diffusion_steps: int
    n_message_passing_steps: int = 4

    def setup(self):
        self.GNN = EncodeProcessDecode(self.n_features_list_nodes,self.n_features_list_nodes,
                                       self.n_features_list_nodes,self.n_features_list_nodes, self.n_features_list_nodes,
                                       edge_updates = False,n_message_passes = 3)

        self.dense_as = nn.Dense(features=1, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)
        self.dense_bs = nn.Dense(features=1, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)
        self.elu = lambda x: jax.nn.elu(x) + 1.
        self.eps = 1e-10

    @flax.linen.jit
    def __call__(self, graph, X_prev, t_idx) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        time_one_hot = t_idx * jnp.ones_like(X_prev[..., -1])
        time_one_hot = jax.nn.one_hot(time_one_hot, num_classes=self.n_diffusion_steps)
        graph = graph._replace(nodes = jnp.concatenate([graph.nodes, X_prev, time_one_hot], axis = -1))
        graph_dict = {"graphs": [graph]}
        decoded_nodes = self.GNN(graph_dict, graph.nodes)
        a_values = self.elu(self.dense_as(decoded_nodes)) + self.eps
        b_values = self.elu(self.dense_bs(decoded_nodes)) + self.eps

        out_dict = {"a_values": a_values[...,0], "b_values": b_values[...,0]}
        return out_dict

    def forward(self, params, graph, X_prev, t_idx, key):

        out_dict = self.apply(params, graph, X_prev, t_idx)
        out_dict["X_prev"] = X_prev
        out_dict, key = self._sample(out_dict, key)
        out_dict = self._get_entropy(out_dict, graph)
        out_dict = self._get_expectation(out_dict)

        return out_dict, key

    def _sample(self, out_dict, key):
        a_values = out_dict["a_values"]
        b_values = out_dict["b_values"]
        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, a_values.shape[0] )
        samples = jax.vmap(sample_from_beta, in_axes=(0,0,0))(a_values, b_values, batched_key)
        out_dict["pdf"] = jax.scipy.stats.beta.pdf(samples, a_values, b_values)
        #out_dict["grads"] = (jax.jacfwd(tfp.math.betainc, argnums = 0)(a, b, samples), jax.jacfwd(tfp.math.betainc, argnums = 1)(a, b, samples))
        out_dict["X_next"] = samples[..., None]
        out_dict["log_probs"] = jax.scipy.stats.beta.logpdf(samples, a_values, b_values)

        return out_dict, key


    def _get_entropy(self,out_dict, graph):
        a_values = out_dict["a_values"]
        b_values = out_dict["b_values"]
        ### TODO make graph aggregation
        nodes = graph.nodes
        n_node = graph.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        neg_entropy_per_node = jnp.log(jax.scipy.special.beta(a_values, b_values)) -(a_values -1) *jax.scipy.special.digamma(a_values) - (b_values -1) *jax.scipy.special.digamma(b_values) + (a_values + b_values - 2)*jax.scipy.special.digamma(a_values + b_values)
        out_dict["entropy"] = jax.ops.segment_sum(neg_entropy_per_node, node_gr_idx, n_graph)
        return out_dict


    def _get_expectation(self, out_dict):
        a_values = out_dict["a_values"]
        b_values = out_dict["b_values"]
        exp_value = a_values/(a_values+b_values)
        out_dict["exp_value"] = exp_value
        return out_dict

    def _sample_prior(self, j_graph, n_basis_states, key, diff_a,diff_b):
        N_nodes = j_graph.nodes.shape[0]

        X_T, key = sample_beta(diff_a, diff_b, key, shape=(N_nodes, n_basis_states))
        X_T_log_prob = jax.scipy.stats.beta.logpdf(X_T, diff_a,diff_b)

        prior_dict = {"samples": X_T[...,None], "log_probs": X_T_log_prob}
        return prior_dict, key




class Trainer():

    def __init__(self, epochs = 1000, lr = 10**-4, n_diffusion_steps = 3, n_basis_states = 10, tau_max = 5., diff_a = 5., diff_b = 5., T_start = 0.):
        N_nodes = 5
        self.epochs = epochs
        j_graph = create_graph(n_nodes=N_nodes)
        _, Gurobi_Energy, _, _ = solveMIS_as_MIP(j_graph)
        self.GurobiEnergy = Gurobi_Energy
        print("Gurobi Energy is", Gurobi_Energy)

        self.T_start = T_start
        self.n_basis_states = n_basis_states
        self.n_diffusion_steps = n_diffusion_steps

        self.diff_a = diff_a
        self.diff_b = diff_b
        self.t_idxs = [tau_max * (i + 1) / self.n_diffusion_steps for i in range(self.n_diffusion_steps)]
        #self.t_idxs = [tau_max * 1 / self.n_diffusion_steps for i in range(self.n_diffusion_steps)]


        self.__init_params(j_graph)
        self.__init_optimizer(lr)
        self._init_vmap_functions()
        self.train_loop(j_graph)

    def _init_vmap_functions(self):
        self.vmap_forward = jax.vmap(self.model.forward, in_axes=(None, None, 1, None, 0), out_axes = (1,1))
        self.vmap_calc_Energy = jax.vmap(calc_Energy, in_axes=(None, 1), out_axes = (0))

    def __init_params(self, j_graph):
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        self.model = GammaModel([20,20],self.n_diffusion_steps)

        self.params = self.model.init({"params": subkey}, j_graph, j_graph.nodes, 1)

    def __init_optimizer(self, lr):
        start_learning_rate = lr
        self.optimizer = optax.adam(start_learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def __init_grads_and_vmaps(self):
        pass

    def train_loop(self, j_graph):
        epochs = self.epochs  # 5
        key = jax.random.PRNGKey(0)
        energies = []

        T_max = self.T_start
        backward = jax.value_and_grad(self.forward, argnums=0, has_aux=True)

        for epoch in range(epochs):
            T = T_max *(1-epoch/epochs)

            (loss, (log_dict, key)), grads  = backward(self.params, j_graph, key, T)

            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

            print("curr_T", T)
            #mean_grads = jax.tree_map(lambda x: jnp.mean(x), grads)
            # print("min and max")
            # print(np.min([el["X_next"] for el in log_dict["log_list"]]))
            # print(np.max([el["X_next"] for el in log_dict["log_list"]]))
            mean_Energy = np.mean(log_dict["Energy_value"])
            energies.append(float(mean_Energy))
            print("mean Energy", mean_Energy, "min energy", np.min(energies), )
            print("Gurobi Energy", self.GurobiEnergy)
            wandb_log_dict = {"mean_Energy": mean_Energy, "loss": loss, "energy_loss": log_dict["energy_loss"], "entropy_loss": log_dict["entropy_loss"], "noise_dist_loss": log_dict["noise_distr_loss"]}
            wandb.log(wandb_log_dict)
            #if (contains_invalid_values(grads)):
            #    raise ValueError("Nan")
            if (np.isnan(mean_Energy)):
                raise ValueError("Nan")

    @partial(jax.jit, static_argnums=0)
    def forward(self, params, j_graph, key, T):

        prior_dict, key = self.model._sample_prior(j_graph, self.n_basis_states, key, self.diff_a, self.diff_b)
        X_T = prior_dict["samples"]
        X_T_log_probs = prior_dict["log_probs"]

        # X_over_diff_steps = jnp.zeros((self.n_diffusion_steps, j_graph.nodes.shape[0], self.n_basis_states, 1))
        # log_probs_over_diffusion_steps = jnp.zeros((self.n_diffusion_steps, self.n_basis_states, 1))
        # Entropies_over_diffusion_steps = jnp.zeros((self.n_diffusion_steps, self.n_basis_states, 1))

        noise_distr_loss = 0.
        entropy_loss = 0.
        noise_distr_over_steps = []
        samples_over_diffusion_steps = []
        entropy_over_diff_steps = []
        log_list = []

        X_prev = X_T

        for i in range(self.n_diffusion_steps):
            ### TODO vmap this function
            key, subkey = jax.random.split(key)
            batched_key = jax.random.split(subkey, self.n_basis_states)
            out_dict, _ = self.vmap_forward( params, j_graph, X_prev, i, batched_key)

            X_next = out_dict["X_next"]
            time_step = self.t_idxs[i]
            noise_distr_loss += -T*jnp.mean(self._log_noise_density(j_graph, X_prev, X_next, time_step))
            entropy_loss += - T* jnp.mean(out_dict["entropy"])

            X_prev = X_next
            noise_distr_over_steps.append(jnp.mean(self._log_noise_density(j_graph, X_prev, X_next, time_step)))
            samples_over_diffusion_steps.append(X_next)
            entropy_over_diff_steps.append(jnp.mean(out_dict["entropy"]))

            #out_dict["noise_grad"] = (jnp.mean(self._log_noise_density(j_graph, X_prev, X_next, time_step)), jnp.mean(self._log_noise_density(j_graph, X_prev, X_next, time_step)) )
            log_list.append(out_dict)

        p_values = out_dict["exp_value"]

        Energy = self.vmap_calc_Energy(j_graph, p_values)
        energy_loss = jnp.mean(Energy)
        overall_loss = noise_distr_loss + entropy_loss + energy_loss

        overall_loss = jnp.mean(overall_loss)
        log_dict = {"Energy_value": np.mean(Energy), "noise_distr_loss": noise_distr_loss, "entropy_loss": entropy_loss, "energy_loss":energy_loss,
                    "noise_distr_over_steps": noise_distr_over_steps, "entropy_over_diff_steps": entropy_over_diff_steps, "log_list": log_list}

        return overall_loss, (log_dict, key)

    def _log_noise_density(self, graph, X_prev, X_next, time_step):
        nodes = graph.nodes
        n_node = graph.n_node
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
        graph_idx = jnp.arange(n_graph)
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
        node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

        x_t_p_1 = X_prev
        x_t = X_next
        log_noise_distr_per_node = log_jacobi_diffusion_density(x_t, x_t_p_1, time_step, self.diff_a, self.diff_b, True, order = 100)
        return jax.ops.segment_sum(log_noise_distr_per_node, node_gr_idx, n_graph)


def contains_invalid_values(tree):
    """
    Check if a JAX tree contains NaN or infinity values.

    Parameters:
    tree (pytree): A JAX pytree (e.g., nested lists, tuples, dicts, or JAX arrays).

    Returns:
    bool: True if the tree contains NaN or infinity values, False otherwise.
    """

    def check_invalid(x):
        # Check for NaN or infinity in the current node
        if isinstance(x, jnp.ndarray):
            return jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x))
        elif isinstance(x, np.ndarray):
            return np.any(np.isnan(x)) or np.any(np.isinf(x))
        elif jnp.isscalar(x):
            return jnp.isnan(x) or jnp.isinf(x)
        return False

    # Flatten the tree to check each element
    leaves = jax.tree_util.tree_leaves(tree)

    for leaf in leaves:
        if check_invalid(leaf):
            return True

    return False



if(__name__ == '__main__'):
    ### TODO find out why Nan occurs
    import os
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    #jax.config.update("jax_debug_nans", True)
    #jax.config.update("jax_disable_jit", True)
    @jax.custom_gradient
    def f(x):
        return x ** 2, lambda g: (g * x,)


    @jax.custom_gradient
    def sample_from_beta(a,b, subkey):
        samples = jax.random.beta(subkey, a,b, )
        return samples, lambda g: (g*jax.grad(tfp.math.betainc, argnums=0)(a, b, samples), g*jax.grad(tfp.math.betainc, argnums=1)(a, b, samples), None)



    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    a = 2.
    b = 2.

    samples = sample_from_beta(a,b,subkey)
    gradient = jax.grad(sample_from_beta, argnums=(0,1))(a,b, subkey)
    #
    jax.config.update("jax_enable_x64", True)

    # x = jnp.array([0.1])
    # raise ValueError("")
    N_nodes = 10
    j_graph = create_graph(N_nodes)
    wandb.init(project="jaccobian_diffusion", settings=wandb.Settings(_service_wait=300))
    jac_trainer = Trainer(epochs = 3000, n_diffusion_steps=3, T_start = 0., lr = 10**-4, n_basis_states=30)

    raise ValueError("")


