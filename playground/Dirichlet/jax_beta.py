#### see code for help https://github.com/jzhoulab/ddsm/blob/main/ddsm.py
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_platform_name', 'cpu')
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
from Networks.Modules.GNNModules.EncodeProcessDecode import LinearMessagePassingLayer
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

@partial(jax.jit, static_argnums=(1))
def f(a,N = 100):
    b = 0
    for i in range(N):
        b += a
    return b
@partial(jax.jit, static_argnames=("order", "speed_balanced"))
def jacobi_diffusion_density_jnp(x0, xt, t, a, b, speed_balanced = True, order = 100):
    #order = 100
    n = jnp.arange(order, dtype = x0.dtype)[None, :]

    if speed_balanced:
        s = 2 / (a + b)
    else:
        s = torch.ones_like(a)
    eigenvalues = (-0.5 * s * n * (n - 1 + a + b))

    logdn = (
            log_rising_factorial_jnp(a, n)
            + log_rising_factorial_jnp(b, n)
            - log_rising_factorial_jnp((a + b), n - 1)
            - jnp.log(2 * n + (a + b) - 1)
            - jax.lax.lgamma(n + 1)
    )
    #print("exp",torch.exp(beta_logp(a, b, xt).unsqueeze(-1)), jacobi(x0 * 2 - 1, alpha=b - 1, beta=a - 1, order=order), jacobi(xt * 2 - 1, alpha=b - 1, beta=a - 1, order=order))
    return (
            jnp.exp(beta_logp_jnp(a, b, xt)[...,None] + (eigenvalues * t - logdn))
            * jacobi_jnp(x0 * 2 - 1., b - 1., a - 1., order)
            * jacobi_jnp(xt * 2 - 1., b - 1., a - 1., order)
    ).sum(-1)

@jax.jit
def dirichlet_logp_jnp(concentration, x):
    x = jnp.stack([x, 1.0 - x], -1)
    x_log_y = jax.scipy.special.xlogy(concentration - 1.0,x)
    # print(x)
    # print((torch.log(x) * (concentration - 1.0)))
    # print("stable", x_log_y)

    return (
            (x_log_y).sum(-1)
            + jax.lax.lgamma(concentration.sum(-1))
            - jax.lax.lgamma(concentration).sum(-1)
    )
@jax.jit
def beta_logp_jnp(alpha, beta, x):
    concentration = jnp.stack([alpha, beta], -1)
    return dirichlet_logp_jnp(concentration, x)

@jax.jit
def log_rising_factorial_jnp(a, n):
    return jax.lax.lgamma(a + n) - jax.lax.lgamma(a)

###TODO make this jitable

@partial(jax.jit, static_argnums=(-1))
def jacobi_jnp(x, alpha, beta, order):
    """
    Compute Jacobi polynomials.
    """
    a = alpha
    b = beta
    recur_fun = lambda p_n, p_n_minus1, n, x: (
                                                      jax.lax.mul(
                                                          x * (2 * n + a + b + 2) * (2 * n + a + b) + (a ** 2 - b ** 2),
                                                          p_n)
                                                      * (2 * n + a + b + 1)
                                                      - p_n_minus1 * (n + a) * (n + b) * (2 * n + a + b + 2) * 2
                                              ) / (2 * (n + 1) * (n + a + b + 1) * (2 * n + a + b))

    ### TODO implement with jax scan
    ys = [jnp.ones_like(x), (a + 1) + (a + b + 2) * (x - 1) * 0.5]
    for i in range(1, order - 1):
        ys.append(recur_fun(ys[i], ys[i - 1], i, x))
    return jnp.stack(ys, -1)



def sample_beta( a,b, key, shape = (2000,)):
    samples = jax.random.beta(key, a,b, shape)
    return samples

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
def Energy(H_graph, bins):
    B = 1.1
    A = 1.

    nodes = H_graph.nodes
    n_node = H_graph.n_node
    n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
    node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)

    n_graph = H_graph.n_node.shape[0]
    nodes = H_graph.nodes
    total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]

    raveled_bins = jnp.reshape(bins, (bins.shape[0], 1))
    Energy_messages = (raveled_bins[H_graph.senders]) * (raveled_bins[H_graph.receivers])

    # print("Energy_per_graph", Energy.shape)
    HA_per_node = - A * raveled_bins
    HB_per_node = B * (jax.ops.segment_sum(Energy_messages, H_graph.receivers, total_num_nodes))

    Energy = jax.ops.segment_sum(HA_per_node + HB_per_node, node_gr_idx, n_graph)
    return jnp.squeeze(Energy)

class GammaModel(nn.Module):
    n_hidden: int = 20
    n_node: int = 2
    n_latents : int = 2

    def setup(self):
        self.hidden_1 = nn.Dense(features=self.n_hidden, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)
        self.ln_1 = nn.LayerNorm()
        self.hidden_2 = nn.Dense(features=self.n_hidden, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)
        self.ln_2 = nn.LayerNorm()
        self.dense_as = nn.Dense(features=1, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)
        self.dense_bs = nn.Dense(features=1, kernel_init=nn.initializers.he_normal(),bias_init=nn.initializers.zeros)

        self.relu = lambda x: jax.nn.elu(x) + 1.
        self.eps = 1e-10
    #@flax.linen.jit
    def __call__(self, node_idx) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        ### TODO remake to GNNs
        #key, subkey = jax.random.split(key)
        #z = jax.random.normal(subkey, shape=(node_idx.shape[0], self.n_latents))
        node_idx = jax.nn.one_hot(node_idx, self.n_node)
        x = self.hidden_1(node_idx)
        x = self.relu(x)
        x = self.ln_1(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.ln_2(x)
        a_values = self.relu(self.dense_as(x)) + self.eps
        b_values = self.relu(self.dense_bs(x)) + self.eps

        out_dict = {"a_values": a_values[...,0], "b_values": b_values[...,0]}
        return out_dict


    def forward(self, params, graph, key, N_basis_states):
        out_dict = self.apply(params, graph.nodes)
        out_dict, key = self._sample(out_dict,key, N_basis_states)
        out_dict = self._get_entropy(out_dict)
        out_dict = self._get_expectation(out_dict)

        return out_dict, key

    def _sample(self, out_dict, key, N_basis_states):
        a_values = out_dict["a_values"]
        b_values = out_dict["b_values"]
        key, subkey = jax.random.split(key)
        samples = jax.random.beta(subkey, a_values, b_values, (N_basis_states, a_values.shape[0]))
        out_dict["samples"] = samples[..., None]
        return out_dict, key

    def _get_entropy(self,out_dict):
        a_values = out_dict["a_values"]
        b_values = out_dict["b_values"]
        entropy = jax.scipy.special.digamma(a_values) - jax.scipy.special.digamma(a_values + b_values)
        out_dict["entropy"] = -entropy
        return out_dict


    def _get_expectation(self, out_dict):
        a_values = out_dict["a_values"]
        b_values = out_dict["b_values"]
        exp_value = a_values/(a_values+b_values)
        out_dict["exp_value"] = exp_value
        return out_dict

    def _get_reparametrization_gradient(self):
        pass

    def _compute_diff_loss(self):
        pass

    def _compute_diff_loss_REINFORCE(self):
        pass


### parameter are a and b
#@partial(jax.jit, static_argnums=(1, 2))
def compute_loss(params, model, j_graph, key):
    N = 100

    #out_dict = {"as": ab[0], "bs":ab[1]}#model.apply(params)
    vmap_energy = jax.vmap(Energy, in_axes=(None, 0))
    Energy_grad = jax.grad(Energy, argnums=(1))
    vmap_Energy_grad = jax.vmap(Energy_grad, in_axes=(None,0))
    vmap_z_grad_beta_CDF = jax.vmap(grad_beta_CDF, in_axes=(None, None, -1, -1), out_axes = (0))
    vmap_grad_beta_CDF = jax.vmap( vmap_z_grad_beta_CDF, in_axes=(None, None, None, 0), out_axes = (0))

    key,subkey = jax.random.split(key)
    shape = (N, j_graph.nodes.shape[0])


    out_dict = model.apply(params, j_graph.nodes)

    xs = sample_beta( out_dict["a_values"], out_dict["b_values"], subkey, shape)

    obj = vmap_energy(j_graph, xs)
    mean_Energy = jnp.mean(obj)

    grad_E_x = vmap_Energy_grad(j_graph, xs)
    grad_CDF_x = jax.scipy.stats.beta.logpdf(xs, out_dict["a_values"], out_dict["b_values"])
    #print("ab for grad", ab[0], ab[1])
    grad_CDF_params = vmap_grad_beta_CDF(params, model, j_graph.nodes, xs)

    grad_E_x_updated = grad_E_x/grad_CDF_x

    print(jax.tree_util.tree_map(lambda x: x.shape,grad_CDF_params))
    print(grad_E_x_updated.shape)
    overall_gradient = jax.tree_util.tree_map(lambda x: 1/x.shape[0]*jnp.tensordot(grad_E_x_updated, -x, axes = ([0,1], [0,1])),grad_CDF_params)
    # grad_x_ab = -grad_CDF_params/ grad_CDF_x[...,None]
    # overall_gradient = jnp.mean(jnp.sum(grad_E_x[...,None]*grad_x_ab, axis = -2), axis = 0)
    return overall_gradient, mean_Energy, xs, key




if(__name__ == '__main__'):
    import optax
    x = 0.1
    jax.grad(jax.scipy.special.digamma)(x)
    # xs = jnp.linspace(0, 1, 1000)
    # abs = [[10,0.01], [0.5, 0.5], [5, 0.01], [5, 0.1]]
    # for ab in abs:
    #     log_pdf_values = jax.scipy.stats.beta.logpdf(xs, ab[0], ab[1])
    #
    #     plt.figure()
    #     plt.title(str(ab))
    #     plt.plot(xs, np.exp(log_pdf_values), "-x")
    #     plt.show()
    #
    jax.config.update("jax_enable_x64", True)
    # x = jnp.array([0.1])
    # raise ValueError("")
    N_nodes = 5
    j_graph = create_graph(n_nodes = N_nodes)
    _, Gurobi_Energy, _, _ = solveMIS_as_MIP(j_graph)
    print("Gurobi Energy is", Gurobi_Energy)
    j_graph = j_graph._replace(nodes = jnp.arange(N_nodes))

    N = 100
    n_latents = 2
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    model = GammaModel(n_node = N_nodes)

    params = model.init({"params": subkey}, j_graph.nodes)

    res = model.forward(params, j_graph, key, 10)
    raise ValueError("")

    epochs = 3*10**4#5
    energies = []
    start_learning_rate = 1e-3
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(params)
    jitted_loss = jax.jit(lambda p, k : compute_loss(p, model, j_graph, k))

    for epoch in range(epochs):
        grads, mean_Energy, samples, key = jitted_loss(params, key)
        ### TODO update params with optax
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        energies.append(float(mean_Energy))
        print("mean Energy", mean_Energy, "min energy", np.min(energies), )
        print("Gurobi Energy", Gurobi_Energy)
        print(samples[0:3])
        if(np.isnan(mean_Energy)):
            raise ValueError("Nan")

    # xs = jnp.linspace(0, 1, 1000)
    # for i in range(2):
    #     log_pdf_values = jax.scipy.stats.beta.logpdf(xs, ab[i,0], ab[i,1])
    #
    #     plt.figure()
    #     plt.title(str(ab[0]))
    #     plt.plot(xs, np.exp(log_pdf_values), "-x")
    #     plt.show()
    # raise ValueError("")
    if(True):
        reps = 10
        t_max = 5
        ts = np.linspace(0.1,t_max,10)
        for t in ts:
            a = jnp.array([5.])
            b = jnp.array([5.])
            x_t = jnp.array([1.])
            x_t_p_1_arr = np.linspace(0.,1, 100)
            diffusion_prob_step_list = []
            for x_t_p_1 in x_t_p_1_arr:
                x_t_p_1 = jnp.array([x_t_p_1])
                diffusion_prob_step = jacobi_diffusion_density_jnp(x_t, x_t_p_1, t, a, b,  True)

                diffusion_prob_step_list.append(diffusion_prob_step)

            plt.figure()
            plt.title(f"a = {np.round(float(a),2)}, b = {np.round(float(b),2)}, X_t = {np.round(float(x_t),2)} {sum(diffusion_prob_step_list)*1/100}")
            plt.plot(x_t_p_1_arr, diffusion_prob_step_list, "-x")

            log_pdf_values = jax.scipy.stats.beta.logpdf(jnp.array(x_t_p_1_arr), float(a), float(b))

            plt.title("v1")
            plt.plot(x_t_p_1_arr, np.exp(log_pdf_values), "-x")
            plt.show()

            print("CDF", sum(diffusion_prob_step_list)*1/100, 100)
            print([np.round(v, 3) for v in diffusion_prob_step_list])

        raise ValueError()



    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)

    aa = np.linspace(0.1, 5, 10)
    for a in aa:
        a = a
        b = a
        sample_beta(key, a,b, shape = (2000,))

        xs = jnp.linspace(0,1, 1000)

        log_pdf_values = jax.scipy.stats.beta.logpdf(xs, a,b)

        plt.figure()
        plt.title("v1")
        plt.plot(xs, np.exp(log_pdf_values), "-x")
        plt.show()

        ### makes no sense
        a = a
        b = a
        sample_beta(key, a,b, shape = (2000,))

        xs = jnp.linspace(0,1*b, 1000)

        log_pdf_values = jax.scipy.stats.beta.logpdf(xs/b, a,1)

        plt.figure()
        plt.title("v2")
        plt.plot(xs, np.exp(log_pdf_values), "-x")
        plt.show()
