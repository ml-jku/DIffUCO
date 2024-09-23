#### see code for help https://github.com/jzhoulab/ddsm/blob/main/ddsm.py
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_platform_name', 'cpu')
from numbers import Real
from matplotlib import pyplot as plt
from functools import partial
import torch
from noise_distribution import jacobi_diffusion_density, log_jacobi_diffusion_density


def sample_beta( a,b, key, shape = (2000,)):
    samples = jax.random.beta(key, a,b, shape)
    return samples

def beta_CDF(x, ab):
    return jax.scipy.stats.beta.cdf(x, ab[0], ab[1], loc=0, scale=1)

def grad_beta_CDF(x,ab):
    return jax.grad(beta_CDF)(x,ab)

### TODO implement simple toy example

### TODO implemend gamma distribution that minimizes this
def Energy(x):
    B = 1.1
    print("x shape",x.shape)
    return - jnp.sum(x, axis = -1) + B*jnp.sum(x[:,None]*x[None,:]) - B * jnp.sum(x**2)

### parameter are a and b
def compute_loss(ab,key, N = 100, dims = 2):

    vmap_energy = jax.vmap(Energy, in_axes=(0,))
    Energy_grad = jax.grad(Energy)
    vmap_Energy_grad = jax.vmap(Energy_grad, in_axes=(0,))
    vmap_z_grad_beta_CDF = jax.vmap(grad_beta_CDF, in_axes=(-1, -1))
    vmap_grad_beta_CDF = jax.vmap( vmap_z_grad_beta_CDF, in_axes=(0,None))

    key,subkey = jax.random.split(key)
    shape = (N, dims)
    xs = sample_beta( ab[0],ab[1], key, shape)

    vmap_energy(xs)

    grad_E_x = vmap_Energy_grad(xs)
    grad_CDF_x = jax.scipy.stats.beta.logpdf(xs, ab[0], ab[1])
    grad_CDF_ab = vmap_grad_beta_CDF(xs,ab)

    grad_x_ab = -grad_CDF_ab/ grad_CDF_x
    overall_gradient = jnp.mean(jnp.sum(grad_E_x*grad_x_ab, axis = -1))
    return overall_gradient





if(__name__ == '__main__'):
    jax.config.update("jax_enable_x64", True)
    ### TODO solve numerical instability
    #### a and b < 1 are dangerous, this causes numerical instability
    def log_func(x):
        return jnp.log(x)

    @jax.custom_gradient
    def log_func2(x):
        return jnp.log(jnp.clip(x, a_min=10**-30)), lambda g: (g*jax.grad(log_func)(jnp.clip(x, a_min=10**-30)))

    print(jax.grad(log_func)(0.))
    print(jax.grad(log_func2)(0.))


    key = jax.random.PRNGKey(100)
    a = jnp.array([2.,5.])
    b = jnp.array([2.,5.])
    print(sample_beta(a, b, key, shape = (100,2)))
    print(beta_CDF(0.5, [5.,5.]))
    ab = jnp.array([[2,2],[5,5]])
    overall_gradient = compute_loss(ab, key)

    tau_max = 5
    n_diff_steps = 20
    ts = [tau_max * (i + 1) / n_diff_steps for i in range(n_diff_steps)]
    if(True):
        reps = 10

        ts = [ tau_max*(i+1)/n_diff_steps for i in range(n_diff_steps)]
        for t in ts:
            a = jnp.array([5.])
            b = jnp.array([5.])
            x_t = jnp.array([(t+1)/n_diff_steps])
            x_t_p_1_arr = np.linspace(0.,1, 100)
            diffusion_prob_step_list = []
            diffusion_prob_step_list_log = []
            for x_t_p_1 in x_t_p_1_arr:
                x_t_p_1 = jnp.array([x_t_p_1])
                diffusion_prob_step = jacobi_diffusion_density(x_t, x_t_p_1, t, a, b,  True)
                stable = log_jacobi_diffusion_density(x_t, x_t_p_1, t, a, b,  True)
                diffusion_prob_step_list.append(diffusion_prob_step)
                diffusion_prob_step_list_log.append(stable)

                print("here")
                print("stable", stable)
                print("unstable", diffusion_prob_step)

            plt.figure()
            plt.title(f"a = {np.round(float(a),2)}, b = {np.round(float(b),2)}, X_t = {np.round(float(x_t),2)} {sum(diffusion_prob_step_list)*1/100}")
            plt.plot(x_t_p_1_arr, np.log(np.array(diffusion_prob_step_list)), "-<", label = "unstable")
            plt.plot(x_t_p_1_arr, diffusion_prob_step_list_log, "-x", label = "stable")
            print(diffusion_prob_step_list_log)

            # log_pdf_values = jax.scipy.stats.beta.logpdf(jnp.array(x_t_p_1_arr), float(a), float(b))
            #
            # plt.title("v1")
            # plt.plot(x_t_p_1_arr, np.exp(log_pdf_values), "-x")
            plt.legend()
            plt.show()

            plt.figure()
            plt.title(f"a = {np.round(float(a),2)}, b = {np.round(float(b),2)}, X_t = {np.round(float(x_t),2)} {sum(diffusion_prob_step_list)*1/100}")
            plt.plot(x_t_p_1_arr, jnp.array(diffusion_prob_step_list), "-x", label = "unstable")
            plt.plot(x_t_p_1_arr, jnp.exp(jnp.array(diffusion_prob_step_list_log)), "-<", label = "stable")

            # log_pdf_values = jax.scipy.stats.beta.logpdf(jnp.array(x_t_p_1_arr), float(a), float(b))
            #
            # plt.title("v1")
            # plt.plot(x_t_p_1_arr, np.exp(log_pdf_values), "-x")
            plt.legend()
            plt.show()

            print("CDF", sum(diffusion_prob_step_list)*1/100, 100)
            print([np.round(v, 3) for v in diffusion_prob_step_list])

        raise ValueError()

    if(False):
        ### TORCH
        reps = 10
        t_max = 5
        ts = np.linspace(0.1,t_max,10)
        for t in ts:
            a = torch.tensor([5])
            b = torch.tensor([5])
            x_t = torch.tensor([1.])
            order = 100
            x_t_p_1_arr = np.linspace(0.,1, 100)
            diffusion_prob_step_list = []
            for x_t_p_1 in x_t_p_1_arr:
                x_t_p_1 = torch.tensor([x_t_p_1])
                diffusion_prob_step = jacobi_diffusion_density(x_t, x_t_p_1, t, a, b, order=order, speed_balanced=True)

                diffusion_prob_step_list.append(diffusion_prob_step)

            plt.figure()
            plt.title(f"a = {np.round(float(a),2)}, b = {np.round(float(b),2)}, X_t = {np.round(float(x_t),2)} {sum(diffusion_prob_step_list)*1/100}")
            plt.plot(x_t_p_1_arr, diffusion_prob_step_list, "-x")

            log_pdf_values = jax.scipy.stats.beta.logpdf(jnp.array(x_t_p_1_arr), float(a), float(b))

            plt.title("v1")
            plt.plot(x_t_p_1_arr, np.exp(log_pdf_values), "-x")
            plt.show()

            print("CDF", sum(diffusion_prob_step_list)*1/100, order)
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
