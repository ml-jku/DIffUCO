import jax
import jax.numpy as jnp
from functools import partial

import numpy as np


@partial(jax.jit, static_argnames=("order", "speed_balanced"))
def jacobi_diffusion_density(x0, xt, t, a, b, speed_balanced = True, order = 100):
    #order = 100
    n = jnp.arange(order, dtype = x0.dtype)[None, :]

    if speed_balanced:
        s = 2 / (a + b)
    else:
        s = jnp.ones_like(a)
    eigenvalues = (-0.5 * s * n * (n - 1 + a + b))

    logdn = (
            log_rising_factorial_jnp(a, n)
            + log_rising_factorial_jnp(b, n)
            - log_rising_factorial_jnp((a + b), n - 1)
            - jnp.log(2 * n + (a + b) - 1)
            - jax.lax.lgamma(n + 1)
    )
    #print("exp",torch.exp(beta_logp(a, b, xt).unsqueeze(-1)), jacobi(x0 * 2 - 1, alpha=b - 1, beta=a - 1, order=order), jacobi(xt * 2 - 1, alpha=b - 1, beta=a - 1, order=order))
    res =  (jnp.exp(beta_logp_jnp(a, b, xt)[...,None] + (eigenvalues * t - logdn))
            * jacobi_jnp(x0 * 2 - 1., b - 1., a - 1., order)
            * jacobi_jnp(xt * 2 - 1., b - 1., a - 1., order)).sum(-1)

    return res

@partial(jax.jit, static_argnames=("order", "speed_balanced"))
def log_jacobi_diffusion_density(x0, xt, t, a, b, speed_balanced = True, order = 100):
    n = jnp.arange(order, dtype = x0.dtype)[None, :]

    if speed_balanced:
        s = 2 / (a + b)
    else:
        s = jnp.ones_like(a)
    eigenvalues = (-0.5 * s * n * (n - 1 + a + b))

    logdn = (
            log_rising_factorial_jnp(a, n)
            + log_rising_factorial_jnp(b, n)
            - log_rising_factorial_jnp((a + b), n - 1)
            - jnp.log(2 * n + (a + b) - 1)
            - jax.lax.lgamma(n + 1)
    )
    #print("exp",torch.exp(beta_logp(a, b, xt).unsqueeze(-1)), jacobi(x0 * 2 - 1, alpha=b - 1, beta=a - 1, order=order), jacobi(xt * 2 - 1, alpha=b - 1, beta=a - 1, order=order))
    return beta_logp_jnp(a, b, xt) + safe_for_grad_log((jnp.exp(eigenvalues * t - logdn)*jacobi_jnp(x0 * 2 - 1., b - 1., a - 1., order)*jacobi_jnp(xt * 2 - 1., b - 1., a - 1., order)).sum(-1))


@jax.custom_gradient
def log_for_grad(x):
    return jnp.log(jnp.clip(x, a_min=10**-16)), lambda g: (g*jax.jacrev(jnp.log)(jnp.clip(x, a_min=10**-16)))

def safe_for_grad_log(x):
  return jnp.log(jnp.where(x > 10**-26, x, 10**-26))
@jax.jit
def dirichlet_logp_jnp(concentration, x):
    x = jnp.stack([x, 1.0 - x], -1)
    #x_log_y = jax.scipy.special.xlogy(concentration - 1.0,x)
    # print(x)
    # print((torch.log(x) * (concentration - 1.0)))
    # print("stable", x_log_y)
    x_log_y = safe_for_grad_log(x) * (concentration - 1.0)
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