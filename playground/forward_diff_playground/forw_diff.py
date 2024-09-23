import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt
import numpy as np
def sample_X_0_tilde(eps, shape, key):
    key, subkey = jax.random.split(key)
    sampled_ps = jax.random.uniform(subkey, shape)

    key, subkey = jax.random.split(key)
    X_0_proposal = jax.random.randint(subkey, shape, minval=0, maxval=2)
    X_0_hat = jnp.where(sampled_ps < eps, X_0_proposal, -1*jnp.ones_like(X_0_proposal))

    mask = jnp.where(X_0_hat != -1, 1, 0)
    return X_0_hat, mask, key

n_diff_steps = 10
tau = 6
gemma_t = [ 1/2*jnp.exp(-step/n_diff_steps*tau) for step in range(n_diff_steps)]
def sample_forward_diff_process(X_t_m1, t_idx, key):

    gamma_t = gemma_t[t_idx]
    X_next_down = 1 - X_t_m1

    log_p_up = X_t_m1 * jnp.log(1 - gamma_t) + (X_next_down) * jnp.log(gamma_t)
    log_p_down = (X_next_down) * jnp.log(1 - gamma_t) + X_t_m1 * jnp.log(gamma_t)
    log_p_per_node = jnp.concatenate([log_p_down[...,None], log_p_up[...,None]], axis=-1)

    key, subkey = jax.random.split(key)
    X_next = jax.random.categorical(key=subkey,
                                    logits=log_p_per_node,
                                    axis=-1,
                                    shape=log_p_per_node.shape[:-1])

    one_hot_state = jax.nn.one_hot(X_next, num_classes=2)
    spin_log_probs = jnp.sum(log_p_per_node * one_hot_state, axis=-1)


    return X_next, spin_log_probs, key

def plot_figure(arr_list, N):
    plt.figure()
    for idx, arr in enumerate(arr_list):
        string = r"\tilde{X}"+ f"_{idx}"
        plt.title(rf"${string}$")
        plt.subplot(2, (len(arr_list)+1)//2, 1 + idx)
        plt.imshow(np.reshape(arr, (N,N)))
    plt.tight_layout()
    plt.show()

if(__name__ == "__main__"):

    eps = 0.9
    N = 10
    shape = (N*N,)
    key = jax.random.PRNGKey(42)
    X_0_hat, mask, key = sample_X_0_tilde(eps, shape, key)
    X_curr = X_0_hat

    X_sequence = [X_0_hat]

    for i in range(n_diff_steps):
        X_next, spin_log_probs, key = sample_forward_diff_process(X_curr, n_diff_steps - i - 1, key)
        X_next = jnp.where(mask == 1, X_next, X_curr)

        X_curr = X_next
        X_sequence.append(X_next)

    plot_figure(X_sequence, N)