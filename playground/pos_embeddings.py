



if(__name__ == "__main__"):

    import numpy as np
    import math
    from matplotlib import pyplot as plt

    def get_sinusoidal_positional_encoding(timestep, embedding_dim, max_position):
        """
        Create a sinusoidal positional encoding as described in the
        "Attention is All You Need" paper.

        Args:
            timestep (int): The current time step.
            embedding_dim (int): The dimensionality of the encoding.

        Returns:
            A 1D tensor of shape (embedding_dim,) representing the
            positional encoding for the given timestep.
        """
        position = timestep
        div_term = np.exp(np.arange(0, embedding_dim, 2) * (-np.log(max_position) / embedding_dim))
        return np.concatenate([np.sin(position * div_term), np.cos(position * div_term)], axis=-1)

    n_diff_steps = 30
    embedding_dim = 32
    step_list = np.arange(0,n_diff_steps)

    res_list = []
    for step in step_list:
        step = step
        cos_embeddings = get_sinusoidal_positional_encoding(step, embedding_dim, n_diff_steps)

        res_list.append(cos_embeddings)

    plt.figure()
    plt.plot(step_list, res_list, "-x")
    plt.show()
    raise ValueError("")

    import numpy as jnp
    from matplotlib import pyplot as plt

    N = 100
    dim = 15
    L = 1.41
    edges = jnp.linspace(0, L, N)[:, None]

    i = jnp.arange(0, dim, 2)
    L_prime_sin = L * ((dim - i + 1) / dim)[None, :]
    L_prime_cos = L * ((dim - i) / dim)[None, :]
    cos_edges = jnp.cos(2 * jnp.pi * edges / L_prime_cos)
    sin_edges = jnp.sin(2 * jnp.pi * edges / L_prime_sin)

    for j, ii in enumerate(i):
        plt.figure()
        plt.title(str(j))
        plt.plot(jnp.arange(0, N), cos_edges[:, j])
        plt.plot(jnp.arange(0, N), sin_edges[:, j])
        plt.show()