import numpy as np
import jax.numpy as jnp

if(__name__ == "__main__"):
    Engery = np.arange(0,1000, 10)

    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
        for el in Engery:
            el = el.astype(dtype)
            Energy = np.exp(el)

            print("Energy:", el, Energy, type(Energy))