
import numpy as np

def unravel_dict(x):
    if (len(x.shape) >= 1):
        x = x.reshape(-1, *x.shape[2:])
    else:
        x = np.mean(x)
    return x