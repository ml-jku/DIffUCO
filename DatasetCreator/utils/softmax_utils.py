
import jax.numpy as jnp
from jax import lax
import numpy as np

def log_softmax( logits, mask):
    masked_logits = jnp.where(mask, logits, -jnp.inf * jnp.ones_like(logits))
    max_logits = lax.stop_gradient(jnp.max(masked_logits, axis=-1))
    shifted = masked_logits - max_logits[:, jnp.newaxis]
    shifted_logsumexp = jnp.log( jnp.sum(jnp.exp(shifted), axis = -1, keepdims=True))
    res = shifted - shifted_logsumexp
    res = jnp.where(mask, res, -jnp.inf * jnp.ones_like(res))
    #print(jnp.sum(jnp.exp(res), axis = -1))
    return res

def log_softmax_np( logits, mask):
    masked_logits = np.where(mask, logits, -np.inf * np.ones_like(logits))
    max_logits = np.max(masked_logits, axis=-1)
    shifted = masked_logits - max_logits[:, np.newaxis]
    shifted_logsumexp = np.log( np.sum(jnp.exp(shifted), axis = -1, keepdims=True))
    res = shifted - shifted_logsumexp
    #print(jnp.sum(jnp.exp(res), axis = -1))
    return res