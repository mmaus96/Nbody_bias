import jax.numpy as jnp
from jaxmapse import primordial_Pk

def postprocessing(params, output, D, emu):
    return 10.0 ** output
