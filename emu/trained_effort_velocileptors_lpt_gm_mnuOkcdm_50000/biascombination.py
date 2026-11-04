import jax.numpy as jnp


def BiasCombination(biases):
    """
    Compute the bias combination coefficients for VelocILEPTors (Reduced).

    Args:
        biases: Array of 11 bias parameters
                [b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4]

    Returns:
        Array of 14 coefficients matching the basis:
        [1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1b3, alpha0, sn]
    """
    # 1. Unpack all 11 parameters (even if some are unused)
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases

    # 2. Construct the 14-element coefficient vector
    bias_coeffs = jnp.array(
        [
            1.0,  # Constant term
            b1,
            b1**2,  # Linear bias terms
            b2,
            b1 * b2,
            b2**2,  # Quadratic bias terms
            bs,
            b1 * bs,
            b2 * bs,
            bs**2,  # Tidal bias terms
            b3,
            b1 * b3,  # Third-order bias terms
            alpha0,  # Counterterm (0th order)
            sn,  # Stochastic noise (0th order)
        ]
    )

    return bias_coeffs
