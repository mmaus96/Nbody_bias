import jax.numpy as jnp


def JacobianBiasCombination(biases):
    """
    Compute the Jacobian of the simplified bias combination.

    Args:
        biases: Array of 11 bias parameters
                [b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4]

    Returns:
        Jacobian matrix of shape (14, 11) where:
        - Rows correspond to the 14 output terms
        - Columns correspond to derivatives w.r.t. the 11 input parameters

    Output terms indices:
    0: 1          7: b1*bs
    1: b1         8: b2*bs
    2: b1^2       9: bs^2
    3: b2         10: b3
    4: b1*b2      11: b1*b3
    5: b2^2       12: alpha0
    6: bs         13: sn
    """
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases

    # Initialize a 14x11 Jacobian matrix with all zeros
    J = jnp.zeros((14, 11))

    # --- Column 0: Derivatives with respect to b1 ---
    # indices: [1: b1, 2: b1^2, 4: b1*b2, 7: b1*bs, 11: b1*b3]
    J = J.at[1, 0].set(1.0)  # d(b1)/db1
    J = J.at[2, 0].set(2 * b1)  # d(b1^2)/db1
    J = J.at[4, 0].set(b2)  # d(b1*b2)/db1
    J = J.at[7, 0].set(bs)  # d(b1*bs)/db1
    J = J.at[11, 0].set(b3)  # d(b1*b3)/db1

    # --- Column 1: Derivatives with respect to b2 ---
    # indices: [3: b2, 4: b1*b2, 5: b2^2, 8: b2*bs]
    J = J.at[3, 1].set(1.0)  # d(b2)/db2
    J = J.at[4, 1].set(b1)  # d(b1*b2)/db2
    J = J.at[5, 1].set(2 * b2)  # d(b2^2)/db2
    J = J.at[8, 1].set(bs)  # d(b2*bs)/db2

    # --- Column 2: Derivatives with respect to bs ---
    # indices: [6: bs, 7: b1*bs, 8: b2*bs, 9: bs^2]
    J = J.at[6, 2].set(1.0)  # d(bs)/dbs
    J = J.at[7, 2].set(b1)  # d(b1*bs)/dbs
    J = J.at[8, 2].set(b2)  # d(b2*bs)/dbs
    J = J.at[9, 2].set(2 * bs)  # d(bs^2)/dbs

    # --- Column 3: Derivatives with respect to b3 ---
    # indices: [10: b3, 11: b1*b3]
    J = J.at[10, 3].set(1.0)  # d(b3)/db3
    J = J.at[11, 3].set(b1)  # d(b1*b3)/db3

    # --- Column 4: Derivatives with respect to alpha0 ---
    # indices: [12: alpha0]
    J = J.at[12, 4].set(1.0)  # d(alpha0)/dalpha0

    # --- Column 8: Derivatives with respect to sn ---
    # indices: [13: sn] (Note: sn is at index 8 of input parameters)
    J = J.at[13, 8].set(1.0)  # d(sn)/dsn

    # Columns 5, 6, 7 (alpha2, alpha4, alpha6) and 9, 10 (sn2, sn4)
    # remain all zeros because they do not appear in the 14 output terms.

    return J
