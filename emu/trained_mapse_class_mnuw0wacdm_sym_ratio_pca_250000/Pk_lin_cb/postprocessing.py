import jax.numpy as jnp
from jaxmapse import primordial_Pk

def postprocessing(params, output, D, emu):
    # params: [ln10As, ns, H0, ...]
    ln10As = params[0]
    ns = params[1]
    As = jnp.exp(ln10As) * 1e-10

    ωb = params[3]
    ωc = params[4]
    Mν = params[5]

    k = emu.k_grid
    P_prim = primordial_Pk(As, ns, k)

    log10_k = jnp.log10(k)
    ων = Mν / 93.14
    Δω = ωc + ων - ωb
    ωm = ωb + ωc + ων

    # Intermediate terms for the Symbolic Expression
    # Mapping: X1 = log10_k, X2 = ωm, X3 = Δω

    # Left-hand side of the subtraction (The Sigmoid-heavy term)
    term_top = (jnp.sqrt(jnp.abs(jnp.exp(jnp.sqrt(jnp.abs(-0.0763 - (-0.0187 * log10_k)))) ** jnp.exp(0.6347 * log10_k))) - (-0.2897 * log10_k))

    # Internal logic for the denominator hierarchy
    denom_a = jnp.sqrt(1 + ((11716.6816 * Δω) ** 0.2043) ** 2)
    part_a = ((32.1095 * ωm) - 3.8999) / denom_a

    denom_b1 = jnp.sqrt(1 + (1.0774 * log10_k) ** 2)
    denom_b2 = jnp.sqrt(1 + (-1.9426 / denom_b1) ** 2)
    part_b1 = jnp.exp(-0.2740 * log10_k) / denom_b2

    part_c_const = jnp.sqrt(jnp.abs(-1.5068))
    denom_c = jnp.sqrt(1 + (0.4092 * log10_k + ((1.5647 ** (-1.1865 * log10_k)) - part_c_const)) ** 2)
    part_b2 = (5.8964 * Δω) / denom_c

    denom_d = jnp.sqrt(1 + (part_b1 + part_b2) ** 2)
    part_d = ((-6.7412 * ωm) + part_a) / denom_d

    denom_e = jnp.sqrt(1 + (jnp.exp(-0.6853 * log10_k) - 0.9870 + 0.8661 * log10_k) ** 2)
    part_e = (((-0.1613 * log10_k) - (-0.3748 * ωm)) + part_d) / denom_e

    # Final assembly of the inner ratio
    final_num = term_top / jnp.sqrt(1 + jnp.exp(part_e) ** 2)

    # The full log-expression
    log_DIFF = ((-23.7061 * (final_num - 0.2076 * log10_k)) - jnp.log(jnp.abs(165.7727 * ωm)))

    DIFF = jnp.exp(log_DIFF)

    # Invert log10: 10.0 ** output
    return (output * DIFF**2) * (D**2) * P_prim
