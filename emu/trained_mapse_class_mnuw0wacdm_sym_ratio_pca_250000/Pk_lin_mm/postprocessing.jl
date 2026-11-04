using Mapse
(params::AbstractVector, output, D, emu) -> begin
    # params: [ln10As, ns, H0, ...]
    ln10As = params[1]
    ns = params[2]
    As = exp(ln10As) * 1e-10

    k = Mapse.get_kgrid(emu)
    P_prim = Mapse.primordial_Pk(As, ns, k)

    # Invert log10: 10.0 .^ output
    return (output) .* D^2 .* P_prim
end
