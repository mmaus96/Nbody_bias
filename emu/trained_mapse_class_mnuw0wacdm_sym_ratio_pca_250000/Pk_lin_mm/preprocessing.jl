params::AbstractVector -> begin
    # params expected order: [ln10As, ns, H0, ombh2, omch2, Mν, w0, wa, Omega_k]
    # We return: [H0, ombh2, omch2, Mν, w0, wa, Omega_k]
    return params[3:end]
end
