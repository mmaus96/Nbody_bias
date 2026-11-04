(params::AbstractVector, output, D, emu) -> begin
    # D is ignored for Boost but included for signature consistency
    # Invert log10: 10.0 .^ output
    return 10.0 .^ output
end
