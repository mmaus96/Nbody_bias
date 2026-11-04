def preprocessing(params):
    # params: [ln10As, ns, H0, ombh2, omch2, Mν, w0, wa, Omega_k, log10T_heat]
    # We return: [H0, ombh2, omch2, Mν, w0, wa, Omega_k]
    # In Python 0-based indexing:
    # Drop first 2 (0,1)
    # Drop last 1 (9)
    # Take slice [2:9] -> indices 2,3,4,5,6,7,8
    return params[2:8]
