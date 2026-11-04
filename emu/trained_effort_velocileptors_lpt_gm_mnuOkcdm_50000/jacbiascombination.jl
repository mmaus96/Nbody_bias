function (biases)
    b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = biases
    T = eltype(biases)

    # --- Mappings for the 14x11 Jacobian ---

    # I = Row indices (Output term index 1..14)
    # J = Column indices (Input parameter index 1..11)

    I = [2, 3, 5, 8, 12,    # col 1 (b1) -> [b1, b1^2, b1b2, b1bs, b1b3]
         4, 5, 6, 9,        # col 2 (b2) -> [b2, b1b2, b2^2, b2bs]
         7, 8, 9, 10,       # col 3 (bs) -> [bs, b1bs, b2bs, bs^2]
         11, 12,            # col 4 (b3) -> [b3, b1b3]
         13,                # col 5 (alpha0)
         14]                # col 9 (sn) - Note: Input index is 9

    J = [1, 1, 1, 1, 1,     # derivatives w.r.t b1
         2, 2, 2, 2,        # derivatives w.r.t b2
         3, 3, 3, 3,        # derivatives w.r.t bs
         4, 4,              # derivatives w.r.t b3
         5,                 # derivative w.r.t alpha0
         9]                 # derivative w.r.t sn (Skip 6,7,8)

    V = [one(T), 2 * b1, b2, bs, b3,    # values for col 1
         one(T), b1, 2 * b2, bs,        # values for col 2
         one(T), b1, b2, 2 * bs,        # values for col 3
         one(T), b1,                    # values for col 4
         one(T),                        # value for col 5
         one(T)]                        # value for col 9

    return sparse(I, J, V, 14, 11)
end
