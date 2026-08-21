from __future__ import annotations


Edge = tuple[int, int]
Matching = tuple[Edge, ...]


def gan_matchings(
    n_molecular: int,
    n_metal: int,
) -> tuple[Matching, ...]:
    """
    Construct the hopping matchings used for the GAN fragmentation.

    Orbital indexing:

        molecular: 0, ..., n_molecular - 1
        metal:     n_molecular, ..., n_molecular + n_metal - 1

    For the current implementation we require an even number
    of molecular orbitals and n_molecular <= n_metal.
    """
    if n_molecular <= 0:
        raise ValueError("n_molecular must be positive")

    if n_metal <= 0:
        raise ValueError("n_metal must be positive")

    if n_molecular % 2 != 0:
        raise ValueError(
            "current implementation requires an even "
            "number of molecular orbitals"
        )

    if n_molecular > n_metal:
        raise ValueError(
            "current implementation requires "
            "n_molecular <= n_metal"
        )

    matchings: list[Matching] = []

    # -------------------------------------------------
    # Molecular-molecular complete graph K_Nmol
    # -------------------------------------------------
    #
    # Round-robin decomposition into Nmol - 1
    # perfect matchings.
    # -------------------------------------------------

    vertices = list(range(n_molecular))

    for _ in range(n_molecular - 1):
        matching = []

        for k in range(n_molecular // 2):
            i = vertices[k]
            j = vertices[-(k + 1)]

            matching.append((i, j))

        matchings.append(tuple(matching))

        # Keep first vertex fixed and rotate the rest.
        vertices = [
            vertices[0],
            vertices[-1],
            *vertices[1:-1],
        ]

    # -------------------------------------------------
    # Molecular-metal complete bipartite graph
    # K_{Nmol,Nmetal}
    # -------------------------------------------------
    #
    # One cyclic matching for each metal orbital.
    # -------------------------------------------------

    metal_offset = n_molecular

    for shift in range(n_metal):
        matching = []

        for i in range(n_molecular):
            metal_local = (i + shift) % n_metal

            j = metal_offset + metal_local

            matching.append((i, j))

        matchings.append(tuple(matching))

    return tuple(matchings)