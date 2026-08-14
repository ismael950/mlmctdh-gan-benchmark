import sys
from itertools import combinations

import numpy as np

from ganbench.model import load_config
from ganbench.exact.hamiltonian import _build_electronic_space


def build_one_particle_hamiltonian(config):
    """
    Build the one-particle electronic Hamiltonian in the orbital basis

        (d1, d2, ..., c1, c2, ...).
    """

    n_orbitals = config.n_orbitals
    n_mol = config.n_molecular_orbitals

    h = np.zeros(
        (n_orbitals, n_orbitals),
        dtype=complex,
    )

    # Orbital energies
    h[np.arange(n_mol), np.arange(n_mol)] = (
        config.molecular_energies
    )

    metal_indices = np.arange(
        n_mol,
        n_orbitals,
    )

    h[metal_indices, metal_indices] = (
        config.metal_energies
    )

    # Molecule-metal couplings
    for i in range(config.n_molecular_orbitals):
        for k in range(config.n_metal_orbitals):

            metal_index = n_mol + k

            coupling = (
                config.molecule_metal_couplings[i, k]
            )

            h[i, metal_index] = coupling
            h[metal_index, i] = np.conjugate(coupling)

    return h


def main():
    config_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "configs/test_two_electron.yaml"
    )

    config = load_config(config_file)

    # -------------------------------------------------
    # One-particle spectrum
    # -------------------------------------------------
    h_one = build_one_particle_hamiltonian(config)

    one_particle_energies = np.linalg.eigvalsh(h_one)

    # -------------------------------------------------
    # Expected fixed-N many-body spectrum
    #
    # For a quadratic fermionic Hamiltonian:
    #
    # E_(i1,...,iN) = e_i1 + ... + e_iN
    # -------------------------------------------------
    expected_many_body_energies = np.array(
        [
            sum(one_particle_energies[list(indices)])
            for indices in combinations(
                range(config.n_orbitals),
                config.n_electrons,
            )
        ]
    )

    expected_many_body_energies.sort()

    # -------------------------------------------------
    # Spectrum produced by our Fock-space construction
    # -------------------------------------------------
    (
        h_many,
        _,
        _,
        electronic_basis,
    ) = _build_electronic_space(config)

    computed_many_body_energies = np.linalg.eigvalsh(
        h_many.toarray()
    )

    computed_many_body_energies.sort()

    # -------------------------------------------------
    # Compare
    # -------------------------------------------------
    errors = np.abs(
        computed_many_body_energies
        - expected_many_body_energies
    )

    print("Config:", config_file)

    print("\nElectronic Fock basis:")
    for state in electronic_basis:
        print("|" + "".join(map(str, state)) + ">")

    print("\nOne-particle eigenvalues:")
    for energy in one_particle_energies:
        print(f"{energy:.12f}")

    print("\nExpected vs computed many-body energies:")
    for expected, computed, error in zip(
        expected_many_body_energies,
        computed_many_body_energies,
        errors,
    ):
        print(
            f"{expected: .12f}   "
            f"{computed: .12f}   "
            f"error = {error:.3e}"
        )

    print(
        "\nMaximum spectral error:",
        np.max(errors),
    )

    if np.max(errors) < 1e-12:
        print(
            "PASS: fixed-N Fock-space spectrum "
            "matches the exact quadratic-fermion spectrum."
        )
    else:
        print(
            "FAIL: fermionic construction requires inspection."
        )


if __name__ == "__main__":
    main()