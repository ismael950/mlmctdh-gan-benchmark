from __future__ import annotations

import numpy as np

from ganbench.quantum.fermions import (
    hopping_operator,
    matching_unitary,
    pauli_z_operator,
)
from ganbench.quantum.matchings import Matching


def hopping_fragment(
    matching: Matching,
    couplings: dict[tuple[int, int], float],
    n_orbitals: int,
) -> np.ndarray:
    """
    Construct one GAN hopping fragment

        F_s = sum_(i,j in matching)
              g_ij (a_i^dagger a_j + a_j^dagger a_i).
    """
    dimension = 2 ** n_orbitals

    fragment = np.zeros(
        (dimension, dimension),
        dtype=complex,
    )

    for i, j in matching:
        edge = (min(i, j), max(i, j))

        if edge not in couplings:
            raise KeyError(
                f"No coupling supplied for edge {edge}"
            )

        fragment += (
            couplings[edge]
            * hopping_operator(
                i,
                j,
                n_orbitals,
            )
        )

    return fragment


def diagonal_hopping_fragment(
    matching: Matching,
    couplings: dict[tuple[int, int], float],
    n_orbitals: int,
) -> np.ndarray:
    """
    Expected diagonal form

        D_s = sum_(i,j)
              g_ij (Z_i - Z_j) / 2.
    """
    dimension = 2 ** n_orbitals

    diagonal = np.zeros(
        (dimension, dimension),
        dtype=complex,
    )

    for i, j in matching:
        edge = (min(i, j), max(i, j))

        if edge not in couplings:
            raise KeyError(
                f"No coupling supplied for edge {edge}"
            )

        diagonal += (
            couplings[edge]
            * (
                pauli_z_operator(i, n_orbitals)
                - pauli_z_operator(j, n_orbitals)
            )
            / 2.0
        )

    return diagonal


def check_fragment_diagonalization(
    matching: Matching,
    couplings: dict[tuple[int, int], float],
    n_orbitals: int,
) -> float:
    """
    Return

        || U_s^dagger F_s U_s - D_s ||.
    """
    fragment = hopping_fragment(
        matching,
        couplings,
        n_orbitals,
    )

    diagonal = diagonal_hopping_fragment(
        matching,
        couplings,
        n_orbitals,
    )

    unitary = matching_unitary(
        list(matching),
        n_orbitals,
    )

    return float(
        np.linalg.norm(
            unitary.conj().T
            @ fragment
            @ unitary
            - diagonal
        )
    )

from ganbench.quantum.fermions import number_operator


def diagonal_fragment_f0(
    molecular_energies: dict[int, float],
    density_interactions: dict[tuple[int, int], float],
    u0: float,
    n_orbitals: int,
) -> np.ndarray:
    """
    Construct the diagonal GAN fragment

        F0 =
            U0
            + sum_i epsilon_i n_i
            + sum_(i,j) V_ij n_i n_j.

    For now all coefficients are constants.
    Nuclear-coordinate dependence will be added later.
    """
    dimension = 2 ** n_orbitals
    identity = np.eye(dimension, dtype=complex)

    fragment = u0 * identity

    for i, energy in molecular_energies.items():
        fragment += (
            energy
            * number_operator(i, n_orbitals)
        )

    for (i, j), interaction in density_interactions.items():
        ni = number_operator(i, n_orbitals)
        nj = number_operator(j, n_orbitals)

        fragment += (
            interaction
            * (ni @ nj)
        )

    return fragment


def diagonal_fragment_f0_z(
    molecular_energies: dict[int, float],
    density_interactions: dict[tuple[int, int], float],
    u0: float,
    n_orbitals: int,
) -> np.ndarray:
    """
    Same F0 written explicitly in terms of Pauli Z:

        n_i = (I - Z_i) / 2.
    """
    dimension = 2 ** n_orbitals
    identity = np.eye(dimension, dtype=complex)

    fragment = u0 * identity

    occupations = {
        i: (
            identity
            - pauli_z_operator(i, n_orbitals)
        ) / 2.0
        for i in molecular_energies
    }

    for i, energy in molecular_energies.items():
        fragment += energy * occupations[i]

    for (i, j), interaction in density_interactions.items():
        ni = (
            identity
            - pauli_z_operator(i, n_orbitals)
        ) / 2.0

        nj = (
            identity
            - pauli_z_operator(j, n_orbitals)
        ) / 2.0

        fragment += interaction * (ni @ nj)

    return fragment

def hopping_fragment_propagator(
    matching: Matching,
    couplings: dict[tuple[int, int], float],
    n_orbitals: int,
    dt: float,
) -> np.ndarray:
    """
    Propagator for one hopping fragment using the
    diagonalization employed in the GAN algorithm:

        exp(-i F_s dt)
        = U_s exp(-i D_s dt) U_s^dagger.
    """
    unitary = matching_unitary(
        list(matching),
        n_orbitals,
    )

    diagonal = diagonal_hopping_fragment(
        matching,
        couplings,
        n_orbitals,
    )

    phases = np.exp(
        -1.0j
        * dt
        * np.diag(diagonal)
    )

    diagonal_propagator = np.diag(phases)

    return (
        unitary
        @ diagonal_propagator
        @ unitary.conj().T
    )