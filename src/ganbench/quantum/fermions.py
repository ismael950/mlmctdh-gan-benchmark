from __future__ import annotations

import numpy as np


I2 = np.array(
    [[1.0, 0.0],
     [0.0, 1.0]],
    dtype=complex,
)

X = np.array(
    [[0.0, 1.0],
     [1.0, 0.0]],
    dtype=complex,
)

Y = np.array(
    [[0.0, -1.0j],
     [1.0j, 0.0]],
    dtype=complex,
)

Z = np.array(
    [[1.0, 0.0],
     [0.0, -1.0]],
    dtype=complex,
)


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    """Kronecker product of a list of operators."""
    result = np.array([[1.0 + 0.0j]])

    for operator in operators:
        result = np.kron(result, operator)

    return result


def jordan_wigner_annihilation(
    orbital: int,
    n_orbitals: int,
) -> np.ndarray:
    """
    Fermionic annihilation operator a_i under Jordan-Wigner.

        a_i =
        Z_0 ... Z_{i-1} (X_i + i Y_i) / 2

    The computational state |0> means unoccupied
    and |1> means occupied.
    """
    if not 0 <= orbital < n_orbitals:
        raise ValueError("orbital index out of range")

    operators = []

    for q in range(n_orbitals):
        if q < orbital:
            operators.append(Z)
        elif q == orbital:
            operators.append((X + 1.0j * Y) / 2.0)
        else:
            operators.append(I2)

    return kron_all(operators)


def jordan_wigner_creation(
    orbital: int,
    n_orbitals: int,
) -> np.ndarray:
    """Fermionic creation operator a_i^dagger."""
    return jordan_wigner_annihilation(
        orbital,
        n_orbitals,
    ).conj().T


def number_operator(
    orbital: int,
    n_orbitals: int,
) -> np.ndarray:
    """Number operator n_i = a_i^dagger a_i."""
    a = jordan_wigner_annihilation(
        orbital,
        n_orbitals,
    )

    return a.conj().T @ a


def majorana(
    index: int,
    n_orbitals: int,
) -> np.ndarray:
    """
    Majorana operators used in the GAN paper.

        gamma_{2i}   = a_i + a_i^dagger
        gamma_{2i+1} = -i(a_i - a_i^dagger)

    Therefore each fermionic orbital i corresponds
    to two Majorana operators.
    """
    if not 0 <= index < 2 * n_orbitals:
        raise ValueError("Majorana index out of range")

    orbital = index // 2

    a = jordan_wigner_annihilation(
        orbital,
        n_orbitals,
    )
    adag = a.conj().T

    if index % 2 == 0:
        return a + adag

    return -1.0j * (a - adag)


def hopping_operator(
    i: int,
    j: int,
    n_orbitals: int,
) -> np.ndarray:
    """
    Fermionic hopping operator

        a_i^dagger a_j + a_j^dagger a_i.
    """
    ai = jordan_wigner_annihilation(i, n_orbitals)
    aj = jordan_wigner_annihilation(j, n_orbitals)

    return (
        ai.conj().T @ aj
        + aj.conj().T @ ai
    )


def hopping_operator_majorana(
    i: int,
    j: int,
    n_orbitals: int,
) -> np.ndarray:
    """
    Same hopping operator written in the Majorana basis:

        i/2 [
            gamma_{2i} gamma_{2j+1}
            + gamma_{2j} gamma_{2i+1}
        ].
    """
    gamma_2i = majorana(2 * i, n_orbitals)
    gamma_2i1 = majorana(2 * i + 1, n_orbitals)

    gamma_2j = majorana(2 * j, n_orbitals)
    gamma_2j1 = majorana(2 * j + 1, n_orbitals)

    return (
        0.5j
        * (
            gamma_2i @ gamma_2j1
            + gamma_2j @ gamma_2i1
        )
    )

from scipy.linalg import expm


def pauli_z_operator(
    orbital: int,
    n_orbitals: int,
) -> np.ndarray:
    """Pauli Z acting on one fermionic orbital."""
    if not 0 <= orbital < n_orbitals:
        raise ValueError("orbital index out of range")

    operators = [
        Z if q == orbital else I2
        for q in range(n_orbitals)
    ]

    return kron_all(operators)


def hopping_diagonalization_unitary(
    i: int,
    j: int,
    n_orbitals: int,
) -> np.ndarray:
    """
    Clifford rotation used in the GAN algorithm:

        U_ij = exp[
            -(pi/4) gamma_{2j+1} gamma_{2i+1}
        ].

    With our Majorana/Jordan-Wigner convention,

        U_ij^dagger
        (a_i^dagger a_j + a_j^dagger a_i)
        U_ij

        = (Z_i - Z_j) / 2.
    """
    gamma_i = majorana(
        2 * i + 1,
        n_orbitals,
    )

    gamma_j = majorana(
        2 * j + 1,
        n_orbitals,
    )

    generator = gamma_j @ gamma_i

    return expm(
        -(np.pi / 4.0) * generator
    )

def matching_unitary(
    edges: list[tuple[int, int]],
    n_orbitals: int,
) -> np.ndarray:
    """
    Product of the Clifford rotations U_ij for one matching.

    A matching requires that no orbital appears
    in more than one edge.
    """
    used_orbitals: set[int] = set()

    for i, j in edges:
        if i == j:
            raise ValueError("matching edge cannot connect an orbital to itself")

        if i in used_orbitals or j in used_orbitals:
            raise ValueError(
                "edges do not form a matching: "
                "an orbital appears more than once"
            )

        used_orbitals.add(i)
        used_orbitals.add(j)

    dimension = 2 ** n_orbitals
    result = np.eye(dimension, dtype=complex)

    for i, j in edges:
        result = (
            result
            @ hopping_diagonalization_unitary(
                i,
                j,
                n_orbitals,
            )
        )

    return result