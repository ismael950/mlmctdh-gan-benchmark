from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def occupation_table(
    n_orbitals: int,
) -> np.ndarray:
    """
    Occupations of every electronic computational-basis state.

    Row b corresponds to the basis state

        |n0 n1 ... n_{N-1}>.

    The ordering matches the Jordan-Wigner convention used
    elsewhere in this project.
    """
    n_states = 2 ** n_orbitals

    indices = np.arange(
        n_states,
        dtype=np.uint64,
    )

    occupations = np.zeros(
        (n_states, n_orbitals),
        dtype=float,
    )

    for orbital in range(n_orbitals):
        shift = n_orbitals - 1 - orbital

        occupations[:, orbital] = (
            (indices >> shift) & 1
        )

    return occupations


@dataclass(frozen=True)
class DiagonalFunctionExpansion:
    """
    Representation

        D(Q,n) = sum_lambda
                 f_lambda(Q) c_lambda(n).

    function_values:
        shape (n_lambda, n_nuclear)

    coefficient_values:
        shape (n_lambda, n_electronic)

    This is the numerical analogue of the function/coefficient
    decomposition used in the GAN quantum algorithm.
    """

    function_values: np.ndarray
    coefficient_values: np.ndarray

    def energy_table(self) -> np.ndarray:
        """
        Values D(n,Q) for every joint basis state.

        Shape:
            (n_electronic, n_nuclear)
        """
        return (
            self.coefficient_values.T
            @ self.function_values
        )

    def apply(
        self,
        state: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Apply

            exp[-i dt D(Q,n)]

        directly in the joint |n,Q> basis.
        """
        energies = self.energy_table()

        expected_size = energies.size

        if state.size != expected_size:
            raise ValueError(
                "state dimension does not match expansion"
            )

        state_grid = state.reshape(
            energies.shape
        )

        phases = np.exp(
            -1.0j * dt * energies
        )

        return (
            phases * state_grid
        ).reshape(-1)


def build_f0_function_expansion(
    points: np.ndarray,
    n_orbitals: int,
    molecular_energies: dict[int, float],
    linear_couplings: dict[int, float],
    u0_quadratic: float,
    density_interactions: (
        dict[tuple[int, int], float] | None
    ) = None,
) -> DiagonalFunctionExpansion:
    """
    Build

        F0 = sum_lambda f_lambda(Q)c_lambda(n)

    using the basis

        f_0(Q) = 1
        f_1(Q) = Q
        f_2(Q) = Q^2.

    Currently supports

        epsilon_i n_i
        lambda_i Q n_i
        V_ij n_i n_j
        a Q^2.
    """
    points = np.asarray(
        points,
        dtype=float,
    )

    occupations = occupation_table(
        n_orbitals
    )

    n_electronic = occupations.shape[0]

    # ---------------------------------------------
    # f_lambda(Q)
    # ---------------------------------------------

    function_values = np.vstack(
        [
            np.ones_like(points),
            points,
            points**2,
        ]
    )

    # ---------------------------------------------
    # c_lambda(n)
    # ---------------------------------------------

    c_constant = np.zeros(
        n_electronic,
        dtype=float,
    )

    c_linear = np.zeros(
        n_electronic,
        dtype=float,
    )

    c_quadratic = np.full(
        n_electronic,
        u0_quadratic,
        dtype=float,
    )

    for orbital, energy in molecular_energies.items():
        c_constant += (
            energy
            * occupations[:, orbital]
        )

    for orbital, coupling in linear_couplings.items():
        c_linear += (
            coupling
            * occupations[:, orbital]
        )

    if density_interactions is not None:
        for (i, j), interaction in (
            density_interactions.items()
        ):
            c_constant += (
                interaction
                * occupations[:, i]
                * occupations[:, j]
            )

    coefficient_values = np.vstack(
        [
            c_constant,
            c_linear,
            c_quadratic,
        ]
    )

    return DiagonalFunctionExpansion(
        function_values=function_values,
        coefficient_values=coefficient_values,
    )