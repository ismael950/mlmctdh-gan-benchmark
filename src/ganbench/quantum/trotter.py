from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ganbench.quantum.toy_model import QuantumToyGAN

from ganbench.quantum.fragments import (
    apply_qdependent_hopping_fragment,
)
from ganbench.quantum.diagonal_functions import (
    occupation_table,
)

def apply_final_fragment(
    state: np.ndarray,
    model: QuantumToyGAN,
    dt: float,
) -> np.ndarray:
    """
    Apply

        exp[-i dt (
            T_nuc
            + sum_k epsilon_k n_k
        )]

    without diagonalizing the full joint matrix.

    The electronic-energy part is diagonal in the
    occupation basis.

    The nuclear kinetic energy is diagonal in the
    momentum basis, reached through the discrete
    Fourier transform.
    """

    n_el = model.electronic_dimension
    n_nuc = model.nuclear_dimension

    state_grid = state.reshape(
        n_el,
        n_nuc,
    )

    # ---------------------------------------------
    # Metal orbital energies
    # ---------------------------------------------

    occupations = occupation_table(
        model.n_electronic_orbitals
    )

    electronic_energies = np.zeros(
        n_el,
        dtype=float,
    )

    for orbital, energy in (
        model.metal_energies.items()
    ):
        electronic_energies += (
            energy
            * occupations[:, orbital]
        )

    electronic_phases = np.exp(
        -1.0j
        * dt
        * electronic_energies
    )

    state_grid = (
        electronic_phases[:, None]
        * state_grid
    )

    # ---------------------------------------------
    # Nuclear kinetic energy
    #
    # T = F^dagger diag(p^2 / 2m) F
    # ---------------------------------------------

    fourier = model.nuclear_grid.fourier

    momentum_state = (
        fourier @ state_grid.T
    ).T

    kinetic_energies = np.diag(
        fourier
        @ model.nuclear_grid.kinetic
        @ fourier.conj().T
    ).real

    kinetic_phases = np.exp(
        -1.0j
        * dt
        * kinetic_energies
    )

    momentum_state *= kinetic_phases[None, :]

    state_grid = (
        fourier.conj().T
        @ momentum_state.T
    ).T

    return state_grid.reshape(-1)

@dataclass(frozen=True)
class SpectralFragment:
    """
    Exact spectral representation of one Hermitian fragment.

        F = V diag(lambda) V^dagger

    so

        exp(-i F dt)
        = V exp(-i lambda dt) V^dagger.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @classmethod
    def from_operator(
        cls,
        operator: np.ndarray,
    ) -> "SpectralFragment":

        eigenvalues, eigenvectors = np.linalg.eigh(
            operator
        )

        return cls(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    def apply(
        self,
        state: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Apply exp(-i F dt) to a state.
        """
        coefficients = (
            self.eigenvectors.conj().T
            @ state
        )

        phases = np.exp(
            -1.0j
            * dt
            * self.eigenvalues
        )

        return (
            self.eigenvectors
            @ (phases * coefficients)
        )


class FirstOrderTrotter:
    """
    First-order product formula for

        H = F0 + F1 + ... + FN.

    During one Trotter step the state experiences

        F0 -> F1 -> ... -> FN

    sequentially.
    """

    def __init__(
        self,
        model: QuantumToyGAN,
    ) -> None:

        operators = (
            model.f0,
            *model.hopping_fragments,
            model.f_last,
        )

        self.fragments = tuple(
            SpectralFragment.from_operator(operator)
            for operator in operators
        )

    @property
    def n_fragments(self) -> int:
        return len(self.fragments)

    def step(
        self,
        state: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Apply one first-order Trotter step.
        """
        result = state.copy()

        for fragment in self.fragments:
            result = fragment.apply(
                result,
                dt,
            )

        return result

    def final_state(
        self,
        initial_state: np.ndarray,
        total_time: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Approximate

            exp(-i H T) |psi(0)>

        with n_steps first-order Trotter steps.
        """
        if n_steps <= 0:
            raise ValueError(
                "n_steps must be positive"
            )

        dt = total_time / n_steps

        state = initial_state.copy()

        for _ in range(n_steps):
            state = self.step(
                state,
                dt,
            )

        return state


def state_fidelity(
    exact: np.ndarray,
    approximate: np.ndarray,
) -> float:
    """
    Pure-state fidelity

        |<psi_exact | psi_approx>|^2.
    """
    overlap = np.vdot(
        exact,
        approximate,
    )

    return float(
        np.abs(overlap) ** 2
    )


def phase_aligned_state_error(
    exact: np.ndarray,
    approximate: np.ndarray,
) -> float:
    """
    State-vector error after removing an irrelevant
    global phase.
    """
    overlap = np.vdot(
        exact,
        approximate,
    )

    if np.abs(overlap) == 0.0:
        return float(
            np.linalg.norm(
                exact - approximate
            )
        )

    aligned = (
        approximate
        * np.exp(
            -1.0j * np.angle(overlap)
        )
    )

    return float(
        np.linalg.norm(
            exact - aligned
        )
    )

class GANAlgorithmTrotter:
    """
    First-order Trotter implementation using the
    hopping-fragment diagonalization of the GAN paper.

    F0 and F_last are still exponentiated exactly
    numerically for now.

    Each hopping fragment is implemented as

        exp(-i F_s dt)
        =
        U_s exp(-i D_s dt) U_s^dagger.
    """

    def __init__(
        self,
        model: QuantumToyGAN,
    ) -> None:

        self.model = model

    @property
    def n_fragments(self) -> int:
        return (
            2
            + len(self.model.matchings)
        )

    def step(
        self,
        state: np.ndarray,
        dt: float,
    ) -> np.ndarray:

        # F0:
        # exp[-i dt sum_lambda f_lambda(Q)c_lambda(n)]
        result = self.model.f0_expansion.apply(
            state,
            dt,
        )

        # Coordinate-dependent hopping fragments
        for matching in self.model.matchings:
            result = (
                apply_qdependent_hopping_fragment(
                    result,
                    matching,
                    self.model.hopping_profiles,
                    self.model.n_electronic_orbitals,
                    dt,
                )
            )

        # Final fragment:
        # metal energies + nuclear kinetic evolution
        result = apply_final_fragment(
            result,
            self.model,
            dt,
        )

        return result

    def final_state(
        self,
        initial_state: np.ndarray,
        total_time: float,
        n_steps: int,
    ) -> np.ndarray:

        if n_steps <= 0:
            raise ValueError(
                "n_steps must be positive"
            )

        dt = total_time / n_steps

        state = initial_state.copy()

        for _ in range(n_steps):
            state = self.step(
                state,
                dt,
            )

        return state