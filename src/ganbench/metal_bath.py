from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscretizedMetalBath:
    """
    Finite explicit representation of a metallic bath.

    energies_ev:
        Single-particle energies epsilon_k in eV.

    quadrature_weights_ev:
        Energy quadrature weights associated with the
        discrete metal orbitals.

    coupling_prefactors_ev:
        Factors multiplying the geometry-dependent
        molecule-metal coupling function.
    """

    energies_ev: np.ndarray
    quadrature_weights_ev: np.ndarray
    coupling_prefactors_ev: np.ndarray
    chemical_potential_ev: float

    @property
    def n_orbitals(self) -> int:
        return len(self.energies_ev)

    @property
    def occupied_indices(self) -> tuple[int, ...]:
        """
        Zero-temperature Fermi sea:
        all states below mu are initially occupied.
        """

        return tuple(
            int(index)
            for index, energy in enumerate(self.energies_ev)
            if energy < self.chemical_potential_ev
        )


def discretize_flat_metal_bath(
    n_orbitals: int,
    gamma_ev: float,
    bandwidth_ev: float,
    chemical_potential_ev: float = 0.0,
) -> DiscretizedMetalBath:
    """
    Discretize a finite wide-band metallic bath using
    Gauss-Legendre quadrature.

    The total interval is

        [mu - bandwidth/2, mu + bandwidth/2]

    and is split at the Fermi level:

        [mu - bandwidth/2, mu]
        [mu, mu + bandwidth/2].

    Each half contains the same number of states.

    For a constant hybridization Gamma,

        Gamma(epsilon)
        =
        2*pi*sum_k |V_k|^2 delta(epsilon-epsilon_k),

    the discrete coupling associated with a quadrature
    weight w_k is

        |V_k|^2 = Gamma*w_k/(2*pi).

    Parameters
    ----------
    n_orbitals
        Total number of discrete metal states.
        Must be even.

    gamma_ev
        Hybridization strength Gamma in eV.

    bandwidth_ev
        Total finite metallic bandwidth in eV.

    chemical_potential_ev
        Chemical potential / Fermi energy in eV.
    """

    if n_orbitals < 2 or n_orbitals % 2 != 0:
        raise ValueError(
            "n_orbitals must be a positive even integer."
        )

    if gamma_ev <= 0:
        raise ValueError(
            "gamma_ev must be positive."
        )

    if bandwidth_ev <= 0:
        raise ValueError(
            "bandwidth_ev must be positive."
        )

    n_half = n_orbitals // 2

    # Standard Gauss-Legendre quadrature on [-1, 1].
    nodes, weights = np.polynomial.legendre.leggauss(
        n_half
    )

    half_bandwidth = bandwidth_ev / 2.0

    # --------------------------------------------------------
    # Occupied side:
    #
    # [mu - bandwidth/2, mu]
    # --------------------------------------------------------

    lower_a = (
        chemical_potential_ev
        - half_bandwidth
    )

    lower_b = chemical_potential_ev

    energies_minus = (
        0.5 * (lower_b - lower_a) * nodes
        + 0.5 * (lower_b + lower_a)
    )

    weights_minus = (
        0.5
        * (lower_b - lower_a)
        * weights
    )

    # --------------------------------------------------------
    # Empty side:
    #
    # [mu, mu + bandwidth/2]
    # --------------------------------------------------------

    upper_a = chemical_potential_ev

    upper_b = (
        chemical_potential_ev
        + half_bandwidth
    )

    energies_plus = (
        0.5 * (upper_b - upper_a) * nodes
        + 0.5 * (upper_b + upper_a)
    )

    weights_plus = (
        0.5
        * (upper_b - upper_a)
        * weights
    )

    energies_ev = np.concatenate(
        (
            energies_minus,
            energies_plus,
        )
    )

    quadrature_weights_ev = np.concatenate(
        (
            weights_minus,
            weights_plus,
        )
    )

    # --------------------------------------------------------
    # Discretized molecule-metal coupling
    #
    # Gamma =
    # 2*pi * |V_k|^2 / w_k
    #
    # => V_k = sqrt(Gamma*w_k/(2*pi))
    # --------------------------------------------------------

    coupling_prefactors_ev = np.sqrt(
        gamma_ev
        * quadrature_weights_ev
        / (2.0 * np.pi)
    )

    # Sort from lowest to highest energy.
    order = np.argsort(
        energies_ev
    )

    return DiscretizedMetalBath(
        energies_ev=energies_ev[order],
        quadrature_weights_ev=(
            quadrature_weights_ev[order]
        ),
        coupling_prefactors_ev=(
            coupling_prefactors_ev[order]
        ),
        chemical_potential_ev=(
            chemical_potential_ev
        ),
    )