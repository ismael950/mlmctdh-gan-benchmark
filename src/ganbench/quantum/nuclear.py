from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NuclearGrid:
    points: np.ndarray
    momenta: np.ndarray

    coordinate: np.ndarray
    kinetic: np.ndarray

    fourier: np.ndarray

    @property
    def size(self) -> int:
        return len(self.points)


def build_periodic_nuclear_grid(
    n_points: int,
    q_min: float,
    q_max: float,
    mass: float = 1.0,
) -> NuclearGrid:
    """
    Uniform periodic position grid.

    The coordinate operator Q is diagonal in the
    position/grid basis.

    The kinetic operator is diagonal in momentum space:

        T = P^2 / (2m)

    and transformed back to the position basis with a
    discrete Fourier transform.
    """
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1")

    if q_max <= q_min:
        raise ValueError("q_max must be greater than q_min")

    if mass <= 0:
        raise ValueError("mass must be positive")

    dq = (q_max - q_min) / n_points

    points = (
        q_min
        + dq * np.arange(n_points)
    )

    momenta = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            n_points,
            d=dq,
        )
    )

    coordinate = np.diag(
        points.astype(complex)
    )

    kinetic_momentum = np.diag(
        momenta**2 / (2.0 * mass)
    ).astype(complex)

    j = np.arange(n_points)
    k = np.arange(n_points)

    fourier = (
        np.exp(
            2.0j
            * np.pi
            * np.outer(j, k)
            / n_points
        )
        / np.sqrt(n_points)
    )

    kinetic = (
        fourier.conj().T
        @ kinetic_momentum
        @ fourier
    )

    return NuclearGrid(
        points=points,
        momenta=momenta,
        coordinate=coordinate,
        kinetic=kinetic,
        fourier=fourier,
    )