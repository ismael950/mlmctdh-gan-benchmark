from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicNuclearGrid:
    """
    Periodic real-space grid using the convention of
    Lang et al. for the quantum GAN algorithm.

    Computational basis index:
        0, 1, ..., K-1

    is interpreted as a two's-complement signed integer

        0, 1, ..., K/2-1, -K/2, ..., -1.

    Position eigenvalues:

        Q |x> = Delta * x |x>

    with

        Delta = sqrt(2*pi/K).

    Momentum is defined by

        P = QFT^dagger Q QFT.
    """

    size: int
    delta: float

    signed_indices: np.ndarray
    points: np.ndarray

    fourier: np.ndarray

    coordinate: np.ndarray
    momentum: np.ndarray
    kinetic: np.ndarray


def build_periodic_nuclear_grid(
    size: int,
    mass: float,
) -> PeriodicNuclearGrid:
    """
    Construct the finite nuclear representation used by
    the quantum GAN algorithm.
    """

    if size < 2:
        raise ValueError("size must be at least 2")

    if size & (size - 1):
        raise ValueError(
            "size must be a power of two"
        )

    if mass <= 0.0:
        raise ValueError(
            "mass must be positive"
        )

    # -------------------------------------------------
    # Two's-complement signed integers
    #
    # K=8:
    # 000 ->  0
    # 001 ->  1
    # 010 ->  2
    # 011 ->  3
    # 100 -> -4
    # 101 -> -3
    # 110 -> -2
    # 111 -> -1
    # -------------------------------------------------

    unsigned = np.arange(
        size,
        dtype=int,
    )

    signed_indices = np.where(
        unsigned < size // 2,
        unsigned,
        unsigned - size,
    )

    # Eq. 10 of Lang et al.
    delta = np.sqrt(
        2.0 * np.pi / size
    )

    points = (
        delta
        * signed_indices.astype(float)
    )

    coordinate = np.diag(
        points.astype(complex)
    )

    # -------------------------------------------------
    # QFT matrix
    # -------------------------------------------------

    j = np.arange(size)
    k = np.arange(size)

    fourier = (
        np.exp(
            2.0j
            * np.pi
            * np.outer(j, k)
            / size
        )
        / np.sqrt(size)
    )

    # Eq. 11:
    #
    # P = QFT^dagger Q QFT
    #
    momentum = (
        fourier.conj().T
        @ coordinate
        @ fourier
    )

    kinetic = (
        momentum @ momentum
        / (2.0 * mass)
    )

    return PeriodicNuclearGrid(
        size=size,
        delta=delta,
        signed_indices=signed_indices,
        points=points,
        fourier=fourier,
        coordinate=coordinate,
        momentum=momentum,
        kinetic=kinetic,
    )