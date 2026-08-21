from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply


def electronic_basis_state(
    occupied_orbitals: list[int],
    n_orbitals: int,
) -> np.ndarray:
    """
    Construct the fermionic occupation state

        |n0 n1 ... n_{N-1}>,

    where |0> = unoccupied and |1> = occupied.
    """
    occupied = set(occupied_orbitals)

    state = np.array([1.0 + 0.0j])

    for orbital in range(n_orbitals):
        if orbital in occupied:
            local = np.array(
                [0.0, 1.0],
                dtype=complex,
            )
        else:
            local = np.array(
                [1.0, 0.0],
                dtype=complex,
            )

        state = np.kron(state, local)

    return state


def gaussian_nuclear_state(
    points: np.ndarray,
    center: float = 0.0,
    sigma: float = 1.0,
    momentum: float = 0.0,
) -> np.ndarray:
    """
    Normalized Gaussian wavepacket on the nuclear grid.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    state = np.exp(
        -0.5 * ((points - center) / sigma) ** 2
        + 1.0j * momentum * points
    )

    norm = np.linalg.norm(state)

    return state / norm


def product_initial_state(
    electronic_state: np.ndarray,
    nuclear_state: np.ndarray,
) -> np.ndarray:
    """Construct |psi_el> tensor |psi_nuc>."""
    state = np.kron(
        electronic_state,
        nuclear_state,
    )

    return state / np.linalg.norm(state)


def exact_propagation(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute

        |psi(t)> = exp(-i H t) |psi(0)>

    for all requested times.
    """
    times = np.asarray(times, dtype=float)

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")

    if len(times) < 2:
        raise ValueError("at least two times are required")

    if not np.allclose(
        np.diff(times),
        np.diff(times)[0],
    ):
        raise ValueError(
            "current implementation requires equally spaced times"
        )

    return expm_multiply(
        -1.0j * hamiltonian,
        initial_state,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )


def expectation_values(
    states: np.ndarray,
    operator: np.ndarray,
) -> np.ndarray:
    """Expectation <psi(t)|O|psi(t)>."""
    return np.asarray(
        [
            np.vdot(state, operator @ state).real
            for state in states
        ]
    )