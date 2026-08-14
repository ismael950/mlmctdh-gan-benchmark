from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import expm_multiply

from ganbench.exact.hamiltonian import ExactSystem


@dataclass
class PropagationResult:
    """Resultado de la propagación temporal exacta."""

    times: NDArray[np.float64]
    states: NDArray[np.complex128]
    norms: NDArray[np.float64]


def propagate_exact(
    system: ExactSystem,
    t_final: float,
    n_times: int,
) -> PropagationResult:
    """
    Propaga el estado inicial usando

        |psi(t)> = exp(-i H t) |psi(0)>

    con hbar = 1.

    Parameters
    ----------
    system:
        Sistema que contiene el Hamiltoniano y el estado inicial.
    t_final:
        Tiempo final de la propagación.
    n_times:
        Número de tiempos en los que se guarda el estado.

    Returns
    -------
    PropagationResult
        Tiempos, estados propagados y norma en cada tiempo.
    """

    if t_final <= 0:
        raise ValueError("t_final must be positive.")

    if n_times < 2:
        raise ValueError("n_times must be at least 2.")

    times = np.linspace(
        0.0,
        t_final,
        n_times,
        dtype=np.float64,
    )

    generator = -1j * system.hamiltonian

    states = expm_multiply(
        generator,
        system.initial_state,
        start=0.0,
        stop=t_final,
        num=n_times,
        endpoint=True,
    )

    states = np.asarray(states, dtype=np.complex128)

    norms = np.sum(
        np.abs(states) ** 2,
        axis=1,
    ).real

    return PropagationResult(
        times=times,
        states=states,
        norms=norms,
    )