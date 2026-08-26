from __future__ import annotations

import numpy as np

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    gaussian_nuclear_state,
    product_initial_state,
)
from ganbench.quantum.pennylane_backend import (
    gan_trotter_final_state,
)
from ganbench.quantum.toy_model import (
    build_quantum_toy_gan,
)
from ganbench.quantum.trotter import (
    GANAlgorithmTrotter,
    phase_aligned_state_error,
    state_fidelity,
)


TOL = 1.0e-11


def compare(
    label: str,
    state: np.ndarray,
    model,
    total_time: float,
    n_steps: int,
) -> None:
    operator_state = GANAlgorithmTrotter(
        model
    ).final_state(
        state,
        total_time,
        n_steps,
    )

    circuit_state = gan_trotter_final_state(
        state,
        model,
        total_time,
        n_steps,
    )

    error = phase_aligned_state_error(
        operator_state,
        circuit_state,
    )

    fidelity = state_fidelity(
        operator_state,
        circuit_state,
    )

    print(label)
    print("  circuit/operator error =", error)
    print("  fidelity =", fidelity)

    if error > TOL:
        raise RuntimeError(
            f"{label} failed: error={error}"
        )


def main() -> None:
    model = build_quantum_toy_gan(
        nuclear_size=32,
    )

    electronic = electronic_basis_state(
        [0, 2, 3],
        6,
    )

    nuclear = gaussian_nuclear_state(
        model.nuclear_grid.points,
        center=0.0,
        sigma=1.0,
    )

    physical_state = product_initial_state(
        electronic,
        nuclear,
    )

    print("dimension =", physical_state.size)
    print(
        "qubits =",
        int(np.log2(physical_state.size)),
    )

    compare(
        "Physical state, 1 step",
        physical_state,
        model,
        total_time=2.5,
        n_steps=1,
    )

    compare(
        "Physical state, 4 steps",
        physical_state,
        model,
        total_time=10.0,
        n_steps=4,
    )

    rng = np.random.default_rng(12345)

    random_state = (
        rng.normal(size=physical_state.size)
        + 1.0j
        * rng.normal(size=physical_state.size)
    )

    random_state /= np.linalg.norm(
        random_state
    )

    compare(
        "Random state, 1 step",
        random_state,
        model,
        total_time=2.5,
        n_steps=1,
    )

    print("K32 CIRCUIT VALIDATION PASSED")


if __name__ == "__main__":
    main()