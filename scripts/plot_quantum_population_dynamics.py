from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    exact_propagation,
    gaussian_nuclear_state,
    product_initial_state,
)
from ganbench.quantum.fermions import number_operator
from ganbench.quantum.space import lift_electronic
from ganbench.quantum.toy_model import build_quantum_toy_gan
from ganbench.quantum.trotter import GANAlgorithmTrotter


def expectation(state, operator):
    return float(
        np.vdot(
            state,
            operator @ state,
        ).real
    )


def main():
    model = build_quantum_toy_gan()

    electronic = electronic_basis_state(
        [0, 2, 3],
        6,
    )

    nuclear = gaussian_nuclear_state(
        model.nuclear_grid.points,
        center=0.0,
        sigma=1.0,
    )

    psi0 = product_initial_state(
        electronic,
        nuclear,
    )

    total_time = 2000.0
    n_steps = 64

    times = np.linspace(
        0.0,
        total_time,
        n_steps + 1,
    )

    exact_states = exact_propagation(
        model.hamiltonian,
        psi0,
        times,
    )

    n0 = lift_electronic(
        number_operator(0, 6),
        model.nuclear_dimension,
    )

    n1 = lift_electronic(
        number_operator(1, 6),
        model.nuclear_dimension,
    )

    exact_n0 = np.array([
        expectation(state, n0)
        for state in exact_states
    ])

    exact_n1 = np.array([
        expectation(state, n1)
        for state in exact_states
    ])

    trotter = GANAlgorithmTrotter(model)

    dt = total_time / n_steps

    state = psi0.copy()

    trotter_n0 = [expectation(state, n0)]
    trotter_n1 = [expectation(state, n1)]

    for _ in range(n_steps):
        state = trotter.step(
            state,
            dt,
        )

        trotter_n0.append(
            expectation(state, n0)
        )

        trotter_n1.append(
            expectation(state, n1)
        )

    trotter_n0 = np.asarray(trotter_n0)
    trotter_n1 = np.asarray(trotter_n1)

    output = Path("results/quantum_toy")
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        output / "population_dynamics.txt",
        np.column_stack(
            (
                times,
                exact_n0,
                trotter_n0,
                exact_n1,
                trotter_n1,
            )
        ),
        header=(
            "time exact_n0 trotter_n0 "
            "exact_n1 trotter_n1"
        ),
    )

    plt.plot(
        times,
        exact_n0,
        "-",
        label="Exact n0",
    )

    plt.plot(
        times,
        trotter_n0,
        "--",
        label="Trotter n0",
    )

    plt.plot(
        times,
        exact_n1,
        "-",
        label="Exact n1",
    )

    plt.plot(
        times,
        trotter_n1,
        "--",
        label="Trotter n1",
    )

    plt.xlabel("Time (a.u.)")
    plt.ylabel("Molecular orbital population")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output / "population_dynamics.png",
        dpi=300,
    )

    print(
        "max |n0 exact - Trotter| =",
        np.max(
            np.abs(
                exact_n0 - trotter_n0
            )
        ),
    )

    print(
        "max |n1 exact - Trotter| =",
        np.max(
            np.abs(
                exact_n1 - trotter_n1
            )
        ),
    )


if __name__ == "__main__":
    main()