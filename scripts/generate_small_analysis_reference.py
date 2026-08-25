from pathlib import Path
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


def expectation(state, operator):
    return float(np.vdot(state, operator @ state).real)


def main():
    model = build_quantum_toy_gan(nuclear_size=32)

    electronic = electronic_basis_state(
        [0, 2, 3],
        model.n_electronic_orbitals,
    )
    nuclear = gaussian_nuclear_state(
        model.nuclear_grid.points,
        center=0.0,
        sigma=1.0,
    )
    psi0 = product_initial_state(electronic, nuclear)

    times = np.arange(0.0, 2000.0 + 5.0, 10.0)
    states = exact_propagation(
        model.hamiltonian,
        psi0,
        times,
    )

    populations = []
    for orbital in range(6):
        op = lift_electronic(
            number_operator(
                orbital,
                model.n_electronic_orbitals,
            ),
            model.nuclear_dimension,
        )
        populations.append(
            np.asarray([
                expectation(state, op)
                for state in states
            ])
        )

    norms = np.asarray([
        float(np.vdot(state, state).real)
        for state in states
    ])

    out = Path("results/small_direct_benchmark/exact")
    out.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        out / "observables.csv",
        np.column_stack(
            [times, *populations, norms]
        ),
        delimiter=",",
        header="time,d1,d2,c1,c2,c3,c4,norm",
        comments="",
    )

    print("Exact K32 reference generated")
    print("points =", len(times))
    print("max norm error =", np.max(np.abs(norms - 1.0)))
    print("final d1 =", populations[0][-1])
    print("final d2 =", populations[1][-1])


if __name__ == "__main__":
    main()
