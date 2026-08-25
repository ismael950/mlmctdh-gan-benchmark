from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    exact_propagation,
    gaussian_nuclear_state,
    product_initial_state,
)
from ganbench.quantum.fermions import number_operator
from ganbench.quantum.toy_model import build_quantum_toy_gan


AU_TO_FS = 0.02418884326505

BASIS_SIZES = (8, 16, 32, 64)

TOTAL_TIME = 2000.0
TIME_STEP = 10.0

OCCUPIED = [0, 2, 3]


def main():

    times = np.arange(
        0.0,
        TOTAL_TIME + 0.5 * TIME_STEP,
        TIME_STEP,
    )

    root = Path(
        "results/small_direct_benchmark/"
        "basis_convergence"
    )
    root.mkdir(parents=True, exist_ok=True)

    results = {}

    for K in BASIS_SIZES:

        print()
        print("=" * 60)
        print(f"K = {K}")
        print("=" * 60)

        t0 = time.time()

        model = build_quantum_toy_gan(
            nuclear_size=K
        )

        electronic = electronic_basis_state(
            OCCUPIED,
            model.n_electronic_orbitals,
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

        print(
            "Dimension =",
            model.total_dimension,
        )

        states = exact_propagation(
            model.hamiltonian,
            psi0,
            times,
        )

        # electronic probabilities after tracing
        # over the nuclear coordinate
        reshaped = states.reshape(
            len(times),
            model.electronic_dimension,
            model.nuclear_dimension,
        )

        p_el = np.sum(
            np.abs(reshaped) ** 2,
            axis=2,
        )

        n0_diag = np.real(
            np.diag(
                number_operator(
                    0,
                    model.n_electronic_orbitals,
                )
            )
        )

        n1_diag = np.real(
            np.diag(
                number_operator(
                    1,
                    model.n_electronic_orbitals,
                )
            )
        )

        n0 = p_el @ n0_diag
        n1 = p_el @ n1_diag

        norms = np.sum(
            np.abs(states) ** 2,
            axis=1,
        )

        results[K] = {
            "n0": n0.copy(),
            "n1": n1.copy(),
            "norm": norms.copy(),
        }

        table = np.column_stack(
            (
                times,
                times * AU_TO_FS,
                n0,
                n1,
                norms,
            )
        )

        np.savetxt(
            root / f"K{K:02d}.csv",
            table,
            delimiter=",",
            header=(
                "time_au,time_fs,"
                "n0,n1,norm"
            ),
            comments="",
        )

        print(
            "Runtime =",
            time.time() - t0,
            "s",
        )

        print(
            "Max norm error =",
            np.max(np.abs(norms - 1.0)),
        )

        print(
            f"Final populations: "
            f"n0={n0[-1]:.12f}, "
            f"n1={n1[-1]:.12f}"
        )

        del states
        del reshaped
        del p_el
        del model
        gc.collect()

    print()
    print("=" * 60)
    print("Consecutive nuclear-grid changes")
    print("=" * 60)

    for K1, K2 in zip(
        BASIS_SIZES[:-1],
        BASIS_SIZES[1:],
    ):

        e0 = np.max(
            np.abs(
                results[K2]["n0"]
                - results[K1]["n0"]
            )
        )

        e1 = np.max(
            np.abs(
                results[K2]["n1"]
                - results[K1]["n1"]
            )
        )

        E = max(e0, e1)

        print(
            f"K={K1} -> {K2}: "
            f"E_n0={e0:.8e}, "
            f"E_n1={e1:.8e}, "
            f"E={E:.8e}"
        )


if __name__ == "__main__":
    main()