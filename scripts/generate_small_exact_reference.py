from __future__ import annotations

import json
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


AU_TO_FS = 0.02418884326505


def expectation(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.vdot(state, operator @ state).real)


def main() -> None:
    model = build_quantum_toy_gan(
        nuclear_size=32,
    )

    # Canonical initial condition for the direct benchmark.
    occupied_orbitals = [0, 2, 3]
    gaussian_center = 0.0
    gaussian_sigma = 1.0

    electronic = electronic_basis_state(
        occupied_orbitals,
        model.n_electronic_orbitals,
    )

    nuclear = gaussian_nuclear_state(
        model.nuclear_grid.points,
        center=gaussian_center,
        sigma=gaussian_sigma,
    )

    psi0 = product_initial_state(
        electronic,
        nuclear,
    )

    # 0 ... 2000 a.u. in 10 a.u. intervals.
    total_time = 2000.0
    time_step = 10.0
    times = np.arange(
        0.0,
        total_time + 0.5 * time_step,
        time_step,
    )

    exact_states = exact_propagation(
        model.hamiltonian,
        psi0,
        times,
    )

    n0_operator = lift_electronic(
        number_operator(
            0,
            model.n_electronic_orbitals,
        ),
        model.nuclear_dimension,
    )

    n1_operator = lift_electronic(
        number_operator(
            1,
            model.n_electronic_orbitals,
        ),
        model.nuclear_dimension,
    )

    n0 = np.asarray([
        expectation(state, n0_operator)
        for state in exact_states
    ])

    n1 = np.asarray([
        expectation(state, n1_operator)
        for state in exact_states
    ])

    norms = np.asarray([
        float(np.vdot(state, state).real)
        for state in exact_states
    ])

    output_dir = Path(
        "results/small_direct_benchmark/exact"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        output_dir / "reference_populations.csv",
        table,
        delimiter=",",
        header="time_au,time_fs,n0_exact,n1_exact,norm",
        comments="",
    )

    metadata = {
        "description": (
            "Canonical exact reference for the "
            "small direct GAN benchmark"
        ),
        "total_dimension": int(model.total_dimension),
        "n_electronic_orbitals": int(
            model.n_electronic_orbitals
        ),
        "nuclear_dimension": int(
            model.nuclear_dimension
        ),
        "grid_points": [
            float(x)
            for x in model.nuclear_grid.points
        ],
        "molecular_energies": {
            str(k): float(v)
            for k, v in model.molecular_energies.items()
        },
        "linear_couplings": {
            str(k): float(v)
            for k, v in model.linear_couplings.items()
        },
        "u0_quadratic": float(
            model.u0_quadratic
        ),
        "metal_energies": {
            str(k): float(v)
            for k, v in model.metal_energies.items()
        },
        "hopping_couplings": {
            f"{i}-{j}": float(value)
            for (i, j), value
            in model.hopping_couplings.items()
        },
        "hopping_profiles": {
            f"{i}-{j}": [
                float(x)
                for x in values
            ]
            for (i, j), values
            in model.hopping_profiles.items()
        },
        "initial_condition": {
            "occupied_orbitals": occupied_orbitals,
            "nuclear_gaussian_center": gaussian_center,
            "nuclear_gaussian_sigma": gaussian_sigma,
        },
        "propagation": {
            "total_time_au": total_time,
            "time_step_au": time_step,
            "n_time_points": int(len(times)),
        },
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print(
        "Saved:",
        output_dir / "reference_populations.csv",
    )
    print(
        "Saved:",
        output_dir / "metadata.json",
    )
    print("Hilbert-space dimension =", model.total_dimension)
    print(
        "Maximum norm error =",
        np.max(np.abs(norms - 1.0)),
    )
    print(
        "Initial populations:",
        f"n0={n0[0]:.12f}",
        f"n1={n1[0]:.12f}",
    )
    print(
        "Final populations:",
        f"n0={n0[-1]:.12f}",
        f"n1={n1[-1]:.12f}",
    )


if __name__ == "__main__":
    main()