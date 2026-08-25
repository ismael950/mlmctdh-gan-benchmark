from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    gaussian_nuclear_state,
    product_initial_state,
)
from ganbench.quantum.fermions import number_operator
from ganbench.quantum.space import lift_electronic
from ganbench.quantum.toy_model import build_quantum_toy_gan
from ganbench.quantum.trotter import GANAlgorithmTrotter


def expectation(
    state: np.ndarray,
    operator: np.ndarray,
) -> float:
    return float(
        np.vdot(
            state,
            operator @ state,
        ).real
    )


def main() -> None:
    exact_path = Path(
        "results/small_direct_benchmark/"
        "exact/reference_populations.csv"
    )

    exact = np.genfromtxt(
        exact_path,
        delimiter=",",
        names=True,
    )

    times = np.asarray(
        exact["time_au"],
        dtype=float,
    )

    exact_n0 = np.asarray(
        exact["n0_exact"],
        dtype=float,
    )

    exact_n1 = np.asarray(
        exact["n1_exact"],
        dtype=float,
    )

    if len(times) < 2:
        raise RuntimeError(
            "Exact reference has too few time points."
        )

    dts = np.diff(times)

    if not np.allclose(dts, dts[0]):
        raise RuntimeError(
            "Exact reference time grid is not uniform."
        )

    dt = float(dts[0])

    model = build_quantum_toy_gan()

    electronic = electronic_basis_state(
        [0, 2, 3],
        model.n_electronic_orbitals,
    )

    nuclear = gaussian_nuclear_state(
        model.nuclear_grid.points,
        center=0.0,
        sigma=1.0,
    )

    state = product_initial_state(
        electronic,
        nuclear,
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

    propagator = GANAlgorithmTrotter(model)

    quantum_n0 = [
        expectation(state, n0_operator)
    ]

    quantum_n1 = [
        expectation(state, n1_operator)
    ]

    norms = [
        float(np.vdot(state, state).real)
    ]

    for _ in range(1, len(times)):
        state = propagator.step(
            state,
            dt,
        )

        quantum_n0.append(
            expectation(state, n0_operator)
        )

        quantum_n1.append(
            expectation(state, n1_operator)
        )

        norms.append(
            float(np.vdot(state, state).real)
        )

    quantum_n0 = np.asarray(quantum_n0)
    quantum_n1 = np.asarray(quantum_n1)
    norms = np.asarray(norms)

    error_n0 = np.abs(
        quantum_n0 - exact_n0
    )

    error_n1 = np.abs(
        quantum_n1 - exact_n1
    )

    max_molecular_error_vs_time = np.maximum(
        error_n0,
        error_n1,
    )

    global_index = int(
        np.argmax(max_molecular_error_vs_time)
    )

    global_error = float(
        max_molecular_error_vs_time[
            global_index
        ]
    )

    if (
        error_n0[global_index]
        >= error_n1[global_index]
    ):
        worst_orbital = 0
    else:
        worst_orbital = 1

    output_dir = Path(
        "results/small_direct_benchmark/quantum"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = np.column_stack(
        (
            exact["time_au"],
            exact["time_fs"],
            quantum_n0,
            quantum_n1,
            error_n0,
            error_n1,
            max_molecular_error_vs_time,
            norms,
        )
    )

    np.savetxt(
        output_dir / "quantum_populations.csv",
        table,
        delimiter=",",
        header=(
            "time_au,time_fs,"
            "n0_quantum,n1_quantum,"
            "error_n0,error_n1,"
            "max_molecular_error,norm"
        ),
        comments="",
    )

    metadata = {
        "method": (
            "First-order GAN quantum "
            "algorithm / Trotter emulation"
        ),
        "implementation": (
            "Operator-level implementation "
            "previously validated against "
            "the explicit PennyLane circuit"
        ),
        "trotter_step_au": dt,
        "n_trotter_steps": int(
            len(times) - 1
        ),
        "total_time_au": float(times[-1]),
        "global_error_definition": (
            "max_t max_i "
            "|n_i_quantum(t)-n_i_exact(t)|"
        ),
        "global_max_molecular_error": (
            global_error
        ),
        "worst_time_au": float(
            times[global_index]
        ),
        "worst_orbital": worst_orbital,
        "maximum_norm_error": float(
            np.max(
                np.abs(norms - 1.0)
            )
        ),
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

    print("Trotter dt =", dt, "a.u.")
    print(
        "Number of Trotter steps =",
        len(times) - 1,
    )
    print(
        "E_quantum =",
        global_error,
    )
    print(
        "Worst time =",
        times[global_index],
        "a.u.",
    )
    print(
        "Worst molecular orbital =",
        worst_orbital,
    )
    print(
        "Maximum norm error =",
        np.max(
            np.abs(norms - 1.0)
        ),
    )


if __name__ == "__main__":
    main()