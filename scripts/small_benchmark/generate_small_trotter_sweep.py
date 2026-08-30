from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    gaussian_nuclear_state,
    product_initial_state,
)
from ganbench.quantum.fermions import (
    number_operator,
)
from ganbench.quantum.space import (
    lift_electronic,
)
from ganbench.quantum.toy_model import (
    build_quantum_toy_gan,
)
from ganbench.quantum.trotter import (
    GANAlgorithmTrotter,
)


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "results"
    / "small_direct_benchmark"
)

OUT = BASE / "trotter_sweep"

N_STEPS_LIST = [
    200,
    400,
    600,
    800,
    1000,
    1200,
    1400,
    1600,
]


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

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Exact trajectory = source of truth
    # ------------------------------------------------------------

    exact = pd.read_csv(
        BASE
        / "exact"
        / "reference_populations.csv"
    )

    times = exact["time_au"].to_numpy()
    times_fs = exact["time_fs"].to_numpy()

    exact_n0 = exact["n0_exact"].to_numpy()
    exact_n1 = exact["n1_exact"].to_numpy()

    output_dt = float(
        times[1] - times[0]
    )

    total_time = float(times[-1])

    if not np.allclose(
        np.diff(times),
        output_dt,
    ):
        raise RuntimeError(
            "Exact observation grid is not uniform."
        )

    # ------------------------------------------------------------
    # Same K32 physical model for every Trotter run
    # ------------------------------------------------------------

    model = build_quantum_toy_gan(
        nuclear_size=32,
    )

    electronic = electronic_basis_state(
        [0, 2, 3],
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

    trotter = GANAlgorithmTrotter(
        model
    )

    summary_rows = []

    # ------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------

    for n_steps in N_STEPS_LIST:

        dt = total_time / n_steps

        substeps_float = (
            output_dt / dt
        )

        substeps = int(
            round(substeps_float)
        )

        if not np.isclose(
            substeps_float,
            substeps,
        ):
            raise RuntimeError(
                f"{n_steps} steps do not align "
                "with the exact observation grid."
            )

        print()
        print(
            f"Running {n_steps} Trotter steps"
        )
        print(
            f"dt = {dt} a.u."
        )

        state = psi0.copy()

        n0_values = [
            expectation(
                state,
                n0_operator,
            )
        ]

        n1_values = [
            expectation(
                state,
                n1_operator,
            )
        ]

        norms = [
            float(
                np.vdot(
                    state,
                    state,
                ).real
            )
        ]

        # Propagate from one exact observation
        # time to the next.
        for _ in range(
            1,
            len(times),
        ):

            for _ in range(
                substeps
            ):
                state = trotter.step(
                    state,
                    dt,
                )

            n0_values.append(
                expectation(
                    state,
                    n0_operator,
                )
            )

            n1_values.append(
                expectation(
                    state,
                    n1_operator,
                )
            )

            norms.append(
                float(
                    np.vdot(
                        state,
                        state,
                    ).real
                )
            )

        n0_values = np.asarray(
            n0_values
        )

        n1_values = np.asarray(
            n1_values
        )

        norms = np.asarray(
            norms
        )

        error_n0 = np.abs(
            n0_values - exact_n0
        )

        error_n1 = np.abs(
            n1_values - exact_n1
        )

        max_error_vs_time = np.maximum(
            error_n0,
            error_n1,
        )

        global_index = int(
            np.argmax(
                max_error_vs_time
            )
        )

        global_error = float(
            max_error_vs_time[
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

        # --------------------------------------------------------
        # Save the entire trajectory
        # --------------------------------------------------------

        trajectory = pd.DataFrame(
            {
                "time_au": times,
                "time_fs": times_fs,
                "n0_quantum": n0_values,
                "n1_quantum": n1_values,
                "error_n0": error_n0,
                "error_n1": error_n1,
                "max_molecular_error":
                    max_error_vs_time,
                "norm": norms,
            }
        )

        trajectory_file = (
            OUT
            / f"steps_{n_steps:04d}.csv"
        )

        trajectory.to_csv(
            trajectory_file,
            index=False,
        )

        summary_rows.append(
            {
                "n_trotter_steps":
                    n_steps,
                "dt_au":
                    dt,
                "max_error":
                    global_error,
                "worst_time_au":
                    float(
                        times[
                            global_index
                        ]
                    ),
                "worst_orbital":
                    worst_orbital,
                "max_norm_error":
                    float(
                        np.max(
                            np.abs(
                                norms - 1.0
                            )
                        )
                    ),
            }
        )

        print(
            "Emax =",
            global_error,
        )

    # ------------------------------------------------------------
    # Final convergence table
    # ------------------------------------------------------------

    convergence = pd.DataFrame(
        summary_rows
    )

    convergence.to_csv(
        OUT / "convergence.csv",
        index=False,
    )

    print()
    print(
        "Trotter convergence:"
    )
    print(
        convergence.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved to:",
        OUT,
    )


if __name__ == "__main__":
    main()