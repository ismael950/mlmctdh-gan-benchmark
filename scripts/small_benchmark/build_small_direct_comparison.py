from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "small_direct_benchmark"
OUT = BASE / "comparison"


def same_time_grid(a, b, label):
    if len(a) != len(b) or not np.allclose(a, b):
        raise RuntimeError(
            f"Time grids do not match: {label}"
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Exact, quantum and converged ML-MCTDH dynamics
    # ============================================================

    exact = pd.read_csv(
        BASE / "exact" / "reference_populations.csv"
    )

    # ============================================================
    # Select the most highly resolved Trotter trajectory
    # automatically
    # ============================================================

    trotter_table = pd.read_csv(
        BASE
        / "trotter_sweep"
        / "convergence.csv"
    )

    best_quantum_row = trotter_table.loc[
        trotter_table[
            "n_trotter_steps"
        ].idxmax()
    ]

    best_n_steps = int(
        best_quantum_row[
            "n_trotter_steps"
        ]
    )

    quantum = pd.read_csv(
        BASE
        / "trotter_sweep"
        / f"steps_{best_n_steps:04d}.csv"
    )

    print(
        "Selected quantum trajectory:",
        best_n_steps,
        "Trotter steps, dt =",
        best_quantum_row["dt_au"],
        "a.u.",
    )

    # ============================================================
    # Select the most highly resolved ML-MCTDH run automatically
    # ============================================================

    ml_candidates = []

    for run_dir in sorted(
        (BASE / "mlmctdh").glob("run_*")
    ):
        summary_file = run_dir / "summary.json"

        if not summary_file.exists():
            continue

        with summary_file.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        ml_candidates.append(
            (
                int(summary["n_ml_coefficients"]),
                run_dir,
                summary,
            )
        )

    if not ml_candidates:
        raise RuntimeError(
            "No ML-MCTDH runs found."
        )

    _, best_ml_dir, best_ml_summary = max(
        ml_candidates,
        key=lambda item: item[0],
    )

    ml = pd.read_csv(
        best_ml_dir / "observables.csv"
    )

    print(
        "Selected ML-MCTDH trajectory:",
        best_ml_dir.name,
        "with",
        best_ml_summary["n_ml_coefficients"],
        "coefficients",
    )

    same_time_grid(
        exact["time_au"],
        quantum["time_au"],
        "exact vs quantum",
    )

    same_time_grid(
        exact["time_au"],
        ml["time"],
        "exact vs ML-MCTDH",
    )

    population = pd.DataFrame(
        {
            "time_au": exact["time_au"],
            "time_fs": exact["time_fs"],
            "n0_exact": exact["n0_exact"],
            "n1_exact": exact["n1_exact"],
            "n0_mlmctdh": ml["d1"],
            "n1_mlmctdh": ml["d2"],
            "n0_quantum": quantum["n0_quantum"],
            "n1_quantum": quantum["n1_quantum"],
        }
    )

    population.to_csv(
        OUT / "population_dynamics.csv",
        index=False,
    )

    ml_error_n0 = np.abs(
        ml["d1"].to_numpy()
        - exact["n0_exact"].to_numpy()
    )

    ml_error_n1 = np.abs(
        ml["d2"].to_numpy()
        - exact["n1_exact"].to_numpy()
    )

    ml_error = np.maximum(
        ml_error_n0,
        ml_error_n1,
    )

    q_error_n0 = np.abs(
        quantum["n0_quantum"].to_numpy()
        - exact["n0_exact"].to_numpy()
    )

    q_error_n1 = np.abs(
        quantum["n1_quantum"].to_numpy()
        - exact["n1_exact"].to_numpy()
    )

    q_error = np.maximum(
        q_error_n0,
        q_error_n1,
    )

    error_dynamics = pd.DataFrame(
        {
            "time_au": exact["time_au"],
            "time_fs": exact["time_fs"],
            "mlmctdh_error": ml_error,
            "quantum_error": q_error,
        }
    )

    error_dynamics.to_csv(
        OUT / "error_dynamics.csv",
        index=False,
    )

    # ============================================================
    # ML-MCTDH convergence
    # ============================================================

    rows = []

    ml_root = BASE / "mlmctdh"

    for run_dir in sorted(ml_root.glob("run_*")):
        summary_file = run_dir / "summary.json"

        if not summary_file.exists():
            continue

        with summary_file.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        rows.append(
            {
                "run": run_dir.name,
                "n_ml_coefficients":
                    summary["n_ml_coefficients"],
                "max_error":
                    summary[
                        "max_molecular_orbital_population_error"
                    ],
                "final_error":
                    summary[
                        "final_max_molecular_orbital_population_error"
                    ],
                "wall_seconds":
                    summary.get("wall_seconds"),
                "cpu_seconds":
                    summary.get("cpu_seconds"),
                "max_norm_error":
                    summary.get("max_norm_error"),
                "lowest_natural_population":
                    summary.get(
                        "largest_expandable_lowest_natural_population"
                    ),
            }
        )

    convergence = pd.DataFrame(rows)

    convergence.to_csv(
        OUT / "mlmctdh_convergence.csv",
        index=False,
    )

    # ============================================================
    # Nuclear basis convergence
    # ============================================================

    basis = {}

    for file in (
        BASE / "basis_convergence"
    ).glob("K*.csv"):

        k = int(file.stem[1:])
        basis[k] = pd.read_csv(file)

    ks = sorted(basis)

    basis_rows = []

    for k1, k2 in zip(ks[:-1], ks[1:]):
        a = basis[k1]
        b = basis[k2]

        same_time_grid(
            a["time_au"],
            b["time_au"],
            f"K{k1} vs K{k2}",
        )

        error = np.maximum(
            np.abs(
                a["n0"].to_numpy()
                - b["n0"].to_numpy()
            ),
            np.abs(
                a["n1"].to_numpy()
                - b["n1"].to_numpy()
            ),
        )

        basis_rows.append(
            {
                "K_from": k1,
                "K_to": k2,
                "successive_max_error":
                    float(np.max(error)),
            }
        )

    basis_convergence = pd.DataFrame(
        basis_rows
    )

    basis_convergence.to_csv(
        OUT / "basis_convergence.csv",
        index=False,
    )

    # ============================================================
    # Trotter convergence
    # ============================================================

    trotter = pd.read_csv(
        BASE
        / "trotter_sweep"
        / "convergence.csv"
    )

    trotter.to_csv(
        OUT / "trotter_convergence.csv",
        index=False,
    )

    # ============================================================
    # Compact benchmark summary
    # ============================================================

    # Report the runs that were actually used above (selected
    # automatically) rather than hard-coded names.
    metrics = {
        "mlmctdh_run": best_ml_dir.name,
        "mlmctdh_n_coefficients":
            int(
                best_ml_summary[
                    "n_ml_coefficients"
                ]
            ),
        "mlmctdh_max_error":
            float(np.max(ml_error)),
        "quantum_trotter_step_au":
            float(best_quantum_row["dt_au"]),
        "quantum_n_trotter_steps":
            best_n_steps,
        "quantum_max_error":
            float(np.max(q_error)),
    }

    with open(
        OUT / "benchmark_metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
        )

    # ============================================================

    print("Comparison tables generated")
    print()
    print(
        "ML-MCTDH Emax =",
        np.max(ml_error),
    )
    print(
        "Quantum Emax =",
        np.max(q_error),
    )

    print()
    print("Basis convergence:")
    print(
        basis_convergence.to_string(
            index=False
        )
    )

    print()
    print("Trotter convergence available:")
    print(
        trotter.to_string(
            index=False
        )
    )

    print()
    print("Saved to:")
    print(OUT)


if __name__ == "__main__":
    main()