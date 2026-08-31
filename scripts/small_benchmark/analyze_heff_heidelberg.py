"""
Analyse the H_eff ML-MCTDH sweep run with Heidelberg MCTDH.

Reads the raw Heidelberg `expectation` files under
    results/small_direct_benchmark/heidelberg_heff/<run>/raw/
and writes, mirroring the original ML-MCTDH runs,
    results/small_direct_benchmark/mlmctdh_heff/<run>/observables.csv
    results/small_direct_benchmark/mlmctdh_heff/<run>/summary.json
    results/small_direct_benchmark/mlmctdh_heff/convergence.csv

`convergence.csv` is the payload: for each dt it gives

    E_eff_mlmctdh = max_t max_i | n_i^{H_eff, ML-MCTDH}(t) - n_i^{exact}(t) |

next to E_eff_exact (from effective_hamiltonian/convergence.csv) and E_trot
(from trotter_sweep/convergence.csv).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "small_direct_benchmark"
RAW = BASE / "heidelberg_heff"
OUT = BASE / "mlmctdh_heff"

N_STEPS_LIST = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
T_FINAL = 2000.0


def read_expectation(path: Path) -> pd.DataFrame:
    """Parse a Heidelberg `expectation` file (real-only, columns nd1..Qmean)."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        rows.append(vals)
    arr = np.array(rows)
    # columns: time, norm, nd1, nd2, nc1, nc2, nc3, nc4, Qmean
    if arr.shape[1] != 9:
        raise RuntimeError(
            f"{path}: expected 9 columns (real-only), got {arr.shape[1]}"
        )
    df = pd.DataFrame(
        arr,
        columns=["time", "norm", "d1", "d2", "c1", "c2", "c3", "c4", "Q1"],
    )
    df["P_mol"] = df["d1"] + df["d2"]
    df["N_total"] = df[["d1", "d2", "c1", "c2", "c3", "c4"]].sum(axis=1)
    return df[["time", "norm", "d1", "d2", "P_mol",
              "c1", "c2", "c3", "c4", "N_total", "Q1"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    exact = pd.read_csv(BASE / "exact" / "reference_populations.csv")
    ex0 = exact["n0_exact"].to_numpy()
    ex1 = exact["n1_exact"].to_numpy()
    times = exact["time_au"].to_numpy()

    heff_conv = pd.read_csv(
        BASE / "effective_hamiltonian" / "convergence.csv"
    ).set_index("n_trotter_steps")
    trot_conv = pd.read_csv(
        BASE / "trotter_sweep" / "convergence.csv"
    ).set_index("n_trotter_steps")

    def analyse_run(name: str):
        exp_path = RAW / name / "raw" / "expectation"
        if not exp_path.exists():
            print(f"  {name}: missing {exp_path}")
            return None
        obs = read_expectation(exp_path)
        run_out = OUT / name
        run_out.mkdir(exist_ok=True)
        obs.to_csv(run_out / "observables.csv", index=False)
        if not np.allclose(obs["time"].to_numpy(), times):
            raise RuntimeError(f"{name}: time grid differs from exact reference")
        e0 = np.abs(obs["d1"].to_numpy() - ex0)
        e1 = np.abs(obs["d2"].to_numpy() - ex1)
        combined = np.maximum(e0, e1)
        k = int(np.argmax(combined))
        summary = {
            "run_id": name,
            "max_molecular_orbital_population_error": float(combined[k]),
            "molecular_orbital_of_max_error": int(0 if e0[k] >= e1[k] else 1),
            "time_of_max_error_au": float(times[k]),
            "max_norm_error": float(np.abs(obs["norm"].to_numpy() - 1.0).max()),
            "max_particle_number_error": float(
                np.abs(obs["N_total"].to_numpy() - 3.0).max()
            ),
        }
        (run_out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary["max_molecular_orbital_population_error"]

    print("H_ref:", analyse_run("H_ref"))

    rows = []
    for idx, n_steps in enumerate(N_STEPS_LIST, start=1):
        dt = T_FINAL / n_steps
        e_ml = analyse_run(f"run_{idx:03d}")
        rows.append(
            {
                "run": f"run_{idx:03d}",
                "n_trotter_steps": n_steps,
                "dt_au": dt,
                "E_eff_mlmctdh": e_ml,
                "E_eff_exact": float(heff_conv.loc[n_steps, "E_eff"]),
                "E_trot": float(trot_conv.loc[n_steps, "max_error"]),
            }
        )
    conv = pd.DataFrame(rows)
    conv.to_csv(OUT / "convergence.csv", index=False)
    print("\n" + conv.to_string(index=False))
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
