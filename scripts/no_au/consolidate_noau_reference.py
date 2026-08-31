"""
Consolidate the NO/Au(111) ML-MCTDH rank-convergence study (true H, 50 fs).

There is no exact reference for this system, so "convergence" is measured as the
self-consistent change of the molecular population between successive SPF trees:

    delta_prev(run)  = max_t | P_mol^run(t) - P_mol^{prev run}(t) |
    delta_final(run) = max_t | P_mol^run(t) - P_mol^{last run}(t) |

Writes
    results/benchmark3_no_au_scattering/heidelberg/convergence.csv
    figures/benchmark3_no_au_scattering/mlmctdh_reference_convergence.png

and prints the ML-MCTDH self-consistency floor, which is the noise floor the
Trotter-error estimator has to clear (or cancel as common mode).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HEID = ROOT / "results" / "benchmark3_no_au_scattering" / "heidelberg"
FIGDIR = ROOT / "figures" / "benchmark3_no_au_scattering"

RUN_RE = re.compile(r"^run_(\d+)$")


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(HEID.glob("run_*"), key=lambda p: int(RUN_RE.match(p.name).group(1))):
        obs_path = d / "analysis" / "observables.csv"
        sum_path = d / "analysis" / "summary.json"
        if not obs_path.exists() or not sum_path.exists():
            print(f"  skip {d.name}: missing analysis")
            continue
        obs = pd.read_csv(obs_path)
        summary = json.loads(sum_path.read_text())
        runs.append(
            {
                "run": d.name,
                "n_ml_coefficients": summary.get("n_ml_coefficients"),
                "wall_seconds": summary.get("wall_seconds"),
                "max_norm_error": summary.get("max_norm_error"),
                "largest_expandable_lowest_natural_population": summary.get(
                    "largest_expandable_lowest_natural_population"
                ),
                "t": obs["time"].to_numpy(),
                "P_mol": obs["P_mol"].to_numpy(),
            }
        )
    if not runs:
        raise RuntimeError(f"no analyzed runs under {HEID}")
    return runs


def max_abs_diff(t_a, p_a, t_b, p_b) -> float:
    lo, hi = max(t_a.min(), t_b.min()), min(t_a.max(), t_b.max())
    m = (t_a >= lo) & (t_a <= hi)
    p_b_on_a = np.interp(t_a[m], t_b, p_b)
    return float(np.max(np.abs(p_a[m] - p_b_on_a)))


def main() -> None:
    runs = load_runs()
    last = runs[-1]

    rows = []
    for i, r in enumerate(runs):
        prev = runs[i - 1] if i > 0 else None
        rows.append(
            {
                "run": r["run"],
                "n_ml_coefficients": r["n_ml_coefficients"],
                "wall_hours": None if r["wall_seconds"] is None else r["wall_seconds"] / 3600.0,
                "P_mol_final": float(r["P_mol"][-1]),
                "P_mol_max": float(r["P_mol"].max()),
                "delta_prev": (
                    np.nan
                    if prev is None
                    else max_abs_diff(r["t"], r["P_mol"], prev["t"], prev["P_mol"])
                ),
                "delta_final_run": max_abs_diff(
                    r["t"], r["P_mol"], last["t"], last["P_mol"]
                ),
                "max_norm_error": r["max_norm_error"],
                "largest_expandable_lowest_natural_population": r[
                    "largest_expandable_lowest_natural_population"
                ],
            }
        )
    df = pd.DataFrame(rows)

    out_csv = HEID / "convergence.csv"
    df.to_csv(out_csv, index=False)

    # ML-MCTDH self-consistency floor: median |delta_prev| over the last few
    # trees that are already SPF-saturated.
    tail = df["delta_prev"].to_numpy()[-4:]
    floor = float(np.nanmedian(tail))

    print(df.to_string(index=False))
    print()
    print(f"ML-MCTDH self-consistency floor (median delta_prev, last 4): {floor:.2e}")
    print(f"  -> Trotter target epsilon must sit at/above this; the H_eff vs H")
    print(f"     same-tree difference cancels it as common mode.")
    print(f"Saved: {out_csv}")

    # -------------------------------------------------------------- figure ----
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    nc = df["n_ml_coefficients"].to_numpy()
    ax1.plot(nc, df["delta_prev"], "o-", label=r"$\delta_{\rm prev}$ (successive trees)")
    ax1.plot(nc, df["delta_final_run"], "s--", label=rf"$\delta$ vs {last['run']}")
    ax1.axhline(floor, color="0.5", ls=":", label=f"floor $\\approx${floor:.1e}")
    ax1.set_yscale("log")
    ax1.set_xlabel("# time-dependent ML coefficients")
    ax1.set_ylabel(r"$\max_t\,|\Delta P_{\rm mol}(t)|$")
    ax1.set_title("NO/Au ML-MCTDH self-convergence (true $H$, 50 fs)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8)

    for _, row in df.iterrows():
        ax2.plot(nc, df["P_mol_max"], "o-", color="C2")
    ax2.set_xlabel("# time-dependent ML coefficients")
    ax2.set_ylabel(r"$\max_t P_{\rm mol}(t)$")
    ax2.set_title("Peak molecular population vs tree size")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGDIR / "mlmctdh_reference_convergence.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIGDIR / 'mlmctdh_reference_convergence.png'}")


if __name__ == "__main__":
    main()
