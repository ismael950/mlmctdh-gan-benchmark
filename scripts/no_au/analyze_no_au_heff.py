"""
Analyse the NO/Au H_eff sweep and extract the Trotter-step estimate.

Reads the raw Heidelberg `expectation` files written by
    backend_inputs/benchmark3_no_au_scattering/heidelberg_heff/run_no_au_heff_heidelberg.sh
i.e.
    results/benchmark3_no_au_scattering/heidelberg_heff/{H_ref,run_001..004}/raw/expectation

Estimator (arXiv:2606.30738 IV.B.2 + App. B, same-tree difference):

    delta_diff(dt) = max_{t <= 50 fs} | P_mol^{H_eff,dt}(t) - P_mol^{H_ref}(t) |

    fit  log delta_diff = alpha * log dt + const     (gate: alpha in [0.85, 1.15])
    a    = slope through the origin over the alpha-ok window
    r*(eps) = ceil(a * t_total / eps) ,   t_total = 2067.0687 a.u. (50 fs)
    band [r*/1.27, r*]  from the small-system calibration E_eff/E_trot -> 1.27

Writes results/benchmark3_no_au_scattering/heidelberg_heff/{convergence.csv,estimate.json}
and figures/benchmark3_no_au_scattering/noau_heff_scaling.png .
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "benchmark3_no_au_scattering" / "heidelberg_heff"
FIGDIR = ROOT / "figures" / "benchmark3_no_au_scattering"

T_TOTAL = 2067.06866675911            # a.u.  (50 fs)
DT_SWEEP = {"run_001": 10.0, "run_002": 6.0, "run_003": 4.0, "run_004": 3.0}
CALIB = 1.27                          # small-system E_eff/E_trot -> ~1.27 as dt->0
EPS_LIST = [1e-2, 3e-3, 1e-3]
ALPHA_LO, ALPHA_HI = 0.85, 1.15


def read_expectation(path: Path) -> pd.DataFrame:
    """time, norm, nd, nc1..nc32, rmean, zmean  (real-only)."""
    rows = [
        [float(x) for x in ln.split()]
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    arr = np.array(rows)
    cols = ["time", "norm", "nd"] + [f"nc{k}" for k in range(1, 33)] + ["rmean", "zmean"]
    if arr.shape[1] != len(cols):
        raise RuntimeError(f"{path}: got {arr.shape[1]} cols, expected {len(cols)}")
    return pd.DataFrame(arr, columns=cols)


def max_abs_diff(t_a, p_a, t_b, p_b) -> float:
    lo, hi = max(t_a.min(), t_b.min()), min(t_a.max(), t_b.max())
    m = (t_a >= lo) & (t_a <= hi)
    return float(np.max(np.abs(p_a[m] - np.interp(t_a[m], t_b, p_b))))


def main() -> None:
    href_path = RAW / "H_ref" / "raw" / "expectation"
    if not href_path.exists():
        raise SystemExit(f"missing {href_path} -- run the sweep first")
    href = read_expectation(href_path)
    t_ref, p_ref = href["time"].to_numpy(), href["nd"].to_numpy()

    rows = []
    for run, dt in sorted(DT_SWEEP.items(), key=lambda kv: -kv[1]):
        ep = RAW / run / "raw" / "expectation"
        if not ep.exists():
            print(f"  {run}: not present yet, skipping")
            continue
        obs = read_expectation(ep)
        d = max_abs_diff(obs["time"].to_numpy(), obs["nd"].to_numpy(), t_ref, p_ref)
        rows.append({"run": run, "dt_au": dt, "r_steps": round(T_TOTAL / dt),
                     "delta_diff": d})
    if len(rows) < 2:
        raise SystemExit("need >= 2 completed H_eff runs to fit")
    df = pd.DataFrame(rows).sort_values("dt_au")

    dt = df["dt_au"].to_numpy()
    dd = df["delta_diff"].to_numpy()

    # local log-log slopes between consecutive points
    local = np.diff(np.log(dd)) / np.diff(np.log(dt))
    df["local_slope_to_next"] = list(local) + [np.nan]

    # global fit + alpha-ok window (drop largest dt while slope out of band)
    def fit(x, y):
        a1 = np.polyfit(np.log(x), np.log(y), 1)
        return a1[0], np.exp(a1[1])

    alpha, _ = fit(dt, dd)
    mask = np.ones(len(dt), bool)
    while mask.sum() >= 2:
        alpha, _ = fit(dt[mask], dd[mask])
        if ALPHA_LO <= alpha <= ALPHA_HI:
            break
        mask[np.argmax(dt * mask)] = False          # drop current largest dt

    a = float(np.sum(dd[mask] * dt[mask]) / np.sum(dt[mask] ** 2))   # slope through origin
    used = df["run"].to_numpy()[mask].tolist()

    est = {
        "t_total_au": T_TOTAL, "window_fs": 50.0,
        "alpha_fit": float(alpha), "alpha_gate": [ALPHA_LO, ALPHA_HI],
        "alpha_ok": bool(ALPHA_LO <= alpha <= ALPHA_HI),
        "slope_a_per_au": a, "window_runs": used,
        "r_star": {}, "note": "r*_350fs ~= 7 * r*_50fs (conservative)",
    }
    for eps in EPS_LIST:
        r = int(np.ceil(a * T_TOTAL / eps))
        est["r_star"][f"{eps:.0e}"] = {"r": r, "band": [int(np.ceil(r / CALIB)), r],
                                      "r_350fs_est": 7 * r}

    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "convergence.csv", index=False)
    (RAW / "estimate.json").write_text(json.dumps(est, indent=2))

    print(df.to_string(index=False))
    print(f"\nalpha (log-log fit)      : {alpha:.3f}   gate [{ALPHA_LO},{ALPHA_HI}]  "
          f"{'OK' if est['alpha_ok'] else 'OUT OF BAND -- estimate unreliable'}")
    print(f"window runs               : {used}")
    print(f"slope a                   : {a:.4e}  (per a.u.)")
    for eps in EPS_LIST:
        b = est["r_star"][f"{eps:.0e}"]
        print(f"  eps={eps:.0e}:  r* = {b['r']:5d}   band {b['band']}   "
              f"(~{b['r_350fs_est']} over 350 fs)")

    try:
        import matplotlib.pyplot as plt
        FIGDIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.loglog(dt, dd, "o", ms=8, label=r"$\delta_{\rm diff}(\Delta t)$ (same-tree)")
        xs = np.linspace(dt.min() * 0.8, dt.max() * 1.2, 50)
        ax.loglog(xs, a * xs, "-", color="0.4", label=rf"fit $a\,\Delta t$,  $\alpha$={alpha:.2f}")
        for eps in EPS_LIST:
            ax.axhline(eps, ls=":", lw=1, color="C3")
            ax.text(dt.max(), eps * 1.1, f"$\\epsilon$={eps:.0e}", fontsize=8, ha="right")
        ax.set_xlabel(r"Trotter step $\Delta t$ (a.u.)")
        ax.set_ylabel(r"max$_t$ $|\Delta P_{\rm mol}|$ over 50 fs")
        ax.set_title("NO/Au: effective-Hamiltonian Trotter-error estimate")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGDIR / "noau_heff_scaling.png", dpi=200)
        print(f"\nsaved {FIGDIR / 'noau_heff_scaling.png'}")
    except Exception as e:            # noqa: BLE001
        print(f"(plot skipped: {e})")

    print(f"saved {RAW / 'convergence.csv'}  and  {RAW / 'estimate.json'}")


if __name__ == "__main__":
    main()
