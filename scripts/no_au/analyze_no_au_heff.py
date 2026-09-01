"""
Analyse the NO/Au H_eff sweep and extract the Trotter-step estimate.

Reads  results/benchmark3_no_au_scattering/heidelberg_heff/{H_ref,run_001..,check_full}/raw/expectation
(written by run_no_au_heff_heidelberg.sh / the sbatch array).

Estimator (arXiv:2606.30738 IV.B.2 + App. B, same-tree difference):

    delta(dt) = < robust functional of | P_mol^{H_eff,dt}(t) - P_mol^{H_ref}(t) | >

    fit  log delta = alpha * log dt + c      (alpha = observable scaling exponent)
    dt* = dt_obs * (eps / delta_obs)^(1/alpha)
    r*(eps) = ceil(t_total / dt*)

Three delta metrics are reported (they disagree when the sweep is not yet in a
clean power-law regime):
  final      |dP| at t = 50 fs
  late_mean  mean |dP| over the last 10 output times
  max_t      max_t |dP|  (noise-contaminated by transients -- shown, not used)

Noise floor = max_t | P_mol^{H_ref}  -  P_mol^{run_009 (same H, same tree)} |.
check_full  = same as run at CHECK_FULL_DT but with the full E1 ([H_j,H_k] back);
             compared against the matching reduced run to validate the truncation.

Writes results/.../heidelberg_heff/{convergence.csv, estimate.json} and a figure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "benchmark3_no_au_scattering" / "heidelberg_heff"
RUN009_OBS = ROOT / "results" / "benchmark3_no_au_scattering" / "heidelberg" / "run_009" / "analysis" / "observables.csv"
FIGDIR = ROOT / "figures" / "benchmark3_no_au_scattering"

T_TOTAL = 2067.06866675911
DT_SWEEP = {"run_001": 3.0, "run_002": 2.0, "run_003": 1.5, "run_004": 1.0,
            "run_005": 0.5, "run_006": 0.25, "run_007": 1.25, "run_008": 1.75}
CHECK_FULL_DT = 1.0
CALIB = 1.27
EPS_LIST = [1e-2, 3e-3, 1e-3]
NCOL = 37                                   # time,norm,nd,nc1..nc32,rmean,zmean
FLOOR_MULT = 5.0                            # keep a point only if  delta >= FLOOR_MULT * floor
SAT_ABS = 5.0e-3                            # ...and  delta <= SAT_ABS  (above this P_mol is saturating)


def load_pmol(path: Path):
    a = np.array([[float(x) for x in ln.split()]
                  for ln in path.read_text().splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")])
    if a.shape[1] != NCOL:
        raise RuntimeError(f"{path}: {a.shape[1]} cols, expected {NCOL}")
    return a[:, 0], a[:, 2]                  # time, P_mol (= nd)


def metrics(t_e, p_e, t_r, p_r) -> dict:
    lo, hi = max(t_e.min(), t_r.min()), min(t_e.max(), t_r.max())
    m = (t_e >= lo) & (t_e <= hi)
    d = np.abs(p_e[m] - np.interp(t_e[m], t_r, p_r))
    return {"final": float(d[-1]), "late_mean": float(d[-10:].mean()),
            "max_t": float(d.max()), "t_last": float(t_e[m][-1])}


def loglog_fit(x, y):
    s, c = np.polyfit(np.log(x), np.log(y), 1)
    return float(s), float(np.exp(c))


def main() -> None:
    t_ref, p_ref = load_pmol(RAW / "H_ref" / "raw" / "expectation")

    # ---- noise floor: H_ref vs the original run_009 (same H, same tree) --------
    floor = None
    if RUN009_OBS.exists():
        o = list(csv.DictReader(open(RUN009_OBS)))
        t9 = np.array([float(r["time"]) for r in o])
        p9 = np.array([float(r["P_mol"]) for r in o])
        fm = metrics(t_ref, p_ref, t9, p9)
        floor = fm
        print(f"NOISE FLOOR (H_ref vs run_009):  final={fm['final']:.2e}  "
              f"late_mean={fm['late_mean']:.2e}  max_t={fm['max_t']:.2e}\n")

    # ---- sweep ---------------------------------------------------------------
    rows = []
    for run, dt in sorted(DT_SWEEP.items(), key=lambda kv: -kv[1]):
        ep = RAW / run / "raw" / "expectation"
        if not ep.exists():
            print(f"  {run} (dt={dt}): not present, skipping")
            continue
        t_e, p_e = load_pmol(ep)
        mm = metrics(t_e, p_e, t_ref, p_ref)
        rows.append({"run": run, "dt_au": dt, "r_steps": round(T_TOTAL / dt), **mm})
    if len(rows) < 2:
        raise SystemExit("need >= 2 completed H_eff runs")
    rows.sort(key=lambda r: r["dt_au"])
    dt = np.array([r["dt_au"] for r in rows])

    # ---- check_full vs matching reduced run --------------------------------
    cf_line = ""
    cfp = RAW / "check_full" / "raw" / "expectation"
    if cfp.exists():
        t_c, p_c = load_pmol(cfp)
        mc = metrics(t_c, p_c, t_ref, p_ref)
        match = next((r for r in rows if abs(r["dt_au"] - CHECK_FULL_DT) < 1e-9), None)
        if match:
            rel = {k: mc[k] / match[k] for k in ("final", "late_mean", "max_t")}
            cf_line = (f"\ncheck_full (dt={CHECK_FULL_DT}, FULL E1) vs reduced run:\n"
                       f"  final     {mc['final']:.3e} / {match['final']:.3e}  = {rel['final']:.2f}x\n"
                       f"  late_mean {mc['late_mean']:.3e} / {match['late_mean']:.3e}  = {rel['late_mean']:.2f}x\n"
                       f"  -> dropping [H_j,H_k] {'OK (<1.3x)' if rel['late_mean'] < 1.3 else 'CHANGES THE ANSWER'}")

    # ---- fit + r* for each metric ----------------------------------------
    print(f"{'run':>8}{'dt':>7}{'r':>7}{'final':>12}{'late_mean':>12}{'max_t':>12}{'t_last':>9}")
    for r in rows:
        print(f"{r['run']:>8}{r['dt_au']:>7}{r['r_steps']:>7}{r['final']:>12.3e}"
              f"{r['late_mean']:>12.3e}{r['max_t']:>12.3e}{r['t_last']:>9.0f}")

    est = {"t_total_au": T_TOTAL, "window_fs": 50.0, "calib": CALIB,
           "noise_floor": floor, "floor_mult": FLOOR_MULT, "sat_abs": SAT_ABS,
           "metrics": {}}
    for metric in ("final", "late_mean", "max_t"):
        y = np.array([r[metric] for r in rows])
        fl = floor[metric] if floor else 0.0
        keep = (y >= FLOOR_MULT * fl) & (y <= SAT_ABS)
        pair = np.diff(np.log(y)) / np.diff(np.log(dt))
        print(f"\n--- metric = {metric}   (floor={fl:.1e}, window: "
              f"{FLOOR_MULT:.0f}x floor .. {SAT_ABS:.0e}) ---")
        for i, r in enumerate(rows):
            tag = "used" if keep[i] else ("<floor" if y[i] < FLOOR_MULT * fl else "sat")
            sl = f"  slope_to_next={pair[i]:+.2f}" if i < len(pair) else ""
            print(f"    dt={r['dt_au']:>4}  {metric}={y[i]:.3e}  [{tag}]{sl}")

        m = {"floor": fl, "window_runs": [rows[i]["run"] for i in np.where(keep)[0]]}
        if keep.sum() < 2:
            m["status"] = "INSUFFICIENT POINTS in the clean window"
            print(f"  -> only {keep.sum()} usable point(s); cannot fit.")
        else:
            xw, yw = dt[keep], y[keep]
            alpha, coef = loglog_fit(xw, yw)
            m["alpha"] = alpha
            m["r_star"] = {}
            dt0, d0 = xw[0], yw[0]                    # anchor on smallest kept dt
            print(f"  fit over {list(xw)}:  alpha = {alpha:.2f}")
            for eps in EPS_LIST:
                dt_star = dt0 * (eps / d0) ** (1.0 / alpha)
                r = int(np.ceil(T_TOTAL / dt_star))
                m["r_star"][f"{eps:.0e}"] = {
                    "r_50fs": r, "band_50fs": [int(np.ceil(r / CALIB)), r],
                    "r_350fs_lb": 7 * r}
                print(f"    eps={eps:.0e}:  dt*={dt_star:.3f}  r*(50fs)={r}"
                      f"  band[{int(np.ceil(r/CALIB))},{r}]  (>~{7*r} @350fs)")
        est["metrics"][metric] = m

    if cf_line:
        print(cf_line)

    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "dt_au", "r_steps",
                                           "final", "late_mean", "max_t", "t_last"])
        w.writeheader()
        w.writerows(rows)
    (RAW / "estimate.json").write_text(json.dumps(est, indent=2))
    print(f"\nsaved {RAW/'convergence.csv'}  {RAW/'estimate.json'}")

    try:
        import matplotlib.pyplot as plt
        FIGDIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        for metric, mk in [("final", "o"), ("late_mean", "s"), ("max_t", "^")]:
            y = np.array([r[metric] for r in rows])
            a, c = loglog_fit(dt, y)
            ax.loglog(dt, y, mk, ms=8, label=f"{metric}  (α={a:.2f})")
            xs = np.linspace(dt.min() * 0.7, dt.max() * 1.3, 40)
            ax.loglog(xs, c * xs ** a, "-", lw=1, alpha=0.5)
        if floor:
            ax.axhline(floor["final"], color="0.5", ls=":", label="noise floor (final)")
        for eps in EPS_LIST:
            ax.axhline(eps, color="C3", ls=":", lw=0.8)
        ax.set_xlabel(r"$\Delta t$ (a.u.)")
        ax.set_ylabel(r"$|\Delta P_{\rm mol}|$")
        ax.set_title("NO/Au effective-Hamiltonian Trotter-error estimate")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGDIR / "noau_heff_scaling.png", dpi=200)
        print(f"saved {FIGDIR/'noau_heff_scaling.png'}")
    except Exception as e:                    # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
