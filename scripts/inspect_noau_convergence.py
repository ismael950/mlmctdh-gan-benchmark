import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("results/benchmark3_no_au_scattering/heidelberg")

runs = sorted(
    [p for p in ROOT.glob("run_*") if p.is_dir()],
    key=lambda p: int(re.search(r"(\d+)$", p.name).group(1))
)

def read_observables(run):
    path = run / "analysis" / "observables.csv"
    if not path.exists():
        return None

    with path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    t = np.array([float(r["time"]) for r in rows])
    p = np.array([float(r["P_mol"]) for r in rows])
    return t, p


def read_summary(run):
    path = run / "analysis" / "summary.json"
    if not path.exists():
        return {}

    with path.open() as f:
        return json.load(f)


def find_ncoeff(obj):
    """
    Search summary.json recursively for a plausible total coefficient count.
    """
    candidates = []

    def walk(x, prefix=""):
        if isinstance(x, dict):
            for k, v in x.items():
                key = f"{prefix}.{k}" if prefix else k
                kl = k.lower()

                if (
                    isinstance(v, (int, float))
                    and ("coeff" in kl or "ncoeff" in kl)
                    and v > 0
                ):
                    candidates.append((key, v))

                walk(v, key)

        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{prefix}[{i}]")

    walk(obj)

    # Prefer keys that look like a total rather than individual pieces.
    for key, value in candidates:
        kl = key.lower()
        if "total" in kl or "ncoeff" in kl:
            return int(value)

    if candidates:
        return int(max(v for _, v in candidates))

    return None


data = {}

for run in runs:
    obs = read_observables(run)
    if obs is not None:
        data[run.name] = obs

if not data:
    raise RuntimeError("No analyzed runs with observables.csv were found.")

names = list(data)
ref_name = names[-1]
tref, pref = data[ref_name]

print()
print(f"Provisional reference: {ref_name}")
print()

header = (
    f"{'run':<10}"
    f"{'Ncoeff':>10}"
    f"{'Pfinal':>14}"
    f"{'Pmax':>14}"
    f"{'delta_prev':>16}"
    f"{'vs_'+ref_name:>16}"
)

print(header)
print("-" * len(header))

previous = None

for name in names:
    t, p = data[name]
    summary = read_summary(ROOT / name)
    ncoeff = find_ncoeff(summary)

    # Successive change.
    delta_prev = np.nan

    if previous is not None:
        tp, pp = data[previous]

        # Interpolate if time grids differ.
        lo = max(t.min(), tp.min())
        hi = min(t.max(), tp.max())

        mask = (t >= lo) & (t <= hi)
        tc = t[mask]
        pp_i = np.interp(tc, tp, pp)

        delta_prev = np.max(np.abs(p[mask] - pp_i))

    # Difference from latest available run.
    lo = max(t.min(), tref.min())
    hi = min(t.max(), tref.max())

    mask = (t >= lo) & (t <= hi)
    tc = t[mask]
    pref_i = np.interp(tc, tref, pref)

    vs_ref = np.max(np.abs(p[mask] - pref_i))

    nc = str(ncoeff) if ncoeff is not None else "?"

    dp = "-" if np.isnan(delta_prev) else f"{delta_prev:.6e}"

    print(
        f"{name:<10}"
        f"{nc:>10}"
        f"{p[-1]:>14.8f}"
        f"{p.max():>14.8f}"
        f"{dp:>16}"
        f"{vs_ref:>16.6e}"
    )

    previous = name

print()
print("delta_prev = max_t |P_run - P_previous|")
print(f"vs_{ref_name} = max_t |P_run - P_{ref_name}|")
print(f"{ref_name} is ONLY a provisional reference here.")