from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results" / "small_direct_benchmark"

OUT = RESULTS / "trotter_error_estimator"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# dt = 2.5 au corresponds to:
#
# T = 2000 au
# N = 800
# ------------------------------------------------------------

source = (
    RESULTS
    / "effective_hamiltonian"
    / "steps_0800.csv"
)

if not source.exists():
    raise FileNotFoundError(source)

print("Using:")
print(source)

data = pd.read_csv(source)


# ------------------------------------------------------------
# IMPORTANT:
# Maximum ONLY over MOLECULAR populations n0 and n1.
# Metal populations are not part of this error metric.
# ------------------------------------------------------------

data["error_trot_molecular"] = np.maximum(
    data["error_trot_n0"],
    data["error_trot_n1"],
)

data["error_heff_molecular"] = np.maximum(
    data["error_heff_n0"],
    data["error_heff_n1"],
)

data["error_residual_molecular"] = np.maximum(
    data["error_model_n0"],
    data["error_model_n1"],
)


# Save the reduced temporal-error table as well.
temporal = data[
    [
        "time_au",
        "time_fs",
        "error_trot_molecular",
        "error_heff_molecular",
        "error_residual_molecular",
    ]
].copy()

csv_out = (
    OUT
    / "trotter_error_estimator_dynamics_dt2p5.csv"
)

temporal.to_csv(
    csv_out,
    index=False,
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 6.5)
)

ax.plot(
    data["time_fs"],
    data["error_trot_molecular"],
    label="True Trotter error",
)

ax.plot(
    data["time_fs"],
    data["error_heff_molecular"],
    linestyle="--",
    label=r"$H_{\rm eff}$ estimate",
)

ax.plot(
    data["time_fs"],
    data["error_residual_molecular"],
    linestyle="-.",
    label=r"$H_{\rm eff}$--Trotter residual",
)

ax.set_xlabel("Time (fs)")

ax.set_ylabel(
    "Maximum molecular-population error"
)

ax.set_title(
    r"Trotter-error estimator dynamics "
    r"($\Delta t = 2.5$ a.u.)"
)

ax.set_yscale("log")

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig_out = (
    OUT
    / "trotter_error_estimator_dynamics_dt2p5.png"
)

fig.savefig(
    fig_out,
    dpi=300,
)

plt.close(fig)


print()
print("Saved:")
print(fig_out)
print(csv_out)

print()
print(
    "Maximum true Trotter error =",
    data["error_trot_molecular"].max(),
)

print(
    "Maximum H_eff estimate     =",
    data["error_heff_molecular"].max(),
)

print(
    "Maximum residual           =",
    data["error_residual_molecular"].max(),
)