from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

# Canonical source: the exact H_eff propagation sweep.
SOURCE = (
    ROOT
    / "results"
    / "small_direct_benchmark"
    / "effective_hamiltonian"
    / "convergence.csv"
)

OUT = (
    ROOT
    / "figures"
    / "small_direct_benchmark"
)

OUT.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"{SOURCE} not found. Run "
        "validate_small_effective_hamiltonian.py first."
    )

convergence = pd.read_csv(SOURCE)

data = pd.DataFrame(
    {
        "N": convergence["n_trotter_steps"],
        "dt_au": convergence["dt_au"],
        "E_trot": convergence["E_trot"],
        "E_eff": convergence["E_eff"],
        # Residual between the H_eff channel and the true
        # Trotter channel (E_model in the source table).
        "E_residual": convergence["E_model"],
    }
)


# Fits in log-log space
def slope(x, y):
    return np.polyfit(
        np.log(x),
        np.log(y),
        1,
    )[0]


s_trot = slope(
    data["dt_au"],
    data["E_trot"],
)

s_eff = slope(
    data["dt_au"],
    data["E_eff"],
)

s_res = slope(
    data["dt_au"],
    data["E_residual"],
)


fig, ax = plt.subplots(
    figsize=(9, 6.5)
)

ax.loglog(
    data["dt_au"],
    data["E_trot"],
    "o-",
    label=(
        rf"True Trotter error "
        rf"($p={s_trot:.2f}$)"
    ),
)

ax.loglog(
    data["dt_au"],
    data["E_eff"],
    "s--",
    label=(
        rf"$H_{{\rm eff}}$ estimate "
        rf"($p={s_eff:.2f}$)"
    ),
)

ax.loglog(
    data["dt_au"],
    data["E_residual"],
    "^-.",
    label=(
        rf"$H_{{\rm eff}}$--Trotter residual "
        rf"($p={s_res:.2f}$)"
    ),
)

ax.set_xlabel(
    r"Trotter step $\Delta t$ (a.u.)"
)

ax.set_ylabel(
    r"Maximum molecular-population error"
)

ax.grid(
    True,
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT / "trotter_error_estimator_scaling.png",
    dpi=300,
)

plt.close(fig)

print("Saved:")
print(
    OUT / "trotter_error_estimator_scaling.png"
)

print()
print("Slopes:")
print("true Trotter :", s_trot)
print("H_eff       :", s_eff)
print("residual    :", s_res)