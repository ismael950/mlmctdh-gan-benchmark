from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "results"
    / "small_direct_benchmark"
    / "trotter_error_estimator"
)

OUT.mkdir(parents=True, exist_ok=True)


data = pd.DataFrame(
    {
        "N": [
            200, 400, 600, 800,
            1000, 1200, 1400, 1600,
        ],
        "dt_au": [
            10.0,
            5.0,
            3.3333333333333335,
            2.5,
            2.0,
            1.6666666666666667,
            1.4285714285714286,
            1.25,
        ],
        "E_trot": [
            2.4607573114399095e-05,
            1.2035559268763762e-05,
            7.964140252725294e-06,
            5.9507734126995615e-06,
            4.749900550327091e-06,
            3.952296370735020e-06,
            3.384036751707953e-06,
            2.958639899741655e-06,
        ],
        "E_eff": [
            7.583936055255425e-05,
            2.4857391111243743e-05,
            1.3664743234942378e-05,
            9.157923910207977e-06,
            6.802692064988847e-06,
            5.377945672124795e-06,
            4.431505625235310e-06,
            3.760638697336560e-06,
        ],
        "E_residual": [
            5.123178743815515e-05,
            1.282183184247998e-05,
            5.7006029822170845e-06,
            3.2071504975084153e-06,
            2.052791514661756e-06,
            1.4256493013897753e-06,
            1.0474688735273574e-06,
            8.01998797594905e-07,
        ],
    }
)

data.to_csv(
    OUT / "trotter_error_estimator.csv",
    index=False,
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
print(
    OUT / "trotter_error_estimator.csv"
)

print()
print("Slopes:")
print("true Trotter :", s_trot)
print("H_eff       :", s_eff)
print("residual    :", s_res)