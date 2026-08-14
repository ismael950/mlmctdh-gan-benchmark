from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "validation_small_comparison"

PYTHON_FILE = DATA_DIR / "python_populations.csv"
HEIDELBERG_FILE = DATA_DIR / "heidelberg_populations.dat"

FIGURE_FILE = DATA_DIR / "python_vs_heidelberg.png"
ERROR_FILE = DATA_DIR / "comparison_errors.csv"


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def main() -> None:
    # Columns: time, Pmol, Pmetal
    python_data = np.loadtxt(
        PYTHON_FILE,
        delimiter=",",
        skiprows=1,
    )

    # The Heidelberg file has comment lines beginning with "#".
    heidelberg_data = np.loadtxt(
        HEIDELBERG_FILE,
        comments="#",
    )

    t_py = python_data[:, 0]
    pmol_py = python_data[:, 1]
    pmetal_py = python_data[:, 2]

    t_hd = heidelberg_data[:, 0]
    pmol_hd_raw = heidelberg_data[:, 1]
    pmetal_hd_raw = heidelberg_data[:, 2]

    # Interpolate only to make the comparison robust in case the
    # two programs used slightly different output time grids.
    pmol_hd = np.interp(t_py, t_hd, pmol_hd_raw)
    pmetal_hd = np.interp(t_py, t_hd, pmetal_hd_raw)

    error_pmol = np.abs(pmol_py - pmol_hd)
    error_pmetal = np.abs(pmetal_py - pmetal_hd)

    conservation_py = np.abs(pmol_py + pmetal_py - 1.0)
    conservation_hd = np.abs(pmol_hd + pmetal_hd - 1.0)

    print("Python vs Heidelberg")
    print("--------------------")
    print(f"Maximum |ΔPmol|   : {np.max(error_pmol):.6e}")
    print(f"RMS     |ΔPmol|   : {rms(error_pmol):.6e}")
    print(f"Maximum |ΔPmetal| : {np.max(error_pmetal):.6e}")
    print(f"RMS     |ΔPmetal| : {rms(error_pmetal):.6e}")
    print()
    print(f"Max population-sum error, Python     : {np.max(conservation_py):.6e}")
    print(f"Max population-sum error, Heidelberg : {np.max(conservation_hd):.6e}")

    output_data = np.column_stack(
        (
            t_py,
            pmol_py,
            pmol_hd,
            error_pmol,
            pmetal_py,
            pmetal_hd,
            error_pmetal,
        )
    )

    np.savetxt(
        ERROR_FILE,
        output_data,
        delimiter=",",
        header=(
            "time,"
            "pmol_python,pmol_heidelberg,error_pmol,"
            "pmetal_python,pmetal_heidelberg,error_pmetal"
        ),
        comments="",
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(t_py, pmol_py, label="Molecule — Python", linewidth=2)
    axes[0].plot(
        t_py,
        pmol_hd,
        "--",
        label="Molecule — Heidelberg",
        linewidth=1.5,
    )
    axes[0].plot(t_py, pmetal_py, label="Metal — Python", linewidth=2)
    axes[0].plot(
        t_py,
        pmetal_hd,
        "--",
        label="Metal — Heidelberg",
        linewidth=1.5,
    )

    axes[0].set_ylabel("Electronic population")
    axes[0].set_title("Exact GAN validation: Python vs Heidelberg")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(
        t_py,
        np.maximum(error_pmol, np.finfo(float).eps),
        label=r"$|\Delta P_{\mathrm{mol}}|$",
    )
    axes[1].semilogy(
        t_py,
        np.maximum(error_pmetal, np.finfo(float).eps),
        label=r"$|\Delta P_{\mathrm{metal}}|$",
    )

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Absolute error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.savefig(FIGURE_FILE, dpi=300)
    plt.close(figure)

    print()
    print(f"Figure saved to: {FIGURE_FILE}")
    print(f"Error data saved to: {ERROR_FILE}")


if __name__ == "__main__":
    main()