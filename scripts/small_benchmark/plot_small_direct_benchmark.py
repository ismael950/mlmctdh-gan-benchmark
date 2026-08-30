from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "results"
    / "small_direct_benchmark"
    / "comparison"
)

FIGURES = (
    ROOT
    / "figures"
    / "small_direct_benchmark"
)


# ============================================================
# Visual style
#
# Change things here without touching the data-processing code.
# ============================================================

FIGSIZE = (6.4, 4.4)
DPI = 300

FONT_SIZE = 11
LABEL_SIZE = 12
LEGEND_SIZE = 10

LINEWIDTH = 2.0
MARKER_SIZE = 6

GRID_ALPHA = 0.20

SAVE_PNG = True
SAVE_PDF = True


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "lines.linewidth": LINEWIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(
    fig: plt.Figure,
    name: str,
) -> None:

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_PNG:
        fig.savefig(
            FIGURES / f"{name}.png",
            dpi=DPI,
            bbox_inches="tight",
        )

    if SAVE_PDF:
        fig.savefig(
            FIGURES / f"{name}.pdf",
            bbox_inches="tight",
        )

    plt.close(fig)


# ============================================================
# 1. Population dynamics
# ============================================================

def plot_population_dynamics() -> None:

    data = pd.read_csv(
        DATA / "population_dynamics.csv"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # Orbital n0
    ax.plot(
        data["time_fs"],
        data["n0_exact"],
        label=r"Exact $n_0$",
    )

    ax.plot(
        data["time_fs"],
        data["n0_mlmctdh"],
        "--",
        label=r"ML-MCTDH $n_0$",
    )

    ax.plot(
        data["time_fs"],
        data["n0_quantum"],
        ":",
        label=r"Quantum $n_0$",
    )

    # Orbital n1
    ax.plot(
        data["time_fs"],
        data["n1_exact"],
        label=r"Exact $n_1$",
    )

    ax.plot(
        data["time_fs"],
        data["n1_mlmctdh"],
        "--",
        label=r"ML-MCTDH $n_1$",
    )

    ax.plot(
        data["time_fs"],
        data["n1_quantum"],
        ":",
        label=r"Quantum $n_1$",
    )

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(
        "Molecular-orbital population"
    )

    ax.grid(
        True,
        alpha=GRID_ALPHA,
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "population_dynamics",
    )


# ============================================================
# 2. Error dynamics
# ============================================================

def plot_error_dynamics() -> None:

    data = pd.read_csv(
        DATA / "error_dynamics.csv"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        data["time_fs"],
        data["mlmctdh_error"],
        label="ML-MCTDH",
    )

    ax.plot(
        data["time_fs"],
        data["quantum_error"],
        label="Quantum",
    )

    ax.set_yscale("log")

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(
        r"$\max_i |n_i(t)-n_i^{\mathrm{exact}}(t)|$"
    )

    ax.grid(
        True,
        which="both",
        alpha=GRID_ALPHA,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "error_dynamics",
    )


# ============================================================
# 3. ML-MCTDH convergence
# ============================================================

def plot_mlmctdh_convergence() -> None:

    data = pd.read_csv(
        DATA / "mlmctdh_convergence.csv"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        data["n_ml_coefficients"],
        data["max_error"],
        marker="o",
        markersize=MARKER_SIZE,
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Number of time-dependent ML coefficients"
    )

    ax.set_ylabel(
        r"$E_{\max}$"
    )

    ax.grid(
        True,
        which="both",
        alpha=GRID_ALPHA,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "mlmctdh_convergence",
    )


# ============================================================
# 4. Nuclear-basis convergence
# ============================================================

def plot_basis_convergence() -> None:

    data = pd.read_csv(
        DATA / "basis_convergence.csv"
    )

    labels = [
        f"{int(a)}→{int(b)}"
        for a, b in zip(
            data["K_from"],
            data["K_to"],
        )
    ]

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        labels,
        data["successive_max_error"],
        marker="o",
        markersize=MARKER_SIZE,
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Nuclear basis refinement"
    )

    ax.set_ylabel(
        "Maximum successive population difference"
    )

    ax.grid(
        True,
        which="both",
        alpha=GRID_ALPHA,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "basis_convergence",
    )


# ============================================================
# 5. Trotter convergence
# ============================================================

def plot_trotter_convergence_steps() -> None:

    data = pd.read_csv(
        DATA / "trotter_convergence.csv"
    )

    data = data.sort_values(
        "n_trotter_steps"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        data["n_trotter_steps"],
        data["max_error"],
        marker="o",
        markersize=MARKER_SIZE,
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Number of Trotter steps"
    )

    ax.set_ylabel(
        r"$E_{\max}$"
    )

    ax.grid(
        True,
        which="both",
        alpha=GRID_ALPHA,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "trotter_convergence_steps",
    )

def plot_trotter_convergence_dt() -> None:

    data = pd.read_csv(
        DATA / "trotter_convergence.csv"
    )

    data = data.sort_values(
        "dt_au"
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.plot(
        data["dt_au"],
        data["max_error"],
        marker="o",
        markersize=MARKER_SIZE,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"Trotter step $\Delta t$ (a.u.)"
    )

    ax.set_ylabel(
        r"$E_{\max}$"
    )

    ax.grid(
        True,
        which="both",
        alpha=GRID_ALPHA,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "trotter_convergence_dt",
    )

# ============================================================
# Main
# ============================================================

def main() -> None:

    configure_style()

    plot_population_dynamics()
    plot_error_dynamics()
    plot_mlmctdh_convergence()
    plot_basis_convergence()
    plot_trotter_convergence_steps()
    plot_trotter_convergence_dt()

    print(
        "Figures generated in:"
    )
    print(FIGURES)


if __name__ == "__main__":
    main()