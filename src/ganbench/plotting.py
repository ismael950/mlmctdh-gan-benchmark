from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_observables(
    results_directory: str | Path,
) -> np.ndarray:
    """Load observables.csv from a results directory."""

    results_directory = Path(
        results_directory
    )

    return np.genfromtxt(
        results_directory / "observables.csv",
        delimiter=",",
        names=True,
    )


def plot_molecular_populations(
    results_directory: str | Path,
    output_directory: str | Path,
) -> None:
    """
    Plot molecular orbital populations and total
    molecular population.
    """

    results_directory = Path(
        results_directory
    )

    output_directory = Path(
        output_directory
    )

    data = load_observables(
        results_directory
    )

    time = data["time"]

    names = list(
        data.dtype.names
    )

    molecular_columns = [
        name
        for name in names
        if (
            name.startswith("d")
            and name[1:].isdigit()
        )
    ]

    plt.figure()

    for name in molecular_columns:
        plt.plot(
            time,
            data[name],
            label=name,
        )

    # Plot total molecular population separately
    # only when there is more than one molecular orbital.
    if (
        "P_mol" in names
        and len(molecular_columns) > 1
    ):
        plt.plot(
            time,
            data["P_mol"],
            label="P_mol",
            linewidth=2,
        )

    plt.xlabel(
        "Time (a.u.)"
    )

    plt.ylabel(
        "Molecular population"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_directory
        / "molecular_populations.png",
        dpi=300,
    )

    plt.close()


def plot_nuclear_coordinates(
    results_directory: str | Path,
    output_directory: str | Path,
) -> None:
    """
    Plot expectation values of all nuclear coordinates.
    """

    results_directory = Path(
        results_directory
    )

    output_directory = Path(
        output_directory
    )

    data = load_observables(
        results_directory
    )

    time = data["time"]

    names = list(
        data.dtype.names
    )

    coordinate_columns = [
        name
        for name in names
        if (
            name.startswith("Q")
            and name[1:].isdigit()
        )
    ]

    if not coordinate_columns:
        return

    plt.figure()

    for name in coordinate_columns:
        plt.plot(
            time,
            data[name],
            label=name,
        )

    plt.xlabel(
        "Time (a.u.)"
    )

    plt.ylabel(
        "Nuclear coordinate expectation value"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_directory
        / "nuclear_coordinates.png",
        dpi=300,
    )

    plt.close()


def plot_vibrational_distributions(
    results_directory: str | Path,
    output_directory: str | Path,
) -> None:
    """
    Plot P_nu(t) for every nuclear mode as a heatmap.
    """

    results_directory = Path(
        results_directory
    )

    output_directory = Path(
        output_directory
    )

    file_path = (
        results_directory
        / "vibrational_distributions.npz"
    )

    data = np.load(
        file_path
    )

    times = data["times"]

    mode_keys = sorted(
        key
        for key in data.files
        if (
            key.startswith("mode_")
            and not key.endswith("_levels")
        )
    )

    for mode_index, key in enumerate(
        mode_keys,
        start=1,
    ):
        distribution = data[key]

        plt.figure()

        plt.imshow(
            distribution.T,
            origin="lower",
            aspect="auto",
            extent=[
                times[0],
                times[-1],
                0,
                distribution.shape[1] - 1,
            ],
        )

        plt.xlabel(
            "Time (a.u.)"
        )

        plt.ylabel(
            "Vibrational level ν"
        )

        plt.colorbar(
            label="Population"
        )

        plt.tight_layout()

        plt.savefig(
            output_directory
            / (
                "vibrational_distribution_"
                f"mode{mode_index}.png"
            ),
            dpi=300,
        )

        plt.close()


def plot_results(
    results_directory: str | Path,
) -> Path:
    """
    Generate all standard plots for one benchmark result.
    """

    results_directory = Path(
        results_directory
    )

    output_directory = (
        results_directory
        / "plots"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_molecular_populations(
        results_directory,
        output_directory,
    )

    plot_nuclear_coordinates(
        results_directory,
        output_directory,
    )

    plot_vibrational_distributions(
        results_directory,
        output_directory,
    )

    return output_directory