import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    input_root = Path("results/basis_convergence")
    output_directory = input_root / "comparison"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_by_basis_size = {}

    for data_file in sorted(input_root.glob("K*/data.npz")):
        with np.load(data_file) as data:
            basis_size = int(data["basis_size"])

            data_by_basis_size[basis_size] = {
                "times": data["times"],
                "electronic_populations": (
                    data["electronic_populations"]
                ),
                "vibrational_occupations": (
                    data["vibrational_occupations"]
                ),
                "vibrational_coordinates": (
                    data["vibrational_coordinates"]
                ),
                "highest_level_population": (
                    data["highest_level_population"]
                ),
            }

    if not data_by_basis_size:
        raise FileNotFoundError(
            "No convergence data were found. "
            "Run scripts/run_basis_convergence.py first."
        )

    basis_sizes = sorted(data_by_basis_size)
    reference_basis_size = max(basis_sizes)
    reference = data_by_basis_size[reference_basis_size]

    reference_times = reference["times"]
    reference_molecular_population = (
        reference["electronic_populations"][:, 0]
    )
    reference_vibrational_occupation = (
        reference["vibrational_occupations"][:, 0]
    )
    reference_vibrational_coordinate = (
        reference["vibrational_coordinates"][:, 0]
    )

    summary_rows = []

    for basis_size in basis_sizes:
        current = data_by_basis_size[basis_size]

        if not np.allclose(
            current["times"],
            reference_times,
        ):
            raise ValueError(
                f"Time grid for K={basis_size} does not match "
                f"the K={reference_basis_size} reference."
            )

        molecular_population = (
            current["electronic_populations"][:, 0]
        )
        vibrational_occupation = (
            current["vibrational_occupations"][:, 0]
        )
        vibrational_coordinate = (
            current["vibrational_coordinates"][:, 0]
        )

        molecular_error = np.max(
            np.abs(
                molecular_population
                - reference_molecular_population
            )
        )

        occupation_error = np.max(
            np.abs(
                vibrational_occupation
                - reference_vibrational_occupation
            )
        )

        coordinate_error = np.max(
            np.abs(
                vibrational_coordinate
                - reference_vibrational_coordinate
            )
        )

        maximum_boundary_population = np.max(
            current["highest_level_population"]
        )

        summary_rows.append(
            {
                "basis_size": basis_size,
                "molecular_population_error": molecular_error,
                "vibrational_occupation_error": occupation_error,
                "vibrational_coordinate_error": coordinate_error,
                "maximum_highest_level_population": (
                    maximum_boundary_population
                ),
            }
        )

    # Guardar resumen numérico
    csv_path = output_directory / "convergence_summary.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # Gráfica 1: población molecular
    plt.figure(figsize=(7, 4.5))

    for basis_size in basis_sizes:
        current = data_by_basis_size[basis_size]

        plt.plot(
            current["times"],
            current["electronic_populations"][:, 0],
            label=f"K = {basis_size}",
        )

    plt.xlabel("Time")
    plt.ylabel(r"$P_{\mathrm{mol}}(t)$")
    plt.legend()
    plt.tight_layout()

    molecular_path = (
        output_directory
        / "molecular_population_convergence.png"
    )

    plt.savefig(
        molecular_path,
        dpi=200,
    )
    plt.close()

    # Gráfica 2: ocupación vibracional
    plt.figure(figsize=(7, 4.5))

    for basis_size in basis_sizes:
        current = data_by_basis_size[basis_size]

        plt.plot(
            current["times"],
            current["vibrational_occupations"][:, 0],
            label=f"K = {basis_size}",
        )

    plt.xlabel("Time")
    plt.ylabel(r"$\langle b^\dagger b\rangle$")
    plt.legend()
    plt.tight_layout()

    occupation_path = (
        output_directory
        / "vibrational_occupation_convergence.png"
    )

    plt.savefig(
        occupation_path,
        dpi=200,
    )
    plt.close()

    # Gráfica 3: coordenada vibracional
    plt.figure(figsize=(7, 4.5))

    for basis_size in basis_sizes:
        current = data_by_basis_size[basis_size]

        plt.plot(
            current["times"],
            current["vibrational_coordinates"][:, 0],
            label=f"K = {basis_size}",
        )

    plt.xlabel("Time")
    plt.ylabel(r"$\langle q\rangle$")
    plt.legend()
    plt.tight_layout()

    coordinate_path = (
        output_directory
        / "vibrational_coordinate_convergence.png"
    )

    plt.savefig(
        coordinate_path,
        dpi=200,
    )
    plt.close()

    print(
        f"\nReference basis: K = {reference_basis_size}"
    )

    for row in summary_rows:
        print(f"\nK = {row['basis_size']}")
        print(
            "Maximum molecular-population difference:",
            row["molecular_population_error"],
        )
        print(
            "Maximum vibrational-occupation difference:",
            row["vibrational_occupation_error"],
        )
        print(
            "Maximum vibrational-coordinate difference:",
            row["vibrational_coordinate_error"],
        )
        print(
            "Maximum population of the highest level:",
            row["maximum_highest_level_population"],
        )

    print("\nFiles created:")
    print(csv_path)
    print(molecular_path)
    print(occupation_path)
    print(coordinate_path)


if __name__ == "__main__":
    main()