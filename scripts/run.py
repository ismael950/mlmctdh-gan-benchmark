from __future__ import annotations

import argparse

import numpy as np

from ganbench.model import load_config
from ganbench.results import save_exact_results
from ganbench.exact.hamiltonian import build_exact_hamiltonian
from ganbench.exact.propagate import propagate_exact
from ganbench.exact.observables import compute_observables


def run_exact(config_file: str) -> None:
    """Run one configuration with the exact solver."""

    config = load_config(config_file)

    # -------------------------------------------------
    # Build Hamiltonian and initial state
    # -------------------------------------------------

    system = build_exact_hamiltonian(config)

    # -------------------------------------------------
    # Exact propagation
    # -------------------------------------------------

    propagation = propagate_exact(
        system=system,
        t_final=config.t_final,
        n_times=config.n_times,
    )

    # -------------------------------------------------
    # Observables
    # -------------------------------------------------

    observables = compute_observables(
        system=system,
        propagation=propagation,
    )

    electronic_populations = (
        observables.electronic_populations
    )

    molecular_populations = electronic_populations[
        :,
        : config.n_molecular_orbitals,
    ]

    metal_populations = electronic_populations[
        :,
        config.n_molecular_orbitals :,
    ]

    total_molecular_population = np.sum(
        molecular_populations,
        axis=1,
    )

    total_metal_population = np.sum(
        metal_populations,
        axis=1,
    )

    total_particle_number = (
        observables.total_electronic_population
    )

    # -------------------------------------------------
    # Conservation checks
    # -------------------------------------------------

    norm_error = np.max(
        np.abs(
            propagation.norms - 1.0
        )
    )

    particle_number_error = np.max(
        np.abs(
            total_particle_number
            - config.n_electrons
        )
    )

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    results_directory = save_exact_results(
        config_file=config_file,
        config=config,
        system=system,
        propagation=propagation,
        observables=observables,
        norm_error=norm_error,
        particle_number_error=particle_number_error,
    )

    # -------------------------------------------------
    # Output
    # -------------------------------------------------

    print("Backend: exact")
    print("Config:", config_file)
    print("Propagation completed.")

    print(
        "States array shape:",
        propagation.states.shape,
    )

    print("\nConservation checks")

    print(
        "Maximum norm error:",
        norm_error,
    )

    print(
        "Maximum particle-number error:",
        particle_number_error,
    )

    print("\nInitial electronic occupations")

    for i in range(
        config.n_molecular_orbitals
    ):
        print(
            f"d{i + 1}:",
            molecular_populations[0, i],
        )

    for k in range(
        config.n_metal_orbitals
    ):
        print(
            f"c{k + 1}:",
            metal_populations[0, k],
        )

    print(
        "Total molecular occupation:",
        total_molecular_population[0],
    )

    print(
        "Total metal occupation:",
        total_metal_population[0],
    )

    print(
        "Total particle number:",
        total_particle_number[0],
    )

    print("\nFinal electronic occupations")

    for i in range(
        config.n_molecular_orbitals
    ):
        print(
            f"d{i + 1}:",
            molecular_populations[-1, i],
        )

    for k in range(
        config.n_metal_orbitals
    ):
        print(
            f"c{k + 1}:",
            metal_populations[-1, k],
        )

    print(
        "Total molecular occupation:",
        total_molecular_population[-1],
    )

    print(
        "Total metal occupation:",
        total_metal_population[-1],
    )

    print(
        "Total particle number:",
        total_particle_number[-1],
    )

    print("\nVibrational observables")

    for mode in range(
        config.n_vibrational_modes
    ):
        occupation = (
            observables.vibrational_occupations[
                :,
                mode,
            ]
        )

        coordinate = (
            observables.vibrational_coordinates[
                :,
                mode,
            ]
        )

        print(f"\nMode {mode + 1}")

        print(
            "Initial occupation:",
            occupation[0],
        )

        print(
            "Final occupation:",
            occupation[-1],
        )

        print(
            "Maximum occupation:",
            np.max(occupation),
        )

        print(
            "Initial coordinate:",
            coordinate[0],
        )

        print(
            "Final coordinate:",
            coordinate[-1],
        )

    print("\nPopulation ranges")

    print(
        "Molecular occupation:",
        np.min(
            total_molecular_population
        ),
        "to",
        np.max(
            total_molecular_population
        ),
    )

    print(
        "Metal occupation:",
        np.min(
            total_metal_population
        ),
        "to",
        np.max(
            total_metal_population
        ),
    )

    print(
    "\nResults saved to:",
    results_directory,
)


def run_heidelberg(config_file: str) -> None:
    """
    Generate Heidelberg input files.

    This backend will be implemented next.
    """

    raise SystemExit(
        "Heidelberg backend is not implemented yet."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or generate a GAN benchmark model."
        )
    )

    parser.add_argument(
        "config",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--backend",
        choices=(
            "exact",
            "heidelberg",
        ),
        required=True,
        help=(
            "Choose the numerical backend."
        ),
    )

    args = parser.parse_args()

    if args.backend == "exact":
        run_exact(
            args.config
        )

    elif args.backend == "heidelberg":
        run_heidelberg(
            args.config
        )


if __name__ == "__main__":
    main()