from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import numpy as np

from ganbench.model import load_config
from ganbench.results import save_exact_results
from ganbench.exact.hamiltonian import build_exact_hamiltonian
from ganbench.exact.propagate import propagate_exact
from ganbench.exact.observables import compute_observables

from ganbench.heidelberg.convergence import (
    evaluate_plateau_history,
    make_adaptive_decision,
    read_branch_states,
)
from ganbench.heidelberg.input import write_refined_input

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
    Analyse the completed Heidelberg convergence history and
    prepare the next adaptive ML-MCTDH input.
    """

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(config_file)
    config = load_config(config_path)

    benchmark = config_path.stem

    results_root = (
        project_root
        / "results"
        / benchmark
        / "heidelberg"
    )

    if not results_root.exists():
        raise SystemExit(
            f"No Heidelberg results found for {benchmark}."
        )

    completed_runs = []

    for path in results_root.glob("run_*"):
        try:
            run_number = int(path.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        expectation = path / "raw" / "expectation"

        if expectation.exists():
            completed_runs.append(
                (run_number, path)
            )

    completed_runs.sort(
        key=lambda item: item[0]
    )

    if not completed_runs:
        raise SystemExit(
            "No completed Heidelberg runs were found."
        )

    latest_number, latest_results = completed_runs[-1]
    latest_run_id = f"run_{latest_number:03d}"

    expectation_paths = [
        path / "raw" / "expectation"
        for _, path in completed_runs
    ]

    _, plateau_status = evaluate_plateau_history(
        expectation_paths,
        config.n_molecular_orbitals,
    )

    natural_populations_path = (
        latest_results
        / "analysis"
        / "natural_populations.csv"
    )

    if not natural_populations_path.exists():
        raise SystemExit(
            f"Missing analysis for {latest_run_id}: "
            f"{natural_populations_path}"
        )

    branch_states = read_branch_states(
        natural_populations_path
    )

    decision = make_adaptive_decision(
        branch_states,
        plateau_status,
    )

    print(
        f"Latest completed run: {latest_run_id}"
    )
    print(
        f"Adaptive action: {decision.action}"
    )
    print(
        f"Plateau status: {plateau_status}"
    )

    if decision.action != "refine":
        print(
            "No additional Heidelberg input is required."
        )
        return

    next_number = latest_number + 1
    next_run_id = f"run_{next_number:03d}"

    source_directory = (
        project_root
        / "backend_inputs"
        / benchmark
        / "heidelberg"
        / latest_run_id
    )

    destination_directory = (
        project_root
        / "backend_inputs"
        / benchmark
        / "heidelberg"
        / next_run_id
    )

    if destination_directory.exists():
        raise SystemExit(
            f"{destination_directory} already exists. "
            "Refusing to overwrite it."
        )

    destination_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_refined_input(
        source_directory / "benchmark.inp",
        destination_directory / "benchmark.inp",
        next_number,
        decision.rank_updates,
    )

    shutil.copy2(
        source_directory / "benchmark.op",
        destination_directory / "benchmark.op",
    )

    print(
        f"Prepared {next_run_id}"
    )
    print(
        f"Rank updates: {decision.rank_updates}"
    )
    print(
        "Input directory:",
        destination_directory,
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