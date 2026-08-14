from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ganbench.exact.hamiltonian import ExactSystem
from ganbench.exact.observables import ObservableResult
from ganbench.exact.propagate import PropagationResult


def get_results_directory(
    config_file: str,
    backend: str,
) -> Path:
    """
    Return the results directory associated with a
    configuration and backend.

    Example:
        configs/benchmark1_newns_anderson.yaml

    becomes:
        results/benchmark1_newns_anderson/exact/
    """

    config_path = Path(config_file)

    project_root = Path(__file__).resolve().parents[2]

    output_directory = (
        project_root
        / "results"
        / config_path.stem
        / backend
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def save_exact_results(
    config_file: str,
    config,
    system: ExactSystem,
    propagation: PropagationResult,
    observables: ObservableResult,
    norm_error: float,
    particle_number_error: float,
) -> Path:
    """
    Save the results of an exact propagation.
    """

    output_directory = get_results_directory(
        config_file=config_file,
        backend="exact",
    )

    # -------------------------------------------------
    # 1. Main observables table
    # -------------------------------------------------

    columns = [
        observables.times,
        propagation.norms,
    ]

    headers = [
        "time",
        "norm",
    ]

    # Molecular orbital populations
    for i in range(
        system.n_molecular_orbitals
    ):
        columns.append(
            observables.electronic_populations[:, i]
        )

        headers.append(
            f"d{i + 1}"
        )

    # Metal orbital populations
    metal_start = (
        system.n_molecular_orbitals
    )

    for k in range(
        system.n_metal_orbitals
    ):
        columns.append(
            observables.electronic_populations[
                :,
                metal_start + k,
            ]
        )

        headers.append(
            f"c{k + 1}"
        )

    # Aggregate electronic populations
    columns.extend(
        [
            observables.molecular_population,
            observables.metal_population,
            observables.total_electronic_population,
        ]
    )

    headers.extend(
        [
            "P_mol",
            "P_metal",
            "N_total",
        ]
    )

    # Nuclear coordinates
    for mode in range(
        len(system.vibrational_basis_sizes)
    ):
        columns.append(
            observables.vibrational_coordinates[
                :,
                mode,
            ]
        )

        headers.append(
            f"Q{mode + 1}"
        )

    # Vibrational occupation is retained only
    # as a diagnostic quantity.
    for mode in range(
        len(system.vibrational_basis_sizes)
    ):
        columns.append(
            observables.vibrational_occupations[
                :,
                mode,
            ]
        )

        headers.append(
            f"nvib{mode + 1}"
        )

    data = np.column_stack(
        columns
    )

    np.savetxt(
        output_directory / "observables.csv",
        data,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )

    # -------------------------------------------------
    # 2. Vibrational energy distributions
    # -------------------------------------------------

    vibrational_data = {
        "times": observables.times,
    }

    for mode, distribution in enumerate(
        observables.vibrational_distributions
    ):
        vibrational_data[
            f"mode_{mode + 1}"
        ] = distribution

        vibrational_data[
            f"mode_{mode + 1}_levels"
        ] = np.arange(
            distribution.shape[1],
            dtype=int,
        )

    np.savez_compressed(
        output_directory
        / "vibrational_distributions.npz",
        **vibrational_data,
    )

    # -------------------------------------------------
    # 3. Optional full quantum-state trajectory
    # -------------------------------------------------

    states_file = (
        output_directory
        / "states.npy"
    )

    if config.save_states:
        np.save(
            states_file,
            propagation.states,
        )

    elif states_file.exists():
        # Remove an old states file if this run explicitly
        # does not request state storage.
        states_file.unlink()

    # -------------------------------------------------
    # 4. Summary / metadata
    # -------------------------------------------------

    summary = {
        "backend": "exact",
        "config": Path(config_file).name,

        "n_molecular_orbitals":
            system.n_molecular_orbitals,

        "n_metal_orbitals":
            system.n_metal_orbitals,

        "n_electrons":
            config.n_electrons,

        "n_vibrational_modes":
            len(
                system.vibrational_basis_sizes
            ),

        "mode_types":
            list(
                config.mode_types
            ),

        "vibrational_basis_sizes":
            list(
                system.vibrational_basis_sizes
            ),

        "electronic_dimension":
            system.electronic_dimension,

        "vibrational_dimension":
            system.vibrational_dimension,

        "total_dimension":
            system.total_dimension,

        "t_final":
            float(
                config.t_final
            ),

        "n_times":
            int(
                config.n_times
            ),

        "states_saved":
            bool(
                config.save_states
            ),

        "maximum_norm_error":
            float(
                norm_error
            ),

        "maximum_particle_number_error":
            float(
                particle_number_error
            ),

        "initial_molecular_population":
            float(
                observables.molecular_population[0]
            ),

        "final_molecular_population":
            float(
                observables.molecular_population[-1]
            ),

        "minimum_molecular_population":
            float(
                np.min(
                    observables.molecular_population
                )
            ),

        "maximum_molecular_population":
            float(
                np.max(
                    observables.molecular_population
                )
            ),
    }

    with open(
        output_directory / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    # -------------------------------------------------
    # 5. Copy exact configuration used
    # -------------------------------------------------

    source_config = Path(
        config_file
    ).resolve()

    shutil.copy2(
        source_config,
        output_directory / "config.yaml",
    )

    return output_directory