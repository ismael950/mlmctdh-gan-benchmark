from dataclasses import replace
from pathlib import Path

import numpy as np

from ganbench.model import load_config
from ganbench.exact.hamiltonian import build_exact_hamiltonian
from ganbench.exact.propagate import propagate_exact
from ganbench.exact.observables import compute_observables


def main() -> None:
    base_config = load_config("configs/validation_small.yaml")

    basis_sizes = [8, 12, 16, 20]

    output_root = Path("results/basis_convergence")
    output_root.mkdir(parents=True, exist_ok=True)

    for basis_size in basis_sizes:
        print(f"\nRunning K = {basis_size}")

        # Crear una copia de la configuración cambiando únicamente K
        config = replace(
            base_config,
            basis_sizes=np.array(
                [basis_size],
                dtype=np.int64,
            ),
        )

        system = build_exact_hamiltonian(config)

        propagation = propagate_exact(
            system=system,
            t_final=config.t_final,
            n_times=config.n_times,
        )

        observables = compute_observables(
            system=system,
            propagation=propagation,
        )

        # Como tenemos un modo:
        # states.shape = (n_times, electronic_dimension, K)
        reshaped_states = propagation.states.reshape(
            len(propagation.times),
            system.electronic_dimension,
            basis_size,
        )

        # Población del último nivel vibracional |K-1>
        highest_level_population = np.sum(
            np.abs(reshaped_states[:, :, -1]) ** 2,
            axis=1,
        )

        output_directory = output_root / f"K{basis_size:02d}"
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = output_directory / "data.npz"

        np.savez_compressed(
            output_file,
            basis_size=np.array(basis_size),
            times=propagation.times,
            states=propagation.states,
            norms=propagation.norms,
            electronic_populations=observables.electronic_populations,
            total_electronic_population=(
                observables.total_electronic_population
            ),
            vibrational_occupations=(
                observables.vibrational_occupations
            ),
            vibrational_coordinates=(
                observables.vibrational_coordinates
            ),
            highest_level_population=highest_level_population,
        )

        norm_error = np.max(
            np.abs(propagation.norms - 1.0)
        )

        particle_number_error = np.max(
            np.abs(
                observables.total_electronic_population - 1.0
            )
        )

        print("Total Hilbert-space dimension:", system.total_dimension)
        print("Maximum norm error:", norm_error)
        print(
            "Maximum particle-number error:",
            particle_number_error,
        )
        print(
            "Maximum highest-level population:",
            np.max(highest_level_population),
        )
        print("Saved:", output_file)


if __name__ == "__main__":
    main()