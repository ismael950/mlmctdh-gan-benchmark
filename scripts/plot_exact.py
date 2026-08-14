from pathlib import Path

import matplotlib.pyplot as plt

from ganbench.model import load_config
from ganbench.exact.hamiltonian import build_exact_hamiltonian
from ganbench.exact.propagate import propagate_exact
from ganbench.exact.observables import compute_observables


def main() -> None:
    # Leer configuración
    config = load_config("configs/validation_small.yaml")

    # Construir y propagar el sistema
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

    # Extraer resultados
    times = observables.times

    molecular_population = observables.electronic_populations[:, 0]
    metal_population = observables.electronic_populations[:, 1]

    vibrational_occupation = observables.vibrational_occupations[:, 0]
    vibrational_coordinate = observables.vibrational_coordinates[:, 0]

    # Carpeta de salida
    output_directory = Path("results/validation_small")
    output_directory.mkdir(parents=True, exist_ok=True)

    # 1. Poblaciones electrónicas
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        times,
        molecular_population,
        label=r"$P_{\mathrm{mol}}(t)$",
    )

    plt.plot(
        times,
        metal_population,
        label=r"$P_{\mathrm{metal}}(t)$",
    )

    plt.xlabel("Time")
    plt.ylabel("Electronic population")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()

    electronic_path = output_directory / "electronic_populations.png"

    plt.savefig(
        electronic_path,
        dpi=200,
    )

    plt.close()

    # 2. Ocupación vibracional
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        times,
        vibrational_occupation,
    )

    plt.xlabel("Time")
    plt.ylabel(r"$\langle b^\dagger b\rangle$")
    plt.tight_layout()

    occupation_path = output_directory / "vibrational_occupation.png"

    plt.savefig(
        occupation_path,
        dpi=200,
    )

    plt.close()

    # 3. Coordenada vibracional
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        times,
        vibrational_coordinate,
    )

    plt.xlabel("Time")
    plt.ylabel(r"$\langle q\rangle$")
    plt.tight_layout()

    coordinate_path = output_directory / "vibrational_coordinate.png"

    plt.savefig(
        coordinate_path,
        dpi=200,
    )

    plt.close()

    print("Plots created:")
    print(electronic_path)
    print(occupation_path)
    print(coordinate_path)


if __name__ == "__main__":
    main()