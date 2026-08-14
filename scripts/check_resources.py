import sys

from ganbench.model import load_config
from ganbench.resources import estimate_resources


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/check_resources.py "
            "configs/file.yaml"
        )

    config_file = sys.argv[1]

    config = load_config(
        config_file
    )

    estimate = estimate_resources(
        config
    )

    print(f"Config: {config_file}")
    print()

    print("Hilbert-space dimensions")
    print(
        "Electronic:",
        estimate.electronic_dimension,
    )
    print(
        "Vibrational:",
        estimate.vibrational_dimension,
    )
    print(
        "Total:",
        estimate.total_dimension,
    )

    print()
    print("Memory estimates")

    print(
        "State vector:",
        f"{estimate.state_memory_mb:.3f} MB",
    )

    print(
        "Dense Hamiltonian:",
        f"{estimate.dense_hamiltonian_memory_mb:.3f} MB",
    )

    print(
        "Dense Hamiltonian:",
        f"{estimate.dense_hamiltonian_memory_gb:.3f} GB",
    )

    print(
        "Full trajectory:",
        f"{estimate.trajectory_memory_mb:.3f} MB",
    )

    print(
        "Full trajectory:",
        f"{estimate.trajectory_memory_gb:.3f} GB",
    )


if __name__ == "__main__":
    main()