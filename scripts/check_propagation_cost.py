import sys
import time

from ganbench.model import load_config
from ganbench.resources import estimate_resources
from ganbench.exact.hamiltonian import (
    build_exact_hamiltonian,
)
from ganbench.exact.propagate import (
    propagate_exact,
)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/check_propagation_cost.py "
            "configs/file.yaml"
        )

    config_file = sys.argv[1]

    config = load_config(config_file)
    estimate = estimate_resources(config)

    # Hamiltonian construction
    start = time.perf_counter()

    system = build_exact_hamiltonian(
        config
    )

    build_time = (
        time.perf_counter() - start
    )

    # Exact propagation
    start = time.perf_counter()

    result = propagate_exact(
        system,
        t_final=config.t_final,
        n_times=config.n_times,
    )

    propagation_time = (
        time.perf_counter() - start
    )

    maximum_norm_error = max(
        abs(result.norms - 1.0)
    )

    print(f"Config: {config_file}")
    print()

    print("Dimensions")
    print(
        f"Electronic: {estimate.electronic_dimension}"
    )
    print(
        f"Vibrational: {estimate.vibrational_dimension}"
    )
    print(
        f"Total: {estimate.total_dimension}"
    )

    print()
    print("Timing")
    print(
        f"Hamiltonian build: "
        f"{build_time:.6f} s"
    )
    print(
        f"Propagation: "
        f"{propagation_time:.6f} s"
    )
    print(
        f"Total: "
        f"{build_time + propagation_time:.6f} s"
    )

    print()
    print("Propagation storage")
    print(
        f"Saved states: {config.n_times}"
    )
    print(
        "Estimated trajectory memory: "
        f"{estimate.trajectory_memory_mb:.3f} MB"
    )

    print()
    print("Numerical check")
    print(
        "Maximum norm error: "
        f"{maximum_norm_error:.3e}"
    )


if __name__ == "__main__":
    main()