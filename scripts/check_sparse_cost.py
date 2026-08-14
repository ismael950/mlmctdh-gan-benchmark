import sys
import time

from ganbench.model import load_config
from ganbench.resources import estimate_resources
from ganbench.exact.hamiltonian import (
    build_exact_hamiltonian,
)


def sparse_memory_bytes(matrix):
    """
    Actual memory used by a CSR sparse matrix.
    """

    return (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/check_sparse_cost.py "
            "configs/file.yaml"
        )

    config_file = sys.argv[1]

    config = load_config(config_file)
    estimate = estimate_resources(config)

    start = time.perf_counter()

    system = build_exact_hamiltonian(config)

    build_time = (
        time.perf_counter() - start
    )

    H = system.hamiltonian

    sparse_bytes = sparse_memory_bytes(H)

    total_entries = (
        H.shape[0] * H.shape[1]
    )

    density = (
        H.nnz / total_entries
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
    print("Sparse Hamiltonian")

    print(
        f"Nonzero elements: {H.nnz}"
    )

    print(
        f"Matrix density: {density:.6e}"
    )

    print(
        "Actual sparse memory: "
        f"{sparse_bytes / 1024**2:.3f} MB"
    )

    print(
        "Dense-equivalent memory: "
        f"{estimate.dense_hamiltonian_memory_mb:.3f} MB"
    )

    print()
    print(
        f"Hamiltonian build time: "
        f"{build_time:.6f} s"
    )


if __name__ == "__main__":
    main()