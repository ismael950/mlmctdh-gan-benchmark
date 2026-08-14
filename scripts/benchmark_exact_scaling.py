from __future__ import annotations

import csv
import time
import statistics
from pathlib import Path

import numpy as np

from ganbench.model import (
    GANConfig,
    NuclearFactorSpec,
    NuclearProductSpec,
    UijTermSpec,
    VijTermSpec,
    WikTermSpec,
)
from ganbench.resources import estimate_resources
from ganbench.exact.hamiltonian import (
    build_exact_hamiltonian,
)
from ganbench.exact.propagate import (
    propagate_exact,
)


# ============================================================
# Benchmark settings
# ============================================================

# (n_metal_orbitals, vibrational_basis_size)
#
# With:
#   n_molecular_orbitals = 2
#   n_electrons = 2
#   n_vibrational_modes = 2
#
# these give total Hilbert-space dimensions approximately:
#
#   96, 240, 1008, 2880, 9100

CASES = [
    (16, 12),
    (20, 14),
    (24, 16),
    (30, 18),
]

T_FINAL = 2.0
N_TIMES = 51
N_REPEATS = 3

OUTPUT_FILE = Path(
    "validation/exact_scaling.csv"
)


# ============================================================
# Helpers
# ============================================================


def sparse_memory_bytes(matrix) -> int:
    """
    Actual memory occupied by a CSR sparse matrix.
    """

    return (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    )


def make_config(
    n_metal: int,
    basis_size: int,
) -> GANConfig:
    """
    Build one synthetic but representative GAN model.

    The physical structure is kept the same while
    the number of metal orbitals and vibrational
    primitive-basis sizes are increased.
    """

    n_mol = 2
    n_electrons = 2
    n_modes = 2

    # --------------------------------------------------------
    # Electronic parameters
    # --------------------------------------------------------

    molecular_energies = np.asarray(
        [0.0, 0.10],
        dtype=float,
    )

    # Keep the metal bandwidth fixed while increasing
    # the discretization.
    metal_energies = np.linspace(
        -0.5,
        0.5,
        n_metal,
        dtype=float,
    )

    # Old special-purpose molecule-metal coupling is off.
    molecule_metal_couplings = np.zeros(
        (n_mol, n_metal),
        dtype=float,
    )

    # --------------------------------------------------------
    # Vibrational parameters
    # --------------------------------------------------------

    frequencies = np.asarray(
        [0.5, 0.8],
        dtype=float,
    )

    basis_sizes = (
        basis_size,
        basis_size,
    )

    # Old special-purpose electron-vibration term is off.
    electron_vibration_couplings = np.zeros(
        (n_mol, n_modes),
        dtype=float,
    )

    # --------------------------------------------------------
    # U_0(Q)
    # --------------------------------------------------------

    u0_terms = (
        NuclearProductSpec(
            coefficient=0.25,
            factors=(
                NuclearFactorSpec(
                    mode=0,
                    kind="quadratic",
                    parameters={},
                ),
            ),
        ),
        NuclearProductSpec(
            coefficient=0.40,
            factors=(
                NuclearFactorSpec(
                    mode=1,
                    kind="quadratic",
                    parameters={},
                ),
            ),
        ),
    )

    # --------------------------------------------------------
    # U_ij(Q)
    #
    # Coordinate-dependent molecular coupling.
    # --------------------------------------------------------

    uij_terms = (
        UijTermSpec(
            i=0,
            j=1,
            nuclear=NuclearProductSpec(
                coefficient=0.05,
                factors=(
                    NuclearFactorSpec(
                        mode=0,
                        kind="linear",
                        parameters={},
                    ),
                ),
            ),
        ),
    )

    # --------------------------------------------------------
    # V_ij(Q)
    # --------------------------------------------------------

    vij_terms = (
        VijTermSpec(
            i=0,
            j=1,
            nuclear=NuclearProductSpec(
                coefficient=0.03,
                factors=(
                    NuclearFactorSpec(
                        mode=1,
                        kind="linear",
                        parameters={},
                    ),
                ),
            ),
        ),
    )

    # --------------------------------------------------------
    # W_ik(Q)
    #
    # Couple molecular orbital 0 to every metal orbital.
    #
    # Scale the individual coupling approximately as
    # 1 / sqrt(N_metal) so increasing N_metal does not
    # automatically make the total hybridization grow.
    # --------------------------------------------------------

    coupling_scale = (
        0.10 / np.sqrt(n_metal)
    )

    wik_terms = tuple(
        WikTermSpec(
            i=0,
            k=k,
            nuclear=NuclearProductSpec(
                coefficient=float(
                    coupling_scale
                ),
                factors=(
                    NuclearFactorSpec(
                        mode=0,
                        kind="exponential",
                        parameters={
                            "alpha": 0.3,
                            "center": 0.0,
                        },
                    ),
                ),
            ),
        )
        for k in range(n_metal)
    )

    # --------------------------------------------------------
    # Initial state
    #
    # Orbital ordering:
    #   0, 1 = molecule
    #   2, ... = metal
    #
    # One electron starts on molecular orbital 0
    # and one in the first metal orbital.
    # --------------------------------------------------------

    occupied_orbitals = (
        0,
        2,
    )

    vibrational_levels = (
        0,
        0,
    )

    return GANConfig(
        n_molecular_orbitals=n_mol,
        n_metal_orbitals=n_metal,
        n_electrons=n_electrons,
        n_vibrational_modes=n_modes,

        molecular_energies=molecular_energies,
        metal_energies=metal_energies,
        molecule_metal_couplings=(
            molecule_metal_couplings
        ),

        frequencies=frequencies,
        basis_sizes=basis_sizes,
        electron_vibration_couplings=(
            electron_vibration_couplings
        ),

        occupied_orbitals=occupied_orbitals,
        vibrational_levels=vibrational_levels,

        t_final=T_FINAL,
        n_times=N_TIMES,

        u0_terms=u0_terms,
        uij_terms=uij_terms,
        vij_terms=vij_terms,
        wik_terms=wik_terms,
    )


# ============================================================
# Main benchmark
# ============================================================


def main() -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print(
        "\nExact GAN scaling calibration\n"
    )

    header = (
        f"{'Nmetal':>7} "
        f"{'K':>5} "
        f"{'Del':>8} "
        f"{'Dvib':>8} "
        f"{'Dtotal':>10} "
        f"{'nnz':>12} "
        f"{'H MB':>10} "
        f"{'build s':>10} "
        f"{'prop s':>10}"
    )

    print(header)
    print("-" * len(header))

    for (
        n_metal,
        basis_size,
    ) in CASES:

        config = make_config(
            n_metal=n_metal,
            basis_size=basis_size,
        )

        estimate = estimate_resources(
            config
        )

        # ----------------------------------------------------
        # Hamiltonian construction
        # ----------------------------------------------------

        start = time.perf_counter()

        system = build_exact_hamiltonian(
            config
        )

        build_time = (
            time.perf_counter()
            - start
        )

        H = system.hamiltonian

        sparse_mb = (
            sparse_memory_bytes(H)
            / (1024**2)
        )

        # ----------------------------------------------------
        # Propagation
        # ----------------------------------------------------

        # Warm-up run.
        propagation = propagate_exact(
            system=system,
            t_final=config.t_final,
            n_times=config.n_times,
        )

        propagation_times = []

        for _ in range(N_REPEATS):

            start = time.perf_counter()

            propagation = propagate_exact(
                system=system,
                t_final=config.t_final,
                n_times=config.n_times,
            )

            propagation_times.append(
                time.perf_counter() - start
            )

        propagation_time = statistics.median(
            propagation_times
        )

        norm_error = float(
            np.max(
                np.abs(
                    propagation.norms
                    - 1.0
                )
            )
        )

        propagation_min = min(propagation_times)
        propagation_max = max(propagation_times)

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        row = {
            "n_metal_orbitals":
                n_metal,

            "basis_size":
                basis_size,

            "electronic_dimension":
                estimate.electronic_dimension,

            "vibrational_dimension":
                estimate.vibrational_dimension,

            "total_dimension":
                estimate.total_dimension,

            "hamiltonian_nnz":
                H.nnz,

            "hamiltonian_density":
                H.nnz
                / (
                    H.shape[0]
                    * H.shape[1]
                ),

            "sparse_memory_mb":
                sparse_mb,

            "trajectory_memory_mb":
                estimate.trajectory_memory_mb,

            "build_time_s":
                build_time,

            "propagation_time_s":
                propagation_time,

            "total_time_s":
                build_time
                + propagation_time,

            "maximum_norm_error":
                norm_error,
        }

        results.append(
            row
        )

        print(
            f"{n_metal:7d} "
            f"{basis_size:5d} "
            f"{estimate.electronic_dimension:8d} "
            f"{estimate.vibrational_dimension:8d} "
            f"{estimate.total_dimension:10d} "
            f"{H.nnz:12d} "
            f"{sparse_mb:10.3f} "
            f"{build_time:10.3f} "
            f"{propagation_time:10.3f}"
        )

    # ========================================================
    # Write CSV
    # ========================================================

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print()
    print(
        "Results written to:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Important:"
    )
    print(
        "This is a computational calibration, "
        "not a physical benchmark."
    )

    print(
        f"    repeats: "
        f"{propagation_min:.3f} - "
        f"{propagation_max:.3f} s"
    )


if __name__ == "__main__":
    main()