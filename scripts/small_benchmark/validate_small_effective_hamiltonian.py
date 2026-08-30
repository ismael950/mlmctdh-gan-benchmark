from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import pennylane.labs.trotter_error as te

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    exact_propagation,
    gaussian_nuclear_state,
    product_initial_state,
)
from ganbench.quantum.fermions import number_operator
from ganbench.quantum.space import lift_electronic
from ganbench.quantum.toy_model import build_quantum_toy_gan


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "results"
    / "small_direct_benchmark"
)

TROTTER_DIR = BASE / "trotter_sweep"
OUT = BASE / "effective_hamiltonian"

N_STEPS_LIST = [
    200,
    400,
    600,
    800,
    1000,
    1200,
    1400,
    1600,
]

BCH_ORDER = 2


def expectation(
    state: np.ndarray,
    operator: csr_matrix,
) -> float:
    return float(
        np.vdot(
            state,
            operator @ state,
        ).real
    )


def sparse_max_abs(
    matrix: csr_matrix,
) -> float:
    if matrix.nnz == 0:
        return 0.0

    return float(
        np.max(np.abs(matrix.data))
    )


def max_population_error(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> tuple[float, int, int]:

    e0 = np.abs(a0 - b0)
    e1 = np.abs(a1 - b1)

    combined = np.maximum(
        e0,
        e1,
    )

    idx = int(
        np.argmax(combined)
    )

    orbital = (
        0
        if e0[idx] >= e1[idx]
        else 1
    )

    return (
        float(combined[idx]),
        idx,
        orbital,
    )


def fit_slope(
    dt: np.ndarray,
    error: np.ndarray,
) -> float:

    mask = error > 0.0

    return float(
        np.polyfit(
            np.log(dt[mask]),
            np.log(error[mask]),
            1,
        )[0]
    )


def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Existing exact reference
    # ------------------------------------------------------------

    exact = pd.read_csv(
        BASE
        / "exact"
        / "reference_populations.csv"
    )

    times = exact[
        "time_au"
    ].to_numpy()

    times_fs = exact[
        "time_fs"
    ].to_numpy()

    exact_n0 = exact[
        "n0_exact"
    ].to_numpy()

    exact_n1 = exact[
        "n1_exact"
    ].to_numpy()

    total_time = float(
        times[-1]
    )

    # ------------------------------------------------------------
    # Canonical K32 model and initial state
    # ------------------------------------------------------------

    model = build_quantum_toy_gan(
        nuclear_size=32,
    )

    electronic = electronic_basis_state(
        [0, 2, 3],
        model.n_electronic_orbitals,
    )

    nuclear = gaussian_nuclear_state(
        model.nuclear_grid.points,
        center=0.0,
        sigma=1.0,
    )

    psi0 = product_initial_state(
        electronic,
        nuclear,
    )

    n0_operator = csr_matrix(
        lift_electronic(
            number_operator(
                0,
                model.n_electronic_orbitals,
            ),
            model.nuclear_dimension,
        )
    )

    n1_operator = csr_matrix(
        lift_electronic(
            number_operator(
                1,
                model.n_electronic_orbitals,
            ),
            model.nuclear_dimension,
        )
    )

    # ------------------------------------------------------------
    # Physical Trotter fragments
    #
    # GANAlgorithmTrotter applies:
    #
    #     F0 -> F1 -> ... -> Flast
    #
    # to the state.
    #
    # Therefore the operator product is
    #
    #     exp(-i dt Flast) ... exp(-i dt F1) exp(-i dt F0)
    #
    # ------------------------------------------------------------

    fragment_matrices = [
        csr_matrix(model.f0),
        *[
            csr_matrix(fragment)
            for fragment
            in model.hopping_fragments
        ],
        csr_matrix(model.f_last),
    ]

    labels = [
        f"F{i}"
        for i in range(
            len(fragment_matrices)
        )
    ]

    wrapped = te.generic_fragments(
        fragment_matrices
    )

    fragments = dict(
        zip(
            labels,
            wrapped,
        )
    )

    # Reverse because operators act on the ket
    # right-to-left.
    product_labels = list(
        reversed(labels)
    )

    # PennyLane defines exp(+i t alpha H).
    # Our physical evolution is exp(-i dt H).
    product_coeffs = [
        -1.0
        for _ in product_labels
    ]

    product_formula = te.ProductFormula(
        product_labels,
        coeffs=product_coeffs,
    )

    print()
    print(
        "Product formula:"
    )
    print(
        product_formula
    )

    print()
    print(
        "Physical application order:",
        " -> ".join(labels),
    )

    print(
        "Matrix product order:",
        " @ ".join(product_labels),
    )

    # ------------------------------------------------------------
    # Sanity check:
    #
    # BCH order 1 must reproduce H exactly after
    # converting Omega -> H_eff:
    #
    #     Omega = -i dt H_eff
    #     H_eff = (i/dt) Omega
    #
    # ------------------------------------------------------------

    test_dt = 1.0

    omega_order1 = (
        te.effective_hamiltonian(
            product_formula,
            fragments,
            order=1,
            timestep=test_dt,
        )
    )

    h_order1 = (
        (1.0j / test_dt)
        * omega_order1.fragment
    )

    exact_h_sparse = csr_matrix(
        model.hamiltonian
    )

    order1_difference = (
        h_order1
        - exact_h_sparse
    )

    order1_error = sparse_max_abs(
        order1_difference
    )

    print()
    print(
        "Order-1 BCH sanity check:"
    )
    print(
        "max |H_BCH(order=1) - H| =",
        order1_error,
    )

    if order1_error > 1.0e-12:
        raise RuntimeError(
            "BCH convention check failed."
        )

    # ------------------------------------------------------------
    # Effective-Hamiltonian sweep
    # ------------------------------------------------------------

    summary_rows = []

    for n_steps in N_STEPS_LIST:

        dt = (
            total_time
            / n_steps
        )

        print()
        print(
            "=" * 60
        )
        print(
            f"{n_steps} steps"
        )
        print(
            f"dt = {dt:.12g} a.u."
        )

        # --------------------------------------------------------
        # Build truncated BCH generator Omega.
        #
        # effective_hamiltonian() in the installed PennyLane
        # 0.45.0 GenericFragment path returns the BCH generator
        # observed numerically as
        #
        #     Omega = -i dt H_eff
        #
        # for our negative product coefficients.
        # --------------------------------------------------------

        omega = (
            te.effective_hamiltonian(
                product_formula,
                fragments,
                order=BCH_ORDER,
                timestep=dt,
            )
        )

        h_eff = (
            (1.0j / dt)
            * omega.fragment
        ).tocsr()

        # --------------------------------------------------------
        # Hermiticity check
        # --------------------------------------------------------

        antihermitian_part = (
            h_eff
            - h_eff.conjugate().T
        )

        hermiticity_error = (
            sparse_max_abs(
                antihermitian_part
            )
        )

        print(
            "Hermiticity error =",
            hermiticity_error,
        )

        if hermiticity_error > 1.0e-11:
            raise RuntimeError(
                "H_eff is not Hermitian "
                "within numerical tolerance."
            )

        # --------------------------------------------------------
        # Propagate H_eff exactly on the same K32 space
        # --------------------------------------------------------

        eff_states = exact_propagation(
            h_eff,
            psi0,
            times,
        )

        eff_n0 = np.asarray([
            expectation(
                state,
                n0_operator,
            )
            for state in eff_states
        ])

        eff_n1 = np.asarray([
            expectation(
                state,
                n1_operator,
            )
            for state in eff_states
        ])

        eff_norm = np.asarray([
            float(
                np.vdot(
                    state,
                    state,
                ).real
            )
            for state in eff_states
        ])

        # --------------------------------------------------------
        # Existing true Trotter trajectory
        # --------------------------------------------------------

        trotter_file = (
            TROTTER_DIR
            / f"steps_{n_steps:04d}.csv"
        )

        if not trotter_file.exists():
            raise FileNotFoundError(
                trotter_file
            )

        trotter = pd.read_csv(
            trotter_file
        )

        if not np.allclose(
            trotter["time_au"].to_numpy(),
            times,
        ):
            raise RuntimeError(
                "Trotter and exact time grids differ."
            )

        trotter_n0 = trotter[
            "n0_quantum"
        ].to_numpy()

        trotter_n1 = trotter[
            "n1_quantum"
        ].to_numpy()

        # --------------------------------------------------------
        # Three central metrics
        # --------------------------------------------------------

        (
            e_trot,
            idx_trot,
            orb_trot,
        ) = max_population_error(
            trotter_n0,
            trotter_n1,
            exact_n0,
            exact_n1,
        )

        (
            e_eff,
            idx_eff,
            orb_eff,
        ) = max_population_error(
            eff_n0,
            eff_n1,
            exact_n0,
            exact_n1,
        )

        (
            e_model,
            idx_model,
            orb_model,
        ) = max_population_error(
            eff_n0,
            eff_n1,
            trotter_n0,
            trotter_n1,
        )

        print(
            "E_Trot  =",
            e_trot,
        )
        print(
            "E_eff   =",
            e_eff,
        )
        print(
            "E_model =",
            e_model,
        )

        # --------------------------------------------------------
        # Save full trajectory
        # --------------------------------------------------------

        trajectory = pd.DataFrame(
            {
                "time_au":
                    times,
                "time_fs":
                    times_fs,
                "n0_exact":
                    exact_n0,
                "n1_exact":
                    exact_n1,
                "n0_trotter":
                    trotter_n0,
                "n1_trotter":
                    trotter_n1,
                "n0_heff":
                    eff_n0,
                "n1_heff":
                    eff_n1,
                "error_trot_n0":
                    np.abs(
                        trotter_n0
                        - exact_n0
                    ),
                "error_trot_n1":
                    np.abs(
                        trotter_n1
                        - exact_n1
                    ),
                "error_heff_n0":
                    np.abs(
                        eff_n0
                        - exact_n0
                    ),
                "error_heff_n1":
                    np.abs(
                        eff_n1
                        - exact_n1
                    ),
                "error_model_n0":
                    np.abs(
                        eff_n0
                        - trotter_n0
                    ),
                "error_model_n1":
                    np.abs(
                        eff_n1
                        - trotter_n1
                    ),
                "norm_heff":
                    eff_norm,
            }
        )

        trajectory.to_csv(
            OUT
            / f"steps_{n_steps:04d}.csv",
            index=False,
        )

        summary_rows.append(
            {
                "n_trotter_steps":
                    n_steps,
                "dt_au":
                    dt,
                "E_trot":
                    e_trot,
                "E_eff":
                    e_eff,
                "E_model":
                    e_model,
                "E_trot_worst_time_au":
                    float(
                        times[idx_trot]
                    ),
                "E_trot_worst_orbital":
                    orb_trot,
                "E_eff_worst_time_au":
                    float(
                        times[idx_eff]
                    ),
                "E_eff_worst_orbital":
                    orb_eff,
                "E_model_worst_time_au":
                    float(
                        times[idx_model]
                    ),
                "E_model_worst_orbital":
                    orb_model,
                "heff_hermiticity_error":
                    hermiticity_error,
                "heff_max_norm_error":
                    float(
                        np.max(
                            np.abs(
                                eff_norm
                                - 1.0
                            )
                        )
                    ),
            }
        )

    # ------------------------------------------------------------
    # Convergence table and slopes
    # ------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT / "convergence.csv",
        index=False,
    )

    dt = summary[
        "dt_au"
    ].to_numpy()

    e_trot = summary[
        "E_trot"
    ].to_numpy()

    e_eff = summary[
        "E_eff"
    ].to_numpy()

    e_model = summary[
        "E_model"
    ].to_numpy()

    slope_trot_all = fit_slope(
        dt,
        e_trot,
    )

    slope_eff_all = fit_slope(
        dt,
        e_eff,
    )

    slope_model_all = fit_slope(
        dt,
        e_model,
    )

    # Smallest four timesteps = most relevant
    # asymptotic regime.
    slope_trot_small = fit_slope(
        dt[-4:],
        e_trot[-4:],
    )

    slope_eff_small = fit_slope(
        dt[-4:],
        e_eff[-4:],
    )

    slope_model_small = fit_slope(
        dt[-4:],
        e_model[-4:],
    )

    print()
    print(
        "=" * 60
    )
    print(
        "BCH effective-Hamiltonian validation"
    )
    print(
        "=" * 60
    )

    print()
    print(
        summary[
            [
                "n_trotter_steps",
                "dt_au",
                "E_trot",
                "E_eff",
                "E_model",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Power-law slopes using all points:"
    )
    print(
        "Trotter    =",
        slope_trot_all,
    )
    print(
        "H_eff      =",
        slope_eff_all,
    )
    print(
        "Model resid=",
        slope_model_all,
    )

    print()
    print(
        "Power-law slopes using smallest 4 dt:"
    )
    print(
        "Trotter    =",
        slope_trot_small,
    )
    print(
        "H_eff      =",
        slope_eff_small,
    )
    print(
        "Model resid=",
        slope_model_small,
    )

    print()
    print(
        "Expected asymptotic behaviour:"
    )
    print(
        "E_Trot  ~ dt^1"
    )
    print(
        "E_eff   ~ dt^1"
    )
    print(
        "E_model ~ dt^2"
    )

    print()
    print(
        "Saved to:",
        OUT,
    )


if __name__ == "__main__":
    main()