from __future__ import annotations

import numpy as np

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    exact_propagation,
    gaussian_nuclear_state,
    product_initial_state,
)

from ganbench.quantum.fermions import (
    number_operator,
)

from ganbench.quantum.pennylane_backend import (
    gan_trotter_final_state,
)

from ganbench.quantum.space import (
    lift_electronic,
)

from ganbench.quantum.toy_model import (
    build_quantum_toy_gan,
)

from ganbench.quantum.trotter import (
    GANAlgorithmTrotter,
    phase_aligned_state_error,
    state_fidelity,
)


def main() -> None:
    model = build_quantum_toy_gan()

    electronic = electronic_basis_state(
        [0, 2, 3],
        6,
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

    total_time = 2000.0

    # --------------------------------------------------
    # Exact reference
    # --------------------------------------------------

    exact = exact_propagation(
        model.hamiltonian,
        psi0,
        np.array(
            [0.0, total_time]
        ),
    )[-1]

    n0_operator = lift_electronic(
        number_operator(0, 6),
        8,
    )

    exact_n0 = np.vdot(
        exact,
        n0_operator @ exact,
    ).real

    operator_trotter = GANAlgorithmTrotter(
        model
    )

    print(
        "r     dt        "
        "circuit/operator    "
        "infidelity          "
        "state_error         "
        "n0_error"
    )

    for n_steps in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]:
        dt = total_time / n_steps

        # Operator-level implementation
        operator_state = (
            operator_trotter.final_state(
                psi0,
                total_time,
                n_steps,
            )
        )

        # Explicit PennyLane circuit
        circuit_state = (
            gan_trotter_final_state(
                psi0,
                model,
                total_time,
                n_steps,
            )
        )

        circuit_operator_error = (
            phase_aligned_state_error(
                operator_state,
                circuit_state,
            )
        )

        infidelity = (
            1.0
            - state_fidelity(
                exact,
                circuit_state,
            )
        )

        state_error = (
            phase_aligned_state_error(
                exact,
                circuit_state,
            )
        )

        circuit_n0 = np.vdot(
            circuit_state,
            n0_operator @ circuit_state,
        ).real

        n0_error = abs(
            circuit_n0
            - exact_n0
        )

        print(
            f"{n_steps:<5d} "
            f"{dt:<9.3f} "
            f"{circuit_operator_error:<19.8e} "
            f"{infidelity:<19.8e} "
            f"{state_error:<19.8e} "
            f"{n0_error:.8e}"
        )


if __name__ == "__main__":
    main()