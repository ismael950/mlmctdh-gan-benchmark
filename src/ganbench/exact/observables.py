from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from ganbench.exact.hamiltonian import ExactSystem
from ganbench.exact.propagate import PropagationResult


@dataclass
class ObservableResult:
    """Observables calculated during propagation."""

    times: NDArray[np.float64]

    # Individual orbital populations:
    # molecular orbitals first, metal orbitals second.
    electronic_populations: NDArray[np.float64]

    # Aggregate electronic populations.
    molecular_population: NDArray[np.float64]
    metal_population: NDArray[np.float64]
    total_electronic_population: NDArray[np.float64]

    # Nuclear-coordinate expectation values.
    vibrational_coordinates: NDArray[np.float64]

    # Diagnostic only:
    # <a^dagger a> for each vibrational mode.
    vibrational_occupations: NDArray[np.float64]

    # For mode k:
    # vibrational_distributions[k].shape
    #     = (n_times, K_k)
    #
    # Entry [t, nu] is the probability of finding
    # mode k in vibrational basis state |nu>.
    vibrational_distributions: tuple[
        NDArray[np.float64], ...
    ]


def expectation_values(
    states: NDArray[np.complex128],
    operator: sparse.spmatrix,
) -> NDArray[np.float64]:
    """
    Calculate <psi(t)|O|psi(t)> at all times.
    """

    operated_states = operator.dot(states.T).T

    values = np.sum(
        states.conj() * operated_states,
        axis=1,
    )

    maximum_imaginary_part = np.max(
        np.abs(values.imag)
    )

    if maximum_imaginary_part > 1.0e-10:
        raise ValueError(
            "Expectation value has a significant "
            "imaginary part: "
            f"{maximum_imaginary_part}"
        )

    return values.real.astype(np.float64)


def vibrational_distributions(
    system: ExactSystem,
    states: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], ...]:
    """
    Calculate the marginal vibrational-level
    distribution for every nuclear mode.

    For mode k,

        P_nu^(k)(t)

    is obtained by summing |Psi(t)|^2 over the
    electronic degrees of freedom and over all
    vibrational modes except k.

    For the current harmonic benchmark, these
    basis-state populations are the vibrational
    energy distribution.
    """

    basis_sizes = (
        system.vibrational_basis_sizes
    )

    n_times = states.shape[0]

    reshaped_states = states.reshape(
        (
            n_times,
            system.electronic_dimension,
            *basis_sizes,
        )
    )

    probabilities = np.abs(
        reshaped_states
    ) ** 2

    distributions = []

    n_modes = len(basis_sizes)

    for mode in range(n_modes):
        # Axis 0 = time
        # Axis 1 = electronic basis
        # Axis 2 + mode = selected vibrational mode
        selected_axis = 2 + mode

        axes_to_sum = tuple(
            axis
            for axis in range(
                probabilities.ndim
            )
            if axis not in (
                0,
                selected_axis,
            )
        )

        distribution = np.sum(
            probabilities,
            axis=axes_to_sum,
        )

        distributions.append(
            distribution.astype(np.float64)
        )

    return tuple(distributions)


def compute_observables(
    system: ExactSystem,
    propagation: PropagationResult,
) -> ObservableResult:
    """
    Calculate common electronic and nuclear
    observables for the benchmark.
    """

    electronic_identity = sparse.identity(
        system.electronic_dimension,
        dtype=np.complex128,
        format="csr",
    )

    vibrational_identity = sparse.identity(
        system.vibrational_dimension,
        dtype=np.complex128,
        format="csr",
    )

    # O_el -> O_el x I_vib
    full_electronic_operators = [
        sparse.kron(
            operator,
            vibrational_identity,
            format="csr",
        )
        for operator in (
            system.electronic_number_operators
        )
    ]

    # O_vib -> I_el x O_vib
    full_vibrational_number_operators = [
        sparse.kron(
            electronic_identity,
            operator,
            format="csr",
        )
        for operator in (
            system.vibrational_number_operators
        )
    ]

    full_vibrational_coordinate_operators = [
        sparse.kron(
            electronic_identity,
            operator,
            format="csr",
        )
        for operator in (
            system.vibrational_coordinate_operators
        )
    ]

    electronic_populations = np.column_stack(
        [
            expectation_values(
                propagation.states,
                operator,
            )
            for operator in (
                full_electronic_operators
            )
        ]
    )

    molecular_population = np.sum(
        electronic_populations[
            :,
            :system.n_molecular_orbitals,
        ],
        axis=1,
    )

    metal_population = np.sum(
        electronic_populations[
            :,
            system.n_molecular_orbitals:,
        ],
        axis=1,
    )

    total_electronic_population = (
        molecular_population
        + metal_population
    )

    vibrational_occupations = np.column_stack(
        [
            expectation_values(
                propagation.states,
                operator,
            )
            for operator in (
                full_vibrational_number_operators
            )
        ]
    )

    vibrational_coordinates = np.column_stack(
        [
            expectation_values(
                propagation.states,
                operator,
            )
            for operator in (
                full_vibrational_coordinate_operators
            )
        ]
    )

    distributions = vibrational_distributions(
        system,
        propagation.states,
    )

    return ObservableResult(
        times=propagation.times,
        electronic_populations=(
            electronic_populations
        ),
        molecular_population=(
            molecular_population
        ),
        metal_population=metal_population,
        total_electronic_population=(
            total_electronic_population
        ),
        vibrational_coordinates=(
            vibrational_coordinates
        ),
        vibrational_occupations=(
            vibrational_occupations
        ),
        vibrational_distributions=(
            distributions
        ),
    )