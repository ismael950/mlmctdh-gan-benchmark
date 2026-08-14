from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from operator import mul

import numpy as np
from scipy import sparse

from ganbench.model import GANConfig


@dataclass(frozen=True)
class ExactSystem:
    """Hamiltonian, initial state, and useful operators."""

    hamiltonian: sparse.csr_matrix
    initial_state: np.ndarray

    electronic_number_operators: tuple[
        sparse.csr_matrix, ...
    ]

    vibrational_number_operators: tuple[
        sparse.csr_matrix, ...
    ]

    vibrational_coordinate_operators: tuple[
        sparse.csr_matrix, ...
    ]

    electronic_basis: tuple[tuple[int, ...], ...]

    electronic_dimension: int
    vibrational_dimension: int

    n_molecular_orbitals: int
    n_metal_orbitals: int
    vibrational_basis_sizes: tuple[int, ...]

    @property
    def total_dimension(self) -> int:
        return self.hamiltonian.shape[0]


def _kron_all(
    operators: list[sparse.spmatrix],
) -> sparse.csr_matrix:
    """Kronecker product of several operators."""

    result = sparse.csr_matrix([[1.0]])

    for operator in operators:
        result = sparse.kron(
            result,
            operator,
            format="csr",
        )

    return result


def _local_vibrational_operators(
    basis_size: int,
    frequency: float,
) -> dict[str, sparse.csr_matrix]:
    """
    Harmonic-oscillator operators in the truncated
    number-state basis |0>, ..., |basis_size - 1>.

    The dimensionless coordinate is

        Q = (a + a^dagger) / sqrt(2)

    and the harmonic Hamiltonian is separated as

        H = T + V_harm.
    """

    ladder_values = np.sqrt(
        np.arange(1, basis_size, dtype=float)
    )

    annihilation = sparse.diags(
        ladder_values,
        offsets=1,
        shape=(basis_size, basis_size),
        format="csr",
    )

    creation = annihilation.T.conj().tocsr()

    identity = sparse.identity(
        basis_size,
        format="csr",
    )

    number = creation @ annihilation

    coordinate = (
        annihilation + creation
    ) / np.sqrt(2.0)

    # Original harmonic-oscillator Hamiltonian
    hamiltonian = frequency * (
        number + 0.5 * identity
    )

    # Harmonic potential:
    #
    # V(Q) = (omega / 2) Q^2
    harmonic_potential = (
        0.5
        * frequency
        * (coordinate @ coordinate)
    )

    # Define the kinetic operator so that
    #
    # T + V_harm = H_HO
    #
    # exactly within the chosen truncated basis.
    kinetic = (
        hamiltonian
        - harmonic_potential
    )

    return {
        "identity": identity,
        "number": number.tocsr(),
        "coordinate": coordinate.tocsr(),
        "kinetic": kinetic.tocsr(),
        "harmonic_potential": (
            harmonic_potential.tocsr()
        ),
        "hamiltonian": hamiltonian.tocsr(),
    }

def _function_of_coordinate(
    coordinate: sparse.csr_matrix,
    function,
) -> sparse.csr_matrix:
    """
    Construct f(Q) from the finite-dimensional
    coordinate operator Q.

    Since Q is Hermitian, diagonalize it as

        Q = U diag(q_alpha) U^dagger

    and define

        f(Q) = U diag(f(q_alpha)) U^dagger.
    """

    coordinate_dense = coordinate.toarray()

    eigenvalues, eigenvectors = np.linalg.eigh(
        coordinate_dense
    )

    function_values = np.asarray(
        function(eigenvalues),
        dtype=complex,
    )

    operator_dense = (
        eigenvectors
        @ np.diag(function_values)
        @ eigenvectors.conj().T
    )

    return sparse.csr_matrix(
        operator_dense
    )

def _nuclear_factor_operator(
    factor,
    local_operators,
) -> sparse.csr_matrix:
    """
    Construct one local nuclear factor f(Q_mode).
    """

    mode = factor.mode
    kind = factor.kind
    parameters = factor.parameters

    coordinate = (
        local_operators[mode]["coordinate"]
    )

    if kind == "linear":
        center = parameters.get(
            "center",
            0.0,
        )

        return _function_of_coordinate(
            coordinate,
            lambda q: q - center,
        )

    if kind == "quadratic":
        center = parameters.get(
            "center",
            0.0,
        )

        return _function_of_coordinate(
            coordinate,
            lambda q: (q - center) ** 2,
        )

    if kind == "morse":
        depth = parameters["depth"]
        alpha = parameters["alpha"]
        equilibrium = parameters.get(
            "equilibrium",
            0.0,
        )

        return _function_of_coordinate(
            coordinate,
            lambda q: depth
            * (
                1.0
                - np.exp(
                    -alpha
                    * (q - equilibrium)
                )
            ) ** 2,
        )

    if kind == "exponential":
        alpha = parameters["alpha"]
        center = parameters.get(
            "center",
            0.0,
        )

        return _function_of_coordinate(
            coordinate,
            lambda q: np.exp(
                -alpha
                * (q - center)
            ),
        )

    if kind == "tanh":
        alpha = parameters.get(
            "alpha",
            1.0,
        )

        center = parameters.get(
            "center",
            0.0,
        )

        return _function_of_coordinate(
            coordinate,
            lambda q: np.tanh(
                alpha
                * (q - center)
            ),
        )

    raise ValueError(
        f"Unknown nuclear function kind: {kind}"
    )


def _nuclear_product_operator(
    product,
    local_operators,
) -> sparse.csr_matrix:
    """
    Construct

        coefficient * product_r f_r(Q_mode_r)

    in the complete vibrational Hilbert space.
    """

    factors = [
        operators["identity"].copy()
        for operators in local_operators
    ]

    for factor in product.factors:
        mode = factor.mode

        local_factor = (
            _nuclear_factor_operator(
                factor,
                local_operators,
            )
        )

        factors[mode] = (
            factors[mode]
            @ local_factor
        ).tocsr()

    return (
        product.coefficient
        * _kron_all(factors)
    ).tocsr()

def _build_vibrational_space(
    config: GANConfig,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    sparse.csr_matrix,
    sparse.csr_matrix,
    tuple[sparse.csr_matrix, ...],
    tuple[sparse.csr_matrix, ...],
    tuple[dict[str, sparse.csr_matrix], ...],
]:
    """
    Construct the full multimode vibrational space.

    Returns separately:

        H_vib
        T_vib
        V_harm
        I_vib
        number operators
        coordinate operators
    """

    local_operators = [
        _local_vibrational_operators(
            basis_size,
            frequency,
        )
        for basis_size, frequency in zip(
            config.basis_sizes,
            config.frequencies,
        )
    ]

    vibrational_dimension = int(
        reduce(
            mul,
            config.basis_sizes,
            1,
        )
    )

    full_identity = sparse.identity(
        vibrational_dimension,
        format="csr",
    )

    full_kinetic = sparse.csr_matrix(
        (
            vibrational_dimension,
            vibrational_dimension,
        ),
        dtype=complex,
    )

    full_harmonic_potential = sparse.csr_matrix(
        (
            vibrational_dimension,
            vibrational_dimension,
        ),
        dtype=complex,
    )

    number_operators = []
    coordinate_operators = []

    for mode in range(
        config.n_vibrational_modes
    ):
        identities = [
            operators["identity"]
            for operators in local_operators
        ]

        # --------------------------------------------
        # Kinetic energy
        # --------------------------------------------

        kinetic_factors = identities.copy()

        kinetic_factors[mode] = (
            local_operators[mode]["kinetic"]
        )

        full_kinetic += _kron_all(
            kinetic_factors
        )

        # --------------------------------------------
        # Harmonic potential
        # --------------------------------------------

        potential_factors = identities.copy()

        potential_factors[mode] = (
            local_operators[mode][
                "harmonic_potential"
            ]
        )

        full_harmonic_potential += _kron_all(
            potential_factors
        )

        # --------------------------------------------
        # Number operator
        # --------------------------------------------

        number_factors = identities.copy()

        number_factors[mode] = (
            local_operators[mode]["number"]
        )

        number_operators.append(
            _kron_all(
                number_factors
            )
        )

        # --------------------------------------------
        # Coordinate operator
        # --------------------------------------------

        coordinate_factors = identities.copy()

        coordinate_factors[mode] = (
            local_operators[mode]["coordinate"]
        )

        coordinate_operators.append(
            _kron_all(
                coordinate_factors
            )
        )

    # --------------------------------------------------------
    # Base nuclear potential U_0(Q)
    # --------------------------------------------------------

    if config.u0_terms:
        full_base_potential = sparse.csr_matrix(
            (
                vibrational_dimension,
                vibrational_dimension,
            ),
            dtype=complex,
        )

        for term in config.u0_terms:
            full_base_potential += (
                _nuclear_product_operator(
                    term,
                    local_operators,
                )
            )

    else:
        # Backward-compatible behavior:
        # use the original harmonic potential.
        full_base_potential = (
            full_harmonic_potential
        )

    full_hamiltonian = (
        full_kinetic
        + full_base_potential
    )

    return (
        full_hamiltonian.tocsr(),
        full_kinetic.tocsr(),
        full_base_potential.tocsr(),
        full_identity,
        tuple(number_operators),
        tuple(coordinate_operators),
        tuple(local_operators),
    )

def _product_vibrational_state(
    basis_sizes: tuple[int, ...],
    levels: tuple[int, ...],
) -> np.ndarray:
    """Construct |n1> tensor |n2> tensor ..."""

    state = np.asarray(
        [1.0 + 0.0j]
    )

    for basis_size, level in zip(
        basis_sizes,
        levels,
    ):
        local_state = np.zeros(
            basis_size,
            dtype=complex,
        )

        local_state[level] = 1.0

        state = np.kron(
            state,
            local_state,
        )

    return state


def _fixed_particle_basis(
    n_orbitals: int,
    n_electrons: int,
) -> tuple[tuple[int, ...], ...]:
    """
    Build the occupation-number basis with a fixed
    number of fermions.

    Orbital ordering:
        molecular orbitals first,
        metal orbitals second.
    """

    basis = []

    for occupied in combinations(
        range(n_orbitals),
        n_electrons,
    ):
        occupation = [0] * n_orbitals

        for orbital in occupied:
            occupation[orbital] = 1

        basis.append(tuple(occupation))

    return tuple(basis)


def _apply_annihilation(
    occupation: tuple[int, ...],
    orbital: int,
) -> tuple[complex, tuple[int, ...]] | None:
    """Apply a_p to an occupation-number state."""

    if occupation[orbital] == 0:
        return None

    parity = sum(
        occupation[:orbital]
    )

    sign = (
        -1.0 if parity % 2
        else 1.0
    )

    new_occupation = list(
        occupation
    )

    new_occupation[orbital] = 0

    return (
        sign,
        tuple(new_occupation),
    )


def _apply_creation(
    occupation: tuple[int, ...],
    orbital: int,
) -> tuple[complex, tuple[int, ...]] | None:
    """Apply a_p^dagger to an occupation-number state."""

    if occupation[orbital] == 1:
        return None

    parity = sum(
        occupation[:orbital]
    )

    sign = (
        -1.0 if parity % 2
        else 1.0
    )

    new_occupation = list(
        occupation
    )

    new_occupation[orbital] = 1

    return (
        sign,
        tuple(new_occupation),
    )


def _apply_hopping(
    occupation: tuple[int, ...],
    target: int,
    source: int,
) -> tuple[complex, tuple[int, ...]] | None:
    """
    Apply a_target^dagger a_source
    to one Fock state.
    """

    annihilated = _apply_annihilation(
        occupation,
        source,
    )

    if annihilated is None:
        return None

    sign_annihilation, intermediate = (
        annihilated
    )

    created = _apply_creation(
        intermediate,
        target,
    )

    if created is None:
        return None

    sign_creation, final_occupation = (
        created
    )

    return (
        sign_annihilation * sign_creation,
        final_occupation,
    )

def _one_body_operator(
    basis: tuple[tuple[int, ...], ...],
    target: int,
    source: int,
) -> sparse.csr_matrix:
    """
    Construct the fermionic one-body operator

        a_target^dagger a_source

    in the fixed-particle-number Fock basis.
    """

    basis_index = {
        occupation: index
        for index, occupation in enumerate(
            basis
        )
    }

    dimension = len(basis)

    operator = sparse.lil_matrix(
        (
            dimension,
            dimension,
        ),
        dtype=complex,
    )

    for column, occupation in enumerate(
        basis
    ):
        result = _apply_hopping(
            occupation,
            target=target,
            source=source,
        )

        if result is None:
            continue

        sign, final_occupation = result

        row = basis_index[
            final_occupation
        ]

        operator[
            row,
            column,
        ] += sign

    return operator.tocsr()


def _build_electronic_space(
    config: GANConfig,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    tuple[sparse.csr_matrix, ...],
    tuple[tuple[int, ...], ...],
]:
    """Construct the fixed-N fermionic electronic space."""

    n_orbitals = config.n_orbitals

    basis = _fixed_particle_basis(
        n_orbitals,
        config.n_electrons,
    )

    basis_index = {
        occupation: index
        for index, occupation in enumerate(
            basis
        )
    }

    electronic_dimension = len(
        basis
    )

    identity = sparse.identity(
        electronic_dimension,
        dtype=complex,
        format="csr",
    )

    orbital_energies = np.concatenate(
        (
            config.molecular_energies,
            config.metal_energies,
        )
    )

    hamiltonian = sparse.lil_matrix(
        (
            electronic_dimension,
            electronic_dimension,
        ),
        dtype=complex,
    )

    # One-particle energies:
    # sum_p epsilon_p n_p
    for column, occupation in enumerate(
        basis
    ):
        hamiltonian[
            column,
            column,
        ] = np.dot(
            orbital_energies,
            np.asarray(
                occupation,
                dtype=float,
            ),
        )

    # Molecule-metal hopping:
    #
    # sum_i,k V_ik d_i^dagger c_k
    #       + V_ik* c_k^dagger d_i

    metal_start = (
        config.n_molecular_orbitals
    )

    for molecular_orbital in range(
        config.n_molecular_orbitals
    ):
        for metal_orbital in range(
            config.n_metal_orbitals
        ):
            global_metal_orbital = (
                metal_start
                + metal_orbital
            )

            coupling = (
                config.molecule_metal_couplings[
                    molecular_orbital,
                    metal_orbital,
                ]
            )

            for column, occupation in enumerate(
                basis
            ):
                # d_i^dagger c_k
                result = _apply_hopping(
                    occupation,
                    target=molecular_orbital,
                    source=global_metal_orbital,
                )

                if result is not None:
                    sign, final_occupation = (
                        result
                    )

                    row = basis_index[
                        final_occupation
                    ]

                    hamiltonian[
                        row,
                        column,
                    ] += coupling * sign

                # c_k^dagger d_i
                result = _apply_hopping(
                    occupation,
                    target=global_metal_orbital,
                    source=molecular_orbital,
                )

                if result is not None:
                    sign, final_occupation = (
                        result
                    )

                    row = basis_index[
                        final_occupation
                    ]

                    hamiltonian[
                        row,
                        column,
                    ] += (
                        np.conjugate(coupling)
                        * sign
                    )

    number_operators = tuple(
        sparse.diags(
            [
                occupation[orbital]
                for occupation in basis
            ],
            offsets=0,
            dtype=complex,
            format="csr",
        )
        for orbital in range(
            n_orbitals
        )
    )

    return (
        hamiltonian.tocsr(),
        identity,
        number_operators,
        basis,
    )


def build_exact_hamiltonian(
    config: GANConfig,
) -> ExactSystem:
    """
    Construct the exact Hamiltonian in a
    fixed-particle-number fermionic sector.

    Electronic orbital ordering:
        molecular orbitals first,
        metal orbitals second.
    """

    (
        electronic_hamiltonian,
        electronic_identity,
        electronic_number_operators,
        electronic_basis,
    ) = _build_electronic_space(
        config
    )

    electronic_dimension = (
        electronic_hamiltonian.shape[0]
    )

    (
        vibrational_hamiltonian,
        vibrational_kinetic,
        vibrational_base_potential,
        vibrational_identity,
        vibrational_number_operators,
        vibrational_coordinate_operators,
        local_vibrational_operators,
    ) = _build_vibrational_space(
        config
    )

    vibrational_dimension = (
        vibrational_hamiltonian.shape[0]
    )

    total_hamiltonian = sparse.kron(
        electronic_hamiltonian,
        vibrational_identity,
        format="csr",
    )

    total_hamiltonian += sparse.kron(
        electronic_identity,
        vibrational_hamiltonian,
        format="csr",
    )

    # Electron-vibration coupling:
    #
    # sum_i,nu g_i,nu n_i q_nu
    for molecular_orbital in range(
        config.n_molecular_orbitals
    ):
        occupation_operator = (
            electronic_number_operators[
                molecular_orbital
            ]
        )

        for mode in range(
            config.n_vibrational_modes
        ):
            coupling = (
                config.electron_vibration_couplings[
                    molecular_orbital,
                    mode,
                ]
            )

            total_hamiltonian += (
                coupling
                * sparse.kron(
                    occupation_operator,
                    vibrational_coordinate_operators[
                        mode
                    ],
                    format="csr",
                )
            )

    # --------------------------------------------------------
    # General molecular one-particle terms:
    #
    # U_ij(Q) a_i^dagger a_j
    #
    # For i != j we represent one independent
    # Hermitian pair:
    #
    # U_ij(Q) (
    #     a_i^dagger a_j
    #     + a_j^dagger a_i
    # )
    # --------------------------------------------------------

    for term in config.uij_terms:

        electronic_operator = (
            _one_body_operator(
                electronic_basis,
                target=term.i,
                source=term.j,
            )
        )

        if term.i != term.j:
            electronic_operator = (
                electronic_operator
                + electronic_operator.getH()
            )

        nuclear_operator = (
            _nuclear_product_operator(
                term.nuclear,
                local_vibrational_operators,
            )
        )

        total_hamiltonian += sparse.kron(
            electronic_operator,
            nuclear_operator,
            format="csr",
        )

    # --------------------------------------------------------
    # Molecular electron-electron interaction:
    #
    # V_ij(Q) n_i n_j
    # --------------------------------------------------------

    for term in config.vij_terms:

        occupation_i = (
            electronic_number_operators[
                term.i
            ]
        )

        occupation_j = (
            electronic_number_operators[
                term.j
            ]
        )

        electronic_operator = (
            occupation_i
            @ occupation_j
        ).tocsr()

        nuclear_operator = (
            _nuclear_product_operator(
                term.nuclear,
                local_vibrational_operators,
            )
        )

        total_hamiltonian += sparse.kron(
            electronic_operator,
            nuclear_operator,
            format="csr",
        )

    # --------------------------------------------------------
    # General molecule-metal coupling:
    #
    # W_ik(Q) (
    #     d_i^dagger c_k
    #     + c_k^dagger d_i
    # )
    # --------------------------------------------------------

    metal_start = (
        config.n_molecular_orbitals
    )

    for term in config.wik_terms:

        molecular_orbital = term.i

        global_metal_orbital = (
            metal_start
            + term.k
        )

        hopping_operator = (
            _one_body_operator(
                electronic_basis,
                target=molecular_orbital,
                source=global_metal_orbital,
            )
        )

        electronic_operator = (
            hopping_operator
            + hopping_operator.getH()
        )

        nuclear_operator = (
            _nuclear_product_operator(
                term.nuclear,
                local_vibrational_operators,
            )
        )

        total_hamiltonian += sparse.kron(
            electronic_operator,
            nuclear_operator,
            format="csr",
        )

    # Electronic initial state.
    initial_occupation = (
        [0] * config.n_orbitals
    )

    for orbital in (
        config.occupied_orbitals
    ):
        initial_occupation[
            orbital
        ] = 1

    initial_occupation = tuple(
        initial_occupation
    )

    basis_index = {
        occupation: index
        for index, occupation in enumerate(
            electronic_basis
        )
    }

    initial_electronic_state = np.zeros(
        electronic_dimension,
        dtype=complex,
    )

    initial_electronic_state[
        basis_index[
            initial_occupation
        ]
    ] = 1.0

    initial_vibrational_state = (
        _product_vibrational_state(
            config.basis_sizes,
            config.vibrational_levels,
        )
    )

    initial_state = np.kron(
        initial_electronic_state,
        initial_vibrational_state,
    )

    hermiticity_error = sparse.linalg.norm(
        total_hamiltonian
        - total_hamiltonian.getH()
    )

    if hermiticity_error > 1e-12:
        raise ValueError(
            "The Hamiltonian is not Hermitian. "
            f"Error = {hermiticity_error:.3e}"
        )

    initial_norm = np.linalg.norm(
        initial_state
    )

    if not np.isclose(
        initial_norm,
        1.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The initial state is not normalized."
        )

    return ExactSystem(
        hamiltonian=(
            total_hamiltonian.tocsr()
        ),
        initial_state=initial_state,
        electronic_number_operators=(
            electronic_number_operators
        ),
        vibrational_number_operators=(
            vibrational_number_operators
        ),
        vibrational_coordinate_operators=(
            vibrational_coordinate_operators
        ),
        electronic_basis=electronic_basis,
        electronic_dimension=(
            electronic_dimension
        ),
        vibrational_dimension=(
            vibrational_dimension
        ),
                n_molecular_orbitals=(
            config.n_molecular_orbitals
        ),
        n_metal_orbitals=(
            config.n_metal_orbitals
        ),
        vibrational_basis_sizes=tuple(
            config.basis_sizes
        ),
    )