from __future__ import annotations

import numpy as np
import pennylane as qml


N_ELECTRONIC_QUBITS = 6
N_NUCLEAR_QUBITS = 3
N_QUBITS = (
    N_ELECTRONIC_QUBITS
    + N_NUCLEAR_QUBITS
)
NUCLEAR_WIRES = (6, 7, 8)

def _layout_from_nuclear_dimension(
    nuclear_dimension: int,
) -> tuple[int, int, tuple[int, ...]]:
    nuclear_dimension = int(nuclear_dimension)

    if (
        nuclear_dimension <= 0
        or nuclear_dimension
        & (nuclear_dimension - 1)
    ):
        raise ValueError(
            "nuclear_dimension must be a positive power of two"
        )

    n_nuclear_qubits = (
        nuclear_dimension.bit_length() - 1
    )

    n_qubits = (
        N_ELECTRONIC_QUBITS
        + n_nuclear_qubits
    )

    nuclear_wires = tuple(
        range(
            N_ELECTRONIC_QUBITS,
            n_qubits,
        )
    )

    return (
        n_nuclear_qubits,
        n_qubits,
        nuclear_wires,
    )


def _layout_from_model(model):
    return _layout_from_nuclear_dimension(
        model.nuclear_dimension
    )

def apply_f0_circuit(
    model,
    dt: float,
) -> None:
    """
    Apply the diagonal GAN fragment

        F0 =
            a Q^2
            + sum_i epsilon_i n_i
            + sum_i lambda_i Q n_i.

    This is the circuit-level analogue of

        sum_lambda f_lambda(Q)c_lambda(n).

    For the small toy model, the functions of Q are
    represented exactly by diagonal lookup unitaries.
    """

    _, _, nuclear_wires = (
        _layout_from_model(model)
    )

    points = np.asarray(
        model.nuclear_grid.points,
        dtype=float,
    )

    # -------------------------------------------------
    # a Q^2
    # -------------------------------------------------

    base_phases = np.exp(
        -1.0j
        * dt
        * model.u0_quadratic
        * points**2
    )

    qml.DiagonalQubitUnitary(
        base_phases,
        wires=nuclear_wires,
    )

    # -------------------------------------------------
    # epsilon_i n_i
    # -------------------------------------------------

    for orbital, energy in (
        model.molecular_energies.items()
    ):
        # PhaseShift(phi) = diag(1, exp(i phi)).
        # Therefore phi = -epsilon_i dt gives
        # exp(-i epsilon_i n_i dt).
        qml.PhaseShift(
            -energy * dt,
            wires=orbital,
        )

    # -------------------------------------------------
    # lambda_i Q n_i
    # -------------------------------------------------

    for orbital, coupling in (
        model.linear_couplings.items()
    ):
        nuclear_phases = np.exp(
            -1.0j
            * dt
            * coupling
            * points
        )

        nuclear_unitary = np.diag(
            nuclear_phases
        )

        # Apply the Q-dependent phase only when
        # electronic orbital i is occupied.
        qml.ctrl(
            qml.QubitUnitary(
                nuclear_unitary,
                wires=nuclear_wires,
            ),
            control=[orbital],
        )


def apply_f0_to_state(
    initial_state: np.ndarray,
    model,
    dt: float,
) -> np.ndarray:
    """
    Apply the F0 circuit to an arbitrary 9-qubit state.
    """

    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(N_QUBITS),
        )

        apply_f0_circuit(
            model,
            dt,
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def prepare_toy_state_pennylane(
    electronic_bits: list[int],
    nuclear_state: np.ndarray,
) -> np.ndarray:
    """
    Prepare

        |n0 ... n5> tensor |phi_nuc>

    on 9 qubits.

    Wires:
        0,...,5 : electronic
        6,7,8   : nuclear
    """

    if len(electronic_bits) != N_ELECTRONIC_QUBITS:
        raise ValueError(
            "electronic_bits must contain 6 entries"
        )

    nuclear_state = np.asarray(
        nuclear_state,
        dtype=complex,
    )

    if nuclear_state.shape != (2**N_NUCLEAR_QUBITS,):
        raise ValueError(
            "nuclear_state must have dimension 8"
        )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.BasisState(
            np.asarray(
                electronic_bits,
                dtype=int,
            ),
            wires=range(N_ELECTRONIC_QUBITS),
        )

        qml.StatePrep(
            nuclear_state,
            wires=range(
                N_ELECTRONIC_QUBITS,
                N_QUBITS,
            ),
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def apply_hopping_clifford(
    i: int,
    j: int,
) -> None:
    """
    Apply the GAN hopping diagonalization Clifford

        U_ij = exp[-(pi/4) gamma_{2j+1} gamma_{2i+1}]

    using its Jordan-Wigner Pauli-string form.

    For i < j:

        U_ij =
        exp[+i(pi/4) X_i Z_{i+1} ... Z_{j-1} Y_j].
    """
    if i == j:
        raise ValueError("i and j must be different")

    if i > j:
        i, j = j, i

    wires = list(range(i, j + 1))

    pauli_word = (
        "X"
        + "Z" * (j - i - 1)
        + "Y"
    )

    # PennyLane PauliRot(phi, P) implements
    #
    #     exp[-i phi P / 2]
    #
    # so phi = -pi/2 gives exp[+i pi P / 4].
    qml.PauliRot(
        -np.pi / 2.0,
        pauli_word,
        wires=wires,
    )


def apply_hopping_clifford_to_state(
    initial_state: np.ndarray,
    i: int,
    j: int,
) -> np.ndarray:
    """
    Apply U_ij to an arbitrary 9-qubit state
    using PennyLane.
    """
    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    if initial_state.shape != (2**N_QUBITS,):
        raise ValueError(
            "initial_state must have dimension 512"
        )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(N_QUBITS),
        )

        apply_hopping_clifford(
            i,
            j,
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def apply_matching_clifford(
    matching: tuple[tuple[int, int], ...],
) -> None:
    """
    Apply

        U_s = product_(i,j in matching) U_ij

    for one GAN matching.
    """
    for i, j in matching:
        apply_hopping_clifford(
            i,
            j,
        )


def apply_matching_clifford_to_state(
    initial_state: np.ndarray,
    matching: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """
    Apply an entire GAN matching Clifford to
    an arbitrary 9-qubit state.
    """
    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    if initial_state.shape != (2**N_QUBITS,):
        raise ValueError(
            "initial_state must have dimension 512"
        )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(N_QUBITS),
        )

        apply_matching_clifford(
            matching,
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def apply_diagonal_hopping_phase(
    matching: tuple[tuple[int, int], ...],
    profiles: dict[
        tuple[int, int],
        np.ndarray,
    ],
    dt: float,
) -> None:
    """
    Apply

        exp[-i D_s(Q) dt]

    with

        D_s(Q) =
            sum_(i,j)
            g_ij(Q) (Z_i - Z_j) / 2.

    Each edge is diagonal jointly in the
    electronic Z basis and nuclear Q basis.
    """
    for i, j in matching:
        edge = (
            min(i, j),
            max(i, j),
        )

        if edge not in profiles:
            raise KeyError(
                f"No hopping profile supplied "
                f"for edge {edge}"
            )

        profile = np.asarray(
            profiles[edge],
            dtype=float,
        )

        nuclear_dimension = profile.size

        _, _, nuclear_wires = (
            _layout_from_nuclear_dimension(
                nuclear_dimension
            )
        )

        if profile.shape != (
            nuclear_dimension,
        ):
            raise ValueError(
                f"Invalid hopping profile shape "
                f"for edge {edge}"
            )

        phases = np.empty(
            (
                2,
                2,
                nuclear_dimension,
            ),
            dtype=complex,
        )

        phases = np.empty(
            (
                2,
                2,
                nuclear_dimension,
            ),
            dtype=complex,
        )

        for bit_i in (0, 1):
            z_i = 1.0 - 2.0 * bit_i

            for bit_j in (0, 1):
                z_j = 1.0 - 2.0 * bit_j

                energy = (
                    0.5
                    * (z_i - z_j)
                    * profile
                )

                phases[
                    bit_i,
                    bit_j,
                    :,
                ] = np.exp(
                    -1.0j
                    * dt
                    * energy
                )

        qml.DiagonalQubitUnitary(
            phases.reshape(-1),
            wires=[
                i,
                j,
                *nuclear_wires,
            ],
        )


def apply_hopping_fragment_circuit(
    matching: tuple[tuple[int, int], ...],
    profiles: dict[tuple[int, int], np.ndarray],
    dt: float,
) -> None:
    """
    Apply the complete GAN hopping propagator

        exp(-i F_s dt)
        =
        U_s exp(-i D_s dt) U_s^dagger.

    Circuit action on the state is therefore:

        U_s^dagger -> exp(-i D_s dt) -> U_s.
    """

    # First transform into the basis where F_s is diagonal.
    qml.adjoint(
        apply_matching_clifford
    )(matching)

    # Apply the diagonal evolution.
    apply_diagonal_hopping_phase(
        matching,
        profiles,
        dt,
    )

    # Transform back.
    apply_matching_clifford(
        matching,
    )


def apply_hopping_fragment_to_state(
    initial_state: np.ndarray,
    matching: tuple[tuple[int, int], ...],
    profiles: dict[tuple[int, int], np.ndarray],
    dt: float,
) -> np.ndarray:
    """
    Apply one complete hopping-fragment propagator
    to an arbitrary 9-qubit state using PennyLane.
    """
    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(N_QUBITS),
        )

        apply_hopping_fragment_circuit(
            matching,
            profiles,
            dt,
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def apply_all_hopping_fragments_circuit(
    matchings: tuple[
        tuple[tuple[int, int], ...],
        ...
    ],
    profiles: dict[tuple[int, int], np.ndarray],
    dt: float,
) -> None:
    """
    Apply all GAN hopping fragments sequentially:

        exp(-i F_1 dt)
        exp(-i F_2 dt)
        ...
        exp(-i F_S dt)

    in the same order used by the first-order
    Trotter implementation.
    """
    for matching in matchings:
        apply_hopping_fragment_circuit(
            matching,
            profiles,
            dt,
        )


def apply_all_hopping_fragments_to_state(
    initial_state: np.ndarray,
    matchings: tuple[
        tuple[tuple[int, int], ...],
        ...
    ],
    profiles: dict[tuple[int, int], np.ndarray],
    dt: float,
) -> np.ndarray:
    """
    Apply all hopping fragments to an arbitrary
    9-qubit state using one PennyLane circuit.
    """
    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(N_QUBITS),
        )

        apply_all_hopping_fragments_circuit(
            matchings,
            profiles,
            dt,
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def apply_final_fragment_circuit(
    model,
    dt: float,
) -> None:
    """
    Apply

        exp[-i dt (
            T_nuc
            + sum_k epsilon_k n_k
        )]

    as an explicit quantum circuit.

    Metal energies are diagonal in occupation basis.

    Nuclear kinetic energy is diagonal in momentum
    basis, reached with the QFT.
    """

    # -------------------------------------------------
    # Metal orbital energies
    # -------------------------------------------------

    _, _, nuclear_wires = (
        _layout_from_model(model)
    )

    for orbital, energy in (
        model.metal_energies.items()
    ):
        qml.PhaseShift(
            -energy * dt,
            wires=orbital,
        )

    # -------------------------------------------------
    # Nuclear kinetic energy
    #
    # Position -> momentum
    # -------------------------------------------------

    qml.QFT(
        wires=nuclear_wires,
    )

    # Kinetic energies in the momentum basis.
    fourier = model.nuclear_grid.fourier

    kinetic_energies = np.diag(
        fourier
        @ model.nuclear_grid.kinetic
        @ fourier.conj().T
    ).real

    kinetic_phases = np.exp(
        -1.0j
        * dt
        * kinetic_energies
    )

    qml.DiagonalQubitUnitary(
        kinetic_phases,
        wires=nuclear_wires,
    )

    # Momentum -> position
    qml.adjoint(qml.QFT)(
        wires=nuclear_wires,
    )


def apply_final_fragment_pennylane_to_state(
    initial_state: np.ndarray,
    model,
    dt: float,
) -> np.ndarray:
    """
    Apply the final GAN fragment to an arbitrary
    9-qubit state using PennyLane.
    """

    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(N_QUBITS),
        )

        apply_final_fragment_circuit(
            model,
            dt,
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def apply_gan_trotter_step_circuit(
    model,
    dt: float,
) -> None:
    """
    Apply one complete first-order GAN Trotter step:

        F0
        -> hopping fragments
        -> F_last

    entirely as PennyLane circuit operations.
    """

    # 1. Diagonal electronic-nuclear fragment
    apply_f0_circuit(
        model,
        dt,
    )

    # 2. All hopping matchings
    apply_all_hopping_fragments_circuit(
        model.matchings,
        model.hopping_profiles,
        dt,
    )

    # 3. Metal energies + nuclear kinetic energy
    apply_final_fragment_circuit(
        model,
        dt,
    )


def apply_gan_trotter_step_to_state(
    initial_state: np.ndarray,
    model,
    dt: float,
) -> np.ndarray:
    """
    Apply one complete GAN Trotter step to an
    arbitrary 9-qubit state.
    """

    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    if initial_state.shape != (2**N_QUBITS,):
        raise ValueError(
            "initial_state must have dimension 512"
        )

    device = qml.device(
        "default.qubit",
        wires=N_QUBITS,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(N_QUBITS),
        )

        apply_gan_trotter_step_circuit(
            model,
            dt,
        )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )

def gan_trotter_final_state(
    initial_state: np.ndarray,
    model,
    total_time: float,
    n_steps: int,
) -> np.ndarray:
    """
    Apply n_steps complete first-order GAN Trotter
    steps in a single PennyLane circuit.
    """

    if n_steps <= 0:
        raise ValueError(
            "n_steps must be positive"
        )

    initial_state = np.asarray(
        initial_state,
        dtype=complex,
    )

    _, n_qubits, _ = (
        _layout_from_model(model)
    )

    expected_dimension = 2**n_qubits

    if initial_state.shape != (
        expected_dimension,
    ):
        raise ValueError(
            "initial_state has dimension "
            f"{initial_state.size}, expected "
            f"{expected_dimension}"
        )

    dt = total_time / n_steps

    device = qml.device(
        "default.qubit",
        wires=n_qubits,
    )

    @qml.qnode(device)
    def circuit():
        qml.StatePrep(
            initial_state,
            wires=range(n_qubits),
        )

        for _ in range(n_steps):
            apply_gan_trotter_step_circuit(
                model,
                dt,
            )

        return qml.state()

    return np.asarray(
        circuit(),
        dtype=complex,
    )