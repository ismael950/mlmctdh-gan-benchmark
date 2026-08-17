from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


# ============================================================
# General nuclear-dependent GAN terms
# ============================================================


@dataclass(frozen=True)
class NuclearFactorSpec:
    """
    One function acting on one nuclear coordinate.

    Examples:
        Q
        Q^2
        Morse(Q)
        exp(-a Q)
        tanh(a Q)
    """

    mode: int
    kind: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class NuclearProductSpec:
    """
    One nuclear product term:

        coefficient * f_1(Q_i) * f_2(Q_j) * ...

    An empty tuple of factors represents a constant.
    """

    coefficient: float
    factors: tuple[NuclearFactorSpec, ...]


# ============================================================
# Physical nuclear-coordinate representation
# ============================================================


@dataclass(frozen=True)
class NuclearBasisSpec:
    """
    Primitive representation of one physical nuclear coordinate.

    Example:
        sine-DVR for a coordinate expressed in Angstrom.
    """

    kind: str
    size: int
    minimum: float
    maximum: float
    length_unit: str = "angstrom"


@dataclass(frozen=True)
class NuclearInitialStateSpec:
    """
    Initial state associated with one physical nuclear coordinate.

    Examples:
        neutral_pes_eigenstate
        gaussian
    """

    kind: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class NuclearCoordinateSpec:
    """
    One physical nuclear degree of freedom.

    Examples for NO/Au:

        r : reactive N-O stretch
        z : translational molecule-surface coordinate
    """

    name: str
    mode_type: str
    mass_amu: float
    basis: NuclearBasisSpec
    initial_state: NuclearInitialStateSpec


# ============================================================
# General GAN electronic terms
# ============================================================


@dataclass(frozen=True)
class UijTermSpec:
    """
    One Hermitian molecular one-particle contribution.

    If i == j:
        F(Q) n_i

    If i != j:
        F(Q) (
            a_i^dagger a_j
            + a_j^dagger a_i
        )
    """

    i: int
    j: int
    nuclear: NuclearProductSpec


@dataclass(frozen=True)
class VijTermSpec:
    """
    One molecular interaction term:

        F(Q) n_i n_j
    """

    i: int
    j: int
    nuclear: NuclearProductSpec


@dataclass(frozen=True)
class WikTermSpec:
    """
    One molecule-metal coupling term:

        F(Q) (
            d_i^dagger c_k
            + c_k^dagger d_i
        )
    """

    i: int
    k: int
    nuclear: NuclearProductSpec


# ============================================================
# Main configuration
# ============================================================


@dataclass(frozen=True)
class GANConfig:
    """Parameters defining one GAN model."""

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    n_molecular_orbitals: int
    n_metal_orbitals: int
    n_electrons: int

    # Keep this name for backward compatibility.
    #
    # In physical models it means the total number of
    # nuclear coordinates, not necessarily harmonic modes.
    n_vibrational_modes: int

    # --------------------------------------------------------
    # Electronic parameters
    # --------------------------------------------------------

    molecular_energies: np.ndarray
    metal_energies: np.ndarray
    molecule_metal_couplings: np.ndarray

    # --------------------------------------------------------
    # Legacy vibrational parameters
    # --------------------------------------------------------
    #
    # Toy models use harmonic-oscillator primitive bases.
    #
    # Physical nuclear-coordinate models do NOT require
    # harmonic frequencies. For those models:
    #
    #     frequencies = empty array
    #
    # while basis_sizes and mode_types are obtained from
    # nuclear_coordinates.

    frequencies: np.ndarray

    basis_sizes: tuple[
        int, ...
    ]

    electron_vibration_couplings: np.ndarray

    mode_types: tuple[
        str, ...
    ]

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    occupied_orbitals: tuple[
        int, ...
    ]

    # Used only by the legacy harmonic representation.
    #
    # Physical nuclear initial states are instead stored
    # inside NuclearCoordinateSpec.
    vibrational_levels: tuple[
        int, ...
    ]

    # --------------------------------------------------------
    # Propagation
    # --------------------------------------------------------

    t_final: float
    n_times: int

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    save_states: bool = False

    # --------------------------------------------------------
    # General GAN nuclear-dependent terms
    # --------------------------------------------------------

    u0_terms: tuple[
        NuclearProductSpec, ...
    ] = field(
        default_factory=tuple
    )

    uij_terms: tuple[
        UijTermSpec, ...
    ] = field(
        default_factory=tuple
    )

    vij_terms: tuple[
        VijTermSpec, ...
    ] = field(
        default_factory=tuple
    )

    wik_terms: tuple[
        WikTermSpec, ...
    ] = field(
        default_factory=tuple
    )

    # --------------------------------------------------------
    # Physical nuclear coordinates
    # --------------------------------------------------------
    #
    # Empty for old toy-model configurations.

    nuclear_coordinates: tuple[
        NuclearCoordinateSpec, ...
    ] = field(
        default_factory=tuple
    )

    @property
    def n_orbitals(self) -> int:
        """Total number of electronic orbitals."""

        return (
            self.n_molecular_orbitals
            + self.n_metal_orbitals
        )

    @property
    def uses_physical_nuclear_coordinates(
        self,
    ) -> bool:
        """
        True when the configuration uses explicit
        physical nuclear coordinates instead of the
        legacy harmonic-oscillator representation.
        """

        return bool(
            self.nuclear_coordinates
        )


# ============================================================
# Parsing helpers
# ============================================================


def _parse_nuclear_factor(
    raw: dict,
) -> NuclearFactorSpec:
    """
    Parse one nuclear factor f(Q_mode)
    from the YAML representation.
    """

    return NuclearFactorSpec(
        mode=int(
            raw["mode"]
        ),
        kind=str(
            raw["kind"]
        ).lower(),
        parameters={
            str(key): float(value)
            for key, value in raw.get(
                "parameters",
                {},
            ).items()
        },
    )


def _parse_nuclear_product(
    raw: dict,
) -> NuclearProductSpec:
    """
    Parse

        coefficient
        * product_nu f_nu(Q_nu)

    from the YAML representation.
    """

    return NuclearProductSpec(
        coefficient=float(
            raw.get(
                "coefficient",
                1.0,
            )
        ),
        factors=tuple(
            _parse_nuclear_factor(
                factor
            )
            for factor in raw.get(
                "factors",
                [],
            )
        ),
    )


def _parse_nuclear_basis(
    raw: dict,
) -> NuclearBasisSpec:
    """
    Parse the primitive basis of one
    physical nuclear coordinate.
    """

    return NuclearBasisSpec(
        kind=str(
            raw["type"]
        ).lower(),

        size=int(
            raw["size"]
        ),

        minimum=float(
            raw["minimum"]
        ),

        maximum=float(
            raw["maximum"]
        ),

        length_unit=str(
            raw.get(
                "length_unit",
                "angstrom",
            )
        ).lower(),
    )


def _parse_nuclear_initial_state(
    raw: dict,
) -> NuclearInitialStateSpec:
    """
    Parse the initial state of one
    physical nuclear coordinate.
    """

    kind = str(
        raw["type"]
    ).lower()

    parameters = {
        str(key): float(value)
        for key, value in raw.items()
        if key != "type"
    }

    return NuclearInitialStateSpec(
        kind=kind,
        parameters=parameters,
    )


def _parse_nuclear_coordinate(
    raw: dict,
) -> NuclearCoordinateSpec:
    """
    Parse one physical nuclear coordinate.
    """

    return NuclearCoordinateSpec(
        name=str(
            raw["name"]
        ),

        mode_type=str(
            raw["mode_type"]
        ).lower(),

        mass_amu=float(
            raw["mass_amu"]
        ),

        basis=_parse_nuclear_basis(
            raw["basis"]
        ),

        initial_state=(
            _parse_nuclear_initial_state(
                raw["initial_state"]
            )
        ),
    )


# ============================================================
# Configuration loading
# ============================================================


def load_config(
    path: str | Path,
) -> GANConfig:
    """
    Read and validate a YAML configuration file.

    Two nuclear representations are supported:

    1. Legacy:
           vibrational:

    2. Physical:
           nuclear:
               coordinates:

    Exactly one must be present.
    """

    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        raw = yaml.safe_load(
            file
        )

    # --------------------------------------------------------
    # Required top-level sections
    # --------------------------------------------------------

    model = raw[
        "model"
    ]

    electronic = raw[
        "electronic"
    ]

    initial_state = raw[
        "initial_state"
    ]

    propagation = raw[
        "propagation"
    ]

    # --------------------------------------------------------
    # Optional top-level sections
    # --------------------------------------------------------

    vibrational = raw.get(
        "vibrational",
        None,
    )

    output = raw.get(
        "output",
        {},
    )

    gan_terms = raw.get(
        "gan_terms",
        {},
    )

    nuclear = raw.get(
        "nuclear",
        {},
    )

    nuclear_coordinates_raw = nuclear.get(
        "coordinates",
        [],
    )

    uses_physical_nuclear_coordinates = bool(
        nuclear_coordinates_raw
    )

    # --------------------------------------------------------
    # Choose exactly one nuclear representation
    # --------------------------------------------------------

    if (
        vibrational is None
        and not uses_physical_nuclear_coordinates
    ):
        raise ValueError(
            "The configuration must define either "
            "'vibrational' or 'nuclear.coordinates'."
        )

    if (
        vibrational is not None
        and uses_physical_nuclear_coordinates
    ):
        raise ValueError(
            "Use either the legacy 'vibrational' section "
            "or the physical 'nuclear.coordinates' section, "
            "not both."
        )

    n_vibrational_modes = int(
        model[
            "n_vibrational_modes"
        ]
    )

    # ========================================================
    # Physical nuclear-coordinate representation
    # ========================================================

    if uses_physical_nuclear_coordinates:

        parsed_nuclear_coordinates = tuple(
            _parse_nuclear_coordinate(
                coordinate
            )
            for coordinate
            in nuclear_coordinates_raw
        )

        # Physical coordinates do not require fictitious
        # harmonic frequencies.
        frequencies = np.asarray(
            [],
            dtype=float,
        )

        basis_sizes = tuple(
            coordinate.basis.size
            for coordinate
            in parsed_nuclear_coordinates
        )

        mode_types = tuple(
            coordinate.mode_type
            for coordinate
            in parsed_nuclear_coordinates
        )

        # Kept only for backward compatibility with
        # the current GANConfig interface.
        #
        # General Q-dependent terms are represented
        # explicitly through gan_terms.
        electron_vibration_couplings = np.zeros(
            (
                int(
                    model[
                        "n_molecular_orbitals"
                    ]
                ),
                n_vibrational_modes,
            ),
            dtype=float,
        )

        # Physical initial states are stored directly
        # in each NuclearCoordinateSpec.
        vibrational_levels = tuple()

    # ========================================================
    # Legacy harmonic representation
    # ========================================================

    else:

        parsed_nuclear_coordinates = tuple()

        frequencies = np.asarray(
            vibrational[
                "frequencies"
            ],
            dtype=float,
        )

        basis_sizes = tuple(
            int(value)
            for value in vibrational[
                "basis_sizes"
            ]
        )

        electron_vibration_couplings = np.asarray(
            vibrational[
                "electron_vibration_couplings"
            ],
            dtype=float,
        )

        mode_types = tuple(
            str(
                value
            ).lower()
            for value in vibrational.get(
                "mode_types",
                [
                    "unspecified"
                    for _ in range(
                        n_vibrational_modes
                    )
                ],
            )
        )

        vibrational_levels = tuple(
            int(value)
            for value in initial_state[
                "vibrational_levels"
            ]
        )

    # ========================================================
    # Construct GANConfig
    # ========================================================

    config = GANConfig(

        # ----------------------------------------------------
        # Model size
        # ----------------------------------------------------

        n_molecular_orbitals=int(
            model[
                "n_molecular_orbitals"
            ]
        ),

        n_metal_orbitals=int(
            model[
                "n_metal_orbitals"
            ]
        ),

        n_electrons=int(
            model[
                "n_electrons"
            ]
        ),

        n_vibrational_modes=(
            n_vibrational_modes
        ),

        # ----------------------------------------------------
        # Electronic parameters
        # ----------------------------------------------------

        molecular_energies=np.asarray(
            electronic[
                "molecular_energies"
            ],
            dtype=float,
        ),

        metal_energies=np.asarray(
            electronic[
                "metal_energies"
            ],
            dtype=float,
        ),

        molecule_metal_couplings=np.asarray(
            electronic[
                "molecule_metal_couplings"
            ],
            dtype=float,
        ),

        # ----------------------------------------------------
        # Nuclear / vibrational representation
        # ----------------------------------------------------

        frequencies=(
            frequencies
        ),

        basis_sizes=(
            basis_sizes
        ),

        electron_vibration_couplings=(
            electron_vibration_couplings
        ),

        mode_types=(
            mode_types
        ),

        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        occupied_orbitals=tuple(
            int(value)
            for value in initial_state[
                "occupied_orbitals"
            ]
        ),

        vibrational_levels=(
            vibrational_levels
        ),

        # ----------------------------------------------------
        # Propagation
        # ----------------------------------------------------

        t_final=float(
            propagation[
                "t_final"
            ]
        ),

        n_times=int(
            propagation[
                "n_times"
            ]
        ),

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        save_states=bool(
            output.get(
                "save_states",
                False,
            )
        ),

        # ----------------------------------------------------
        # U_0(Q)
        # ----------------------------------------------------

        u0_terms=tuple(
            _parse_nuclear_product(
                term
            )
            for term in gan_terms.get(
                "u0",
                [],
            )
        ),

        # ----------------------------------------------------
        # U_ij(Q) a_i^dagger a_j
        # ----------------------------------------------------

        uij_terms=tuple(
            UijTermSpec(
                i=int(
                    term[
                        "i"
                    ]
                ),

                j=int(
                    term[
                        "j"
                    ]
                ),

                nuclear=(
                    _parse_nuclear_product(
                        term
                    )
                ),
            )

            for term in gan_terms.get(
                "uij",
                [],
            )
        ),

        # ----------------------------------------------------
        # V_ij(Q) n_i n_j
        # ----------------------------------------------------

        vij_terms=tuple(
            VijTermSpec(
                i=int(
                    term[
                        "i"
                    ]
                ),

                j=int(
                    term[
                        "j"
                    ]
                ),

                nuclear=(
                    _parse_nuclear_product(
                        term
                    )
                ),
            )

            for term in gan_terms.get(
                "vij",
                [],
            )
        ),

        # ----------------------------------------------------
        # W_ik(Q) molecule-metal coupling
        # ----------------------------------------------------

        wik_terms=tuple(
            WikTermSpec(
                i=int(
                    term[
                        "i"
                    ]
                ),

                k=int(
                    term[
                        "k"
                    ]
                ),

                nuclear=(
                    _parse_nuclear_product(
                        term
                    )
                ),
            )

            for term in gan_terms.get(
                "wik",
                [],
            )
        ),

        # ----------------------------------------------------
        # Physical nuclear coordinates
        # ----------------------------------------------------

        nuclear_coordinates=(
            parsed_nuclear_coordinates
        ),
    )

    validate_config(
        config
    )

    return config


# ============================================================
# Validation
# ============================================================


def validate_config(
    config: GANConfig,
) -> None:
    """
    Check that all parameter dimensions and indices
    are consistent.
    """

    n_mol = (
        config.n_molecular_orbitals
    )

    n_metal = (
        config.n_metal_orbitals
    )

    n_modes = (
        config.n_vibrational_modes
    )

    uses_physical_nuclear_coordinates = (
        config.uses_physical_nuclear_coordinates
    )

    # ========================================================
    # Basic model size
    # ========================================================

    if n_mol < 1:
        raise ValueError(
            "At least one molecular orbital is required."
        )

    if n_metal < 1:
        raise ValueError(
            "At least one metal orbital is required."
        )

    if n_modes < 1:
        raise ValueError(
            "At least one nuclear mode is required."
        )

    if not (
        0
        <= config.n_electrons
        <= config.n_orbitals
    ):
        raise ValueError(
            "n_electrons must be between "
            "0 and n_orbitals."
        )

    # ========================================================
    # Array dimensions common to both representations
    # ========================================================

    expected_shapes = {
        "molecular_energies": (
            n_mol,
        ),

        "metal_energies": (
            n_metal,
        ),

        "molecule_metal_couplings": (
            n_mol,
            n_metal,
        ),

        "electron_vibration_couplings": (
            n_mol,
            n_modes,
        ),
    }

    arrays = {
        "molecular_energies":
            config.molecular_energies,

        "metal_energies":
            config.metal_energies,

        "molecule_metal_couplings":
            config.molecule_metal_couplings,

        "electron_vibration_couplings":
            config.electron_vibration_couplings,
    }

    for (
        name,
        expected_shape,
    ) in expected_shapes.items():

        actual_shape = (
            arrays[
                name
            ].shape
        )

        if (
            actual_shape
            != expected_shape
        ):
            raise ValueError(
                f"{name} must have shape "
                f"{expected_shape}, but has "
                f"shape {actual_shape}."
            )

    # ========================================================
    # Legacy frequencies
    # ========================================================

    if not uses_physical_nuclear_coordinates:

        if (
            config.frequencies.shape
            != (n_modes,)
        ):
            raise ValueError(
                "frequencies must contain one value "
                "for each vibrational mode."
            )

        if np.any(
            config.frequencies
            <= 0
        ):
            raise ValueError(
                "All vibrational frequencies "
                "must be positive."
            )

    else:

        if (
            config.frequencies.size
            != 0
        ):
            raise ValueError(
                "Physical nuclear-coordinate "
                "configurations must not define "
                "legacy vibrational frequencies."
            )

    # ========================================================
    # Primitive basis sizes
    # ========================================================

    if (
        len(
            config.basis_sizes
        )
        != n_modes
    ):
        raise ValueError(
            "basis_sizes must contain one value "
            "for each nuclear mode."
        )

    if any(
        size < 2
        for size in config.basis_sizes
    ):
        raise ValueError(
            "Every nuclear basis must contain "
            "at least two states."
        )

    # ========================================================
    # Nuclear mode types
    # ========================================================

    if (
        len(
            config.mode_types
        )
        != n_modes
    ):
        raise ValueError(
            "mode_types must contain one value "
            "for each nuclear mode."
        )

    allowed_mode_types = {
        "nonreactive",
        "reactive",
        "translational",
        "unspecified",
    }

    for mode_type in (
        config.mode_types
    ):

        if (
            mode_type
            not in allowed_mode_types
        ):
            raise ValueError(
                f"Unknown nuclear mode type "
                f"'{mode_type}'. Allowed values are: "
                "nonreactive, reactive, translational, "
                "unspecified."
            )

    # ========================================================
    # Physical nuclear-coordinate representation
    # ========================================================

    if uses_physical_nuclear_coordinates:

        if (
            len(
                config.nuclear_coordinates
            )
            != n_modes
        ):
            raise ValueError(
                "The number of entries in "
                "nuclear.coordinates must equal "
                "n_vibrational_modes."
            )

        if config.vibrational_levels:
            raise ValueError(
                "Physical nuclear-coordinate "
                "configurations must not define "
                "legacy vibrational_levels."
            )

        # ----------------------------------------------------
        # Coordinate names must be unique
        # ----------------------------------------------------

        coordinate_names = [
            coordinate.name
            for coordinate
            in config.nuclear_coordinates
        ]

        if (
            len(
                set(
                    coordinate_names
                )
            )
            != len(
                coordinate_names
            )
        ):
            raise ValueError(
                "Physical nuclear-coordinate names "
                "must be unique."
            )

        # ----------------------------------------------------
        # Supported primitive bases
        # ----------------------------------------------------

        allowed_basis_kinds = {
            "sine_dvr",
            "sin_dvr",
            "sine",
        }

        # ----------------------------------------------------
        # Supported physical initial states
        # ----------------------------------------------------

        allowed_initial_state_kinds = {
            "neutral_pes_eigenstate",
            "gaussian",
        }

        # ----------------------------------------------------
        # Validate each coordinate
        # ----------------------------------------------------

        for (
            index,
            coordinate,
        ) in enumerate(
            config.nuclear_coordinates
        ):

            if (
                coordinate.mass_amu
                <= 0
            ):
                raise ValueError(
                    f"Nuclear coordinate "
                    f"'{coordinate.name}' must have "
                    "a positive mass_amu."
                )

            if (
                coordinate.mode_type
                not in allowed_mode_types
            ):
                raise ValueError(
                    f"Unknown mode type "
                    f"'{coordinate.mode_type}' for "
                    f"coordinate '{coordinate.name}'."
                )

            if (
                coordinate.basis.kind
                not in allowed_basis_kinds
            ):
                raise ValueError(
                    f"Unsupported basis type "
                    f"'{coordinate.basis.kind}' for "
                    f"coordinate '{coordinate.name}'. "
                    "Currently supported physical "
                    "basis types are: "
                    "sine_dvr, sin_dvr, sine."
                )

            if (
                coordinate.basis.size
                < 2
            ):
                raise ValueError(
                    f"Nuclear coordinate "
                    f"'{coordinate.name}' must have "
                    "at least two primitive basis states."
                )

            if not (
                coordinate.basis.minimum
                < coordinate.basis.maximum
            ):
                raise ValueError(
                    f"Nuclear coordinate "
                    f"'{coordinate.name}' must satisfy "
                    "minimum < maximum."
                )

            if (
                coordinate.basis.length_unit
                not in {
                    "angstrom",
                    "bohr",
                    "au",
                }
            ):
                raise ValueError(
                    f"Unsupported length unit "
                    f"'{coordinate.basis.length_unit}' "
                    f"for coordinate "
                    f"'{coordinate.name}'."
                )

            if (
                coordinate.initial_state.kind
                not in allowed_initial_state_kinds
            ):
                raise ValueError(
                    f"Unsupported initial-state type "
                    f"'{coordinate.initial_state.kind}' "
                    f"for coordinate "
                    f"'{coordinate.name}'."
                )

            # Ensure derived convenience values are
            # internally consistent.

            if (
                coordinate.basis.size
                != config.basis_sizes[
                    index
                ]
            ):
                raise ValueError(
                    f"basis_sizes is inconsistent with "
                    f"nuclear coordinate "
                    f"'{coordinate.name}'."
                )

            if (
                coordinate.mode_type
                != config.mode_types[
                    index
                ]
            ):
                raise ValueError(
                    f"mode_types is inconsistent with "
                    f"nuclear coordinate "
                    f"'{coordinate.name}'."
                )

    # ========================================================
    # Legacy harmonic representation
    # ========================================================

    else:

        if config.nuclear_coordinates:
            raise ValueError(
                "Legacy vibrational configurations "
                "must not define physical "
                "nuclear_coordinates."
            )

        if (
            len(
                config.vibrational_levels
            )
            != n_modes
        ):
            raise ValueError(
                "vibrational_levels must contain "
                "one value for each vibrational mode."
            )

        for (
            level,
            basis_size,
        ) in zip(
            config.vibrational_levels,
            config.basis_sizes,
        ):

            if not (
                0
                <= level
                < basis_size
            ):
                raise ValueError(
                    "An initial vibrational level lies "
                    "outside its basis."
                )

    # ========================================================
    # Electronic initial state
    # ========================================================

    if (
        len(
            config.occupied_orbitals
        )
        != config.n_electrons
    ):
        raise ValueError(
            "The number of occupied_orbitals must "
            "equal n_electrons."
        )

    if (
        len(
            set(
                config.occupied_orbitals
            )
        )
        != len(
            config.occupied_orbitals
        )
    ):
        raise ValueError(
            "occupied_orbitals contains "
            "repeated indices."
        )

    for orbital in (
        config.occupied_orbitals
    ):

        if not (
            0
            <= orbital
            < config.n_orbitals
        ):
            raise ValueError(
                f"Orbital index {orbital} lies outside "
                f"the range 0 to "
                f"{config.n_orbitals - 1}."
            )

    # ========================================================
    # General GAN nuclear-dependent terms
    # ========================================================

    all_nuclear_products = list(
        config.u0_terms
    )

    all_nuclear_products.extend(
        term.nuclear
        for term in config.uij_terms
    )

    all_nuclear_products.extend(
        term.nuclear
        for term in config.vij_terms
    )

    all_nuclear_products.extend(
        term.nuclear
        for term in config.wik_terms
    )

    for nuclear_product in (
        all_nuclear_products
    ):

        for factor in (
            nuclear_product.factors
        ):

            if not (
                0
                <= factor.mode
                < n_modes
            ):
                raise ValueError(
                    f"Nuclear mode index "
                    f"{factor.mode} lies outside "
                    f"the range 0 to "
                    f"{n_modes - 1}."
                )

    # ========================================================
    # U_ij indices
    # ========================================================

    for term in (
        config.uij_terms
    ):

        if not (
            0
            <= term.i
            < n_mol
        ):
            raise ValueError(
                f"Uij molecular index i={term.i} "
                "is outside the molecular "
                "orbital range."
            )

        if not (
            0
            <= term.j
            < n_mol
        ):
            raise ValueError(
                f"Uij molecular index j={term.j} "
                "is outside the molecular "
                "orbital range."
            )

    # ========================================================
    # V_ij indices
    # ========================================================

    for term in (
        config.vij_terms
    ):

        if (
            term.i
            >= term.j
        ):
            raise ValueError(
                "Vij terms must use i < j "
                "to avoid double counting."
            )

        if not (
            0
            <= term.i
            < n_mol
        ):
            raise ValueError(
                f"Vij molecular index i={term.i} "
                "is outside the molecular "
                "orbital range."
            )

        if not (
            0
            <= term.j
            < n_mol
        ):
            raise ValueError(
                f"Vij molecular index j={term.j} "
                "is outside the molecular "
                "orbital range."
            )

    # ========================================================
    # W_ik indices
    # ========================================================
    #
    # i = molecular orbital index
    # k = metal orbital index

    for term in (
        config.wik_terms
    ):

        if not (
            0
            <= term.i
            < n_mol
        ):
            raise ValueError(
                f"Wik molecular index i={term.i} "
                "is outside the molecular "
                "orbital range."
            )

        if not (
            0
            <= term.k
            < n_metal
        ):
            raise ValueError(
                f"Wik metal index k={term.k} "
                "is outside the metal "
                "orbital range."
            )

    # ========================================================
    # Propagation
    # ========================================================

    if (
        config.t_final
        <= 0
    ):
        raise ValueError(
            "t_final must be positive."
        )

    if (
        config.n_times
        < 2
    ):
        raise ValueError(
            "n_times must be at least 2."
        )