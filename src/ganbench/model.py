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

    # Model size
    n_molecular_orbitals: int
    n_metal_orbitals: int
    n_electrons: int
    n_vibrational_modes: int

    # Electronic parameters
    molecular_energies: np.ndarray
    metal_energies: np.ndarray
    molecule_metal_couplings: np.ndarray

    # Vibrational parameters
    frequencies: np.ndarray
    basis_sizes: tuple[int, ...]
    electron_vibration_couplings: np.ndarray

    # Physical classification of nuclear modes:
    # nonreactive, reactive, translational, or unspecified.
    mode_types: tuple[str, ...]

    # Initial state
    occupied_orbitals: tuple[int, ...]
    vibrational_levels: tuple[int, ...]

    # Propagation parameters
    t_final: float
    n_times: int

    # Output options
    save_states: bool = False

    # General GAN nuclear-dependent terms.
    #
    # These default to empty tuples so old configuration
    # files remain valid.
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

    @property
    def n_orbitals(self) -> int:
        """Total number of electronic orbitals."""

        return (
            self.n_molecular_orbitals
            + self.n_metal_orbitals
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


# ============================================================
# Configuration loading
# ============================================================


def load_config(
    path: str | Path,
) -> GANConfig:
    """Read and validate a YAML configuration file."""

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

    model = raw["model"]
    electronic = raw["electronic"]
    vibrational = raw["vibrational"]
    initial_state = raw["initial_state"]
    propagation = raw["propagation"]

    output = raw.get(
        "output",
        {},
    )

    # Optional section.
    #
    # If it does not exist, all general GAN terms
    # are simply empty.
    gan_terms = raw.get(
        "gan_terms",
        {},
    )

    n_vibrational_modes = int(
        model[
            "n_vibrational_modes"
        ]
    )

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
        # Vibrational parameters
        # ----------------------------------------------------

        frequencies=np.asarray(
            vibrational[
                "frequencies"
            ],
            dtype=float,
        ),

        basis_sizes=tuple(
            int(value)
            for value in vibrational[
                "basis_sizes"
            ]
        ),

        electron_vibration_couplings=np.asarray(
            vibrational[
                "electron_vibration_couplings"
            ],
            dtype=float,
        ),

        mode_types=tuple(
            str(value).lower()
            for value in vibrational.get(
                "mode_types",
                [
                    "unspecified"
                    for _ in range(
                        n_vibrational_modes
                    )
                ],
            )
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

        vibrational_levels=tuple(
            int(value)
            for value in initial_state[
                "vibrational_levels"
            ]
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
                    term["i"]
                ),
                j=int(
                    term["j"]
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
                    term["i"]
                ),
                j=int(
                    term["j"]
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
                    term["i"]
                ),
                k=int(
                    term["k"]
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
    """Check that all parameter dimensions are consistent."""

    n_mol = (
        config.n_molecular_orbitals
    )

    n_metal = (
        config.n_metal_orbitals
    )

    n_modes = (
        config.n_vibrational_modes
    )

    # --------------------------------------------------------
    # Basic model size
    # --------------------------------------------------------

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
            "At least one vibrational mode is required."
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

    # --------------------------------------------------------
    # Array dimensions
    # --------------------------------------------------------

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
        "frequencies": (
            n_modes,
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

        "frequencies":
            config.frequencies,

        "electron_vibration_couplings":
            config.electron_vibration_couplings,
    }

    for (
        name,
        expected_shape,
    ) in expected_shapes.items():

        actual_shape = (
            arrays[name].shape
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

    # --------------------------------------------------------
    # Vibrational bases
    # --------------------------------------------------------

    if (
        len(
            config.basis_sizes
        )
        != n_modes
    ):
        raise ValueError(
            "basis_sizes must contain one value "
            "for each vibrational mode."
        )

    if any(
        size < 2
        for size in config.basis_sizes
    ):
        raise ValueError(
            "Every vibrational basis must contain "
            "at least two states."
        )

    # --------------------------------------------------------
    # Nuclear mode types
    # --------------------------------------------------------

    if (
        len(
            config.mode_types
        )
        != n_modes
    ):
        raise ValueError(
            "mode_types must contain one value "
            "for each vibrational mode."
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

    if (
        len(
            config.vibrational_levels
        )
        != n_modes
    ):
        raise ValueError(
            "vibrational_levels must contain one "
            "value for each vibrational mode."
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

    # --------------------------------------------------------
    # Electronic initial state
    # --------------------------------------------------------

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
            "occupied_orbitals contains repeated indices."
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

    # --------------------------------------------------------
    # Frequencies
    # --------------------------------------------------------

    if np.any(
        config.frequencies
        <= 0
    ):
        raise ValueError(
            "All vibrational frequencies must be positive."
        )

    # --------------------------------------------------------
    # General GAN terms
    # --------------------------------------------------------

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

    # U_ij indices refer only to molecular orbitals.

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
                "is outside the molecular orbital range."
            )

        if not (
            0
            <= term.j
            < n_mol
        ):
            raise ValueError(
                f"Uij molecular index j={term.j} "
                "is outside the molecular orbital range."
            )

    # V_ij indices also refer to molecular orbitals.

    for term in (
        config.vij_terms
    ):

        if term.i >= term.j:
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
                "is outside the molecular orbital range."
            )

        if not (
            0
            <= term.j
            < n_mol
        ):
            raise ValueError(
                f"Vij molecular index j={term.j} "
                "is outside the molecular orbital range."
            )

    # W_ik:
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
                "is outside the molecular orbital range."
            )

        if not (
            0
            <= term.k
            < n_metal
        ):
            raise ValueError(
                f"Wik metal index k={term.k} "
                "is outside the metal orbital range."
            )

    # --------------------------------------------------------
    # Propagation
    # --------------------------------------------------------

    if config.t_final <= 0:
        raise ValueError(
            "t_final must be positive."
        )

    if config.n_times < 2:
        raise ValueError(
            "n_times must be at least 2."
        )