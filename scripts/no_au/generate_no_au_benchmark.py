from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import yaml

from ganbench.metal_bath import (
    discretize_flat_metal_bath,
)
from ganbench.model import load_config


# ============================================================
# Unit conversions
# ============================================================

EV_TO_HARTREE = 1.0 / 27.211386245988
AMU_TO_ELECTRON_MASS = 1822.888486209
BOHR_TO_ANGSTROM = 0.529177210903
FS_TO_ATOMIC_TIME = 1.0 / 0.024188843265857


# ============================================================
# Benchmark size
# ============================================================

N_METAL = 32

# Explicit finite-bath approximation used in the ML-MCTDH/SQR
# calculation.
#
# This is a numerical discretization parameter, not a new
# physical parameter of NO/Au.
FINITE_BANDWIDTH_EV = 100.0


# ============================================================
# Literature parameters: NO/Au(111)
# ============================================================

# ------------------------------------------------------------
# Neutral NO / Au diabatic surface
# ------------------------------------------------------------

R0_ANGSTROM = 1.1510
A0_INV_ANGSTROM = 2.7968
D0_EV = 6.610

B0_INV_ANGSTROM = 1.9535
Z0_ANGSTROM = -0.26876
C0_EV = 6.5713


# ------------------------------------------------------------
# Anionic NO / Au diabatic surface
# ------------------------------------------------------------

R1_ANGSTROM = 1.2950
A1_INV_ANGSTROM = 2.5194
D1_EV = 4.1528

Z1_ANGSTROM = 1.2350
A2_INV_ANGSTROM = 1.0015
D2_EV = 2.4171

C1_EV = 8.9587


# ------------------------------------------------------------
# Molecule-metal electronic coupling
# ------------------------------------------------------------

GAMMA_EV = 1.5
A_TILDE_ANGSTROM = 10.0

CHEMICAL_POTENTIAL_EV = 0.0


# ------------------------------------------------------------
# Reservoir parameters reported in the quantum-dynamics paper
# ------------------------------------------------------------

LITERATURE_RESERVOIR_WIDTH_EV = 50.0
LITERATURE_TEMPERATURE_K = 300.0


# ============================================================
# Nuclear masses
# ============================================================

# Isotopic masses of 14N and 16O in atomic mass units.

MASS_N_AMU = 14.00307400443
MASS_O_AMU = 15.99491461957

MASS_TRANSLATIONAL_AMU = (
    MASS_N_AMU
    + MASS_O_AMU
)

MASS_REDUCED_AMU = (
    MASS_N_AMU
    * MASS_O_AMU
    / MASS_TRANSLATIONAL_AMU
)


# ============================================================
# Initial scattering state
# ============================================================

INITIAL_VIBRATIONAL_LEVEL = 16

INITIAL_Z_ANGSTROM = 5.0

INITIAL_TRANSLATIONAL_KE_EV = 1.0


# ------------------------------------------------------------
# Compute incoming momentum
#
# KE = p^2 / (2m)
# ------------------------------------------------------------

mass_z_au = (
    MASS_TRANSLATIONAL_AMU
    * AMU_TO_ELECTRON_MASS
)

kinetic_energy_au = (
    INITIAL_TRANSLATIONAL_KE_EV
    * EV_TO_HARTREE
)

momentum_magnitude_au = math.sqrt(
    2.0
    * mass_z_au
    * kinetic_energy_au
)

# z grows away from the surface, so negative momentum
# corresponds to an incoming molecule.

INITIAL_MOMENTUM_AU = (
    -momentum_magnitude_au
)


# ------------------------------------------------------------
# Width specified in the Supporting Information:
#
# sigma = 20 au / |p_ini|
# ------------------------------------------------------------

SIGMA_BOHR = (
    20.0
    / momentum_magnitude_au
)

SIGMA_ANGSTROM = (
    SIGMA_BOHR
    * BOHR_TO_ANGSTROM
)


# ============================================================
# Nuclear primitive bases
# ============================================================

# ------------------------------------------------------------
# r: N-O bond stretch
#
# This is our starting DVR.
# It will later be subjected to primitive-basis convergence.
# ------------------------------------------------------------

R_GRID_MIN_ANGSTROM = 0.7
R_GRID_MAX_ANGSTROM = 3.0
R_GRID_SIZE = 128


# ------------------------------------------------------------
# z: molecule-surface translation
#
# These values are the converged values reported in the
# Supporting Information.
# ------------------------------------------------------------

Z_GRID_MIN_ANGSTROM = 0.53
Z_GRID_MAX_ANGSTROM = 6.9
Z_GRID_SIZE = 340


# ============================================================
# Initial propagation window
# ============================================================

PROPAGATION_TIME_FS = 350.0

T_FINAL_AU = (
    PROPAGATION_TIME_FS
    * FS_TO_ATOMIC_TIME
)

# Save approximately every 1 fs.
N_TIMES = 351


# ============================================================
# Helpers
# ============================================================


def morse_factor(
    mode: int,
    depth_ev: float,
    alpha_inv_angstrom: float,
    equilibrium_angstrom: float,
) -> dict:
    """
    Return a shifted-positive Morse factor

        D [1 - exp(-a(Q-Qe))]^2.

    The literature Morse convention is

        D [exp(-2ax) - 2 exp(-ax)].

    These differ only by a constant -D, which is included
    explicitly in the generated GAN terms.
    """

    return {
        "mode": mode,
        "kind": "morse",
        "parameters": {
            "depth": (
                depth_ev
                * EV_TO_HARTREE
            ),
            "alpha": (
                alpha_inv_angstrom
            ),
            "equilibrium": (
                equilibrium_angstrom
            ),
        },
    }


def exponential_factor(
    mode: int,
    alpha_inv_angstrom: float,
    center_angstrom: float,
) -> dict:
    """
    exp[-alpha * (Q - center)]
    """

    return {
        "mode": mode,
        "kind": "exponential",
        "parameters": {
            "alpha": (
                alpha_inv_angstrom
            ),
            "center": (
                center_angstrom
            ),
        },
    }


def tanh_factor(
    mode: int,
    scale_angstrom: float,
) -> dict:
    """
    tanh(Q / scale)
    """

    return {
        "mode": mode,
        "kind": "tanh",
        "parameters": {
            "alpha": (
                1.0
                / scale_angstrom
            ),
            "center": 0.0,
        },
    }


# ============================================================
# Construct finite metal bath
# ============================================================


bath = discretize_flat_metal_bath(
    n_orbitals=N_METAL,
    gamma_ev=GAMMA_EV,
    bandwidth_ev=FINITE_BANDWIDTH_EV,
    chemical_potential_ev=(
        CHEMICAL_POTENTIAL_EV
    ),
)


# Convert the finite electronic Hamiltonian to atomic units.

metal_energies_hartree = (
    bath.energies_ev
    * EV_TO_HARTREE
)

coupling_prefactors_hartree = (
    bath.coupling_prefactors_ev
    * EV_TO_HARTREE
)


# ============================================================
# Initial electronic occupation
# ============================================================

# Orbital ordering:
#
# global index 0       = molecular orbital d
# global index 1 + k   = metal orbital k
#
# The molecular orbital is initially empty.
# All metal levels below mu are occupied.

occupied_orbitals = [
    1 + index
    for index in bath.occupied_indices
]

N_ELECTRONS = len(
    occupied_orbitals
)


# ============================================================
# Construct U_0(r,z)
# ============================================================

# Literature neutral surface:
#
# U0 =
#     V_M(r-r0; D0,a0)
#   + exp[-b0(z-z0)]
#   + c0
#
# Our "morse" primitive is
#
#     D(1-exp[-a(r-r0)])^2
#
# = literature_V_M + D
#
# therefore the required constant is c0-D0.

u0_terms = [

    # N-O Morse potential
    {
        "coefficient": 1.0,
        "factors": [
            morse_factor(
                mode=0,
                depth_ev=D0_EV,
                alpha_inv_angstrom=(
                    A0_INV_ANGSTROM
                ),
                equilibrium_angstrom=(
                    R0_ANGSTROM
                ),
            )
        ],
    },

    # Repulsive molecule-surface exponential.
    #
    # The coefficient is 1 eV in the analytic model.
    {
        "coefficient": (
            1.0
            * EV_TO_HARTREE
        ),
        "factors": [
            exponential_factor(
                mode=1,
                alpha_inv_angstrom=(
                    B0_INV_ANGSTROM
                ),
                center_angstrom=(
                    Z0_ANGSTROM
                ),
            )
        ],
    },

    # Constant energy shift.
    {
        "coefficient": (
            (C0_EV - D0_EV)
            * EV_TO_HARTREE
        ),
        "factors": [],
    },
]


# ============================================================
# Construct U_11(r,z) = U_anion - U_neutral
# ============================================================

# Literature anionic surface:
#
# U1 =
#     V_M(r-r1;D1,a1)
#   + V_M(z-z1;D2,a2)
#   + c1
#
# In GAN notation:
#
#     U_11 = U1 - U0

u11_constant_ev = (
    (
        C1_EV
        - D1_EV
        - D2_EV
    )
    -
    (
        C0_EV
        - D0_EV
    )
)


uij_terms = [

    # + anionic N-O Morse
    {
        "i": 0,
        "j": 0,
        "coefficient": 1.0,
        "factors": [
            morse_factor(
                mode=0,
                depth_ev=D1_EV,
                alpha_inv_angstrom=(
                    A1_INV_ANGSTROM
                ),
                equilibrium_angstrom=(
                    R1_ANGSTROM
                ),
            )
        ],
    },

    # - neutral N-O Morse
    {
        "i": 0,
        "j": 0,
        "coefficient": -1.0,
        "factors": [
            morse_factor(
                mode=0,
                depth_ev=D0_EV,
                alpha_inv_angstrom=(
                    A0_INV_ANGSTROM
                ),
                equilibrium_angstrom=(
                    R0_ANGSTROM
                ),
            )
        ],
    },

    # + anionic molecule-surface Morse
    {
        "i": 0,
        "j": 0,
        "coefficient": 1.0,
        "factors": [
            morse_factor(
                mode=1,
                depth_ev=D2_EV,
                alpha_inv_angstrom=(
                    A2_INV_ANGSTROM
                ),
                equilibrium_angstrom=(
                    Z1_ANGSTROM
                ),
            )
        ],
    },

    # - neutral molecule-surface exponential
    {
        "i": 0,
        "j": 0,
        "coefficient": (
            -1.0
            * EV_TO_HARTREE
        ),
        "factors": [
            exponential_factor(
                mode=1,
                alpha_inv_angstrom=(
                    B0_INV_ANGSTROM
                ),
                center_angstrom=(
                    Z0_ANGSTROM
                ),
            )
        ],
    },

    # Constant part of U1-U0
    {
        "i": 0,
        "j": 0,
        "coefficient": (
            u11_constant_ev
            * EV_TO_HARTREE
        ),
        "factors": [],
    },
]


# ============================================================
# Construct W_1k(z)
# ============================================================

# For every finite metal state:
#
# W_1k(z)
# =
# V_k [1 - tanh(z/a_tilde)]
#
# This is written as two separable GAN terms:
#
# + V_k
# - V_k tanh(z/a_tilde)

wik_terms = []

for k, prefactor_hartree in enumerate(
    coupling_prefactors_hartree
):

    prefactor_hartree = float(
        prefactor_hartree
    )

    # + V_k
    wik_terms.append(
        {
            "i": 0,
            "k": k,
            "coefficient": (
                prefactor_hartree
            ),
            "factors": [],
        }
    )

    # - V_k tanh(z/a_tilde)
    wik_terms.append(
        {
            "i": 0,
            "k": k,
            "coefficient": (
                -prefactor_hartree
            ),
            "factors": [
                tanh_factor(
                    mode=1,
                    scale_angstrom=(
                        A_TILDE_ANGSTROM
                    ),
                )
            ],
        }
    )


# ============================================================
# Complete YAML configuration
# ============================================================

config_data = {

    # --------------------------------------------------------
    # Human-readable metadata
    # --------------------------------------------------------

    "metadata": {
        "description": (
            "Physical 2D NO/Au(111) nonadiabatic "
            "scattering benchmark"
        ),
        "physical_system": "NO/Au(111)",
        "nuclear_coordinates": [
            "r: N-O stretch",
            "z: molecule-surface translation",
        ],
        "energy_unit_for_hamiltonian": (
            "hartree"
        ),
        "coordinate_unit": (
            "angstrom"
        ),
        "finite_bath_note": (
            "The explicit finite SQR bath is a numerical "
            "adaptation used for ML-MCTDH and is not the "
            "same object as the finite-temperature HEOM "
            "pole decomposition."
        ),
    },

    # --------------------------------------------------------
    # Model dimensions
    # --------------------------------------------------------

    "model": {
        "n_molecular_orbitals": 1,
        "n_metal_orbitals": (
            N_METAL
        ),
        "n_electrons": (
            N_ELECTRONS
        ),
        "n_vibrational_modes": 2,
    },

    # --------------------------------------------------------
    # Electronic finite representation
    # --------------------------------------------------------

    "electronic": {

        # The entire molecular energy difference is contained
        # in U_11(r,z), so no additional constant orbital
        # energy is needed here.
        "molecular_energies": [
            0.0
        ],

        "metal_energies": [
            float(value)
            for value
            in metal_energies_hartree
        ],

        # Static molecule-metal couplings are zero because
        # W_1k(z) is represented explicitly in gan_terms.wik.
        "molecule_metal_couplings": [
            [
                0.0
                for _ in range(
                    N_METAL
                )
            ]
        ],
    },

    # --------------------------------------------------------
    # Physical nuclear coordinates
    # --------------------------------------------------------

    "nuclear": {
        "coordinates": [

            # =================================================
            # Q_0 = r
            # =================================================

            {
                "name": "r",
                "mode_type": "reactive",
                "mass_amu": (
                    MASS_REDUCED_AMU
                ),

                "basis": {
                    "type": "sine_dvr",
                    "size": (
                        R_GRID_SIZE
                    ),
                    "minimum": (
                        R_GRID_MIN_ANGSTROM
                    ),
                    "maximum": (
                        R_GRID_MAX_ANGSTROM
                    ),
                    "length_unit": (
                        "angstrom"
                    ),
                },

                "initial_state": {
                    "type": (
                        "neutral_pes_eigenstate"
                    ),
                    "level": float(
                        INITIAL_VIBRATIONAL_LEVEL
                    ),
                },
            },

            # =================================================
            # Q_1 = z
            # =================================================

            {
                "name": "z",
                "mode_type": (
                    "translational"
                ),
                "mass_amu": (
                    MASS_TRANSLATIONAL_AMU
                ),

                "basis": {
                    "type": "sine_dvr",
                    "size": (
                        Z_GRID_SIZE
                    ),
                    "minimum": (
                        Z_GRID_MIN_ANGSTROM
                    ),
                    "maximum": (
                        Z_GRID_MAX_ANGSTROM
                    ),
                    "length_unit": (
                        "angstrom"
                    ),
                },

                "initial_state": {
                    "type": "gaussian",

                    "center_angstrom": (
                        INITIAL_Z_ANGSTROM
                    ),

                    "kinetic_energy_ev": (
                        INITIAL_TRANSLATIONAL_KE_EV
                    ),

                    "momentum_au": (
                        INITIAL_MOMENTUM_AU
                    ),

                    "sigma_bohr": (
                        SIGMA_BOHR
                    ),

                    "sigma_angstrom": (
                        SIGMA_ANGSTROM
                    ),
                },
            },
        ]
    },

    # --------------------------------------------------------
    # Electronic initial state
    # --------------------------------------------------------

    "initial_state": {
        "occupied_orbitals": (
            occupied_orbitals
        )
    },

    # --------------------------------------------------------
    # Propagation
    # --------------------------------------------------------

    "propagation": {
        "t_final": (
            T_FINAL_AU
        ),
        "n_times": (
            N_TIMES
        ),
    },

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    "output": {
        "save_states": False,
    },

    # --------------------------------------------------------
    # General GAN terms
    # --------------------------------------------------------

    "gan_terms": {

        "u0": (
            u0_terms
        ),

        "uij": (
            uij_terms
        ),

        "vij": [],

        "wik": (
            wik_terms
        ),
    },

    # --------------------------------------------------------
    # Original physical parameters retained explicitly
    # for provenance and reproducibility.
    # --------------------------------------------------------

    "physical_parameters": {

        "neutral_surface": {
            "r0_angstrom": R0_ANGSTROM,
            "a0_inv_angstrom": (
                A0_INV_ANGSTROM
            ),
            "D0_ev": D0_EV,
            "b0_inv_angstrom": (
                B0_INV_ANGSTROM
            ),
            "z0_angstrom": Z0_ANGSTROM,
            "c0_ev": C0_EV,
        },

        "anionic_surface": {
            "r1_angstrom": R1_ANGSTROM,
            "a1_inv_angstrom": (
                A1_INV_ANGSTROM
            ),
            "D1_ev": D1_EV,
            "z1_angstrom": Z1_ANGSTROM,
            "a2_inv_angstrom": (
                A2_INV_ANGSTROM
            ),
            "D2_ev": D2_EV,
            "c1_ev": C1_EV,
        },

        "molecule_metal": {
            "gamma_ev": GAMMA_EV,
            "a_tilde_angstrom": (
                A_TILDE_ANGSTROM
            ),
        },

        "literature_reservoir": {
            "temperature_k": (
                LITERATURE_TEMPERATURE_K
            ),
            "chemical_potential_ev": (
                CHEMICAL_POTENTIAL_EV
            ),
            "width_ev": (
                LITERATURE_RESERVOIR_WIDTH_EV
            ),
        },

        "explicit_sqr_bath": {
            "n_orbitals": N_METAL,
            "bandwidth_ev": (
                FINITE_BANDWIDTH_EV
            ),
            "chemical_potential_ev": (
                CHEMICAL_POTENTIAL_EV
            ),
            "temperature_k": 0.0,
            "discretization": (
                "split_gauss_legendre_flat"
            ),
        },
    },
}


# ============================================================
# Write YAML
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark3_no_au_scattering.yaml"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as file:

    yaml.safe_dump(
        config_data,
        file,
        sort_keys=False,
        default_flow_style=False,
    )


# ============================================================
# Validate generated configuration
# ============================================================

config = load_config(
    OUTPUT_PATH
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 65)
print("NO/Au benchmark generated successfully")
print("=" * 65)

print(
    "Config:",
    OUTPUT_PATH,
)

print()

print("Electronic model")
print(
    "  Molecular orbitals:",
    config.n_molecular_orbitals,
)
print(
    "  Metal orbitals:",
    config.n_metal_orbitals,
)
print(
    "  Electrons:",
    config.n_electrons,
)
print(
    "  Occupied metal states:",
    len(
        bath.occupied_indices
    ),
)

print()

print("Metal energy range")
print(
    "  Emin:",
    bath.energies_ev[0],
    "eV",
)
print(
    "  Emax:",
    bath.energies_ev[-1],
    "eV",
)

print()

print("Nuclear coordinates")

for coordinate in (
    config.nuclear_coordinates
):
    print(
        f"  {coordinate.name}: "
        f"{coordinate.mode_type}, "
        f"N={coordinate.basis.size}, "
        f"[{coordinate.basis.minimum}, "
        f"{coordinate.basis.maximum}] "
        f"{coordinate.basis.length_unit}"
    )

print()

print("Initial scattering state")

print(
    "  nu_ini =",
    INITIAL_VIBRATIONAL_LEVEL,
)

print(
    "  z_ini =",
    INITIAL_Z_ANGSTROM,
    "Angstrom",
)

print(
    "  KE_ini =",
    INITIAL_TRANSLATIONAL_KE_EV,
    "eV",
)

print(
    "  p_ini =",
    INITIAL_MOMENTUM_AU,
    "a.u.",
)

print(
    "  sigma =",
    SIGMA_BOHR,
    "bohr",
)

print(
    "  sigma =",
    SIGMA_ANGSTROM,
    "Angstrom",
)

print()

print(
    "Propagation window:",
    PROPAGATION_TIME_FS,
    "fs",
)

print(
    "Number of output times:",
    N_TIMES,
)

print()

print(
    "U0 terms:",
    len(
        config.u0_terms
    ),
)

print(
    "U11 terms:",
    len(
        config.uij_terms
    ),
)

print(
    "W1k terms:",
    len(
        config.wik_terms
    ),
)

print()

print(
    "Configuration validation: OK"
)