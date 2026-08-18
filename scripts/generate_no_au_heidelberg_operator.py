from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ganbench.model import load_config


# ============================================================
# Constants
# ============================================================

EV_TO_HARTREE = 1.0 / 27.211386245988

BOHR_TO_ANGSTROM = 0.529177210903

ANGSTROM_TO_BOHR = (
    1.0
    / BOHR_TO_ANGSTROM
)

AMU_TO_ELECTRON_MASS = 1822.888486209


# ============================================================
# Benchmark paths
# ============================================================

BENCHMARK = "benchmark3_no_au_scattering"

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / f"{BENCHMARK}.yaml"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "backend_inputs"
    / BENCHMARK
    / "heidelberg"
    / "run_001"
)

OPERATOR_PATH = (
    OUTPUT_DIRECTORY
    / "benchmark.op"
)


# ============================================================
# Load configuration
# ============================================================

config = load_config(
    CONFIG_PATH
)

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    raw = yaml.safe_load(
        file
    )


if (
    config.n_molecular_orbitals
    != 1
):
    raise ValueError(
        "This generator assumes exactly "
        "one molecular orbital."
    )


if (
    not config.uses_physical_nuclear_coordinates
):
    raise ValueError(
        "NO/Au must use physical nuclear coordinates."
    )


if not np.allclose(
    config.molecular_energies,
    0.0,
):
    raise ValueError(
        "For this physical NO/Au mapping, "
        "molecular_energies must be zero because "
        "the complete neutral-anion energy difference "
        "is contained in U11(r,z)."
    )


if not np.allclose(
    config.molecule_metal_couplings,
    0.0,
):
    raise ValueError(
        "Static molecule-metal couplings must be zero. "
        "W_1k(z) is represented through wik_terms."
    )


# ============================================================
# Physical parameters
# ============================================================

physical = raw[
    "physical_parameters"
]

neutral = physical[
    "neutral_surface"
]

anion = physical[
    "anionic_surface"
]

molecule_metal = physical[
    "molecule_metal"
]


# ------------------------------------------------------------
# Neutral surface
# ------------------------------------------------------------

R0_ANGSTROM = float(
    neutral[
        "r0_angstrom"
    ]
)

A0_INV_ANGSTROM = float(
    neutral[
        "a0_inv_angstrom"
    ]
)

D0_EV = float(
    neutral[
        "D0_ev"
    ]
)

B0_INV_ANGSTROM = float(
    neutral[
        "b0_inv_angstrom"
    ]
)

Z0_ANGSTROM = float(
    neutral[
        "z0_angstrom"
    ]
)

C0_EV = float(
    neutral[
        "c0_ev"
    ]
)


# ------------------------------------------------------------
# Anionic surface
# ------------------------------------------------------------

R1_ANGSTROM = float(
    anion[
        "r1_angstrom"
    ]
)

A1_INV_ANGSTROM = float(
    anion[
        "a1_inv_angstrom"
    ]
)

D1_EV = float(
    anion[
        "D1_ev"
    ]
)

Z1_ANGSTROM = float(
    anion[
        "z1_angstrom"
    ]
)

A2_INV_ANGSTROM = float(
    anion[
        "a2_inv_angstrom"
    ]
)

D2_EV = float(
    anion[
        "D2_ev"
    ]
)

C1_EV = float(
    anion[
        "c1_ev"
    ]
)


# ------------------------------------------------------------
# Geometry-dependent molecule-metal coupling
# ------------------------------------------------------------

A_TILDE_ANGSTROM = float(
    molecule_metal[
        "a_tilde_angstrom"
    ]
)


# ============================================================
# Nuclear coordinates
# ============================================================

coordinate_map = {
    coordinate.name: coordinate
    for coordinate
    in config.nuclear_coordinates
}

r_coordinate = (
    coordinate_map[
        "r"
    ]
)

z_coordinate = (
    coordinate_map[
        "z"
    ]
)


mass_r_au = (
    r_coordinate.mass_amu
    * AMU_TO_ELECTRON_MASS
)

mass_z_au = (
    z_coordinate.mass_amu
    * AMU_TO_ELECTRON_MASS
)


# ============================================================
# Mode ordering
# ============================================================
#
# Heidelberg operator ordering:
#
#   1      d
#   2      c1
#   3      c2
#   ...
#   33     c32
#   34     r
#   35     z
#
# ============================================================

D_MODE = 1

METAL_MODE_START = 2

R_MODE = (
    1
    + config.n_metal_orbitals
    + 1
)

Z_MODE = (
    R_MODE
    + 1
)


metal_labels = [
    f"c{k}"
    for k in range(
        1,
        config.n_metal_orbitals + 1,
    )
]

all_mode_labels = [
    "d",
    *metal_labels,
    "r",
    "z",
]


# ============================================================
# Formatting helpers
# ============================================================


def fortran_number(
    value: float,
) -> str:
    """
    Heidelberg parameter files use Fortran-style
    double-precision exponential notation.
    """

    return (
        f"{float(value):.16e}"
        .replace(
            "e",
            "d",
        )
    )


def render_modes(
    labels: list[str],
    per_line: int = 10,
) -> list[str]:
    """
    Render long mode lists using several 'modes' lines.

    Heidelberg explicitly permits multiple modes lines.
    """

    lines = []

    for start in range(
        0,
        len(labels),
        per_line,
    ):

        block = labels[
            start:
            start + per_line
        ]

        lines.append(
            "modes"
            + "".join(
                f" | {label}"
                for label in block
            )
        )

    return lines


def render_term(
    coefficient: str,
    operators: list[
        tuple[int, str]
    ],
    max_length: int = 220,
) -> list[str]:
    """
    Render one Heidelberg Hamiltonian product term.

    Long Jordan-Wigner strings are automatically split
    using Heidelberg continuation lines beginning with &&&.
    """

    operators = sorted(
        operators,
        key=lambda item: item[0],
    )

    first_prefix = (
        f"{coefficient:<22}"
    )

    continuation_prefix = (
        f"{'&&&':<22}"
    )

    lines = []

    current = first_prefix

    for (
        mode,
        operator,
    ) in operators:

        piece = (
            f" |{mode} {operator}"
        )

        if (
            len(current)
            + len(piece)
            > max_length
        ):

            lines.append(
                current
            )

            current = (
                continuation_prefix
                + piece
            )

        else:

            current += piece

    lines.append(
        current
    )

    return lines


# ============================================================
# Physical 1D functions
# ============================================================


def literature_morse_ev(
    coordinate_angstrom: np.ndarray,
    equilibrium_angstrom: float,
    alpha_inv_angstrom: float,
    depth_ev: float,
) -> np.ndarray:
    """
    Morse convention used in the NO/Au model:

        D [exp(-2 a x) - 2 exp(-a x)]

    where x = Q - Q_e.

    The minimum is -D.
    """

    displacement = (
        coordinate_angstrom
        - equilibrium_angstrom
    )

    exponential = np.exp(
        -alpha_inv_angstrom
        * displacement
    )

    return (
        depth_ev
        * (
            exponential**2
            - 2.0
            * exponential
        )
    )


def neutral_r_hartree(
    r_bohr: np.ndarray,
) -> np.ndarray:

    r_angstrom = (
        r_bohr
        * BOHR_TO_ANGSTROM
    )

    return (
        literature_morse_ev(
            r_angstrom,
            R0_ANGSTROM,
            A0_INV_ANGSTROM,
            D0_EV,
        )
        * EV_TO_HARTREE
    )


def anionic_r_hartree(
    r_bohr: np.ndarray,
) -> np.ndarray:

    r_angstrom = (
        r_bohr
        * BOHR_TO_ANGSTROM
    )

    return (
        literature_morse_ev(
            r_angstrom,
            R1_ANGSTROM,
            A1_INV_ANGSTROM,
            D1_EV,
        )
        * EV_TO_HARTREE
    )


def neutral_z_hartree(
    z_bohr: np.ndarray,
) -> np.ndarray:
    """
    Neutral molecule-surface contribution:

        exp[-b0 (z-z0)] eV
    """

    z_angstrom = (
        z_bohr
        * BOHR_TO_ANGSTROM
    )

    value_ev = np.exp(
        -B0_INV_ANGSTROM
        * (
            z_angstrom
            - Z0_ANGSTROM
        )
    )

    return (
        value_ev
        * EV_TO_HARTREE
    )


def anionic_z_hartree(
    z_bohr: np.ndarray,
) -> np.ndarray:

    z_angstrom = (
        z_bohr
        * BOHR_TO_ANGSTROM
    )

    return (
        literature_morse_ev(
            z_angstrom,
            Z1_ANGSTROM,
            A2_INV_ANGSTROM,
            D2_EV,
        )
        * EV_TO_HARTREE
    )


def coupling_geometry(
    z_bohr: np.ndarray,
) -> np.ndarray:
    """
    Geometry-dependent factor in

        W_1k(z)
        =
        V_k [1 - tanh(z/a_tilde)].
    """

    z_angstrom = (
        z_bohr
        * BOHR_TO_ANGSTROM
    )

    return (
        1.0
        - np.tanh(
            z_angstrom
            / A_TILDE_ANGSTROM
        )
    )


# ============================================================
# Write external1d files
# ============================================================


def write_external_1d(
    path: Path,
    minimum_bohr: float,
    maximum_bohr: float,
    function,
    n_points: int = 8193,
) -> None:
    """
    external1d requires an equally spaced two-column file:

        x_i    f(x_i)

    We deliberately extend the tabulated range beyond the
    primitive grid so Heidelberg never has to extrapolate at
    the first or last DVR point.
    """

    span = (
        maximum_bohr
        - minimum_bohr
    )

    padding = (
        0.05
        * span
    )

    x = np.linspace(
        minimum_bohr
        - padding,
        maximum_bohr
        + padding,
        n_points,
    )

    y = np.asarray(
        function(
            x
        ),
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            y
        )
    ):
        raise ValueError(
            f"Non-finite values generated for {path.name}."
        )

    data = np.column_stack(
        (
            x,
            y,
        )
    )

    np.savetxt(
        path,
        data,
        fmt="%.16e",
    )


r_min_bohr = (
    r_coordinate.basis.minimum
    * ANGSTROM_TO_BOHR
)

r_max_bohr = (
    r_coordinate.basis.maximum
    * ANGSTROM_TO_BOHR
)

z_min_bohr = (
    z_coordinate.basis.minimum
    * ANGSTROM_TO_BOHR
)

z_max_bohr = (
    z_coordinate.basis.maximum
    * ANGSTROM_TO_BOHR
)


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


write_external_1d(
    OUTPUT_DIRECTORY
    / "neutral_r.dat",
    r_min_bohr,
    r_max_bohr,
    neutral_r_hartree,
)


write_external_1d(
    OUTPUT_DIRECTORY
    / "anionic_r.dat",
    r_min_bohr,
    r_max_bohr,
    anionic_r_hartree,
)


write_external_1d(
    OUTPUT_DIRECTORY
    / "neutral_z.dat",
    z_min_bohr,
    z_max_bohr,
    neutral_z_hartree,
)


write_external_1d(
    OUTPUT_DIRECTORY
    / "anionic_z.dat",
    z_min_bohr,
    z_max_bohr,
    anionic_z_hartree,
)


write_external_1d(
    OUTPUT_DIRECTORY
    / "coupling_z.dat",
    z_min_bohr,
    z_max_bohr,
    coupling_geometry,
)


# ============================================================
# Recover the discrete W_k prefactors from YAML
# ============================================================

coupling_prefactors = []


for k in range(
    config.n_metal_orbitals
):

    terms_for_k = [
        term
        for term in config.wik_terms
        if term.k == k
    ]

    constant_terms = [
        term
        for term in terms_for_k
        if len(
            term.nuclear.factors
        ) == 0
    ]

    if (
        len(
            constant_terms
        )
        != 1
    ):
        raise ValueError(
            f"Expected one constant W_1{k + 1} "
            "prefactor in wik_terms."
        )

    prefactor = float(
        constant_terms[
            0
        ].nuclear.coefficient
    )

    coupling_prefactors.append(
        prefactor
    )


coupling_prefactors = np.asarray(
    coupling_prefactors,
    dtype=float,
)


# ============================================================
# Constants from the diabatic surfaces
# ============================================================
#
# Hamiltonian:
#
# H_nuc =
#     T
#   + U_N
#   + n_d (U_A - U_N)
#
# Since
#
#     n_d = 1/2 + q_d,
#
# this becomes
#
#     T
#   + 1/2 (U_N + U_A)
#   + q_d (U_A - U_N).
#
# This is particularly convenient for SQR.
# ============================================================

C_AVERAGE_HARTREE = (
    0.5
    * (
        C0_EV
        + C1_EV
    )
    * EV_TO_HARTREE
)

C_DIFFERENCE_HARTREE = (
    (
        C1_EV
        - C0_EV
    )
    * EV_TO_HARTREE
)


# ============================================================
# Metal constant energy shift
# ============================================================
#
# epsilon_k n_k
# =
# epsilon_k (1/2 + q_k)
#
# The bath is symmetric, so this should be essentially zero,
# but compute it rather than assume exact floating-point
# cancellation.
# ============================================================

METAL_CONSTANT_HARTREE = (
    0.5
    * float(
        np.sum(
            config.metal_energies
        )
    )
)


# ============================================================
# Parameter section
# ============================================================

parameter_lines = [

    (
        "mass_r = "
        + fortran_number(
            mass_r_au
        )
    ),

    (
        "mass_z = "
        + fortran_number(
            mass_z_au
        )
    ),

    (
        "cavg = "
        + fortran_number(
            C_AVERAGE_HARTREE
        )
    ),

    (
        "cdiff = "
        + fortran_number(
            C_DIFFERENCE_HARTREE
        )
    ),

    (
        "metalconst = "
        + fortran_number(
            METAL_CONSTANT_HARTREE
        )
    ),
]


for (
    k,
    energy,
) in enumerate(
    config.metal_energies,
    start=1,
):

    parameter_lines.append(
        (
            f"ec{k} = "
            + fortran_number(
                float(
                    energy
                )
            )
        )
    )


for (
    k,
    coupling,
) in enumerate(
    coupling_prefactors,
    start=1,
):

    parameter_lines.append(
        (
            f"vk{k} = "
            + fortran_number(
                float(
                    coupling
                )
            )
        )
    )


# ============================================================
# Labels
# ============================================================

label_lines = [
    (
        "VNr = "
        "external1d{neutral_r.dat}"
    ),
    (
        "VAr = "
        "external1d{anionic_r.dat}"
    ),
    (
        "VNz = "
        "external1d{neutral_z.dat}"
    ),
    (
        "VAz = "
        "external1d{anionic_z.dat}"
    ),
    (
        "fz = "
        "external1d{coupling_z.dat}"
    ),
]


# ============================================================
# System Hamiltonian
# ============================================================

hamiltonian_lines = []

hamiltonian_lines.extend(
    render_modes(
        all_mode_labels
    )
)

hamiltonian_lines.append("")


# ------------------------------------------------------------
# Nuclear kinetic energy
# ------------------------------------------------------------

hamiltonian_lines.extend(
    render_term(
        "1.0",
        [
            (
                R_MODE,
                "KE",
            )
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "1.0",
        [
            (
                Z_MODE,
                "KE",
            )
        ],
    )
)


# ------------------------------------------------------------
# Nuclear-only average surface
#
# 1/2 (U_N + U_A)
# ------------------------------------------------------------

hamiltonian_lines.extend(
    render_term(
        "0.5",
        [
            (
                R_MODE,
                "VNr",
            )
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "0.5",
        [
            (
                R_MODE,
                "VAr",
            )
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "0.5",
        [
            (
                Z_MODE,
                "VNz",
            )
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "0.5",
        [
            (
                Z_MODE,
                "VAz",
            )
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "cavg",
        [
            (
                Z_MODE,
                "1",
            )
        ],
    )
)


# ------------------------------------------------------------
# q_d (U_A - U_N)
# ------------------------------------------------------------

hamiltonian_lines.extend(
    render_term(
        "1.0",
        [
            (
                D_MODE,
                "q",
            ),
            (
                R_MODE,
                "VAr",
            ),
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "-1.0",
        [
            (
                D_MODE,
                "q",
            ),
            (
                R_MODE,
                "VNr",
            ),
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "1.0",
        [
            (
                D_MODE,
                "q",
            ),
            (
                Z_MODE,
                "VAz",
            ),
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "-1.0",
        [
            (
                D_MODE,
                "q",
            ),
            (
                Z_MODE,
                "VNz",
            ),
        ],
    )
)

hamiltonian_lines.extend(
    render_term(
        "cdiff",
        [
            (
                D_MODE,
                "q",
            )
        ],
    )
)


# ------------------------------------------------------------
# Metal orbital energies
#
# epsilon_k n_k
# =
# epsilon_k/2
# + epsilon_k q_k
# ------------------------------------------------------------

if abs(
    METAL_CONSTANT_HARTREE
) > 1.0e-15:

    hamiltonian_lines.extend(
        render_term(
            "metalconst",
            [
                (
                    Z_MODE,
                    "1",
                )
            ],
        )
    )


for k in range(
    1,
    config.n_metal_orbitals + 1,
):

    mode = (
        METAL_MODE_START
        + k
        - 1
    )

    hamiltonian_lines.extend(
        render_term(
            f"ec{k}",
            [
                (
                    mode,
                    "q",
                )
            ],
        )
    )


# ============================================================
# Molecule-metal hopping
# ============================================================
#
# Primitive spin representation:
#
#   sigma_x = 2 dq^2
#   sigma_y = -2 i dq
#   sigma_z = 2 q
#
# Following the same Jordan-Wigner convention already used
# in benchmark 2.
#
# Orbital order:
#
#     d, c1, c2, ..., cN
#
# For d <-> ck there are k-1 intermediate parity modes.
#
# Coefficient:
#
#     2 * (-2)^(k-1) * V_k
#
# and the common geometry factor fz operates on z.
# ============================================================

for k in range(
    1,
    config.n_metal_orbitals + 1,
):

    metal_mode = (
        METAL_MODE_START
        + k
        - 1
    )

    n_intermediate = (
        k - 1
    )

    jordan_wigner_factor = (
        2
        * (
            (-2)
            ** n_intermediate
        )
    )

    coefficient = (
        f"{float(jordan_wigner_factor):.1f}"
        f"*vk{k}"
    )


    # --------------------------------------------------------
    # sigma_x-like endpoint contribution
    # --------------------------------------------------------

    operators_x = [
        (
            D_MODE,
            "dq^2",
        )
    ]

    for intermediate_mode in range(
        D_MODE + 1,
        metal_mode,
    ):

        operators_x.append(
            (
                intermediate_mode,
                "q",
            )
        )

    operators_x.append(
        (
            metal_mode,
            "dq^2",
        )
    )

    operators_x.append(
        (
            Z_MODE,
            "fz",
        )
    )

    hamiltonian_lines.extend(
        render_term(
            coefficient,
            operators_x,
        )
    )


    # --------------------------------------------------------
    # sigma_y-like endpoint contribution
    # --------------------------------------------------------

    operators_y = [
        (
            D_MODE,
            "I*dq",
        )
    ]

    for intermediate_mode in range(
        D_MODE + 1,
        metal_mode,
    ):

        operators_y.append(
            (
                intermediate_mode,
                "q",
            )
        )

    operators_y.append(
        (
            metal_mode,
            "I*dq",
        )
    )

    operators_y.append(
        (
            Z_MODE,
            "fz",
        )
    )

    hamiltonian_lines.extend(
        render_term(
            coefficient,
            operators_y,
        )
    )


# ============================================================
# Initial r Hamiltonian
# ============================================================
#
# At z -> infinity, the neutral surface is
#
#     V_M(r-r0) + c0.
#
# c0 is a constant and therefore does not affect eigenvectors.
# We omit it when generating chi_nu(r).
# ============================================================

rinit_lines = [
    "modes | r",
    "",
    "1.0                  |1 KE",
    "1.0                  |1 VNr",
]


# ============================================================
# Observable sections
# ============================================================


def observable_section(
    name: str,
    terms: list[
        tuple[
            str,
            list[
                tuple[int, str]
            ],
        ]
    ],
) -> str:

    lines = [
        (
            "HAMILTONIAN-SECTION_"
            + name
        ),
        "",
    ]

    lines.extend(
        render_modes(
            all_mode_labels
        )
    )

    lines.append("")

    for (
        coefficient,
        operators,
    ) in terms:

        lines.extend(
            render_term(
                coefficient,
                operators,
            )
        )

    lines.extend(
        [
            "",
            (
                "end-HAMILTONIAN-SECTION"
            ),
        ]
    )

    return "\n".join(
        lines
    )


observable_sections = []


# ------------------------------------------------------------
# Molecular occupation
#
# n_d = 1/2 + q_d
# ------------------------------------------------------------

observable_sections.append(
    observable_section(
        "nd",
        [
            (
                "0.5",
                [
                    (
                        D_MODE,
                        "1",
                    )
                ],
            ),
            (
                "1.0",
                [
                    (
                        D_MODE,
                        "q",
                    )
                ],
            ),
        ],
    )
)


# ------------------------------------------------------------
# Metal occupations
# ------------------------------------------------------------

for k in range(
    1,
    config.n_metal_orbitals + 1,
):

    mode = (
        METAL_MODE_START
        + k
        - 1
    )

    observable_sections.append(
        observable_section(
            f"nc{k}",
            [
                (
                    "0.5",
                    [
                        (
                            mode,
                            "1",
                        )
                    ],
                ),
                (
                    "1.0",
                    [
                        (
                            mode,
                            "q",
                        )
                    ],
                ),
            ],
        )
    )


# ------------------------------------------------------------
# Mean nuclear coordinates
# ------------------------------------------------------------

observable_sections.append(
    observable_section(
        "rmean",
        [
            (
                "1.0",
                [
                    (
                        R_MODE,
                        "q",
                    )
                ],
            )
        ],
    )
)

observable_sections.append(
    observable_section(
        "zmean",
        [
            (
                "1.0",
                [
                    (
                        Z_MODE,
                        "q",
                    )
                ],
            )
        ],
    )
)


# ============================================================
# Assemble operator file
# ============================================================

text = f"""#######################################################################
### Benchmark 3: physical NO/Au(111) scattering
### Heidelberg ML-MCTDH/SQR operator
###
### Orbital ordering:
###
###   1       d
###   2..33   c1..c32
###   34      r
###   35      z
###
### Fermionic convention:
###
###   n = 1/2 + q
###   parity = (-1)^n = -2 q
###
### Nuclear potential functions are tabulated in atomic units and read
### as one-dimensional external operators.
#######################################################################


OP_DEFINE-SECTION

title
Physical 2D NO/Au(111) generalized Anderson-Newns Hamiltonian
end-title

end-OP_DEFINE-SECTION


#######################################################################
### Parameters
#######################################################################

PARAMETER-SECTION

{chr(10).join(parameter_lines)}

end-PARAMETER-SECTION


#######################################################################
### One-dimensional physical functions
#######################################################################

LABELS-SECTION

{chr(10).join(label_lines)}

end-LABELS-SECTION


#######################################################################
### System Hamiltonian
###
### H =
###     T_r + T_z
###   + U_N(r,z)
###   + n_d [U_A(r,z)-U_N(r,z)]
###   + sum_k epsilon_k n_k
###   + sum_k W_k(z)(d^dagger c_k + c_k^dagger d)
###
### Using n_d = 1/2 + q_d:
###
### U_N + n_d(U_A-U_N)
### =
### 1/2(U_N+U_A) + q_d(U_A-U_N)
#######################################################################

HAMILTONIAN-SECTION

{chr(10).join(hamiltonian_lines)}

end-HAMILTONIAN-SECTION


#######################################################################
### Asymptotic neutral NO Hamiltonian used to generate chi_nu(r)
#######################################################################

HAMILTONIAN-SECTION_rinit

{chr(10).join(rinit_lines)}

end-HAMILTONIAN-SECTION


#######################################################################
### Observables
#######################################################################

{chr(10).join(observable_sections)}


END-OPERATOR
"""


# ============================================================
# Sanity checks
# ============================================================

operator_lines = (
    text.splitlines()
)

longest_line = max(
    len(line)
    for line in operator_lines
)


if (
    longest_line
    > 240
):
    raise RuntimeError(
        "Generated operator contains a line "
        f"with {longest_line} characters."
    )


if (
    len(
        coupling_prefactors
    )
    != config.n_metal_orbitals
):
    raise RuntimeError(
        "Incorrect number of finite-bath couplings."
    )


if (
    R_MODE
    != 34
    or Z_MODE
    != 35
):
    raise RuntimeError(
        "Unexpected mode ordering for the 32-state bath."
    )


# ============================================================
# Write operator
# ============================================================

OPERATOR_PATH.write_text(
    text,
    encoding="utf-8",
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("NO/Au Heidelberg operator generated")
print("=" * 70)

print()

print(
    "Operator:",
    OPERATOR_PATH,
)

print()

print("Mode ordering")
print(
    "  d  ->",
    D_MODE,
)
print(
    "  c1 ->",
    METAL_MODE_START,
)
print(
    f"  c{config.n_metal_orbitals} ->",
    METAL_MODE_START
    + config.n_metal_orbitals
    - 1,
)
print(
    "  r  ->",
    R_MODE,
)
print(
    "  z  ->",
    Z_MODE,
)

print()

print("Nuclear masses")
print(
    "  mass_r =",
    mass_r_au,
    "electron masses",
)
print(
    "  mass_z =",
    mass_z_au,
    "electron masses",
)

print()

print("External nuclear operators")
print(
    "  neutral_r.dat"
)
print(
    "  anionic_r.dat"
)
print(
    "  neutral_z.dat"
)
print(
    "  anionic_z.dat"
)
print(
    "  coupling_z.dat"
)

print()

print("Bath")
print(
    "  metal states =",
    config.n_metal_orbitals,
)
print(
    "  hopping terms =",
    2
    * config.n_metal_orbitals,
)

print()

print(
    "metal constant shift =",
    METAL_CONSTANT_HARTREE,
    "Hartree",
)

print(
    "longest operator line =",
    longest_line,
)

print()

print(
    "benchmark.op generation: OK"
)