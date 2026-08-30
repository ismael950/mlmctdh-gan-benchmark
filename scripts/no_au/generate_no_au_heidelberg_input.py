from __future__ import annotations

from pathlib import Path

import numpy as np

from ganbench.model import load_config


# ============================================================
# Constants
# ============================================================

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


# ============================================================
# Benchmark
# ============================================================

BENCHMARK = "benchmark3_no_au_scattering"

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
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

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "benchmark.inp"
)


# ============================================================
# Initial ML ranks
# ============================================================

# These are intentionally modest starting ranks.
# They will later be increased by the adaptive convergence
# procedure.

ROOT_RANK = 4
METAL_RANK = 4
NUCLEAR_RANK = 4


# ============================================================
# Load model
# ============================================================

config = load_config(
    CONFIG_PATH
)

if (
    config.n_molecular_orbitals
    != 1
):
    raise ValueError(
        "This generator currently assumes exactly "
        "one molecular orbital."
    )

if (
    not config.uses_physical_nuclear_coordinates
):
    raise ValueError(
        "NO/Au must use physical nuclear coordinates."
    )


# ============================================================
# Nuclear coordinates
# ============================================================

coordinate_map = {
    coordinate.name: coordinate
    for coordinate
    in config.nuclear_coordinates
}

if (
    "r" not in coordinate_map
    or "z" not in coordinate_map
):
    raise ValueError(
        "The NO/Au benchmark requires coordinates r and z."
    )

r_coordinate = (
    coordinate_map["r"]
)

z_coordinate = (
    coordinate_map["z"]
)


# ============================================================
# Unit conversion
# ============================================================


def length_to_bohr(
    value: float,
    unit: str,
) -> float:

    unit = unit.lower()

    if unit == "angstrom":
        return (
            value
            * ANGSTROM_TO_BOHR
        )

    if unit in {
        "bohr",
        "au",
    }:
        return value

    raise ValueError(
        f"Unsupported length unit: {unit}"
    )


# ============================================================
# Physical primitive grids
# ============================================================

r_min_bohr = length_to_bohr(
    r_coordinate.basis.minimum,
    r_coordinate.basis.length_unit,
)

r_max_bohr = length_to_bohr(
    r_coordinate.basis.maximum,
    r_coordinate.basis.length_unit,
)

z_min_bohr = length_to_bohr(
    z_coordinate.basis.minimum,
    z_coordinate.basis.length_unit,
)

z_max_bohr = length_to_bohr(
    z_coordinate.basis.maximum,
    z_coordinate.basis.length_unit,
)


# ============================================================
# Initial nuclear state
# ============================================================

if (
    r_coordinate.initial_state.kind
    != "neutral_pes_eigenstate"
):
    raise ValueError(
        "The r initial state must be "
        "neutral_pes_eigenstate."
    )

nu_ini = int(
    r_coordinate.initial_state.parameters[
        "level"
    ]
)

# Heidelberg eigenf uses 1-based state counting:
#
# pop=1 -> ground state
# pop=17 -> nu=16

r_eigenf_pop = (
    nu_ini + 1
)


if (
    z_coordinate.initial_state.kind
    != "gaussian"
):
    raise ValueError(
        "The z initial state must be gaussian."
    )

z_initial = (
    z_coordinate.initial_state.parameters
)

z_center_bohr = (
    z_initial[
        "center_angstrom"
    ]
    * ANGSTROM_TO_BOHR
)

z_momentum_au = (
    z_initial[
        "momentum_au"
    ]
)

z_sigma_bohr = (
    z_initial[
        "sigma_bohr"
    ]
)


# ============================================================
# Electronic labels
# ============================================================

molecular_label = "d"

metal_labels = [
    f"c{index}"
    for index in range(
        1,
        config.n_metal_orbitals + 1,
    )
]


# ============================================================
# Split metal orbitals around the Fermi level
# ============================================================

occupied_metal_labels = []

empty_metal_labels = []

for (
    index,
    energy,
) in enumerate(
    config.metal_energies
):

    label = metal_labels[
        index
    ]

    if energy < 0.0:
        occupied_metal_labels.append(
            label
        )

    elif energy > 0.0:
        empty_metal_labels.append(
            label
        )

    else:
        raise ValueError(
            "A finite metal orbital lies exactly "
            "at the chemical potential."
        )


if (
    not occupied_metal_labels
    or not empty_metal_labels
):
    raise ValueError(
        "The metal bath must contain both occupied "
        "and empty sectors."
    )


# ============================================================
# Balanced metal trees
# ============================================================


def build_binary_tree(
    labels: list[str],
):
    """
    Recursively build a balanced binary tree.

    At the bottom, up to two fermionic orbitals are
    grouped into one primitive SQR mode.
    """

    if len(labels) <= 2:
        return (
            "leaf",
            labels,
        )

    midpoint = (
        len(labels)
        // 2
    )

    return (
        "node",
        build_binary_tree(
            labels[
                :midpoint
            ]
        ),
        build_binary_tree(
            labels[
                midpoint:
            ]
        ),
    )


def child_rank(
    tree,
) -> int:
    """
    Rank assigned to a child branch.

    A primitive group containing:
        1 fermion -> dimension 2
        2 fermions -> dimension 4
    """

    if tree[0] == "leaf":

        n_orbitals = len(
            tree[1]
        )

        return (
            2 ** n_orbitals
        )

    return METAL_RANK


def render_tree(
    tree,
    layer: int,
    indent: str,
) -> list[str]:

    if tree[0] == "leaf":

        labels = " ".join(
            tree[1]
        )

        return [
            f"{indent}{layer}> [{labels}]"
        ]

    left = tree[1]
    right = tree[2]

    left_rank = child_rank(
        left
    )

    right_rank = child_rank(
        right
    )

    lines = [
        (
            f"{indent}{layer}> "
            f"{left_rank} {right_rank}"
        )
    ]

    lines.extend(
        render_tree(
            left,
            layer + 1,
            indent + "    ",
        )
    )

    lines.extend(
        render_tree(
            right,
            layer + 1,
            indent + "    ",
        )
    )

    return lines


occupied_tree = build_binary_tree(
    occupied_metal_labels
)

empty_tree = build_binary_tree(
    empty_metal_labels
)


# ============================================================
# ML-BASIS section
# ============================================================

ml_lines = []

# ------------------------------------------------------------
# ROOT
#
#              ROOT
#             /    \
#       ELECTRONIC NUCLEAR
# ------------------------------------------------------------

ml_lines.append(
    (
        f"0> {ROOT_RANK} "
        f"{ROOT_RANK}"
    )
)

ml_lines.append("")

# ------------------------------------------------------------
# Electronic node
#
#          ELECTRONIC
#         /    |     \
#        d   METAL- METAL+
# ------------------------------------------------------------

ml_lines.append(
    (
        f"    1> 2 "
        f"{METAL_RANK} "
        f"{METAL_RANK}"
    )
)

# Molecular orbital

ml_lines.append(
    "        2> [d]"
)

# Occupied metal sector

ml_lines.extend(
    render_tree(
        occupied_tree,
        layer=2,
        indent="        ",
    )
)

# Empty metal sector

ml_lines.extend(
    render_tree(
        empty_tree,
        layer=2,
        indent="        ",
    )
)

ml_lines.append("")

# ------------------------------------------------------------
# Nuclear node
#
#            NUCLEAR
#             /  \
#            r    z
# ------------------------------------------------------------

ml_lines.append(
    (
        f"    1> "
        f"{NUCLEAR_RANK} "
        f"{NUCLEAR_RANK}"
    )
)

ml_lines.append(
    "        2> [r]"
)

ml_lines.append(
    "        2> [z]"
)


# ============================================================
# Initial electronic state
# ============================================================

occupied_global_indices = set(
    config.occupied_orbitals
)

electronic_init_lines = []

# Global index 0 = molecular d

d_state = (
    2
    if 0 in occupied_global_indices
    else 1
)

electronic_init_lines.append(
    f"d      euclid    {d_state}"
)


# Global indices:
#
# c1 -> 1
# c2 -> 2
# ...

for (
    metal_index,
    label,
) in enumerate(
    metal_labels,
    start=1,
):

    state = (
        2
        if metal_index
        in occupied_global_indices
        else 1
    )

    electronic_init_lines.append(
        (
            f"{label:<6} "
            f"euclid    {state}"
        )
    )


# ============================================================
# Primitive basis
# ============================================================

primitive_lines = []

primitive_lines.append(
    "d      sin      2     -0.5     0.5     spin"
)

for label in metal_labels:

    primitive_lines.append(
        (
            f"{label:<6} "
            "sin      2     -0.5     0.5     spin"
        )
    )

primitive_lines.append("")

# SHORT is deliberate:
#
# xi and xf are interpreted by Heidelberg as the
# first and last grid points of the sine-DVR.

primitive_lines.append(
    (
        f"r      sin      "
        f"{r_coordinate.basis.size:<5} "
        f"{r_min_bohr:.12f} "
        f"{r_max_bohr:.12f} "
        "short"
    )
)

primitive_lines.append(
    (
        f"z      sin      "
        f"{z_coordinate.basis.size:<5} "
        f"{z_min_bohr:.12f} "
        f"{z_max_bohr:.12f} "
        "short"
    )
)


# ============================================================
# Expectation values
# ============================================================

expectation_labels = [
    "nd",
]

expectation_labels.extend(
    f"nc{index}"
    for index in range(
        1,
        config.n_metal_orbitals + 1,
    )
)

expectation_labels.extend(
    [
        "rmean",
        "zmean",
    ]
)

expectation_line = (
    "expect = real-only, "
    + ", ".join(
        expectation_labels
    )
)


# ============================================================
# Time grid
# ============================================================

tout = (
    config.t_final
    / (
        config.n_times
        - 1
    )
)

# We only need the final psi file initially.
# Natural populations and expectation values are printed
# independently during propagation.

tpsi = (
    config.t_final
)


# ============================================================
# Complete Heidelberg input
# ============================================================

text = f"""#######################################################################
### Benchmark 3: physical NO/Au(111) scattering
### ML-MCTDH/SQR
###
### Nuclear coordinates:
###   r = N-O stretch
###   z = molecule-surface translation
###
### Electronic representation:
###   d     = active molecular orbital
###   c1... = explicit finite Au bath
#######################################################################

RUN-SECTION

name = ../../../../results/{BENCHMARK}/heidelberg/run_001/raw

propagate

time-not-fs
energy-not-eV

tout   = {tout:.15g}
tpsi   = {tpsi:.15g}
tfinal = {config.t_final:.15g}

steps
psi = double

{expectation_line}

end-run-section


#######################################################################
### Operator
#######################################################################

OPERATOR-SECTION

opname = benchmark
oppath = .

end-operator-section


#######################################################################
### Primitive basis
#######################################################################

PBASIS-SECTION

{chr(10).join(primitive_lines)}

end-PBASIS-SECTION


#######################################################################
### ML-MCTDH tree
###
###                         ROOT
###                      /        \\
###               ELECTRONIC      NUCLEAR
###              /    |    \\       /   \\
###             d   METAL- METAL+  r     z
###
### Metal states are sorted by energy.
### METAL- contains epsilon_k < mu.
### METAL+ contains epsilon_k > mu.
###
### The metal sectors are recursively split into balanced
### binary energy-ordered subtrees.
###
### The topology remains fixed during adaptive convergence.
### Only ML ranks are increased.
#######################################################################

ML-BASIS-SECTION

{chr(10).join(ml_lines)}

end-ML-BASIS-SECTION


#######################################################################
### Integrator
#######################################################################

INTEGRATOR-SECTION

VMF
RK8 = 1.0d-8, 1.0d-5

end-INTEGRATOR-SECTION


#######################################################################
### Initial state
###
### Electronic:
###   neutral molecule: d empty
###   T=0 finite Fermi sea in the metal
###
### Nuclear:
###   r = nu={nu_ini} eigenstate of asymptotic neutral NO
###   z = incoming Gaussian scattering packet
#######################################################################

INIT_WF-SECTION

build

{chr(10).join(electronic_init_lines)}

r      eigenf    rinit    pop={r_eigenf_pop}

z      gauss     {z_center_bohr:.15g}    {z_momentum_au:.15g}    {z_sigma_bohr:.15g}

end-build

end-INIT_WF-SECTION


END-INPUT
"""


# ============================================================
# Basic checks
# ============================================================

if (
    len(
        occupied_metal_labels
    )
    + len(
        empty_metal_labels
    )
    != config.n_metal_orbitals
):
    raise RuntimeError(
        "Metal tree does not contain every metal orbital."
    )


# Heidelberg operator lines have strict line-size limits.
# Keeping the input lines short is also good practice.

longest_line = max(
    len(line)
    for line in text.splitlines()
)

if longest_line > 240:
    raise RuntimeError(
        f"Generated input contains a line of "
        f"{longest_line} characters."
    )


# ============================================================
# Write
# ============================================================

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    text,
    encoding="utf-8",
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 68)
print("NO/Au Heidelberg input generated")
print("=" * 68)

print()
print(
    "Input:",
    OUTPUT_PATH,
)

print()
print("Electronic tree")

print(
    "  Molecular:",
    molecular_label,
)

print(
    "  METAL-:",
    len(
        occupied_metal_labels
    ),
    "orbitals",
)

print(
    "  METAL+:",
    len(
        empty_metal_labels
    ),
    "orbitals",
)

print()
print("Nuclear primitive basis")

print(
    f"  r: sin "
    f"{r_coordinate.basis.size} "
    f"[{r_coordinate.basis.minimum}, "
    f"{r_coordinate.basis.maximum}] "
    f"{r_coordinate.basis.length_unit}"
)

print(
    f"  z: sin "
    f"{z_coordinate.basis.size} "
    f"[{z_coordinate.basis.minimum}, "
    f"{z_coordinate.basis.maximum}] "
    f"{z_coordinate.basis.length_unit}"
)

print()
print("Initial nuclear state")

print(
    "  r eigenstate nu =",
    nu_ini,
)

print(
    "  Heidelberg eigenf pop =",
    r_eigenf_pop,
)

print(
    "  z center =",
    z_center_bohr,
    "bohr",
)

print(
    "  z momentum =",
    z_momentum_au,
    "au",
)

print(
    "  z sigma =",
    z_sigma_bohr,
    "bohr",
)

print()
print(
    "tout =",
    tout,
    "au",
)

print(
    "tfinal =",
    config.t_final,
    "au",
)

print()
print(
    "Longest input line:",
    longest_line,
)

print()
print(
    "benchmark.inp generation: OK"
)