from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.linalg import eigh


# ============================================================
# Constants
# ============================================================

EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_EV = 27.211386245988

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
AMU_TO_ELECTRON_MASS = 1822.888486209


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "benchmark3_no_au_scattering.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmark3_no_au_scattering"
    / "inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load YAML
# ============================================================

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    raw = yaml.safe_load(file)


params = raw["physical_parameters"]

neutral = params["neutral_surface"]
anion = params["anionic_surface"]
mol_metal = params["molecule_metal"]

coordinates = raw["nuclear"]["coordinates"]

r_spec = coordinates[0]
z_spec = coordinates[1]


# ============================================================
# Literature parameters
# ============================================================

r0 = neutral["r0_angstrom"]
a0 = neutral["a0_inv_angstrom"]
D0 = neutral["D0_ev"]

b0 = neutral["b0_inv_angstrom"]
z0 = neutral["z0_angstrom"]
c0 = neutral["c0_ev"]

r1 = anion["r1_angstrom"]
a1 = anion["a1_inv_angstrom"]
D1 = anion["D1_ev"]

z1 = anion["z1_angstrom"]
a2 = anion["a2_inv_angstrom"]
D2 = anion["D2_ev"]

c1 = anion["c1_ev"]

gamma = mol_metal["gamma_ev"]
a_tilde = mol_metal["a_tilde_angstrom"]


# ============================================================
# PES definitions
# ============================================================


def literature_morse(
    x,
    depth,
    alpha,
):
    """
    Literature convention:

        V_M(x)
        =
        D [exp(-2 a x) - 2 exp(-a x)]

    Minimum = -D.
    """

    return depth * (
        np.exp(
            -2.0 * alpha * x
        )
        - 2.0
        * np.exp(
            -alpha * x
        )
    )


def U0(
    r,
    z,
):
    """
    Neutral diabatic surface.
    """

    return (
        literature_morse(
            r - r0,
            D0,
            a0,
        )
        + np.exp(
            -b0 * (z - z0)
        )
        + c0
    )


def U1(
    r,
    z,
):
    """
    Anionic diabatic surface.
    """

    return (
        literature_morse(
            r - r1,
            D1,
            a1,
        )
        + literature_morse(
            z - z1,
            D2,
            a2,
        )
        + c1
    )


def coupling_geometry_factor(
    z,
):
    """
    Common geometry dependence of W_1k(z):

        1 - tanh(z / a_tilde)
    """

    return (
        1.0
        - np.tanh(
            z / a_tilde
        )
    )


# ============================================================
# Plot grids
# ============================================================

r_plot = np.linspace(
    r_spec["basis"]["minimum"],
    r_spec["basis"]["maximum"],
    300,
)

z_plot = np.linspace(
    z_spec["basis"]["minimum"],
    z_spec["basis"]["maximum"],
    300,
)

R, Z = np.meshgrid(
    r_plot,
    z_plot,
    indexing="ij",
)

U0_grid = U0(
    R,
    Z,
)

U1_grid = U1(
    R,
    Z,
)


# ============================================================
# Neutral PES
# ============================================================

plt.figure(
    figsize=(7, 5)
)

contour = plt.contourf(
    Z,
    R,
    U0_grid,
    levels=60,
)

plt.colorbar(
    contour,
    label="Energy (eV)",
)

plt.xlabel(
    r"$z$ ($\AA$)"
)

plt.ylabel(
    r"$r$ ($\AA$)"
)

plt.title(
    "Neutral diabatic surface $U_0(r,z)$"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "neutral_pes.png",
    dpi=200,
)

plt.close()


# ============================================================
# Anionic PES
# ============================================================

plt.figure(
    figsize=(7, 5)
)

contour = plt.contourf(
    Z,
    R,
    U1_grid,
    levels=60,
)

plt.colorbar(
    contour,
    label="Energy (eV)",
)

plt.xlabel(
    r"$z$ ($\AA$)"
)

plt.ylabel(
    r"$r$ ($\AA$)"
)

plt.title(
    "Anionic diabatic surface $U_1(r,z)$"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "anionic_pes.png",
    dpi=200,
)

plt.close()


# ============================================================
# Difference U1 - U0
#
# This is precisely U_11(r,z) in GAN notation.
# ============================================================

plt.figure(
    figsize=(7, 5)
)

difference = (
    U1_grid
    - U0_grid
)

contour = plt.contourf(
    Z,
    R,
    difference,
    levels=60,
)

plt.colorbar(
    contour,
    label="Energy (eV)",
)

plt.xlabel(
    r"$z$ ($\AA$)"
)

plt.ylabel(
    r"$r$ ($\AA$)"
)

plt.title(
    r"$U_{11}(r,z)=U_1(r,z)-U_0(r,z)$"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "u11_difference.png",
    dpi=200,
)

plt.close()


# ============================================================
# Sine-DVR helper
# ============================================================


def sine_dvr_grid(
    minimum_angstrom: float,
    maximum_angstrom: float,
    n_points: int,
):
    """
    Interior sine-DVR points for a box with boundaries
    minimum and maximum.
    """

    length = (
        maximum_angstrom
        - minimum_angstrom
    )

    indices = np.arange(
        1,
        n_points + 1,
    )

    return (
        minimum_angstrom
        + indices
        * length
        / (n_points + 1)
    )


def sine_dvr_kinetic(
    minimum_angstrom: float,
    maximum_angstrom: float,
    n_points: int,
    mass_amu: float,
):
    """
    Construct the kinetic-energy matrix in a sine-DVR.

    Atomic units are used internally.
    """

    minimum_bohr = (
        minimum_angstrom
        * ANGSTROM_TO_BOHR
    )

    maximum_bohr = (
        maximum_angstrom
        * ANGSTROM_TO_BOHR
    )

    length_bohr = (
        maximum_bohr
        - minimum_bohr
    )

    mass_au = (
        mass_amu
        * AMU_TO_ELECTRON_MASS
    )

    i = np.arange(
        1,
        n_points + 1,
    )[:, None]

    n = np.arange(
        1,
        n_points + 1,
    )[None, :]

    transform = np.sqrt(
        2.0
        / (n_points + 1)
    ) * np.sin(
        np.pi
        * i
        * n
        / (n_points + 1)
    )

    quantum_numbers = np.arange(
        1,
        n_points + 1,
    )

    kinetic_eigenvalues = (
        (
            quantum_numbers
            * np.pi
            / length_bohr
        ) ** 2
        / (
            2.0
            * mass_au
        )
    )

    kinetic = (
        transform
        @ np.diag(
            kinetic_eigenvalues
        )
        @ transform.T
    )

    return kinetic


# ============================================================
# Compute chi_16(r)
# ============================================================

r_min = (
    r_spec["basis"]["minimum"]
)

r_max = (
    r_spec["basis"]["maximum"]
)

n_r = (
    r_spec["basis"]["size"]
)

mass_r = (
    r_spec["mass_amu"]
)

r_dvr = sine_dvr_grid(
    r_min,
    r_max,
    n_r,
)

T_r = sine_dvr_kinetic(
    r_min,
    r_max,
    n_r,
    mass_r,
)


# ------------------------------------------------------------
# Isolated neutral NO potential
#
# z -> infinity:
#
# U0(r,infinity)
# =
# V_M(r-r0) + c0
#
# Constant c0 does not affect eigenvectors.
# We shift the Morse minimum to zero for clarity.
# ------------------------------------------------------------

V_r_ev = (
    D0
    * (
        1.0
        - np.exp(
            -a0
            * (
                r_dvr
                - r0
            )
        )
    ) ** 2
)

V_r_au = (
    V_r_ev
    * EV_TO_HARTREE
)

H_r = (
    T_r
    + np.diag(
        V_r_au
    )
)

energies_au, states = eigh(
    H_r
)

energies_ev = (
    energies_au
    * HARTREE_TO_EV
)

nu_ini = int(
    r_spec[
        "initial_state"
    ][
        "level"
    ]
)

chi = states[
    :,
    nu_ini,
]


# Convert discrete normalization into an approximate
# continuous probability density.

delta_r = (
    (r_max - r_min)
    / (n_r + 1)
)

chi_probability_density = (
    np.abs(
        chi
    ) ** 2
    / delta_r
)


# ============================================================
# Plot chi_16
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    r_dvr,
    chi_probability_density,
)

plt.xlabel(
    r"$r$ ($\AA$)"
)

plt.ylabel(
    r"$|\chi_{16}(r)|^2$"
)

plt.title(
    "Initial N-O vibrational state"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "chi16.png",
    dpi=200,
)

plt.close()


# ============================================================
# Initial Gaussian in z
# ============================================================

z_min = (
    z_spec["basis"]["minimum"]
)

z_max = (
    z_spec["basis"]["maximum"]
)

n_z = (
    z_spec["basis"]["size"]
)

z_dvr = sine_dvr_grid(
    z_min,
    z_max,
    n_z,
)

initial_z = (
    z_spec[
        "initial_state"
    ]
)

z_center = (
    initial_z[
        "center_angstrom"
    ]
)

sigma = (
    initial_z[
        "sigma_angstrom"
    ]
)

momentum_au = (
    initial_z[
        "momentum_au"
    ]
)

z_bohr = (
    z_dvr
    * ANGSTROM_TO_BOHR
)

z_center_bohr = (
    z_center
    * ANGSTROM_TO_BOHR
)

sigma_bohr = (
    sigma
    * ANGSTROM_TO_BOHR
)

gaussian = np.exp(
    -(
        z_bohr
        - z_center_bohr
    ) ** 2
    / (
        4.0
        * sigma_bohr**2
    )
    + 1j
    * momentum_au
    * (
        z_bohr
        - z_center_bohr
    )
)

delta_z_bohr = (
    (
        z_max
        - z_min
    )
    * ANGSTROM_TO_BOHR
    / (n_z + 1)
)

norm = np.sqrt(
    np.sum(
        np.abs(
            gaussian
        ) ** 2
    )
    * delta_z_bohr
)

gaussian /= norm

gaussian_density = (
    np.abs(
        gaussian
    ) ** 2
)


# ============================================================
# Plot initial translational density
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    z_dvr,
    gaussian_density,
)

plt.axvline(
    z_center,
    linestyle="--",
)

plt.xlabel(
    r"$z$ ($\AA$)"
)

plt.ylabel(
    r"$|\Phi(z)|^2$"
)

plt.title(
    "Initial translational wavepacket"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "initial_z_wavepacket.png",
    dpi=200,
)

plt.close()


# ============================================================
# Geometry dependence of molecule-metal coupling
# ============================================================

coupling_factor = (
    coupling_geometry_factor(
        z_plot
    )
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    z_plot,
    coupling_factor,
)

plt.xlabel(
    r"$z$ ($\AA$)"
)

plt.ylabel(
    r"$1-\tanh(z/\tilde a)$"
)

plt.title(
    "Geometry dependence of molecule-metal coupling"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "coupling_vs_z.png",
    dpi=200,
)

plt.close()


# ============================================================
# Metal spectrum
# ============================================================

metal_energies_au = np.asarray(
    raw[
        "electronic"
    ][
        "metal_energies"
    ],
    dtype=float,
)

metal_energies_ev = (
    metal_energies_au
    * HARTREE_TO_EV
)

orbital_indices = np.arange(
    1,
    len(
        metal_energies_ev
    )
    + 1,
)

plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    orbital_indices,
    metal_energies_ev,
)

plt.axhline(
    0.0,
    linestyle="--",
)

plt.xlabel(
    "Metal orbital index"
)

plt.ylabel(
    r"$\epsilon_k$ (eV)"
)

plt.title(
    "Finite explicit Au bath"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "metal_spectrum.png",
    dpi=200,
)

plt.close()


# ============================================================
# Numerical checks
# ============================================================

left_r_probability = np.sum(
    np.abs(
        chi[:5]
    ) ** 2
)

right_r_probability = np.sum(
    np.abs(
        chi[-5:]
    ) ** 2
)

left_z_probability = np.sum(
    np.abs(
        gaussian[:5]
    ) ** 2
) * delta_z_bohr

right_z_probability = np.sum(
    np.abs(
        gaussian[-5:]
    ) ** 2
) * delta_z_bohr


# ============================================================
# Summary
# ============================================================

print()
print("=" * 65)
print("NO/Au physical inspection")
print("=" * 65)

print()

print("Vibrational state")
print(
    "  nu_ini:",
    nu_ini,
)

print(
    "  E_nu relative to Morse minimum:",
    energies_ev[
        nu_ini
    ],
    "eV",
)

print(
    "  probability in first 5 r points:",
    left_r_probability,
)

print(
    "  probability in last 5 r points:",
    right_r_probability,
)

print()

print("Translational wavepacket")

print(
    "  center:",
    z_center,
    "Angstrom",
)

print(
    "  sigma:",
    sigma,
    "Angstrom",
)

print(
    "  probability in first 5 z points:",
    left_z_probability,
)

print(
    "  probability in last 5 z points:",
    right_z_probability,
)

print()

print("Diabatic surfaces")

print(
    "  minimum U0 on plotting grid:",
    np.min(
        U0_grid
    ),
    "eV",
)

print(
    "  minimum U1 on plotting grid:",
    np.min(
        U1_grid
    ),
    "eV",
)

print()

print("Metal bath")

print(
    "  number of metal states:",
    len(
        metal_energies_ev
    ),
)

print(
    "  states below EF:",
    np.sum(
        metal_energies_ev
        < 0.0
    ),
)

print(
    "  states above EF:",
    np.sum(
        metal_energies_ev
        > 0.0
    ),
)

print()

print(
    "Plots saved to:",
    OUTPUT_DIR,
)

print()

print("Inspection completed.")