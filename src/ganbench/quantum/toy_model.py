from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ganbench.quantum.fermions import (
    hopping_operator,
    number_operator,
)
from ganbench.quantum.matchings import (
    Matching,
    gan_matchings,
)
from ganbench.nuclear_grid import (
    PeriodicNuclearGrid,
    build_periodic_nuclear_grid,
)
from ganbench.quantum.space import (
    lift_electronic,
    lift_nuclear,
    coupled_operator,
)
from ganbench.quantum.diagonal_functions import (
    DiagonalFunctionExpansion,
    build_f0_function_expansion,
)

def molecule_metal_switching_profile(
    points: np.ndarray,
    center: float = 0.0,
    width: float = 1.0,
) -> np.ndarray:
    """
    Smooth molecule-metal coupling profile.

    The coupling is stronger for smaller Q
    (molecule closer to the metal) and weaker
    for larger Q.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")

    points = np.asarray(points, dtype=float)

    return 0.5 * (
        1.0
        - np.tanh(
            (points - center) / width
        )
    )

@dataclass(frozen=True)
class QuantumToyGAN:
    hamiltonian: np.ndarray

    f0: np.ndarray
    hopping_fragments: tuple[np.ndarray, ...]
    f_last: np.ndarray

    nuclear_grid: PeriodicNuclearGrid

    n_electronic_orbitals: int
    electronic_dimension: int
    nuclear_dimension: int

    molecular_energies: dict[int, float]
    linear_couplings: dict[int, float]
    u0_quadratic: float

    matchings: tuple[Matching, ...]
    hopping_couplings: dict[tuple[int, int], float]
    hopping_profiles: dict[tuple[int, int], np.ndarray]

    f0_expansion: DiagonalFunctionExpansion

    metal_energies: dict[int, float]
    @property
    def total_dimension(self) -> int:
        return self.hamiltonian.shape[0]

def build_quantum_toy_gan() -> QuantumToyGAN:
    """
    Small GAN model derived from Benchmark 2.

    Electronic orbitals:
        0,1     molecular
        2,3,4,5 metal

    Nuclear grid:
        8 points = 3 nuclear qubits.

    Total:
        6 electronic qubits + 3 nuclear qubits = 9 qubits.
    """

    n_molecular = 2
    n_metal = 4
    n_orbitals = n_molecular + n_metal

    electronic_dimension = 2 ** n_orbitals

    # -------------------------------------------------
    # Nuclear grid
    # -------------------------------------------------

    omega = 0.0002

    grid = build_periodic_nuclear_grid(
        size=8,
        mass=1.0 / omega,
    )

    nuclear_dimension = grid.size

    identity_el = np.eye(
        electronic_dimension,
        dtype=complex,
    )

    identity_nuc = np.eye(
        nuclear_dimension,
        dtype=complex,
    )

    # -------------------------------------------------
    # Electronic occupation operators
    # -------------------------------------------------

    occupations = [
        number_operator(i, n_orbitals)
        for i in range(n_orbitals)
    ]

    # -------------------------------------------------
    # F0
    #
    # U0(Q) = 0.0001 Q^2
    #
    # g_00(Q) = 0.003 - 0.0012 Q
    # g_11(Q) = 0.005 - 0.0008 Q
    # -------------------------------------------------

    q = grid.coordinate
    q2 = q @ q

    f0 = (
        0.0001
        * lift_nuclear(
            q2,
            electronic_dimension,
        )
    )

    molecular_energies = {
        0: 0.003,
        1: 0.005,
    }

    linear_couplings = {
        0: -0.0012,
        1: -0.0008,
    }

    f0_expansion = build_f0_function_expansion(
        points=grid.points,
        n_orbitals=n_orbitals,
        molecular_energies=molecular_energies,
        linear_couplings=linear_couplings,
        u0_quadratic=0.0001,
    )

    for i in range(n_molecular):
        f0 += (
            molecular_energies[i]
            * lift_electronic(
                occupations[i],
                nuclear_dimension,
            )
        )

        f0 += (
            linear_couplings[i]
            * coupled_operator(
                occupations[i],
                q,
            )
        )

    # -------------------------------------------------
    # Hopping fragments
    # -------------------------------------------------

    matchings = gan_matchings(
        n_molecular,
        n_metal,
    )

    couplings: dict[tuple[int, int], float] = {
        # Small molecular-molecular hopping added
        # specifically to exercise every GAN matching.
        (0, 1): 0.00015,
    }

    for j in range(2, 6):
        couplings[(0, j)] = 0.00025
        couplings[(1, j)] = 0.00020

    switching = molecule_metal_switching_profile(
        grid.points,
    )

    hopping_profiles: dict[
        tuple[int, int],
        np.ndarray,
    ] = {
        # Molecular-molecular hopping remains constant.
        (0, 1): np.full(
            nuclear_dimension,
            couplings[(0, 1)],
            dtype=float,
        ),
    }

    for j in range(2, 6):
        hopping_profiles[(0, j)] = (
            couplings[(0, j)] * switching
        )
        hopping_profiles[(1, j)] = (
            couplings[(1, j)] * switching
        )

    hopping_fragments = []

    for matching in matchings:
        fragment = np.zeros(
            (
                electronic_dimension
                * nuclear_dimension,
                electronic_dimension
                * nuclear_dimension,
            ),
            dtype=complex,
        )

        for i, j in matching:
            edge = (
                min(i, j),
                max(i, j),
            )

            fragment += coupled_operator(
                hopping_operator(
                    i,
                    j,
                    n_orbitals,
                ),
                np.diag(
                    hopping_profiles[edge]
                ),
            )

        hopping_fragments.append(fragment)

    # -------------------------------------------------
    # Final fragment:
    #
    # T_nuc + sum_k epsilon_k n_k
    # -------------------------------------------------

    f_last = lift_nuclear(
        grid.kinetic,
        electronic_dimension,
    )

    metal_energies = {
        2: -0.003,
        3: -0.001,
        4: 0.001,
        5: 0.003,
    }

    for i, energy in metal_energies.items():
        f_last += (
            energy
            * lift_electronic(
                occupations[i],
                nuclear_dimension,
            )
        )

    # -------------------------------------------------
    # Complete Hamiltonian
    # -------------------------------------------------

    hamiltonian = f0.copy()

    for fragment in hopping_fragments:
        hamiltonian += fragment

    hamiltonian += f_last

    return QuantumToyGAN(
        hamiltonian=hamiltonian,
        f0=f0,
        hopping_fragments=tuple(
            hopping_fragments
        ),
        f_last=f_last,
        nuclear_grid=grid,
        n_electronic_orbitals=n_orbitals,
        electronic_dimension=electronic_dimension,
        nuclear_dimension=nuclear_dimension,
        molecular_energies=molecular_energies,
        linear_couplings=linear_couplings,
        u0_quadratic=0.0001,
        matchings=matchings,
        hopping_couplings=couplings,
        hopping_profiles=hopping_profiles,
        f0_expansion=f0_expansion,
        metal_energies=metal_energies,
    )