from __future__ import annotations

from dataclasses import dataclass
from math import comb
from functools import reduce
from operator import mul

from ganbench.model import GANConfig


@dataclass(frozen=True)
class ResourceEstimate:
    """Basic size estimates for one GAN configuration."""

    electronic_dimension: int
    vibrational_dimension: int
    total_dimension: int
    trajectory_memory_bytes: int

    state_memory_bytes: int
    dense_hamiltonian_memory_bytes: int

    @property
    def state_memory_mb(self) -> float:
        
        return self.state_memory_bytes / (1024**2)

    @property
    def dense_hamiltonian_memory_mb(self) -> float:
        return (
            self.dense_hamiltonian_memory_bytes
            / (1024**2)
        )

    @property
    def dense_hamiltonian_memory_gb(self) -> float:
        return (
            self.dense_hamiltonian_memory_bytes
            / (1024**3)
        )

    @property
    def trajectory_memory_mb(self) -> float:
        return (
            self.trajectory_memory_bytes
            / (1024**2)
        )

    @property
    def trajectory_memory_gb(self) -> float:
        return (
            self.trajectory_memory_bytes
            / (1024**3)
        )


def estimate_resources(
    config: GANConfig,
) -> ResourceEstimate:
    """
    Estimate the Hilbert-space dimensions and
    basic memory requirements.

    Assumes complex128 numbers:
        16 bytes per complex value.
    """

    electronic_dimension = comb(
        config.n_orbitals,
        config.n_electrons,
    )

    vibrational_dimension = int(
        reduce(
            mul,
            config.basis_sizes,
            1,
        )
    )

    total_dimension = (
        electronic_dimension
        * vibrational_dimension
    )

    bytes_per_complex = 16

    state_memory_bytes = (
        total_dimension
        * bytes_per_complex
    )

    dense_hamiltonian_memory_bytes = (
        total_dimension
        * total_dimension
        * bytes_per_complex
    )

    trajectory_memory_bytes = (
        config.n_times
        * total_dimension
        * bytes_per_complex
    )

    return ResourceEstimate(
        electronic_dimension=electronic_dimension,
        vibrational_dimension=vibrational_dimension,
        total_dimension=total_dimension,
        state_memory_bytes=state_memory_bytes,
        trajectory_memory_bytes=trajectory_memory_bytes,
        dense_hamiltonian_memory_bytes=(
            dense_hamiltonian_memory_bytes
        ),
    )