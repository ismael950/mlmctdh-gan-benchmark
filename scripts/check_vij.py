import numpy as np
from scipy import sparse

from ganbench.model import load_config
from ganbench.exact.hamiltonian import (
    build_exact_hamiltonian,
)


reference_config = load_config(
    "configs/test_vij_reference.yaml"
)

vij_config = load_config(
    "configs/test_vij.yaml"
)

reference_system = build_exact_hamiltonian(
    reference_config
)

vij_system = build_exact_hamiltonian(
    vij_config
)

# Actual Hamiltonian change
delta_h = (
    vij_system.hamiltonian
    - reference_system.hamiltonian
)

# Expected electronic operator n_0 n_1
n0 = vij_system.electronic_number_operators[0]
n1 = vij_system.electronic_number_operators[1]

electronic_expected = (
    n0 @ n1
).tocsr()

# Constant Vij means identity in vibrational space
vibrational_identity = sparse.identity(
    vij_system.vibrational_dimension,
    dtype=complex,
    format="csr",
)

expected_delta_h = (
    0.7
    * sparse.kron(
        electronic_expected,
        vibrational_identity,
        format="csr",
    )
)

error = sparse.linalg.norm(
    delta_h - expected_delta_h
)

print("Vij test")
print("Electronic basis:")
for index, state in enumerate(
    vij_system.electronic_basis
):
    print(
        f"{index}: |{''.join(map(str, state))}>"
    )

print()
print(
    "Expected occupied-pair operator diagonal:",
    electronic_expected.diagonal(),
)

print(
    f"Maximum operator error: {error:.3e}"
)