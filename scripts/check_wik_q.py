from scipy import sparse

from ganbench.model import load_config
from ganbench.exact.hamiltonian import (
    build_exact_hamiltonian,
    _one_body_operator,
)


reference = build_exact_hamiltonian(
    load_config(
        "configs/test_wik_q_reference.yaml"
    )
)

system = build_exact_hamiltonian(
    load_config(
        "configs/test_wik_q.yaml"
    )
)

delta_h = (
    system.hamiltonian
    - reference.hamiltonian
)

# Orbital ordering:
# 0 = molecular
# 1 = metal

hopping = _one_body_operator(
    system.electronic_basis,
    target=0,
    source=1,
)

electronic_operator = (
    hopping
    + hopping.getH()
)

Q = (
    system.vibrational_coordinate_operators[0]
)

expected = (
    0.2
    * sparse.kron(
        electronic_operator,
        Q,
        format="csr",
    )
)

error = sparse.linalg.norm(
    delta_h - expected
)

print("Wik(Q) test")
print(
    f"Operator error: {error:.3e}"
)