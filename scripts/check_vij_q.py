from scipy import sparse

from ganbench.model import load_config
from ganbench.exact.hamiltonian import (
    build_exact_hamiltonian,
)


reference = build_exact_hamiltonian(
    load_config(
        "configs/test_vij_reference.yaml"
    )
)

system = build_exact_hamiltonian(
    load_config(
        "configs/test_vij_q.yaml"
    )
)

delta_h = (
    system.hamiltonian
    - reference.hamiltonian
)

n0 = system.electronic_number_operators[0]
n1 = system.electronic_number_operators[1]

electronic_operator = (
    n0 @ n1
).tocsr()

Q = system.vibrational_coordinate_operators[0]

expected = (
    0.7
    * sparse.kron(
        electronic_operator,
        Q,
        format="csr",
    )
)

error = sparse.linalg.norm(
    delta_h - expected
)

print("Vij(Q) test")
print(
    f"Operator error: {error:.3e}"
)