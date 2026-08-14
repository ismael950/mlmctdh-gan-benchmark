import numpy as np

from ganbench.exact.hamiltonian import (
    _local_vibrational_operators,
    _function_of_coordinate,
)


basis_size = 12
frequency = 0.5

operators = _local_vibrational_operators(
    basis_size,
    frequency,
)

Q = operators["coordinate"]

# f(Q) = Q
Q_from_function = _function_of_coordinate(
    Q,
    lambda x: x,
)

error_Q = np.linalg.norm(
    (Q_from_function - Q).toarray()
)

# f(Q) = Q^2
Q2_from_function = _function_of_coordinate(
    Q,
    lambda x: x**2,
)

Q2_direct = Q @ Q

error_Q2 = np.linalg.norm(
    (Q2_from_function - Q2_direct).toarray()
)

print("Coordinate-function test")
print(f"Error f(Q)=Q:   {error_Q:.3e}")
print(f"Error f(Q)=Q^2: {error_Q2:.3e}")