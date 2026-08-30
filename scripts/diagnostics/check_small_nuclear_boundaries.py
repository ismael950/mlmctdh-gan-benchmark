import numpy as np

from ganbench.quantum.dynamics import (
    electronic_basis_state,
    gaussian_nuclear_state,
    product_initial_state,
    exact_propagation,
)
from ganbench.quantum.toy_model import build_quantum_toy_gan


K = 32

model = build_quantum_toy_gan(
    nuclear_size=K,
)

electronic = electronic_basis_state(
    [0, 2, 3],
    model.n_electronic_orbitals,
)

nuclear = gaussian_nuclear_state(
    model.nuclear_grid.points,
    center=0.0,
    sigma=1.0,
)

psi0 = product_initial_state(
    electronic,
    nuclear,
)

times = np.arange(
    0.0,
    2000.0 + 5.0,
    10.0,
)

states = exact_propagation(
    model.hamiltonian,
    psi0,
    times,
)

reshaped = states.reshape(
    len(times),
    model.electronic_dimension,
    model.nuclear_dimension,
)

nuclear_probability = np.sum(
    np.abs(reshaped) ** 2,
    axis=1,
)

points = model.nuclear_grid.points

# Four grid states furthest from Q = 0.
boundary_indices = np.argsort(
    np.abs(points)
)[-4:]

boundary_probability = np.sum(
    nuclear_probability[:, boundary_indices],
    axis=1,
)

imax = int(np.argmax(boundary_probability))

print("K =", K)
print("Q range =", np.min(points), np.max(points))
print("Boundary Q points =", points[boundary_indices])

print(
    "Maximum boundary probability =",
    boundary_probability[imax],
)

print(
    "Time of maximum =",
    times[imax],
    "a.u.",
)

print(
    "Final boundary probability =",
    boundary_probability[-1],
)