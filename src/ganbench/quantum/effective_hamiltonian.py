"""
BCH effective Hamiltonian for the small GAN first-order Trotter formula.

The first-order product formula applies the fragments in the order

    F0 -> F1 -> ... -> F_last

to the state, i.e. the operator product exp(-i dt F_last) ... exp(-i dt F0).
Treating the r-fold product as exact evolution under an effective Hamiltonian,

    S1(dt)^{t/dt} = exp(-i t H_eff),   H_eff = H + dt * E + O(dt^2),

this returns ``H_eff`` truncated at the leading BCH correction (``bch_order=2``
in PennyLane's ``labs.trotter_error`` counting), as a sparse matrix on the same
Hilbert space as ``model.hamiltonian``.

Requires ``pennylane.labs.trotter_error`` (PennyLane >= 0.45, the ``.venv-estimator``
environment).  The import is deferred so this module can be imported without it.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def build_effective_hamiltonian(model, dt: float, bch_order: int = 2) -> csr_matrix:
    """Return ``H_eff = (i/dt) * Omega`` for the first-order GAN product formula."""
    import pennylane.labs.trotter_error as te

    fragment_matrices = [
        csr_matrix(model.f0),
        *[csr_matrix(fragment) for fragment in model.hopping_fragments],
        csr_matrix(model.f_last),
    ]
    labels = [f"F{i}" for i in range(len(fragment_matrices))]
    fragments = dict(zip(labels, te.generic_fragments(fragment_matrices)))

    # Operators act on the ket right-to-left, and PennyLane's ProductFormula
    # convention is exp(+i t a H); our evolution is exp(-i dt H).
    product_formula = te.ProductFormula(
        list(reversed(labels)),
        coeffs=[-1.0] * len(labels),
    )

    omega = te.effective_hamiltonian(
        product_formula, fragments, order=bch_order, timestep=dt
    )
    h_eff = ((1.0j / dt) * omega.fragment).tocsr()

    antihermitian = sparse_max_abs(h_eff - h_eff.conjugate().T)
    if antihermitian > 1.0e-10:
        raise RuntimeError(
            f"H_eff is not Hermitian within tolerance ({antihermitian:.2e})"
        )
    return h_eff


def sparse_max_abs(matrix: csr_matrix) -> float:
    return 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))
