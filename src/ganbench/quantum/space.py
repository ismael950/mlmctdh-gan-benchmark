from __future__ import annotations

import numpy as np


def lift_electronic(
    operator: np.ndarray,
    nuclear_dimension: int,
) -> np.ndarray:
    """
    Embed an electronic operator into

        H_el tensor H_nuc

    as

        O_el tensor I_nuc.
    """
    identity_nuclear = np.eye(
        nuclear_dimension,
        dtype=complex,
    )

    return np.kron(
        operator,
        identity_nuclear,
    )


def lift_nuclear(
    operator: np.ndarray,
    electronic_dimension: int,
) -> np.ndarray:
    """
    Embed a nuclear operator into

        H_el tensor H_nuc

    as

        I_el tensor O_nuc.
    """
    identity_electronic = np.eye(
        electronic_dimension,
        dtype=complex,
    )

    return np.kron(
        identity_electronic,
        operator,
    )


def coupled_operator(
    electronic_operator: np.ndarray,
    nuclear_operator: np.ndarray,
) -> np.ndarray:
    """
    Construct a product operator

        O_el tensor O_nuc.
    """
    return np.kron(
        electronic_operator,
        nuclear_operator,
    )