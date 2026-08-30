from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pennylane.labs.trotter_error as te


ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "results"
    / "benchmark3_no_au_scattering"
    / "bch_structure"
)

N_METAL = 32


def hopping_label(k: int) -> str:
    return f"H{k:02d}"


def is_hopping(label: str) -> bool:
    return label.startswith("H")


def classify_pair(a: str, b: str) -> str:
    pair = {a, b}

    if pair == {"F0", "Flast"}:
        return "F0-Flast"

    if "F0" in pair:
        other = b if a == "F0" else a
        if is_hopping(other):
            return "F0-hopping"

    if "Flast" in pair:
        other = b if a == "Flast" else a
        if is_hopping(other):
            return "hopping-Flast"

    if is_hopping(a) and is_hopping(b):
        return "hopping-hopping"

    return "other"


def clean_complex(value: complex, tol: float = 1.0e-14) -> complex:
    real = 0.0 if abs(value.real) < tol else value.real
    imag = 0.0 if abs(value.imag) < tol else value.imag
    return complex(real, imag)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Physical GAN fragments
    #
    # F0:
    #   U_N + n_d (U_A - U_N)
    #
    # H01 ... H32:
    #   W_k(z) (d^dag c_k + c_k^dag d)
    #
    # Flast:
    #   T_r + T_z + sum_k eps_k n_k
    # ------------------------------------------------------------

    physical_order = [
        "F0",
        *[
            hopping_label(k)
            for k in range(1, N_METAL + 1)
        ],
        "Flast",
    ]

    # GAN algorithm applies:
    #
    # F0 -> H01 -> ... -> H32 -> Flast
    #
    # Therefore the matrix product acting on the ket is:
    #
    # exp(-i dt Flast) ... exp(-i dt H01) exp(-i dt F0)
    #
    matrix_order = list(reversed(physical_order))

    product_formula = te.ProductFormula(
        matrix_order,
        coeffs=[-1.0] * len(matrix_order),
    )

    # ------------------------------------------------------------
    # Reproduce the same convention validated in small_direct.
    #
    # effective_hamiltonian internally evaluates the BCH after
    # multiplying the product-formula exponent by i*dt.
    #
    # Set dt = 1 here:
    #
    # Omega = log(U_Trot)
    #
    # and
    #
    # H_eff = i * Omega
    #
    # for dt = 1.
    #
    # BCH order 1 -> H
    # BCH order 2 -> leading dt * E1 correction.
    # ------------------------------------------------------------

    bch = te.bch_expansion(
        product_formula(1.0j),
        order=2,
    )

    first_order = bch[0]
    second_order = bch[1]

    print()
    print("=" * 72)
    print("NO/Au first-order Trotter BCH structure")
    print("=" * 72)

    print()
    print("Number of physical fragments:")
    print(len(physical_order))

    print()
    print("Physical application order:")
    print(
        "F0 -> H01 -> ... -> H32 -> Flast"
    )

    print()
    print("Matrix product order:")
    print(
        "Flast @ H32 @ ... @ H01 @ F0"
    )

    # ------------------------------------------------------------
    # First-order BCH sanity check
    # ------------------------------------------------------------

    print()
    print("First-order BCH coefficients after Omega -> H:")

    first_order_ok = True

    for commutator, omega_coeff in first_order.items():
        heff_coeff = clean_complex(
            1.0j * omega_coeff
        )

        print(
            f"  {commutator}: {heff_coeff}"
        )

        if abs(heff_coeff - 1.0) > 1.0e-12:
            first_order_ok = False

    if not first_order_ok:
        raise RuntimeError(
            "First-order BCH convention check failed."
        )

    print()
    print(
        "Sanity check passed: "
        "BCH order 1 reproduces H."
    )

    # ------------------------------------------------------------
    # Leading BCH correction E1
    # ------------------------------------------------------------

    rows = []
    family_counts = Counter()

    for commutator, omega_coeff in second_order.items():

        if len(commutator) != 2:
            raise RuntimeError(
                "Unexpected non-pair commutator "
                f"at BCH order 2: {commutator}"
            )

        a, b = commutator

        # Since dt=1:
        #
        # H_eff = i Omega
        #
        # This coefficient is the coefficient multiplying
        # the commutator in E1.
        heff_coeff = clean_complex(
            1.0j * omega_coeff
        )

        family = classify_pair(a, b)
        family_counts[family] += 1

        rows.append(
            {
                "left": a,
                "right": b,
                "family": family,
                "omega_coeff_real":
                    clean_complex(omega_coeff).real,
                "omega_coeff_imag":
                    clean_complex(omega_coeff).imag,
                "E1_coeff_real":
                    heff_coeff.real,
                "E1_coeff_imag":
                    heff_coeff.imag,
            }
        )

    rows.sort(
        key=lambda row: (
            row["family"],
            row["left"],
            row["right"],
        )
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("Symbolic pair-commutator families:")
    print()

    for family in sorted(family_counts):
        print(
            f"  {family:20s}: "
            f"{family_counts[family]}"
        )

    print()
    print(
        "Total symbolic pair commutators:",
        len(rows),
    )

    expected_pairs = (
        len(physical_order)
        * (len(physical_order) - 1)
        // 2
    )

    print(
        "Expected pairs C(n,2):",
        expected_pairs,
    )

    if len(rows) != expected_pairs:
        raise RuntimeError(
            "Unexpected number of BCH pair commutators."
        )

    # ------------------------------------------------------------
    # Show representative terms only
    # ------------------------------------------------------------

    print()
    print("Representative E1 terms:")

    shown = set()

    for row in rows:
        family = row["family"]

        if family in shown:
            continue

        shown.add(family)

        coeff = complex(
            row["E1_coeff_real"],
            row["E1_coeff_imag"],
        )

        print(
            f"  {family:20s}: "
            f"{coeff} "
            f"[{row['left']}, {row['right']}]"
        )

    # ------------------------------------------------------------
    # Save complete symbolic structure
    # ------------------------------------------------------------

    output_file = OUT / "leading_bch_commutators.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "left",
                "right",
                "family",
                "omega_coeff_real",
                "omega_coeff_imag",
                "E1_coeff_real",
                "E1_coeff_imag",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved:")
    print(output_file)

    print()
    print(
        "NOTE: these are symbolic BCH pairs. "
        "Some commutators may vanish after inserting "
        "the actual GAN operator algebra."
    )


if __name__ == "__main__":
    main()