"""
Generate Heidelberg ML-MCTDH inputs for the effective Hamiltonian H_eff(dt).

Goal: propagate H_eff with a *converged* ML-MCTDH tree and see how the resulting
Trotter-error estimate

    E_eff^{ML-MCTDH}(dt) = max_t max_i | n_i^{H_eff,ML-MCTDH}(t) - n_i^{exact}(t) |

compares with the exact-propagation estimate E_eff(dt) already in
results/small_direct_benchmark/effective_hamiltonian/convergence.csv.

For each dt in the sweep this writes a self-contained run directory under

    backend_inputs/small_direct_benchmark/heidelberg_heff/
        run_NNN/                 (NNN = 001..008  <->  n_steps 200..1600, dt 10..1.25 a.u.)
            benchmark.inp        converged (run_007) tree, exp DVR on Q
            benchmark.op         H + dt*E  (E = leading first-order-GAN BCH error operator)
            switching.dat        s(Q) grid values, same file as the reference runs
        H_ref/                   the reference H run on the same tree / DVR

Every benchmark.op is reconstructed as a matrix and checked against the PennyLane
H_eff matrix (max abs difference < 1e-9) before being written.

The nuclear primitive is switched from FFT to `exp` DVR: the two are the same
periodic Fourier basis, but `exp` stores explicit operator matrices so MCTDH can
form the operator products (q*dq^2, sw*dq^2, ...) that the BCH momentum terms
[A, KE] = -1/(2 mass) (A*dq^2 - dq^2*A) require.  FFT DVR rejects such products.

Run with the .venv-estimator environment.
"""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path

import numpy as np

from ganbench.quantum.effective_hamiltonian import build_effective_hamiltonian
from ganbench.quantum.toy_model import build_quantum_toy_gan

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "backend_inputs" / "small_direct_benchmark" / "heidelberg" / "K32_smoke"
TREE_INP = ROOT / "backend_inputs" / "small_direct_benchmark" / "heidelberg" / "run_007" / "benchmark.inp"
OUT = ROOT / "backend_inputs" / "small_direct_benchmark" / "heidelberg_heff"

N_STEPS_LIST = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
T_FINAL = 2000.0
MASS_Q = 5000.0
BCH_ORDER = 2
FIT_TOL = 1.0e-9          # per-string Q-fit residual must be below this
RECON_TOL = 1.0e-9        # full .op reconstruction vs PennyLane H_eff

# ------------------------------------------------------------------ operators
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}

# 'spin' primitive basis (sin 2 -0.5 0.5 spin), basis order (n=0, n=1):
#   q = diag(-1/2, 1/2) = -Z/2      dq^2 = [[0,1/2],[1/2,0]] = X/2
#   I*dq = [[0,-i/2],[i/2,0]] = Y/2
# so   Z = -2 q,   X = 2 dq^2,   Y = 2 (I*dq).
_PAULI_TO_SPIN = {
    "I": (1.0, "1"),
    "Z": (-2.0, "q"),
    "X": (2.0, "dq^2"),
    "Y": (2.0, "I*dq"),
}

# multiplicative Q operators  ->  .op token
_MULT_TOKEN = {"1": "1", "Q": "q", "sw": "sw", "sw*Q": "sw*q", "sw2": "sw*sw"}
# momentum Q operators  [A, KE]  ->  A token  (KE = -1/(2 mass) dq^2)
_MOM_A_TOKEN = {"[Q,KE]": "q", "[Q2,KE]": "q^2", "[sw,KE]": "sw"}


def q_operators(model):
    g = model.nuclear_grid
    Q = np.asarray(g.coordinate, dtype=complex)
    KE = np.asarray(g.kinetic, dtype=complex)
    pts = np.asarray(g.points, dtype=float)
    sw = np.diag(0.5 * (1.0 - np.tanh(pts))).astype(complex)
    I32 = np.eye(len(pts), dtype=complex)

    def cke(A):
        return A @ KE - KE @ A

    return {
        "1": I32, "Q": Q, "sw": sw, "sw*Q": sw @ Q, "sw2": sw @ sw,
        "[Q,KE]": cke(Q), "[Q2,KE]": cke(Q @ Q), "[sw,KE]": cke(sw),
    }


def pauli_strings(n):
    return ["".join(c) for c in itertools.product("IXYZ", repeat=n)]


def spin_factor_and_tokens(pauli: str):
    scalar = 1.0
    tokens = []
    for ch in pauli:
        f, tok = _PAULI_TO_SPIN[ch]
        scalar *= f
        tokens.append(tok)
    return scalar, tokens


# ------------------------------------------------------------------ term model
class Term:
    """One .op line:  coeff * (prod of single-mode operators)."""

    __slots__ = ("coeff", "spin_tokens", "q_token")

    def __init__(self, coeff, spin_tokens, q_token):
        self.coeff = complex(coeff)
        self.spin_tokens = list(spin_tokens)   # 6 tokens, mode 1..6
        self.q_token = q_token                 # mode 7 token string

    def op_line(self) -> str:
        c = self.coeff

        def fnum(x: float) -> str:
            # MCTDH wants Fortran-style 'd' exponents, not 'e'.
            return f"{x:.12e}".replace("e", "d")

        if abs(c.imag) < 1e-14 * max(abs(c.real), 1.0):
            cstr = fnum(c.real)
        elif abs(c.real) < 1e-14 * max(abs(c.imag), 1.0):
            cstr = fnum(c.imag) + "*I"
        else:                                   # not expected here
            raise ValueError(f"mixed complex coeff {c}")
        parts = [cstr.strip()]
        for k, tok in enumerate(self.spin_tokens, start=1):
            if tok != "1":
                parts.append(f"|{k} {tok}")
        parts.append(f"|7 {self.q_token}")
        return "   ".join(parts)


# ------------------------------------------------------------------ H terms
def h_terms():
    """The K=32 GAN Hamiltonian, term by term (matches K32_smoke/benchmark.op)."""
    ed = [0.003, 0.005]
    ec = [-0.003, -0.001, 0.001, 0.003]
    g1, g2, g0 = -0.0012, -0.0008, -0.001
    econst, Vmm, V1, V2, kQ = 0.004, 0.00015, 0.00025, 0.00020, 0.0001
    S6 = ["1"] * 6
    T = []
    T.append(Term(1.0, S6, "KE"))
    T.append(Term(kQ, S6, "q^2"))
    T.append(Term(econst, S6, "1"))
    for i, e in enumerate(ed):
        T.append(Term(e, ["q" if k == i else "1" for k in range(6)], "1"))
    for i, e in enumerate(ec):
        T.append(Term(e, ["q" if k == i + 2 else "1" for k in range(6)], "1"))
    T.append(Term(g0, S6, "q"))
    T.append(Term(g1, ["q", "1", "1", "1", "1", "1"], "q"))
    T.append(Term(g2, ["1", "q", "1", "1", "1", "1"], "q"))

    def hop(coeff, i, j, parity, qtok):
        base = ["1"] * 6
        for k in parity:
            base[k] = "q"
        t_dq2 = list(base); t_dq2[i] = "dq^2"; t_dq2[j] = "dq^2"
        t_idq = list(base); t_idq[i] = "I*dq"; t_idq[j] = "I*dq"
        return [Term(coeff, t_dq2, qtok), Term(coeff, t_idq, qtok)]

    T += hop(2 * Vmm, 0, 1, [], "1")
    T += hop(-4 * V1, 0, 2, [1], "sw")
    T += hop(8 * V1, 0, 3, [1, 2], "sw")
    T += hop(-16 * V1, 0, 4, [1, 2, 3], "sw")
    T += hop(32 * V1, 0, 5, [1, 2, 3, 4], "sw")
    T += hop(2 * V2, 1, 2, [], "sw")
    T += hop(-4 * V2, 1, 3, [2], "sw")
    T += hop(8 * V2, 1, 4, [2, 3], "sw")
    T += hop(-16 * V2, 1, 5, [2, 3, 4], "sw")
    return T


# ------------------------------------------------------------------ E terms
def e_terms(E, qops, n_orb):
    """Decompose E = sum_P P (x) M_P and turn each piece into .op Terms."""
    d_el = 1 << n_orb
    d_nuc = qops["1"].shape[0]
    E4 = E.reshape(d_el, d_nuc, d_el, d_nuc)

    strings = pauli_strings(n_orb)
    Pstack = np.empty((len(strings), d_el, d_el), dtype=complex)
    for i, s in enumerate(strings):
        M = np.array([[1.0 + 0j]])
        for ch in s:
            M = np.kron(M, _PAULI[ch])
        Pstack[i] = M
    MP = np.einsum("pab,axby->pxy", Pstack.conj(), E4, optimize=True) / d_el

    names = list(qops)
    A = np.stack([qops[n].reshape(-1) for n in names], axis=1)

    terms = []
    worst = 0.0
    for i, s in enumerate(strings):
        m = MP[i]
        if np.linalg.norm(m) <= 1e-11:
            continue
        coef, *_ = np.linalg.lstsq(A, m.reshape(-1), rcond=None)
        resid = float(np.abs(A @ coef - m.reshape(-1)).max())
        worst = max(worst, resid)

        s_spin, spin_toks = spin_factor_and_tokens(s)
        for name, c in zip(names, coef):
            if abs(c) < 1e-12:
                continue
            if name in _MULT_TOKEN:
                terms.append(Term(s_spin * c.real, spin_toks, _MULT_TOKEN[name]))
            else:  # [A, KE] = -1/(2 mass) (A dq^2 - dq^2 A), c is imaginary
                A_tok = _MOM_A_TOKEN[name]
                g = c.imag * (-1.0 / (2.0 * MASS_Q))       # real
                terms.append(Term((s_spin * g) * 1j, spin_toks, f"{A_tok}*dq^2"))
                terms.append(Term(-(s_spin * g) * 1j, spin_toks, f"dq^2*{A_tok}"))
    return terms, worst


# ------------------------------------------------------------------ reconstruction
_SPIN_MAT = {
    "1": _I2,
    "q": np.diag([-0.5, 0.5]).astype(complex),
    "dq^2": np.array([[0, 0.5], [0.5, 0]], dtype=complex),
    "I*dq": np.array([[0, -0.5j], [0.5j, 0]], dtype=complex),
}


def term_matrix(term: Term, qops, model):
    g = model.nuclear_grid
    Q = np.asarray(g.coordinate, dtype=complex)
    KE = np.asarray(g.kinetic, dtype=complex)
    dq2 = -np.asarray(g.momentum, dtype=complex) @ np.asarray(g.momentum, dtype=complex)
    pts = np.asarray(g.points, dtype=float)
    sw = np.diag(0.5 * (1.0 - np.tanh(pts))).astype(complex)
    prim = {
        "1": qops["1"], "q": Q, "q^2": Q @ Q, "KE": KE, "dq^2": dq2, "sw": sw,
    }

    def qmat(tok):
        if "*" in tok:
            a, b = tok.split("*")
            return prim[a] @ prim[b]
        return prim[tok]

    spin = np.array([[1.0 + 0j]])
    for tok in term.spin_tokens:
        spin = np.kron(spin, _SPIN_MAT[tok])
    return term.coeff * np.kron(spin, qmat(term.q_token))


# ------------------------------------------------------------------ .op / .inp
def render_op(all_terms) -> str:
    lines = [
        "#######################################################################",
        "### Effective Hamiltonian  H_eff = H + dt*E  for the small GAN benchmark",
        "### Generated by scripts/small_benchmark/generate_small_heff_heidelberg.py",
        "#######################################################################",
        "",
        "OP_DEFINE-SECTION",
        "title",
        "H_eff for the first-order GAN Trotter formula (leading BCH error operator)",
        "end-title",
        "end-OP_DEFINE-SECTION",
        "",
        "PARAMETER-SECTION",
        f"mass_Q = {MASS_Q}",
        "end-PARAMETER-SECTION",
        "",
        "LABELS-SECTION",
        "sw = read1d{switching.dat ascii}",
        "end-LABELS-SECTION",
        "",
        "HAMILTONIAN-SECTION",
        "modes | d1 | d2 | c1 | c2 | c3 | c4 | Q",
        "",
    ]
    lines += [t.op_line() for t in all_terms]
    lines += ["", "end-HAMILTONIAN-SECTION", ""]

    expect = {
        "nd1": ("0.5 |1 1", "1.0 |1 q"),
        "nd2": ("0.5 |1 1", "1.0 |2 q"),
        "nc1": ("0.5 |1 1", "1.0 |3 q"),
        "nc2": ("0.5 |1 1", "1.0 |4 q"),
        "nc3": ("0.5 |1 1", "1.0 |5 q"),
        "nc4": ("0.5 |1 1", "1.0 |6 q"),
        "Qmean": ("1.0 |7 q",),
    }
    for name, rows in expect.items():
        lines.append(f"HAMILTONIAN-SECTION_{name}")
        lines.append("modes | d1 | d2 | c1 | c2 | c3 | c4 | Q")
        lines += list(rows)
        lines.append("end-HAMILTONIAN-SECTION")
        lines.append("")
    lines.append("END-OPERATOR")
    lines.append("")
    return "\n".join(lines)


def render_inp(name_rel: str) -> str:
    text = TREE_INP.read_text(encoding="utf-8")
    text = text.replace(
        "name = ../../../../results/small_direct_benchmark/heidelberg/run_007/raw",
        f"name = ../../../../results/small_direct_benchmark/heidelberg_heff/{name_rel}/raw",
    )
    # FFT -> exp DVR so operator products are allowed
    text = text.replace("Q      FFT      32", "Q      exp      32")
    return text


# ------------------------------------------------------------------ main
def main() -> None:
    model = build_quantum_toy_gan(nuclear_size=32)
    n_orb = model.n_electronic_orbitals
    H = np.asarray(model.hamiltonian, dtype=complex)
    qops = q_operators(model)

    HT = h_terms()
    Hrec = sum(term_matrix(t, qops, model) for t in HT)
    assert np.abs(Hrec - H).max() < 1e-11, np.abs(Hrec - H).max()
    print(f"H term reconstruction: max|Hrec - H| = {np.abs(Hrec - H).max():.2e}")

    OUT.mkdir(parents=True, exist_ok=True)
    switching = (SMOKE / "switching.dat").read_text(encoding="utf-8")

    # reference H run on the same tree / DVR
    d = OUT / "H_ref"
    d.mkdir(exist_ok=True)
    (d / "benchmark.op").write_text(render_op(HT), encoding="utf-8")
    (d / "benchmark.inp").write_text(render_inp("H_ref"), encoding="utf-8")
    (d / "switching.dat").write_text(switching, encoding="utf-8")
    print("wrote H_ref/")

    for idx, n_steps in enumerate(N_STEPS_LIST, start=1):
        dt = T_FINAL / n_steps
        h_eff = build_effective_hamiltonian(model, dt=dt, bch_order=BCH_ORDER).toarray()
        E = h_eff - H

        ET, worst = e_terms(E, qops, n_orb)
        if worst > FIT_TOL:
            raise RuntimeError(f"dt={dt}: Q-fit residual {worst:.2e} > {FIT_TOL}")

        recon = Hrec + sum(term_matrix(t, qops, model) for t in ET)
        err = float(np.abs(recon - h_eff).max())
        if err > RECON_TOL:
            raise RuntimeError(f"dt={dt}: .op reconstruction error {err:.2e} > {RECON_TOL}")

        run = OUT / f"run_{idx:03d}"
        run.mkdir(exist_ok=True)
        (run / "benchmark.op").write_text(render_op(HT + ET), encoding="utf-8")
        (run / "benchmark.inp").write_text(render_inp(f"run_{idx:03d}"), encoding="utf-8")
        (run / "switching.dat").write_text(switching, encoding="utf-8")
        print(
            f"run_{idx:03d}  n_steps={n_steps:4d}  dt={dt:7.4f}  "
            f"E-terms={len(ET):3d}  fit_resid={worst:.1e}  recon_err={err:.1e}"
        )

    print(f"\nAll inputs written under {OUT}")


if __name__ == "__main__":
    main()
