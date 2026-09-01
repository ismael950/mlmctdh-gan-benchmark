"""
Generate Heidelberg ML-MCTDH inputs for the NO/Au effective Hamiltonian

    H_eff(dt) = H + dt * E1 ,   E1 = -(i/2) sum_{a<b} [F_a, F_b]

(leading BCH error operator of the GAN first-order product formula).

Unlike the small benchmark, NO/Au cannot be built as a dense matrix, so E1 is
emitted analytically, family by family (see docs/no_au_heff_design.md).  The
analytic emission is verified against a dense PennyLane `effective_hamiltonian`
on a reduced NO/Au (few metal orbitals, tiny r/z grids):

    .venv-estimator/Scripts/python scripts/no_au/generate_no_au_heff_heidelberg.py --verify-reduced

If verification passes, run without --verify-reduced to write the full inputs:

    backend_inputs/benchmark3_no_au_scattering/heidelberg_heff/
        H_ref/                 true H on the frozen run_009 tree
        run_001 .. run_004     H_eff at dt = 10, 6, 4, 3 a.u.

Fragments (physical order  F0 -> H_1..H_Nm -> Flast):

    F0     = A(r,z) I_el + q_d B(r,z)                         A = 0.5(VNr+VAr)+0.5(VNz+VAz)+cavg
    H_k    = vk_k fz(z) (d^dag c_k + c_k^dag d)               B = (VAr-VNr)+(VAz-VNz)+cdiff
    Flast  = KE_r + KE_z + sum_k eps_k n_k                    fz(z) = 1 - tanh(z/a_tilde)

d = JW orbital 0, c_k = JW orbital k.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "benchmark3_no_au_scattering.yaml"
TREE_INPUT = ROOT / "results" / "benchmark3_no_au_scattering" / "heidelberg" / "run_009" / "raw" / "input"
BASE_INPUTS = ROOT / "backend_inputs" / "benchmark3_no_au_scattering" / "heidelberg" / "run_008"
OUT = ROOT / "backend_inputs" / "benchmark3_no_au_scattering" / "heidelberg_heff"

DT_SWEEP = [3.0, 2.0, 1.5, 1.0, 0.5, 0.25, 1.25, 1.75]   # run_001..008 (007,008 = window refinement)
CHECK_FULL_DT = 1.0                        # extra dir 'check_full': Δt=1 with the FULL E1 ([H_j,H_k] back)
INCLUDE_METAL_METAL = False                # default for the sweep: drop the 496-term [H_j,H_k] family
RECON_TOL = 1.0e-9

EV_TO_HARTREE = 1.0 / 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
AMU_TO_ELECTRON_MASS = 1822.888486209


# ======================================================================
# Spin-primitive (2-level 'sin ... spin') single-orbital matrices
#   q = -Z/2,  dq^2 = X/2,  I*dq = Y/2
# ======================================================================
_I2 = np.eye(2, dtype=complex)
_SPIN = {
    "1": _I2,
    "q": np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=complex),
    "dq^2": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex),
    "I*dq": np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=complex),
}


def kron_list(mats):
    out = np.array([[1.0 + 0.0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def jw_annihilation(orbital: int, n_orb: int) -> np.ndarray:
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    mats = []
    for q in range(n_orb):
        if q < orbital:
            mats.append(Z)
        elif q == orbital:
            mats.append((X + 1j * Y) / 2.0)
        else:
            mats.append(_I2)
    return kron_list(mats)


# ======================================================================
# Colbert-Miller sine DVR  (matches MCTDH  'sin'  primitive)
# ======================================================================
@dataclass(frozen=True)
class SineDVR:
    points: np.ndarray          # grid points (bohr)
    kinetic: np.ndarray         # KE matrix (hartree), mass folded in
    identity: np.ndarray

    @property
    def n(self) -> int:
        return len(self.points)

    def position(self) -> np.ndarray:
        return np.diag(self.points).astype(complex)

    def diagonal(self, values: np.ndarray) -> np.ndarray:
        return np.diag(np.asarray(values, dtype=float)).astype(complex)


def build_sine_dvr(x_min: float, x_max: float, n: int, mass: float) -> SineDVR:
    length = x_max - x_min
    idx = np.arange(1, n + 1)
    x = x_min + idx * length / (n + 1)
    # DVR <-> sine-FBR transform
    nn, ii = np.meshgrid(idx, idx)
    U = np.sqrt(2.0 / (n + 1)) * np.sin(nn * ii * np.pi / (n + 1))
    k2 = (idx * np.pi / length) ** 2
    KE = U @ np.diag(k2 / (2.0 * mass)) @ U.T
    return SineDVR(points=x, kinetic=KE.astype(complex), identity=np.eye(n, dtype=complex))


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


# ======================================================================
# Physical 1D functions  (hartree, argument in bohr)   -- from the H generator
# ======================================================================
def _load_params():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    p = raw["physical_parameters"]
    n, a = p["neutral_surface"], p["anionic_surface"]
    mm = p["molecule_metal"]
    return raw, n, a, mm


_RAW, _NEU, _ANI, _MM = _load_params()


def _morse_lit_ev(x_ang, xe, alpha, depth):
    e = np.exp(-alpha * (x_ang - xe))
    return depth * (e ** 2 - 2.0 * e)


def VNr(r_bohr):
    return _morse_lit_ev(r_bohr * BOHR_TO_ANGSTROM, _NEU["r0_angstrom"],
                         _NEU["a0_inv_angstrom"], _NEU["D0_ev"]) * EV_TO_HARTREE


def VAr(r_bohr):
    return _morse_lit_ev(r_bohr * BOHR_TO_ANGSTROM, _ANI["r1_angstrom"],
                         _ANI["a1_inv_angstrom"], _ANI["D1_ev"]) * EV_TO_HARTREE


def VNz(z_bohr):
    za = z_bohr * BOHR_TO_ANGSTROM
    return np.exp(-_NEU["b0_inv_angstrom"] * (za - _NEU["z0_angstrom"])) * EV_TO_HARTREE


def VAz(z_bohr):
    return _morse_lit_ev(z_bohr * BOHR_TO_ANGSTROM, _ANI["z1_angstrom"],
                         _ANI["a2_inv_angstrom"], _ANI["D2_ev"]) * EV_TO_HARTREE


def fz(z_bohr):
    return 1.0 - np.tanh(z_bohr * BOHR_TO_ANGSTROM / _MM["a_tilde_angstrom"])


CAVG = 0.5 * (_NEU["c0_ev"] + _ANI["c1_ev"]) * EV_TO_HARTREE
CDIFF = (_ANI["c1_ev"] - _NEU["c0_ev"]) * EV_TO_HARTREE


# ======================================================================
# E1 term container
# ======================================================================
@dataclass
class E1Term:
    """coeff * (prod of spin ops over orbitals) (x) r_op (x) z_op."""
    coeff: complex
    spin: list[str]                      # length n_orb, tokens in _SPIN
    r_op: str = "1"                      # token(s), see _R_OPS / _Z_OPS below
    z_op: str = "1"

    def spin_matrix(self) -> np.ndarray:
        return kron_list(_SPIN[t] for t in self.spin)


# ======================================================================
# Reduced dense NO/Au fragments  (verification only)
# ======================================================================
def build_reduced(n_metal: int, r: SineDVR, z: SineDVR, eps, vk):
    n_orb = 1 + n_metal
    Iel = np.eye(2 ** n_orb, dtype=complex)
    Ir, Iz = r.identity, z.identity

    def El(m):      # electronic op -> full space
        return np.kron(np.kron(m, Ir), Iz)

    def R(m):
        return np.kron(np.kron(Iel, m), Iz)

    def Zc(m):
        return np.kron(np.kron(Iel, Ir), m)

    a = [jw_annihilation(i, n_orb) for i in range(n_orb)]
    ad = [x.conj().T for x in a]
    nd = ad[0] @ a[0]
    qd = nd - 0.5 * Iel

    # A(r,z), B(r,z) as separable sums on the (r,z) grid
    A_r = R(r.diagonal(0.5 * (VNr(r.points) + VAr(r.points))))
    A_z = Zc(z.diagonal(0.5 * (VNz(z.points) + VAz(z.points)))) + CAVG * np.kron(np.kron(Iel, Ir), Iz)
    B_r = R(r.diagonal(VAr(r.points) - VNr(r.points)))
    B_z = Zc(z.diagonal(VAz(z.points) - VNz(z.points))) + CDIFF * np.kron(np.kron(Iel, Ir), Iz)

    F0 = A_r + A_z + El(qd) @ (B_r + B_z)

    Hlist = []
    fzz = Zc(z.diagonal(fz(z.points)))
    for k in range(1, n_metal + 1):
        hop = El(ad[0] @ a[k] + ad[k] @ a[0])
        Hlist.append(vk[k - 1] * (fzz @ hop))

    Flast = R(r.kinetic) + Zc(z.kinetic)
    for k in range(1, n_metal + 1):
        Flast = Flast + eps[k - 1] * El(ad[k] @ a[k])

    H = F0 + sum(Hlist) + Flast
    return F0, Hlist, Flast, H, dict(El=El, R=R, Zc=Zc)


def pennylane_heff(F0, Hlist, Flast, dt: float) -> np.ndarray:
    import pennylane.labs.trotter_error as te
    mats = [F0, *Hlist, Flast]
    labels = [f"F{i}" for i in range(len(mats))]
    frags = dict(zip(labels, te.generic_fragments([m for m in mats])))
    pf = te.ProductFormula(list(reversed(labels)), coeffs=[-1.0] * len(labels))
    omega = te.effective_hamiltonian(pf, frags, order=2, timestep=dt)
    return np.asarray((1j / dt) * omega.fragment, dtype=complex)


# ======================================================================
# Analytic  dt * E1  emission
# ======================================================================
def _zstring(lo: int, hi: int, n_orb: int) -> list[str]:
    """spin tokens: 'q' on orbitals lo..hi-1 (JW Z-string -> (-2)^m prod q)."""
    return ["q" if lo <= i < hi else "1" for i in range(n_orb)]


def _imag_hop(coeff: complex, i: int, j: int, n_orb: int, r_op="1", z_op="1"):
    """coeff * (a_i^dag a_j - a_j^dag a_i),  i<j.

    = coeff * (i/2) Zstr_{i+1..j-1} (X_i Y_j - Y_i X_j)
    -> 2i*coeff*(-2)^{j-1-i} [ dq2_i (Idq)_j - (Idq)_i dq2_j ] prod q
    """
    base = _zstring(i + 1, j, n_orb)
    pref = 2j * coeff * ((-2.0) ** (j - i - 1))          # (-2)^m folds the JW Z-string
    t1 = list(base); t1[i] = "dq^2"; t1[j] = "I*dq"
    t2 = list(base); t2[i] = "I*dq"; t2[j] = "dq^2"
    return [E1Term(pref, t1, r_op, z_op), E1Term(-pref, t2, r_op, z_op)]


def _real_hop(coeff: complex, i: int, j: int, n_orb: int, r_op="1", z_op="1"):
    """coeff * (a_i^dag a_j + a_j^dag a_i),  i<j.

    = coeff * (1/2) Zstr (X_i X_j + Y_i Y_j)
    -> 2*coeff*(-2)^{j-1-i} [ dq2_i dq2_j + (Idq)_i (Idq)_j ] prod q
    """
    base = _zstring(i + 1, j, n_orb)
    pref = 2.0 * coeff * ((-2.0) ** (j - i - 1))         # (-2)^m folds the JW Z-string
    t1 = list(base); t1[i] = "dq^2"; t1[j] = "dq^2"
    t2 = list(base); t2[i] = "I*dq"; t2[j] = "I*dq"
    return [E1Term(pref, t1, r_op, z_op), E1Term(pref, t2, r_op, z_op)]


def e1_terms(n_metal: int, eps, vk, dt: float, include_kinetic: bool = False,
             include_metal_metal: bool = True) -> list[E1Term]:
    """dt * E1 = dt * (+i/2) [ sum_k [F0,H_k] + [F0,Flast]
                               + sum_{j<k}[H_j,H_k] + sum_k [H_k,Flast] ] .

    include_kinetic=False drops the two commutator-with-KE families:
      [H_k,Flast]_T  ~ 1e-5 (negligible),
      [F0,Flast]     ~ 1e-3 (small; needs |mode V |mode KE product syntax).
    include_metal_metal=False drops [H_j,H_k] (the 496-term family): per term
      ~ vk_j vk_k ~ 1e-3 (smallest), but numerous.  Dropping it is a
      DOWNWARD bias on r* -- size reported by verify_reduced().
    """
    n_orb = 1 + n_metal
    pre = 0.5j * dt
    T: list[E1Term] = []

    # ---- (F0, H_k):  W_k(z) B(r,z) (d^dag c_k - c_k^dag d),  B = dVr + dVz + cdiff ----
    for k in range(1, n_metal + 1):
        c = pre * vk[k - 1]
        T += _imag_hop(c, 0, k, n_orb, r_op="VAr", z_op="fz")
        T += _imag_hop(-c, 0, k, n_orb, r_op="VNr", z_op="fz")
        T += _imag_hop(c, 0, k, n_orb, r_op="1", z_op="fzVAz")
        T += _imag_hop(-c, 0, k, n_orb, r_op="1", z_op="fzVNz")
        T += _imag_hop(c * CDIFF, 0, k, n_orb, r_op="1", z_op="fz")

    # ---- (H_j, H_k), j<k:  W_j W_k(z) (c_j^dag c_k - c_k^dag c_j) ----
    if include_metal_metal:
        for j in range(1, n_metal + 1):
            for k in range(j + 1, n_metal + 1):
                T += _imag_hop(pre * vk[j - 1] * vk[k - 1], j, k, n_orb, r_op="1", z_op="fz2")

    # ---- (H_k, Flast)_eps:  W_k(z) eps_k (d^dag c_k - c_k^dag d) ----
    for k in range(1, n_metal + 1):
        T += _imag_hop(pre * vk[k - 1] * eps[k - 1], 0, k, n_orb, r_op="1", z_op="fz")

    if include_kinetic:
        one = ["1"] * n_orb
        qd = ["q" if i == 0 else "1" for i in range(n_orb)]
        for k in range(1, n_metal + 1):
            T += _real_hop(pre * vk[k - 1], 0, k, n_orb, r_op="1", z_op="[fz,KE]")
        T.append(E1Term(pre, list(one), r_op="[Asym,KE]", z_op="1"))
        T.append(E1Term(pre, list(one), r_op="1", z_op="[Asym,KE]"))
        T.append(E1Term(pre, list(qd), r_op="[dVr,KE]", z_op="1"))
        T.append(E1Term(pre, list(qd), r_op="1", z_op="[dVz,KE]"))
    return T


# ---- nuclear-operator dictionaries (dense, for verification) ----
def nuclear_ops(r: SineDVR, z: SineDVR):
    Asym_r = r.diagonal(0.5 * (VNr(r.points) + VAr(r.points)))
    Asym_z = z.diagonal(0.5 * (VNz(z.points) + VAz(z.points)))
    dVr = r.diagonal(VAr(r.points) - VNr(r.points))
    dVz = z.diagonal(VAz(z.points) - VNz(z.points))
    fzz = z.diagonal(fz(z.points))
    VAr_m = r.diagonal(VAr(r.points))
    VNr_m = r.diagonal(VNr(r.points))
    r_ops = {
        "1": r.identity,
        "VAr": VAr_m,
        "VNr": VNr_m,
        "dVr": dVr,
        "[Asym,KE]": comm(Asym_r, r.kinetic),
        "[dVr,KE]": comm(dVr, r.kinetic),
    }
    z_ops = {
        "1": z.identity,
        "fz": fzz,
        "fz2": fzz @ fzz,
        "fzVAz": fzz @ z.diagonal(VAz(z.points)),
        "fzVNz": fzz @ z.diagonal(VNz(z.points)),
        "[fz,KE]": comm(fzz, z.kinetic),
        "[Asym,KE]": comm(Asym_z, z.kinetic),
        "[dVz,KE]": comm(dVz, z.kinetic),
    }
    return r_ops, z_ops


def reconstruct(terms: list[E1Term], r: SineDVR, z: SineDVR) -> np.ndarray:
    r_ops, z_ops = nuclear_ops(r, z)
    total = None
    for t in terms:
        m = np.kron(np.kron(t.spin_matrix(), r_ops[t.r_op]), z_ops[t.z_op])
        total = m * t.coeff if total is None else total + t.coeff * m
    return total


# ======================================================================
# Verification
# ======================================================================
def verify_reduced() -> None:
    ok = True
    for n_metal in (2, 3, 4):
        rng = np.random.default_rng(n_metal)
        eps = np.linspace(-1.4, 1.4, n_metal) + 0.05 * rng.standard_normal(n_metal)
        vk = 0.015 + 0.012 * np.arange(n_metal)
        nr, nz = (6, 6) if n_metal == 4 else (8, 8)
        r = build_sine_dvr(1.6, 4.2, nr, mass=_reduced_mass_r())
        z = build_sine_dvr(3.0, 11.0, nz, mass=_reduced_mass_z())
        F0, Hlist, Flast, H, _ = build_reduced(n_metal, r, z, eps, vk)
        herm = np.abs(H - H.conj().T).max()
        for dt in (3.0, 6.0, 10.0):
            h_eff = pennylane_heff(F0, Hlist, Flast, dt)
            E_pl = h_eff - H
            E_full = reconstruct(e1_terms(n_metal, eps, vk, dt, include_kinetic=True), r, z)
            E_diag = reconstruct(e1_terms(n_metal, eps, vk, dt, include_kinetic=False), r, z)
            E_emit = reconstruct(e1_terms(n_metal, eps, vk, dt, include_kinetic=False,
                                          include_metal_metal=INCLUDE_METAL_METAL), r, z)
            err = float(np.abs(E_full - E_pl).max())
            trunc = float(np.abs(E_diag - E_pl).max())
            emit_miss = float(np.abs(E_emit - E_pl).max())
            scale = float(np.abs(E_pl).max())
            flag = "OK" if err < RECON_TOL else "FAIL"
            if err >= RECON_TOL:
                ok = False
            print(f"  n_metal={n_metal} dt={dt:4.1f}  full|E-E_pl|={err:.1e}  "
                  f"diag-only={100*trunc/scale:.2f}%  EMITTED-miss={100*emit_miss/scale:.1f}% "
                  f"(|E_pl|~{scale:.1e})  [{flag}]")
    # ---- .op string round-trip: emit -> parse -> dense == e1_terms dense ----
    nm = 3
    eps = np.array([-0.9, 0.1, 1.1]); vk = np.array([0.015, 0.027, 0.021])
    r = build_sine_dvr(1.6, 4.2, 8, mass=_reduced_mass_r())
    z = build_sine_dvr(3.0, 11.0, 8, mass=_reduced_mass_z())
    r_ops, z_ops = nuclear_ops(r, z)
    r_mode, z_mode = nm + 2, nm + 3
    lab2r = {v: k for k, v in _R_LABEL.items() if v}
    lab2z = {v: k for k, v in _Z_LABEL.items() if v}
    def parse_terms(op_lines):
        """yield (coeff, [(mode, op), ...]) joining &&& continuations."""
        coeff, ops = None, []
        for ln in op_lines:
            if not ln or ln.startswith("#"):
                continue
            head, rest = ln[:22].strip(), ln[22:]
            toks = [(int(p.split()[0]), p.split(None, 1)[1].strip())
                    for p in rest.split("|") if p.strip()]
            if head == "&&&":
                ops += toks
            else:
                if coeff is not None:
                    yield coeff, ops
                coeff, ops = float(head.replace("d", "e")), toks
        if coeff is not None:
            yield coeff, ops

    for dt in (1.0, 3.0):
        want = reconstruct(e1_terms(nm, eps, vk, dt, include_kinetic=False,
                                    include_metal_metal=INCLUDE_METAL_METAL), r, z)
        got = np.zeros_like(want)
        for coeff, ops in parse_terms(emit_e1_op_lines(dt, vk, eps, nm)):
            spin = ["1"] * (1 + nm)
            r_t, z_t = "1", "1"
            for m, op in ops:
                if m <= 1 + nm:
                    spin[m - 1] = op
                elif m == r_mode:
                    r_t = lab2r[op]
                elif m == z_mode:
                    z_t = lab2z[op]
            got = got + coeff * np.kron(
                np.kron(kron_list(_SPIN[s] for s in spin), r_ops[r_t]), z_ops[z_t])
        rt = float(np.abs(got - want).max())
        tag = "OK" if rt < 1e-9 else "FAIL"
        if rt >= 1e-9:
            ok = False
        print(f"  .op round-trip  n_metal={nm} dt={dt:4.1f}  max|parsed-e1_terms|={rt:.1e}  [{tag}]")

    print("\nVERIFICATION", "PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


# ======================================================================
# Full-system  .op / .dat / .inp  writers   (n_metal = 32, diagonal-only E1)
# ======================================================================
RUN001 = ROOT / "backend_inputs" / "benchmark3_no_au_scattering" / "heidelberg" / "run_001"
BASE_DATS = ["neutral_r.dat", "anionic_r.dat", "neutral_z.dat", "anionic_z.dat", "coupling_z.dat"]
NEW_LABELS = [
    "fz2 = external1d{fz2.dat}",
    "fzVAz = external1d{fzVAz.dat}",
    "fzVNz = external1d{fzVNz.dat}",
]


def _fnum(x: float) -> str:
    return f"{float(x):.12e}".replace("e", "d")


def _op_term(coeff_str: str, ops: list[tuple[int, str]], wrap: int = 205) -> list[str]:
    ops = sorted(ops, key=lambda t: t[0])
    cur = f"{coeff_str:<22}"
    out = []
    for m, o in ops:
        piece = f" |{m} {o}"
        if len(cur) + len(piece) > wrap:
            out.append(cur)
            cur = f"{'&&&':<22}" + piece
        else:
            cur += piece
    out.append(cur)
    return out


_R_LABEL = {"1": None, "VAr": "VAr", "VNr": "VNr", "dVr": None}          # dVr not emitted (split upstream)
_Z_LABEL = {"1": None, "fz": "fz", "fz2": "fz2", "fzVAz": "fzVAz", "fzVNz": "fzVNz"}


def emit_e1_op_lines(dt: float, vk, ec, n_metal: int, metal_metal: bool | None = None) -> list[str]:
    """diagonal-only  dt * E1  as .op HAMILTONIAN-SECTION lines.

    Formats the SAME E1Term list the reduced verifier checks against PennyLane
    (single source of truth; coefficients are baked real numbers).
    """
    mm = INCLUDE_METAL_METAL if metal_metal is None else metal_metal
    r_mode, z_mode = n_metal + 2, n_metal + 3
    lines = ["", f"### ---- dt*E1  (dt = {dt} a.u., metal_metal={mm}) ----"]
    for t in e1_terms(n_metal, ec, vk, dt, include_kinetic=False,
                      include_metal_metal=mm):
        c = complex(t.coeff)
        if abs(c.imag) > 1e-10 * max(abs(c.real), 1.0):
            raise RuntimeError(f"unexpected complex E1 coeff {c}")
        ops = [(i + 1, tok) for i, tok in enumerate(t.spin) if tok != "1"]
        if _R_LABEL[t.r_op]:
            ops.append((r_mode, _R_LABEL[t.r_op]))
        if _Z_LABEL[t.z_op]:
            ops.append((z_mode, _Z_LABEL[t.z_op]))
        lines.extend(_op_term(_fnum(c.real), ops))
    return lines


def write_product_dat(dst_dir: Path) -> None:
    fz = np.loadtxt(RUN001 / "coupling_z.dat")
    vaz = np.loadtxt(RUN001 / "anionic_z.dat")
    vnz = np.loadtxt(RUN001 / "neutral_z.dat")
    x = fz[:, 0]
    assert np.allclose(x, vaz[:, 0]) and np.allclose(x, vnz[:, 0]), "z .dat grids differ"
    for name, y in [
        ("fz2.dat", fz[:, 1] ** 2),
        ("fzVAz.dat", fz[:, 1] * vaz[:, 1]),
        ("fzVNz.dat", fz[:, 1] * vnz[:, 1]),
    ]:
        np.savetxt(dst_dir / name, np.column_stack([x, y]), fmt="%.16e")


def _patched_inp(run_rel: str) -> str:
    text = TREE_INPUT.read_text(encoding="utf-8").replace("\r\n", "\n")
    old = "name = ../../../../results/benchmark3_no_au_scattering/heidelberg/run_009/raw"
    new = f"name = ../../../../results/benchmark3_no_au_scattering/heidelberg_heff/{run_rel}/raw"
    if old not in text:
        raise RuntimeError("run_009 name line not found in tree input")
    return text.replace(old, new)


def _base_op_text() -> str:
    return (RUN001 / "benchmark.op").read_text(encoding="utf-8").replace("\r\n", "\n")


def _op_with_e1(dt: float, vk, ec, n_metal: int, metal_metal: bool | None = None) -> str:
    text = _base_op_text()
    text = text.replace(
        "\nend-LABELS-SECTION",
        "\n" + "\n".join(NEW_LABELS) + "\nend-LABELS-SECTION",
        1,
    )
    e1 = "\n".join(emit_e1_op_lines(dt, vk, ec, n_metal, metal_metal)) + "\n"
    text = text.replace("\nend-HAMILTONIAN-SECTION", "\n" + e1 + "end-HAMILTONIAN-SECTION", 1)
    return text


def write_full_inputs() -> None:
    raw = _RAW
    n_metal = int(raw["model"]["n_metal_orbitals"])
    ec = np.asarray(raw["electronic"]["metal_energies"], dtype=float)
    # vk_k = constant W_1k prefactor (hartree) from wik_terms
    vk = np.zeros(n_metal)
    for term in raw["gan_terms"]["wik"]:
        if not term["factors"]:
            vk[int(term["k"])] = float(term["coefficient"])
    assert np.all(vk > 0), vk

    OUT.mkdir(parents=True, exist_ok=True)
    base_op = _base_op_text()

    # H_ref : true H on the frozen run_009 tree
    d = OUT / "H_ref"
    d.mkdir(exist_ok=True)
    (d / "benchmark.op").write_text(base_op, encoding="utf-8")
    (d / "benchmark.inp").write_text(_patched_inp("H_ref"), encoding="utf-8")
    for f in BASE_DATS:
        (d / f).write_bytes((RUN001 / f).read_bytes())
    print(f"H_ref/            true H, run_009 tree")

    for idx, dt in enumerate(DT_SWEEP, start=1):
        run = f"run_{idx:03d}"
        d = OUT / run
        d.mkdir(exist_ok=True)
        (d / "benchmark.op").write_text(_op_with_e1(dt, vk, ec, n_metal), encoding="utf-8")
        (d / "benchmark.inp").write_text(_patched_inp(run), encoding="utf-8")
        for f in BASE_DATS:
            (d / f).write_bytes((RUN001 / f).read_bytes())
        write_product_dat(d)
        r_steps = round(2067.06866675911 / dt)
        n_e1 = sum(1 for ln in emit_e1_op_lines(dt, vk, ec, n_metal)
                   if ln and not ln.startswith("#") and not ln.startswith("###"))
        print(f"{run}/  dt={dt:5.2f} a.u.  r={r_steps:4d}   E1 .op lines={n_e1}")

    # check_full : full E1 ([H_j,H_k] back in) at CHECK_FULL_DT  -- convergence check
    d = OUT / "check_full"
    d.mkdir(exist_ok=True)
    (d / "benchmark.op").write_text(
        _op_with_e1(CHECK_FULL_DT, vk, ec, n_metal, metal_metal=True), encoding="utf-8")
    (d / "benchmark.inp").write_text(_patched_inp("check_full"), encoding="utf-8")
    for f in BASE_DATS:
        (d / f).write_bytes((RUN001 / f).read_bytes())
    write_product_dat(d)
    n_e1 = sum(1 for ln in emit_e1_op_lines(CHECK_FULL_DT, vk, ec, n_metal, metal_metal=True)
               if ln and not ln.startswith("#") and not ln.startswith("###"))
    print(f"check_full/  dt={CHECK_FULL_DT:.2f} a.u.  FULL E1 .op lines={n_e1}")

    # runner
    sh = OUT / "run_no_au_heff_heidelberg.sh"
    sh.write_text(
        "#!/bin/bash\n"
        "# Run the NO/Au H_eff sweep with Heidelberg MCTDH.\n"
        "#   bash backend_inputs/.../heidelberg_heff/run_no_au_heff_heidelberg.sh\n"
        "# Override RUNS to do a subset, e.g.  RUNS='H_ref run_001' bash ...\n"
        "set -e\n"
        ': "${MCTDH_DIR:=/data/$USER/software/mctdh86.10}"\n'
        ': "${MCTDH_BIN:=$MCTDH_DIR/bin/binary/x86_64/mctdh86}"\n'
        ': "${RUNS:=H_ref run_001 run_002 run_003 run_004}"\n'
        'source "$MCTDH_DIR/install/mctdh.profile"\n'
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'RES="$HERE/../../../results/benchmark3_no_au_scattering/heidelberg_heff"\n'
        'for run in $RUNS; do\n'
        '    d="$HERE/$run"; [ -d "$d" ] || continue\n'
        '    echo "=============== $run  ($(date)) ==============="\n'
        '    mkdir -p "$RES/$run"\n'
        '    cd "$d" && "$MCTDH_BIN" -mnd -w benchmark.inp\n'
        "done\n"
        'echo "done -> python scripts/no_au/analyze_no_au_heff.py"\n',
        encoding="utf-8",
    )
    print(f"\nwrote runner: {sh}")
    print(f"all inputs under {OUT}")


def _reduced_mass_r():
    raw = _RAW
    for c in raw["nuclear"]["coordinates"]:
        if c["name"] == "r":
            return c["mass_amu"] * AMU_TO_ELECTRON_MASS
    raise KeyError("r")


def _reduced_mass_z():
    raw = _RAW
    for c in raw["nuclear"]["coordinates"]:
        if c["name"] == "z":
            return c["mass_amu"] * AMU_TO_ELECTRON_MASS
    raise KeyError("z")


# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-reduced", action="store_true",
                    help="only run the dense reduced-system verification vs PennyLane")
    args = ap.parse_args()

    if args.verify_reduced:
        verify_reduced()
        return

    write_full_inputs()


if __name__ == "__main__":
    main()
