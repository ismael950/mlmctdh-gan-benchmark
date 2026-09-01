# NO/Au `E₁` — approximations and their rigorous justification

This document lists **every approximation** made in the NO/Au Trotter-step
estimate, with the mathematics behind it, its size, its sign (does it push `r*`
up or down), and how we check it.

Companion docs: `no_au_heff_methodology.md` (theory of the method),
`no_au_heff_design.md` (implementation / verification of the generator).

---

## 0. Notation

| symbol | meaning |
|---|---|
| `d, d†` | annihilation / creation for the molecular orbital |
| `c_k, c_k†` | annihilation / creation for metal orbital `k = 1 … M`, `M = 32` |
| `n_d = d†d`, `n_k = c_k†c_k` | number operators |
| `r, z` | N–O bond stretch, molecule–surface distance |
| `p_r, p_z` | conjugate momenta, `[q_μ, p_μ] = i` |
| `T_μ = p_μ² / 2m_μ` | nuclear kinetic energy, `μ ∈ {r, z}` |
| `U_N(r,z), U_A(r,z)` | neutral / anion diabatic potential surfaces |
| `ΔU = U_A − U_N` | diabatic gap |
| `W_k(z) = v_k [1 − tanh(z/ã)]` | molecule–metal coupling, orbital `k` |
| `ε_k` | metal orbital energy, `\|ε_k\| ≤ 1.83` Ha (band ±50 eV) |
| `v_k` | coupling prefactor, `v_k ∈ [0.0148, 0.0391]` Ha |
| `m_r ≈ 1.36·10⁴`, `m_z ≈ 5.47·10⁴` a.u. | reduced / total NO mass |

All energies in Hartree (Ha); 1 Ha = 27.211 eV.

---

## 1. The exact target

The GAN first-order product formula applies the fragments in order

```
F_0  <  F_1  <  …  <  F_M  <  F_last            (application order)
```

with

```
F_0    = U_N(r,z)  +  n_d ΔU(r,z)                            (diagonal potential)
F_k    = W_k(z) ( d†c_k + c_k†d ) ,   k = 1 … M              (molecule–metal hopping)
F_last = T_r + T_z + Σ_{k=1}^{M} ε_k n_k                     (nuclear kinetic + metal energies)
```

`H = Σ_a F_a`. One step `S₁(Δt) = e^{−iF_last Δt} ⋯ e^{−iF_0 Δt}`. By BCH,

```
S₁(Δt) = exp{ −iΔt · H_eff(Δt) } ,     H_eff(Δt) = H + Δt·E₁ + Δt²·E₂ + O(Δt³)

E₁ = (i/2) Σ_{a<b} [F_a, F_b]                    ← the object we want
```

(sign and value verified to 4·10⁻¹⁶ against PennyLane `labs.trotter_error`,
see `no_au_heff_design.md` §4). `E₁` is Hermitian because `[F_a,F_b]† =
−[F_a,F_b]` for Hermitian `F_a`, so `i[F_a,F_b]` is Hermitian.

The pairs `(a,b)` fall into **four families**. §2 derives each commutator
exactly; §3 estimates their sizes; §4–8 justify each approximation.

---

## 2. The four commutator families (exact)

Fermion algebra used repeatedly (`j ≠ k`):
`{d, d†} = 1`, `{c_j, c_k†} = δ_{jk}`, all other anticommutators zero;
`[f(q_μ), p_μ] = i f′(q_μ)`.

### 2.1  `[F_0, F_k]`   —   `M` terms

`U_N(r,z)` is multiplicative in the nuclear coordinates and commutes with every
electronic operator and with `W_k(z)`, so `[U_N, F_k] = 0`. Only the `n_d ΔU`
piece contributes:

```
[F_0, F_k] = ΔU(r,z) W_k(z) [ n_d , d†c_k + c_k†d ] .
```

With `[n_d, d†c_k] = d†c_k` and `[n_d, c_k†d] = −c_k†d`:

```
┌─────────────────────────────────────────────────────────────┐
│  [F_0, F_k]  =  W_k(z) · ΔU(r,z) · ( d†c_k − c_k†d )         │
└─────────────────────────────────────────────────────────────┘
```

Anti-Hermitian electronic factor `(d†c_k − c_k†d)`; nuclear factor is a
**product of multiplicative functions** `W_k(z)·ΔU(r,z)`. **Expressible in the
`.op`.**

### 2.2  `[F_j, F_k]`,  `1 ≤ j < k ≤ M`   —   `C(M,2) = 496` terms

```
[F_j, F_k] = W_j(z) W_k(z) [ d†c_j + c_j†d , d†c_k + c_k†d ] .
```

The four sub-commutators:
`[d†c_j, d†c_k] = [c_j†d, c_k†d] = 0`,
`[d†c_j, c_k†d] = −c_k†c_j`,
`[c_j†d, d†c_k] = +c_j†c_k`
(the `n_d`-dependent pieces cancel between the two non-zero brackets). Hence

```
┌─────────────────────────────────────────────────────────────┐
│  [F_j, F_k]  =  W_j(z) W_k(z) · ( c_j†c_k − c_k†c_j )        │
└─────────────────────────────────────────────────────────────┘
```

**`n_d` has cancelled completely** — this is a pure metal↔metal hopping.
Nuclear factor `W_j W_k(z) ∝ [1 − tanh(z/ã)]²` is multiplicative.
**Expressible in the `.op`** (needs one extra tabulated function `fz²`).

### 2.3  `[F_k, F_last]`   —   `M` terms  (two pieces)

`[F_k, T_r] = 0` (`F_k` has no `r`-dependence). The other two:

```
[F_k, Σ_l ε_l n_l] = W_k(z) ε_k ( d†c_k − c_k†d )
                     (only l = k survives; [d†c_k+c_k†d, n_k] = d†c_k − c_k†d)

[F_k, T_z]         = [ W_k(z) , T_z ] ( d†c_k + c_k†d )
                   = (i / 2m_z) ( W_k′(z) p_z + p_z W_k′(z) ) ( d†c_k + c_k†d )
```

```
┌───────────────────────────────────────────────────────────────────────────┐
│  [F_k, F_last]  =  W_k(z) ε_k ( d†c_k − c_k†d )              ← (ε-piece)    │
│                +  (i/2m_z)( W_k′ p_z + p_z W_k′ )( d†c_k + c_k†d )  ← (T-piece) │
└───────────────────────────────────────────────────────────────────────────┘
```

- **ε-piece**: electronic hop × multiplicative `W_k(z)`. **Expressible.** KEPT.
- **T-piece**: contains `W_k′(z) · p_z`, i.e. **(function of z) × (momentum on z)**.
  **NOT expressible** (see §6). DROPPED.

### 2.4  `[F_0, F_last]`   —   a few terms

`[F_0, Σ_l ε_l n_l] = 0` (`F_0` is diagonal in all occupation numbers). Only the
potential–kinetic brackets survive:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ [F_0, F_last] = Σ_{μ∈{r,z}} (i/2m_μ) {  (∂_μ U_N) p_μ + p_μ (∂_μ U_N)           │
│                                       + n_d [ (∂_μ ΔU) p_μ + p_μ (∂_μ ΔU) ] }  │
└───────────────────────────────────────────────────────────────────────────────┘
```

Every term contains `(∂_μ V) · p_μ` — **(function) × (momentum)**.
**NOT expressible** (see §6). DROPPED. This is the textbook split-operator
(potential vs kinetic) Trotter error.

---

## 3. Sizes and the importance hierarchy

Importance-sampling weight (Maxwell et al. §III.C):
`w([A,B]) ≤ 2‖A‖‖B‖`. With `‖d†c_k ± c_k†d‖ = 1`, `‖W_k(z)‖ ≤ 2|v_k|`,
`‖ΔU‖ ≲ 0.3` Ha, `‖ε_k‖ ≤ 1.83` Ha:

| family | per-term weight `w` | # terms | family Σ‖·‖ (triangle) | true family norm (measured / est.) |
|---|---|---|---|---|
| `[F_k,F_last]_ε` | `2\|v_k\|\|ε_k\|` ≈ **0.05** (up to 0.14) | 32 | `Σ\|v_k ε_k\|` ≈ 0.5–0.7 | **≈ 0.2–0.4** (dominant) |
| `[F_k,F_0]` | `2\|v_k\|‖ΔU‖` ≈ **9·10⁻³** | 32 | ≈ 0.14 | ≈ 0.05–0.1 |
| `[F_j,F_k]` | `4\|v_j v_k\|` ≈ **3.6·10⁻³** | 496 | ≈ 0.8 (**loose**) | **0.7–1.9 % of ‖E₁‖** (PennyLane, reduced system) |
| `[F_0,F_last]` | `‖∂_r U_N‖\|p_r\|/m_r` ≈ **4·10⁻⁴** | few | ≈ 5·10⁻⁴ | ≈ 5·10⁻⁴ Ha  (**≈ 0.1–0.2 % of ‖E₁‖**) |
| `[F_k,F_last]_T` | `‖W_k′‖\|p_z\|/m_z` ≈ **2·10⁻⁶** | 32 | ≈ 6·10⁻⁵ | negligible |

`‖E₁‖ ≈ 0.3–0.5` Ha, set by `[F_k,F_last]_ε` (the metal band is ±50 eV, so
`ε_k` is the large factor). The `[F_j,F_k]` family has a **loose** triangle
bound (0.8 Ha) but its true contribution is 0.7–1.9 % because the 496 terms
carry oscillating signs (from the JW strings and the occupied↔empty structure
of `c_j†c_k` across the Fermi sea) — this is exactly the norm-bound over-estimate
Maxwell et al. quantify.

Per-**term** ranking (= the sampling weight): `[F_k,F_last]_ε` ≫ `[F_k,F_0]` >
`[F_j,F_k]` ≫ `[F_0,F_last]` > `[F_k,F_last]_T`.

---

## 4. Approximation A — BCH truncation at `E₁`

**What.** `H_eff(Δt) = H + Δt·E₁ + Δt²·E₂ + …`  →  keep `H + Δt·E₁`, drop
`Δt²·E₂` and higher.

**Validity.** The BCH series is an expansion in `Δt · (nested commutators of the
F_a)`. Truncation after `E₁` is accurate iff

```
Δt · ‖E₁‖  ≪  1 .
```

For NO/Au `‖E₁‖ ≈ 0.3–0.5` Ha, so the boundary is `Δt ≈ 2–3` a.u.:

| Δt (a.u.) | `Δt·‖E₁‖` | regime |
|---|---|---|
| 10 | ≈ 3–5 | series may not converge; propagation stalled (observed) |
| 3 | ≈ 1–1.5 | borderline — `Δt²E₂` non-negligible |
| 1 | ≈ 0.3–0.5 | marginal |
| ≤ 0.5 | ≤ 0.25 | safe |

This is why the sweep was extended to `Δt = 0.5, 0.25` (`run_005/006`).

**Direction / correction.** On the small validation system, `δ_diff` built from
the **truncated** `H_eff` over-estimates the true Trotter error `E_trot` by a
factor that → **1.27** as `Δt → 0` (higher BCH orders, systematically positive).
Hence `δ_diff → a → r*` is an **over-estimate**; the true value sits in
`[ r*/1.27 , r* ]`.

**Verification.** (i) the `α` gate — a clean power law over a `Δt` window is
evidence that `Δt²E₂` is negligible there; (ii) small-system `δ_diff` vs exact
`E_trot` (open item — WSL run).

---

## 5. Approximation B — importance sampling: drop `[F_j, F_k]`

**What.** Omit the entire `[F_j,F_k]` family (496 commutators, 992 `.op` lines)
from `E₁`.

**Justification (Maxwell et al. §III.C).** Rank commutators by `w([A,B]) ≤
2‖A‖‖B‖` and keep the top-weight ones. From §3, `w([F_j,F_k]) ≈ 3.6·10⁻³`, i.e.
**≈ 15× below** the dominant `w([F_k,F_last]_ε) ≈ 0.05`, and **the smallest of
all families per term**. We drop it as one block rather than term-by-term (a
coarser but same-spirit version of the paper's sampling).

**Physical reason it barely touches the observable.** From §2.2, `n_d` cancels
out of `[F_j,F_k]` — it is metal↔metal hopping and does **not act on the
molecular orbital `d`**. Its effect on `P_mol = ⟨n_d⟩` is therefore **second
order** (metal redistribution → modified effective coupling → `P_mol`), whereas
`[F_k,F_last]_ε` and `[F_k,F_0]` modify the molecule–metal hop **directly**
(first order in `P_mol`).

**Size.** Reconstructing `E₁` densely for a reduced NO/Au (`M ∈ {2,3,4}`, small
`r,z` grids) and comparing with / without `[F_j,F_k]`:

```
‖E₁^{no [F_j,F_k]} − E₁^{full}‖  /  ‖E₁^{full}‖   =   0.7 %  (M=2)  →  1.9 %  (M=4)
```

(operator norm; the observable-level miss is smaller — see the physical
argument). Growth with `M` is slow; a conservative read for `M = 32` is
`≲ 5–10 %` of `‖E₁‖`, likely a few % in `P_mol`.

**Direction.** Dropping terms makes `δ_diff` **smaller** → `a` smaller → `r*`
**smaller**. This is a **non-conservative** bias (under-estimate of the step
count). It is bounded and being measured (below).

**Verification — the `check_full` run.** Re-propagate at `Δt = 1` with
`[F_j,F_k]` **added back** (`INCLUDE_METAL_METAL = True`, 1440 `.op` lines) and
compare `δ_diff`:

```
ratio  R  =  δ_diff^{full}(Δt=1)  /  δ_diff^{reduced}(Δt=1)

R < 1.3   →  the drop is validated; report r* from the reduced sweep,
             cite §III.C importance sampling, quote R as the systematic.
R ≥ 1.3   →  restore [F_j,F_k] (or its top-weight subset, e.g. the ~50 pairs
             with both v_j, v_k near the band centre) and re-run the H_eff sweep.
```

---

## 6. Approximation C — drop the kinetic-commutator families

**What.** Omit `[F_0,F_last]` (§2.4) and the `T`-piece of `[F_k,F_last]` (§2.3).

**Why they cannot be in the `.op` (this is the point you asked about).**

`H` itself is a sum of **simple** operators: on every nuclear mode, each term
carries **exactly one** operator — either the kinetic operator `KE`, or a
multiplicative function `V(r)`, `W_k(z)`, … — **never a product `f(q_μ)·KE` or
`f(q_μ)·p_μ` on the same mode**. The GAN fragmentation is designed this way so
each `F_a` is cheaply exponentiable.

The kinetic commutators break that structure. E.g.

```
[V(r), T_r] = [V(r), p_r²/2m_r] = (i/2m_r)( V′(r) p_r + p_r V′(r) )
```

is `(function of r) × (momentum on r)` — a genuinely different operator type. To
write it in an `.op` line you would need two operators on the same mode,
`|mode Vprime |mode dq`. Heidelberg's `sin` DVR primitive does **not** build
such `function·derivative` products reliably (the small benchmark hit exactly
this and had to switch the nuclear primitive **FFT → `exp`** "so MCTDH can form
the operator products"). NO/Au must keep `sin` DVR because `z` is a **scattering
coordinate** (not periodic), so the periodic `exp`/FFT basis is unavailable.

The other three families produce only `(electronic operators) × (multiplicative
nuclear functions)` — those **are** expressible (pre-tabulate the function
products: `fz²`, `fz·U_A`, …). That is the whole reason B-families work and
C-families do not.

**Size.** From §3:

```
‖[F_0,F_last]‖      ≈ (∂_r U_N) · p_r / m_r  ≈  0.15 · 40 / 1.36·10⁴  ≈  4·10⁻⁴ Ha
‖[F_k,F_last]_T‖·M  ≈ (v_k/ã) · p_z / m_z · M ≈ (0.03/18.9)·63/5.5·10⁴·32 ≈ 6·10⁻⁵ Ha
Σ ≈ 5·10⁻⁴ Ha        →   ≈ 0.1–0.2 %  of  ‖E₁‖ ≈ 0.3–0.5 Ha .
```

Small because the nuclei are heavy (`1/m_μ ∼ 10⁻⁴–10⁻⁵`) — the momentum
factors (`p_r ∼ 40`, `p_z ∼ 63`) do not compensate.

**Direction.** Same as B: `δ_diff` smaller → `r*` smaller → **non-conservative**.

**Caveat (your concern — it is legitimate).** Unlike `[F_j,F_k]`, the kinetic
families:
1. scale as **`Δt¹`** (they are genuine first-order BCH terms), and
2. have **no obvious cancellation** in `P_mol` (they are the standard
   split-operator error, which accumulates coherently along a fast trajectory —
   `p_z = 63` a.u.).

If the dominant families cancel in `P_mol` (giving the observed `α ≈ 2`) but
`[F_0,F_last]` does not, then at **sufficiently small `Δt`** the kinetic error
would eventually dominate and pull the true asymptotic exponent back to
`α = 1`, with a coefficient set by `‖[F_0,F_last]‖`. Concretely: if
`δ_kinetic ≈ (‖[F_0,F_last]‖ / ‖E₁‖) · a · Δt ≈ 10⁻³ · a · Δt`, it overtakes a
cancelling `∝ Δt²` term when `Δt ≲ 10⁻³ · (…)` — i.e. only at very small `Δt`
/ very small target `ε`. For `ε ≳ 10⁻³` (our range) it is sub-dominant, but this
sets the floor of what the estimate can claim.

**How to remove it (future work).** Switch `r, z` to a matrix-storing DVR that
supports operator products (`exp` for a periodic coordinate; for scattering,
tabulate `∂_μ V` and `W_k′` explicitly and write symmetric momentum products
`{f, p_μ}`), or use the Hall-basis grouped BCH (Maxwell et al. §III.B) which
recasts the whole `E₁` and would carry these terms natively.

---

## 7. Approximation D — the classical solver (same-tree difference)

**What.** `δ(Δt)` (true observable Trotter error) is unknowable at 49 qubits;
we estimate it by ML-MCTDH on a fixed tree (ranks from `run_009`):

```
δ_diff(Δt) = f_t | P_mol^{H_eff, ML}(t) − P_mol^{H, ML}(t) | .
```

Writing each ML result as exact + ML error `ε_ML`:

```
δ_diff = ( P_mol^{H_eff,exact} − P_mol^{H,exact} )  +  ( ε_ML(H_eff) − ε_ML(H) )
       =            δ(Δt)                            +            ρ
```

`H_eff = H + Δt·E₁` is a small perturbation of `H`, so on the same tree
`ε_ML(H_eff) ≈ ε_ML(H)` ⇒ `ρ ≈ 0` (common-mode cancellation).

**The floor.** `ρ` is small but non-zero — a systematic contamination on every
`δ_diff`. Setting `Δt = 0` (`H_eff = H`) makes `δ = 0`, so `δ_diff = ρ`
exactly. We measure it by differencing two `H` propagations on the same tree
(`H_ref` here vs the earlier `run_009`):

```
floor  =  f_t | P_mol^{H_ref} − P_mol^{run_009} |
       =  4.7·10⁻⁵ (t = 50 fs) ,  1.7·10⁻⁴ (mean last 10) ,  3.6·10⁻⁴ (max_t)
```

**Rule.** A point is usable only if `δ_diff ≳ 10 × floor` (then `\|ρ\|/δ_diff ≲
10 %`). The `f_t` functional and its floor must **match** (final-time `δ_diff`
↔ final-time floor). `max_t` is discarded: its floor is 7× higher (a transient
mid-trajectory phase spike) and its scaling is non-monotonic.

**Verification.** (i) two-tree check (`run_009` vs `run_005`-tree) — pending;
(ii) `lownatpop` monitoring: the H_eff run's lowest SPF natural populations must
stay comparable to the H_ref run's and `≲ 10⁻³`.

---

## 8. Approximation E — finite window and time extrapolation

**Window.** The sweep is 50 fs (`t = 2067` a.u.), the incoming leg of the
scattering event; the full event is ~350 fs.

**Extrapolation.** The perturbative Trotter error `δ(Δt,t) ≈ a(t)·Δt^α` with
`a(t) = ‖∫₀ᵗ U_H(t−s) E₁ U_H(s) ψ₀ ds‖_O` growing ~linearly in `t`. Hence

```
r*(350 fs)  ≈  (350/50) · r*(50 fs)  =  7 · r*(50 fs) .
```

**Direction.** The 50 fs window has weaker molecule–metal coupling than the
point of closest approach, so `a(350 fs) > 7 a(50 fs)` — the ×7 is a **lower
estimate** of the full-event step count.

---

## 9. Total error budget for `r*`

| # | approximation | parameter | size | pushes `r*` | check |
|---|---|---|---|---|---|
| A | BCH truncation at `E₁` | `Δt‖E₁‖` | calib. factor **1.27** | **up** (band `[r*/1.27, r*]`) | `α` gate; small-system vs exact |
| B | drop `[F_j,F_k]` | `w` ratio ≈ 1/15 | **0.7–1.9 %** of `‖E₁‖` (op.); few % in `P_mol` | **down** | `check_full` run (ratio `R`) |
| C | drop kinetic families | `1/m_μ` | **0.1–0.2 %** of `‖E₁‖` | **down** (may set true `α→1` at tiny `Δt`) | operator-norm bound; different DVR (future) |
| D | ML-MCTDH same-tree diff | `ρ / δ_diff` | **± 5–20 %** (points kept at `≳10×` floor) | ± | floor measurement; two-tree check (pending) |
| E | 50 fs → 350 fs (×7) | linearity of `a(t)` | factor 7, **lower bound** | **down** for the full event | redo fit on the full window (future) |
| — | metric spread | `α_final` vs `α_late_mean` | ≈ 1.0× (`ε=10⁻²`) … 1.5× (`ε=10⁻³`) | ± | reported as part of the band |

**Net.** `r*(50 fs)` is defensible to roughly a **factor ~2**, one-sidedly
biased **low** by B + C (bounded, and B is being measured). `r*(350 fs) ≈ 7 ×`
is a **lower estimate**.

---

## 10. What the pending runs resolve

| run | Δt | E₁ | resolves |
|---|---|---|---|
| `run_005` | 0.5 | reduced | is the sweep asymptotic? (`Δt‖E₁‖ ≤ 0.25`) — pins `α` |
| `run_006` | 0.25 | reduced | same; also whether `δ_diff` hits the floor (→ window bottoms out) |
| `check_full` | 1.0 | **full** | approximation **B**: `R = δ_diff^{full}/δ_diff^{reduced}` |
| *(pending)* | 2.0 | reduced, **`run_005`-tree** | approximation **D**: two-tree convergence of `δ_diff` |
| *(pending, WSL)* | sweep | small system, exact ref | approximation **A**: `δ_diff` vs true `E_trot`, and the 1.27 factor |
