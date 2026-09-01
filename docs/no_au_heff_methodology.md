# NO/Au Trotter-step estimation — theory & methodology

Companion to `no_au_heff_design.md` (implementation).  This one explains *why*
each step is what it is: the effective-Hamiltonian idea, the same-tree
difference and its "floor", the scaling law, the extrapolation, and where
`compactoper` fits (it doesn't — see §10).

Notation: `H` Hamiltonian, `t` total physical time, `Δt` Trotter step,
`r = t/Δt` number of steps, `ε` target accuracy, `‖·‖` operator norm,
`⟨O⟩_X(t) = ⟨ψ(t)|O|ψ(t)⟩` with `|ψ(t)⟩` evolved by method `X`.

---

## 0. The question

The GAN quantum algorithm (Lang et al., arXiv:2601.16264) simulates `e^{-iHt}`
for a molecule–metal Hamiltonian with a **first-order Trotter product formula**.
Its resource table quotes "1000 steps" as a placeholder and says a real
step-count analysis is *out of scope*. We supply that number for the
Anderson–Newns / NO(Au) class:

> **How large must `r` be so that the error in a physical observable stays below
> `ε` over the simulated time?**

Method: Maxwell et al., arXiv:2606.30738, §III–IV.

---

## 1. The algorithm: first-order Trotter of the GAN Hamiltonian

Split `H` into fragments that are each cheap to exponentiate:

```
H = Σ_a F_a ,   a = 0 … n
```

For NO/Au (`n = 33`): `F_0` = diagonal potential `U_N(r,z) + n_d[U_A−U_N](r,z)`;
`F_k = W_k(z)(d†c_k + c_k†d)` for `k = 1…32` (molecule↔metal hopping);
`F_last = T_r + T_z + Σ_k ε_k n_k` (kinetic + metal orbital energies).

One first-order Trotter step:

```
S₁(Δt) = e^{-iF_n Δt} ⋯ e^{-iF_1 Δt} e^{-iF_0 Δt}
```

The algorithm approximates `e^{-iHt} ≈ [S₁(Δt)]^r`, `r = t/Δt`. Each step has a
fixed gate cost (the GAN paper optimises this). The **total** cost is
`r × (cost per step)`, so `r` is the missing multiplier.

---

## 2. Trotter error = evolution under an effective Hamiltonian

By the Baker–Campbell–Hausdorff (BCH) formula, a product of exponentials is a
single exponential:

```
S₁(Δt) = exp{ -iΔt·H  −  (Δt²/2) Σ_{a<b} [F_b, F_a]  +  O(Δt³) }
```

Factor out `-iΔt`:

```
S₁(Δt) = exp{ -iΔt·H_eff(Δt) } ,

H_eff(Δt) = H + Δt·E₁ + Δt²·E₂ + …
E₁ = (i/2) Σ_{a<b} [F_a, F_b]          ← leading Trotter-error operator
```

`E₁` is Hermitian (`[F_a,F_b]` is anti-Hermitian, `i·`(anti-Herm) is Herm).
For NO/Au, `[F_a,F_b]` gives four families (derivation + sign check to 4·10⁻¹⁶
against PennyLane in `no_au_heff_design.md`):

| family | operator | per-term size | # |
|---|---|---|---|
| `[F_k,F_last]` | `W_k(z)·ε_k·(d†c_k − c_k†d)` | `~ε_k W_k ~ 0.07` (**dominant**, ε_k up to 1.83 Ha) | 32 |
| `[F_k,F_0]` | `W_k(z)·ΔU(r,z)·(d†c_k − c_k†d)` | `~W_k·ΔU ~ 0.01` | 32 |
| `[F_j,F_k]` | `W_jW_k(z)·(c_j†c_k − c_k†c_j)` | `~W_jW_k ~ 10⁻³` (smallest) | 496 |
| `[F_0,F_last]` | `[V(r,z), T]` (potential–kinetic) | `~∇V·p/m ~ 10⁻⁴` | few |

**The key identity.** Because `H_eff` does not depend on which step,

```
[S₁(Δt)]^r |ψ₀⟩  =  [e^{-iΔt H_eff}]^r |ψ₀⟩  =  e^{-i(rΔt) H_eff} |ψ₀⟩  =  e^{-it H_eff(Δt)} |ψ₀⟩ .
```

**Propagating `H_eff(Δt)` continuously for the physical time `t` reproduces the
Trotterised state exactly** — no need to actually apply `r` discrete steps. The
only approximation is truncating `H_eff` at `E₁` (dropping `Δt²E₂…`), which is
controlled by keeping `Δt` in the regime where `Δt‖E₁‖ ≪ 1` (see §6).

So the Trotter error we want is

```
δ(Δt,t) = | ⟨O⟩_{e^{-itH_eff(Δt)}}(t)  −  ⟨O⟩_{e^{-itH}}(t) |
```

with `O` a physical observable and both states from `|ψ₀⟩`.

---

## 3. The error metric

`O = P_mol = d†d` — the molecular electronic population (the NO⁻ anion
probability, i.e. the charge-transfer signal). Chosen because the hopping terms
`F_k` drive it directly, so it is *sensitive* to Trotter error, and it is the
quantity the physical study (Preston et al.) reports. We take the worst / final
value over the 50 fs window:

```
δ(Δt) = f_t | P_mol^{H_eff}(t) − P_mol^{H}(t) | ,     f_t ∈ {value at t=50 fs, mean over last 10 outputs, max_t}
```

`δ(Δt)` is one number per `Δt`. `ε` (the target) is defined on the *same*
functional.

---

## 4. Why we can't just compute `δ`

`e^{-itH}` and `e^{-itH_eff}` need an exact solver. At NO/Au size (~49 qubits,
Hilbert space `2^33 × 128 × 340`) that is impossible.

- **Small validation system** (6 orbitals, 1 mode, `K=32`): exact state-vector
  propagation *is* possible → we compute the true `δ` there and check the
  estimator below reproduces it.
- **NO/Au**: the best classical solver is **ML-MCTDH** (Heidelberg). It is
  *approximate* — it carries its own error `ε_ML` from (i) truncating the
  single-particle-function (SPF) bases and (ii) the finite integrator
  tolerance.

---

## 5. The same-tree difference estimator — and "the floor"

We propagate **both** `H` and `H_eff(Δt)` with ML-MCTDH **on the identical tree**
(same topology, same SPF ranks = from convergence run `run_009`), and take the
**difference**:

```
δ_diff(Δt) = f_t | P_mol^{H_eff, ML}(t)  −  P_mol^{H, ML}(t) |
```

Write each ML result as exact + its ML error:

```
P_mol^{H_eff, ML}  =  P_mol^{H_eff, exact}  +  ε_ML(H_eff)
P_mol^{H,     ML}  =  P_mol^{H,     exact}  +  ε_ML(H)

δ_diff  =  ( P_mol^{H_eff,exact} − P_mol^{H,exact} )   +   ( ε_ML(H_eff) − ε_ML(H) )
        =  δ(Δt)  [true Trotter error]                 +   ρ  [common-mode residual]
```

`H_eff = H + Δt·E₁` differs from `H` by a *small* perturbation, so the ML error
is nearly the same for both propagations on the same tree: `ε_ML(H_eff) ≈
ε_ML(H)` ⇒ `ρ ≈ 0` ⇒ **`δ_diff ≈ δ(Δt)`**. The ML representation error cancels
*as common mode*. This is why we do not need the ML propagation to be converged
in an absolute sense — only the *difference* has to be.

### The floor

`ρ` is not exactly zero. **The floor is the size of `ρ`** — the smallest
`δ_diff` we can believe. We measure it directly: propagate `H` twice with the
same tree (once as `H_ref` in this sweep, once = the earlier `run_009`), whose
"true" difference is zero, so

```
floor  =  f_t | P_mol^{H_ref, ML}(t)  −  P_mol^{run_009, ML}(t) |
```

Measured: **`floor` ≈ 5·10⁻⁵ (final), 2·10⁻⁴ (late mean), 4·10⁻⁴ (max_t)**.

- `δ_diff ≫ floor`  → the number is signal, trust it.
- `δ_diff ~ floor`  → the Trotter error is below what this estimator resolves;
  that `Δt` point is unusable (either shrink the noise — tighter integrator,
  bigger tree — or accept that the window stops there).

The `max_t` functional has the highest floor (a transient mid-trajectory
wobble); `final` / `late_mean` are ~10× cleaner and are what we use.

---

## 6. The scaling law

First-order Trotter: local error per step `O(Δt²)`, accumulated over `r = t/Δt`
steps → **global error `O(Δt¹)`**. First-order perturbation theory in the
perturbation `Δt·E₁` gives the leading term explicitly:

```
δ(Δt,t)  ≈  Δt · ‖ ∫₀ᵗ U_H(t−s) · E₁ · U_H(s) |ψ₀⟩ ds ‖_O   ≡   a(t) · Δt
```

`U_H(s) = e^{-iHs}`. The integral `a(t)` is **independent of `Δt`** — it depends
only on `(t, ψ₀, O)`. So the expectation is a straight line on a log–log plot:

```
log δ  =  α · log Δt  +  const ,     α = 1  (ideally)
```

Fit `α` from the sweep. Then `a` = slope.

### Why `α` may not be 1

1. **Observable cancellation.** The norm error is `O(Δt)`, but for an
   *observable* the leading contribution can partially cancel. Maxwell et al.
   found this for naphthalene: the Trotter error shifts the spectrum
   approximately uniformly, and a *population* depends on energy *differences*
   (gaps), which are first-order insensitive → effective `α ≈ 2`. Our dominant
   kept family `[F_k,F_last]` (`∝ ε_k`) is exactly a near-uniform spectral
   shift. **Our data: `α` between 1.4 and 2.2** — consistent with partial
   cancellation.

2. **Δt² curvature.** The second-order term `~ (Δt·a·t)²` adds `∝ Δt²` at large
   `Δt` → apparent `α > 1` there. Fixed by staying at small `Δt`.

3. **The floor.** As `Δt → 0`, `δ` flattens toward `floor` → apparent `α`
   steepens between the flat part and the growing part → *spurious* large `α` at
   small `Δt`.

So we fit in a **window** where `α` is stable *and* `δ ≫ floor`. Points outside
that window are dropped. (Dropping a curved large-`Δt` point biases `a`
*upward* → `r*` conservative; dropping a floor-limited small-`Δt` point is
mandatory or `a` is garbage.)

---

## 7. The extrapolation

We want the step `Δt*` at which the error first meets the target `ε`. From the
fitted law `δ(Δt) = c · Δt^α`:

```
δ(Δt*) = ε          ⇒     Δt*  =  Δt_obs · ( ε / δ_obs )^{1/α}

r*(ε)  =  ⌈ t_total / Δt* ⌉  =  ⌈ (t_total / Δt_obs) · ( δ_obs / ε )^{1/α} ⌉
```

(`Δt_obs, δ_obs` = any point in the fit window, e.g. the smallest `Δt`.)
This is the recipe in arXiv:2606.30738 (main text before Eq. between (6)–(7)
and App. B Eq. (B3), generalised from `α = d` to a fitted `α`).

`α` matters a lot: for `ε/δ_obs = 10⁻²`, `α = 1` gives `r* ∝ 10²` more steps;
`α = 2` gives `r* ∝ 10`. This is the whole reason the extra small-`Δt` runs
matter.

### Two corrections

- **Calibration band `[r*/1.27, r*]`.** On the small system, `δ_diff` (built
  from the *truncated* BCH `H_eff = H + Δt·E₁`) overestimates the *true* Trotter
  error `E_trot` by a factor that → **≈ 1.27** as `Δt → 0` (higher BCH orders,
  systematically positive). So `δ_diff` → `a` → `r*` is an over-estimate; the
  true value sits in `[r*/1.27, r*]`.

- **50 fs → 350 fs.** `a(t)` grows roughly linearly in `t` (the integral in §6
  accumulates), so `r*(350 fs) ≈ 7 · r*(50 fs)`. This is a **lower** estimate:
  the 50 fs window is the incoming leg, which couples more weakly than the point
  of closest approach, so the full-event `a` is likely `> 7×`.

---

## 8. Choosing "good precision" `ε`

- **`ε = 10⁻²`** — matches Maxwell et al.'s naphthalene target (`≤ 0.01` on state
  populations); directly comparable.
- **`ε = 10⁻³`** — physically meaningful (~3 % of the `P_mol` peak ≈ 0.036) and
  it is the *tightest target the reference can certify*: the ML-MCTDH
  self-consistency floor for `P_mol` is ~`10⁻³` (it stops improving with more
  SPFs — integrator/representation limited, not truncation). Going tighter would
  be an unverifiable claim.
- We also report `ε = 3·10⁻³` for a third point on the `r*(ε)` curve.

---

## 9. What we validate, and how (the four legs)

For NO/Au there is **no exact Trotter reference**, so the estimate stands on
four checks; none alone is proof, together they are defensible.

1. **Small-system, against exact.** `E_eff_mlmctdh_diff(Δt)` (same-tree
   difference) tracks `E_trot(Δt)` (exact Trotter vs exact) with ratio → ~1.27
   as `Δt → 0`. → the same-tree difference *works as a concept*.
   *(status: the small H_eff sweep still needs to be run — WSL, ~30 min.)*

2. **Self-consistency.** `α` roughly constant over a `Δt` window *and* every
   point in it `≫ floor`.
   *(status: current window `{3,2,1.5,1}` gives `α ≈ 1.4–2.2`, not yet flat;
   `run_005/006` at `Δt = 0.5, 0.25` are meant to settle it.)*

3. **Tree convergence.** Repeat `H_ref` + one `Δt` on a *smaller* tree
   (`run_005`-tree) and check `δ_diff` is unchanged within ~10–20 %. If it moves,
   the common-mode cancellation is tree-dependent → need a bigger tree.
   *(status: not yet done.)*

4. **Truncation check (`check_full`).** Re-run one `Δt` with the **full** `E₁`
   (`[F_j,F_k]` family back in). If `δ_diff` changes `< 1.3×`, dropping that
   family (§10) was legitimate — cite importance sampling. If it changes a lot,
   put the family (or its top-weight subset) back.
   *(status: `check_full` at `Δt = 1` submitted with `run_005/006`.)*

---

## 10. The operator-size problem, and `compactoper`

`E₁` in full = **~1440 sum-of-products terms** in the `.op` file. ML-MCTDH builds
mean-field operators every RHS evaluation with cost `~O(#terms)`, so the full
`E₁` propagation is `~10×` the base `H` → at `Δt = 10` it did not advance past
`t = 0` in 26 min; at `Δt ≤ 3` it is `~20 min/fs` → `~17 h` per 50 fs run.

### Why `compactoper86` did not help (and why you struggled with it)

`compactoper` compresses a sum-of-products operator by finding **shared
single-mode operator structure** across terms and refitting it (natural-potential
/ POTFIT-style). It shines when the operator is essentially a *coordinate
function* that can be re-represented compactly.

Our `E₁` is the opposite:

- Each term is a product of **spin/SQR operators** (`dq²`, `I·dq`, `q`) on the 33
  Jordan–Wigner modes × a tabulated **nuclear** function.
- The JW string is **different for every term** — `q` on a *different subset* of
  the 32 metal modes (`[F_j,F_k]` alone has `C(32,2) = 496` distinct string
  patterns). There is essentially no shared single-mode structure for
  `compactoper` to collapse.
- The SQR spin primitives + `external1d` labels are not a form `compactoper`'s
  refitting handles cleanly.

So `compactoper` either refuses, produces something numerically wrong, or does
not shrink the term count — matching your experience.

### What we did instead: importance sampling (paper §III.C)

Maxwell et al. §III.C: *"strategies that only sample the top-k most important
commutators"*, weight `w([A,B]) ≤ 2‖A‖‖B‖`. We drop the **`[F_j,F_k]` family**
(992 of the 1440 lines):

- per-term weight `~W_jW_k ~ vk_j vk_k ~ 10⁻³` — the **smallest** family (the
  `[F_k,F_last]` terms are ~60× larger);
- it is metal↔metal hopping and **`n_d` cancels out of that commutator**, so it
  does not touch the molecular orbital — its effect on `P_mol` is *second order*;
- reduced-system check: dropping it misses **0.7–1.9 %** of `‖E₁‖` (operator
  norm; observable-level less), growing slowly with `n_metal`.

Result: `E₁` → **436 lines**, propagation → **6.5 min/fs → ~5.5 h / 50 fs run**.
The omission is a small, quantified, **downward** bias on `r*`, and
`check_full` (§9.4) measures it directly.

Separately, the `[F_0,F_last]` kinetic family (~0.1 % of `‖E₁‖`) is dropped
because it needs `|mode V |mode KE` operator products that the sine-DVR `.op`
dialect does not support — a **technical limitation**, stated as a caveat, not
importance sampling.

### The "proper" fix (future work)

Maxwell et al. §III.B: compute the BCH error operator in the **Hall basis** as
commutators of *partial sums* of fragments, `[Σ α_i F_i , Σ β_i F_i]` → `O(n)`
terms instead of `O(n³)`. That is an algebraic restructuring done *before*
emitting the `.op`, and would let the full `E₁` propagate at base-`H` cost. It is
a coding project, not a `compactoper` invocation.

---

## 11. Current status

| `Δt` | `r` (50 fs) | `\|ΔP_mol\|` final | late-mean |
|---|---|---|---|
| 3.0 | 689 | 1.0·10⁻² | 1.2·10⁻² |
| 2.0 | 1034 | 5.8·10⁻³ | 7.2·10⁻³ |
| 1.5 | 1378 | 3.3·10⁻³ | 4.7·10⁻³ |
| 1.0 | 2067 | 9.0·10⁻⁴ | 1.9·10⁻³ |

- floor (final) ≈ 5·10⁻⁵ → all four points are signal.
- per-pair `α`: `3.2, 2.0, 1.4` (final) — **decreasing with `Δt`**; global fit
  `α ≈ 2.2` (final) / `1.7` (late-mean).
- **Preliminary** `r*` (50 fs, using the fitted `α`, band `[/1.27, ·]`):

| `ε` | `r*` @ `α≈2.2` (final) | `r*` @ `α≈1.7` (late-mean) |
|---|---|---|
| 10⁻² | ~700 `[550, 700]` | ~750 `[600, 750]` |
| 3·10⁻³ | ~1200 `[950, 1200]` | ~1560 `[1230, 1560]` |
| 10⁻³ | ~1970 `[1560, 1970]` | ~3010 `[2370, 3010]` |

`×7` for 350 fs (lower estimate): ~5000 / ~8500 / ~14000–21000.

**What `run_005/006/check_full` decide:**
- `α` clean at `Δt = 0.5, 0.25` and points stay `≫ floor` → `α` fixed, `r*`
  fixed (probably the favourable `α ≈ 2` branch).
- `δ` flattens at `~5·10⁻⁴` for `Δt ≤ 0.5` → we have hit the floor; usable window
  is `Δt ∈ [1,3]`, `r*` as above with a stated resolution limit.
- `check_full` ≈ reduced (`< 1.3×`) → `[F_j,F_k]` drop validated.

---

## 12. Symbol glossary

| symbol | meaning |
|---|---|
| `H` | full NO/Au Newns–Anderson Hamiltonian |
| `F_a` | Trotter fragment (`F_0` potential, `F_k` hopping, `F_last` kinetic+ε_k) |
| `S₁(Δt)` | one first-order Trotter step |
| `Δt` / `r` / `t` | Trotter step / step count / total time (`r = t/Δt`, `t = 2067 a.u. = 50 fs`) |
| `H_eff(Δt)` | effective Hamiltonian, `= H + Δt E₁ + Δt² E₂ + …` |
| `E₁` | leading BCH error operator, `(i/2) Σ_{a<b}[F_a,F_b]` |
| `P_mol = ⟨d†d⟩` | molecular electronic population (the observable `O`) |
| `δ(Δt)` | true Trotter error in `P_mol` (unknowable directly for NO/Au) |
| `δ_diff(Δt)` | same-tree ML-MCTDH estimate of `δ(Δt)` |
| `floor` (`ρ`) | common-mode residual of the same-tree difference; smallest trustworthy `δ_diff` |
| `α` | fitted log–log slope of `δ_diff` vs `Δt` (ideally 1; observed 1.4–2.2) |
| `a` | `δ_diff / Δt^α` in the fit window |
| `ε` | target accuracy (`10⁻²`, `3·10⁻³`, `10⁻³`) |
| `r*(ε)` | estimated required step count |
| `1.27` | small-system calibration: truncated-`H_eff` over-estimates `E_trot` by this as `Δt→0` |
