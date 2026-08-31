# NO/Au H_eff Trotter-error estimate — design & derivation

Status: **generator implemented and verified.**
`scripts/no_au/generate_no_au_heff_heidelberg.py`

- `--verify-reduced`: analytic `dt*E1` == PennyLane `effective_hamiltonian` to
  ~4e-16 for n_metal in {2,3,4}, dt in {3,6,10}; `.op` string round-trip matches
  `e1_terms` to ~1e-15.
- default run: writes `backend_inputs/.../heidelberg_heff/{H_ref,run_001..004}`
  (dt = 10, 6, 4, 3 a.u.  ->  r = 207, 345, 517, 689 over 50 fs), each with
  `benchmark.op` (base H from run_001 + 1440 E1 lines), `benchmark.inp`
  (run_009 tree, repointed), 8 `.dat`, + `run_no_au_heff_heidelberg.sh`.

**Production emits families (a), (b), (c-eps) only** -- pure diagonal-function x
hopping, identical `.op` dialect to the validated 11 runs.  The two
commutator-with-KE families are dropped:
  [H_k,Flast]_T  ~ 1e-5   (negligible),
  [F0,Flast]     -> in the reduced tests the omission is 0.08-0.4% of |E1|,
                   shrinking with n_metal; a small DOWNWARD bias on r*.
`--` re-enable via `include_kinetic=True` in `e1_terms` if the cluster accepts
`|mode V |mode KE` products and a sensitivity check is wanted.

## 1. What we are estimating

Number of **first-order** Trotter steps `r` the GAN algorithm (arXiv:2601.16264)
needs so that

    max_{t <= 50 fs} max_i | P_i^{Trotter}(t) - P_i^{exact}(t) |  <=  epsilon

for the NO/Au(111) benchmark (`benchmark3_no_au_scattering`), `P_i` = molecular
orbital population. Method: arXiv:2606.30738 §IV.B.2 + App. B (empirical
effective-Hamiltonian), with the same-tree difference estimator from
`small_direct_benchmark`.

Targets: **epsilon = 1e-2** (headline, comparable to Maxwell et al.) and
**epsilon = 1e-3** (fine; at the edge of what the ML-MCTDH reference certifies —
its self-consistency floor is ~6e-4). Also report `r*(3e-3)`.

`t = 2067 a.u.` (50 fs window). Full 350 fs event reported separately as the
conservative estimate `r*_350fs ≈ 7 · r*_50fs`.

## 2. Estimator (frozen spec)

1. Tree **T\* = run_009 topology + ranks** (`delta_prev` 5.3e-4 ≈ floor; cheaper
   than run_011). Nuclear DVR changed if needed so MCTDH can form operator
   products (see §5). Integrator tightened to `RK8 = 1d-9, 1d-7`.
2. `H_ref`: propagate psi_0 under the true `H` on T\*, 50 fs, 51 outputs.
3. For each `dt` in the sweep: `H_eff(dt) = H + dt * E1`, propagate on the **same**
   T\* with identical settings.
4. `delta_diff(dt) = max_{t,i} | P_i^{H_eff}(t) - P_i^{H_ref}(t) |`.
   (ML-MCTDH representation + integrator error is common mode and cancels.)
5. Fit `log delta_diff` vs `log dt` -> exponent `alpha`. **Gate: alpha in
   [0.85, 1.15]** (first order => alpha ≈ 1). Drop the largest `dt` (curvature)
   or smallest (floor) and refit if it fails.
6. `a` = slope through origin over the alpha≈1 window.
7. `r*(epsilon) = ceil(a * t / epsilon)`; band `[r*/1.3, r*]` from the small-system
   calibration `E_eff/E_trot -> 1.27`.

**dt sweep:** `{10, 6, 4, 3}` a.u. -> `r = 207, 344, 517, 689`. Start with
`{10, 6, 3}`; add `4` if noisy.

## 3. Fragments (GAN first-order product formula, physical order F0 -> H1..H32 -> Flast)

    F0     = U_N(r,z)  +  n_d [U_A - U_N](r,z)                      (diagonal potential)
    H_k    = W_k(z) ( d^dag c_k + c_k^dag d ),   k = 1..32          (molecule-metal hopping)
    Flast  = T_r + T_z + sum_k eps_k n_k                            (kinetic + metal energies)

with `W_k(z) = vk_k * fz(z)`, `fz(z) = 1 - tanh(z/a_tilde)`,
`Delta U(r,z) = U_A - U_N = (VAr-VNr)(r) + (VAz-VNz)(z) + cdiff`,
`eps_k = ec_k` (metal energies, symmetric, |eps_k| up to 1.83 Ha).

Molecular `eps_d = 0` (the whole neutral–anion gap is inside `Delta U`).

## 4. Leading BCH error operator

PennyLane (`inspect_no_au_bch_structure.py`) gives, for the first-order formula,

    Omega_2 = -1/2 * sum_{a<b} [F_a, F_b]        (every pair, coeff -1/2, MATRIX order)
    E1      = i * Omega_2
    H_eff   = H + dt * E1

VERIFIED sign (reduced dense vs PennyLane, machine precision): in **physical**
fragment order F0 < H_1 < ... < Flast,

    dt * E1 = dt * (+i/2) * sum_{a<b, physical order} [F_a, F_b]

(the +/- flips vs the matrix-order statement above because matrix order is the
reverse of physical order). Emission verified for n_metal in {2,3,4},
dt in {3,6,10}, all four families incl. the kinetic commutators, err ~4e-16.

By the fermion algebra (d = index 0, c_k = index k; `[.,.]` below are the physical
commutators, all anti-Hermitian, so `-(i/2)[.,.]` is Hermitian):

### (a) hopping–hopping   `[H_j, H_k]`, j != k   — 496 pairs, the bulk

    [H_j, H_k] = W_j(z) W_k(z) ( c_j^dag c_k - c_k^dag c_j )

n_d cancels. Metal–metal imaginary hopping, z-dependent via `fz(z)^2`.
For j<k, JW:  c_j^dag c_k - c_k^dag c_j = (i/2) Z_{j+1..k-1} (X_j Y_k - Y_j X_k).

    E1 term = (1/4) vk_j vk_k (-2)^{k-1-j} [ dq2_{j+1} (Idq)_{k+1}
                                            - (Idq)_{j+1} dq2_{k+1} ] * prod_{m=j+2}^{k} q_m * fz2(z)

(Heidelberg modes: c_j -> mode j+1. `fz2` = external1d of `fz(z)^2`.)

### (b) hopping–F0   `[H_k, F0]`   — 32 terms

    [H_k, F0] = W_k(z) * Delta U(r,z) * ( c_k^dag d - d^dag c_k )

z- and r-dependent imaginary molecule–metal hopping. `Delta U` splits into
`(VAr-VNr)(r)`, `(VAz-VNz)(z)`, `cdiff`. Same JW skeleton as the base hopping but
`dq2 <-> Idq` asymmetric (imaginary), coeff carries `dt * (-2)^{k-1}` times each
`Delta U` piece; the r-piece multiplies an r-mode external1d, the z-pieces
multiply `fz*VAz` etc. on the z-mode.

### (c) hopping–Flast   `[H_k, Flast]`   — 32 + 32 terms

Metal-energy part (dominant, |eps_k| large):

    [H_k, Flast]_eps = W_k(z) eps_k ( d^dag c_k - c_k^dag d )

    E1 term = vk_k * ec_k * (-2)^{k-1} [ dq2_0 (Idq)_k - (Idq)_0 dq2_k ]
              * prod_{m=1}^{k-1} q_m * fz(z)

Kinetic-z part:

    [H_k, Flast]_T = [ W_k(z), T_z ] ( d^dag c_k + c_k^dag d )
                   = (i / 2 m_z) ( W_k'(z) p_z + p_z W_k'(z) ) ( d^dag c_k + c_k^dag d )

needs the `fz`–`KE_z` operator product (see §5).

### (d) F0–Flast   `[F0, Flast]`   — few terms

    [F0, Flast] = [ U_N + n_d Delta U ,  T_r + T_z ]

standard potential–kinetic commutators on r and z, with an n_d-independent piece
(`U_N` + `1/2` of the average) and an `n_d` (i.e. `q_d`) piece (`Delta U`). Needs
`V`–`KE` products on both nuclear modes.

Metal–metal-energy, F0–metal-energy: zero (all diagonal in occupation).

## 5. Heidelberg `.op` mechanics — open questions to settle at verification

- **Operator products with `KE`.** Families (c)-kinetic and (d) need
  `A*KE - KE*A` on the nuclear modes. The `small_direct_benchmark` H_eff runs
  switched the nuclear primitive **FFT -> exp** for exactly this reason. Here the
  primitive is **`sin`** (particle-in-box) DVR, which *does* store explicit
  matrices — need to confirm MCTDH accepts `fz*KE` / `VNr*KE` products on `sin`.
  If not: switch r,z to a matrix-storing DVR, or tabulate `W_k'(z)` and use
  `{fzp, KE}`-style symmetric products.
- **`fz2.dat`**: new external1d = `(1 - tanh(z/a_tilde))^2` on the padded z range.
- Term count: ~992 (a) + ~192 (b) + ~128 (c) + ~8 (d) ≈ **1400 extra lines**.
  Fine for MCTDH; watch `.op` parser line limits (use `&&&` continuation, already
  supported by `render_term`).
- Cost: each `H_eff` propagation ≈ same order as an `H` run on T\* (the mean-field
  build sees more terms; expect a modest slowdown, not a blow-up).

## 6. Verification plan (no full dense matrix possible)

`generate_no_au_heff_heidelberg.py --verify-reduced`:

1. Build a **reduced NO/Au**: `N_metal = 3`, `r,z` on 8 points each. Dense
   dimension 2^4 * 8 * 8 = 1024 -> fine.
2. Assemble `F0, H_1..H_3, Flast` as dense matrices; get `H_eff_ref` from
   PennyLane `te.effective_hamiltonian(order=2, timestep=dt)`.
3. Reconstruct `H_eff` from the generated `.op` lines (same tokenizer as the
   small script) as a dense matrix.
4. Require `max |H_eff_op - H_eff_ref| < 1e-9`, Hermiticity < 1e-12.
5. Then trust the generator (index-parametric in `k`, `(j,k)`) for `N_metal = 32`.

## 7. Fallback if §5 blocks us before the cluster window

Report the **naive triangle-inequality bound** `r_UB(eps) = (t^2 / 2 eps) *
sum_{a<b} ||[F_a,F_b]||` (Eq. 2 of arXiv:2606.30738) with the commutator norms
from a reduced NO/Au, plus the **perturbative** `d * |<psi|E1|psi>|` (Thm 3) if
`E1` on the ML-MCTDH state is reachable. Still a correct implementation of the
paper's methodology, just looser than the empirical `delta_diff`.

## 8. Provenance TODO

Confirm the exact citation for the NO/Au parameters (D0=6.61, a0=2.7968,
r0=1.151 eV/Å, Γ=1.5 eV, ã=10 Å, ν=16, KE=1 eV, σ=20 a.u./|p|). Shenvi–Roy–Tully
NO/Au(111) family; the finite-reservoir + grid details point to a specific
quantum-dynamics reproduction — pin it before the writeup.
