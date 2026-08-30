# `small_direct_benchmark` — results layout

Everything under `results/`, `figures/` and `checkpoints/` is **not versioned**
(see `.gitignore`); it regenerates from the scripts below. This file is the
contract for what each path means.

## Model

One GAN Hamiltonian (`configs/small_direct_benchmark.yaml`):
2 molecular + 4 metal orbitals, 1 nuclear mode, K=32 FFT grid,
`t_final = 2000` a.u., 201 output times. Error metric everywhere:

```
E_X = max_t max_i |n_i^X(t) - n_i^exact(t)|,   i over the 2 molecular orbitals.
```

## `results/small_direct_benchmark/`

| Path | Produced by | Contents |
|---|---|---|
| `exact/reference_populations.csv`, `exact/metadata.json` | `small_benchmark/generate_small_exact_reference.py` | Exact propagation of `H` (state vector, K=32). The ground truth. |
| `basis_convergence/K08.csv … K64.csv` | `small_benchmark/check_small_nuclear_basis_convergence.py` | Nuclear-grid convergence, K = 8/16/32/64. K=32 is canonical. |
| `trotter_sweep/steps_0200.csv … steps_1600.csv`, `trotter_sweep/convergence.csv` | `small_benchmark/generate_small_trotter_sweep.py` | GAN **first-order Trotter** trajectories and `E_trot(dt)`. `dt = 2000/N` for `N` in {200,400,…,1600} ⇒ `dt` in {10,…,1.25} a.u. |
| `effective_hamiltonian/steps_XXXX.csv`, `effective_hamiltonian/convergence.csv` | `small_benchmark/validate_small_effective_hamiltonian.py` (needs `.venv-estimator`) | Exact propagation of `H_eff = H + Δt²·E` (BCH order 2, PennyLane `labs.trotter_error`). `convergence.csv` columns: `E_trot` (= Trotter vs exact), `E_eff` (= H_eff vs exact, the **estimator**), `E_model` (= H_eff vs Trotter, the residual, ~Δt²). |
| `mlmctdh/run_001/ … run_008/` (`observables.csv`, `summary.json`) | Heidelberg ML-MCTDH on the cluster; copied in | SPF-rank convergence sweep, 448 → 4144 coefficients. Converged tree: `run_007`. Versioned inputs: `backend_inputs/small_direct_benchmark/heidelberg/run_00X/benchmark.inp`. |
| `comparison/population_dynamics.csv`, `error_dynamics.csv`, `mlmctdh_convergence.csv`, `basis_convergence.csv`, `trotter_convergence.csv`, `benchmark_metrics.json` | `small_benchmark/build_small_direct_comparison.py` | Assembled 3-way comparison. Auto-selects the finest Trotter trajectory and the most-converged ML-MCTDH run; `trotter_convergence.csv` is a copy of `trotter_sweep/convergence.csv` kept so `comparison/` is self-contained. |
| `_archive/` | — | Nothing deleted, just parked: `exact_K8_legacy/`, `quantum_K8_legacy/` (superseded by K=32); `quantum/`, `quantum_dt2p5/`, `quantum_dt5/` (orphaned single-`dt` copies of `trotter_sweep/steps_*`); `trotter_error_estimator/` (superseded by `effective_hamiltonian/` + figures). |

## `figures/small_direct_benchmark/`

| File | Produced by |
|---|---|
| `population_dynamics.*`, `error_dynamics.*`, `mlmctdh_convergence.*`, `basis_convergence.*`, `trotter_convergence*.*` | `small_benchmark/plot_small_direct_benchmark.py` |
| `trotter_error_estimator_scaling.png` | `small_benchmark/plot_trotter_error_estimator.py` (reads `effective_hamiltonian/convergence.csv`) |
| `trotter_error_estimator_dynamics_dt2p5.{png,csv}` | `small_benchmark/plot_trotter_estimator_dynamics.py` (reads `effective_hamiltonian/steps_0800.csv`) |

## Regeneration order

Run from the repository root.

```bash
python scripts/small_benchmark/generate_small_exact_reference.py
python scripts/small_benchmark/check_small_nuclear_basis_convergence.py
python scripts/small_benchmark/generate_small_trotter_sweep.py
.venv-estimator/Scripts/python scripts/small_benchmark/validate_small_effective_hamiltonian.py
# (ML-MCTDH runs happen on the cluster; drop outputs into results/.../mlmctdh/run_XXX/)
python scripts/small_benchmark/build_small_direct_comparison.py
python scripts/small_benchmark/plot_small_direct_benchmark.py
python scripts/small_benchmark/plot_trotter_error_estimator.py
python scripts/small_benchmark/plot_trotter_estimator_dynamics.py
```
