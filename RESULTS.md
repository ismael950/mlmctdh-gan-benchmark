# Results inventory

## benchmark1_newns_anderson
Historical ML-MCTDH convergence benchmark.
Convergence data: results/benchmark1_newns_anderson/heidelberg/convergence/mlmctdh_convergence.csv
Best preserved run: run_015
Maximum molecular-population error: 1.5227385725347098e-05
Full raw/checkpoint preserved only for run_015. Intermediate runs retain analysis and reproducible inputs.

## benchmark2_two_molecular
Historical two-molecular-orbital ML-MCTDH benchmark.
Convergence data: results/benchmark2_two_molecular/heidelberg/convergence/mlmctdh_convergence.csv
Best preserved run: run_010
Maximum molecular-population error: 3.729729694867956e-06
Full raw/checkpoint preserved only for run_010. Intermediate runs retain analysis and reproducible inputs.

## benchmark3_no_au_scattering
Primary large ML-MCTDH scaling benchmark: NO/Au(111).
Current rank-convergence study uses 50 fs scouting propagations.

run_005 completed and its full checkpoint was preserved.
run_006 completed and its full 50 fs checkpoint was preserved and verified locally.

Maximum molecular-population change:
- run_004 -> run_005: 7.604649e-04
- run_005 -> run_006: 8.396022e-04

The adaptive refinement target after run_006 was:
layer 2, node 19, mode 1, rank 8 -> 12.

run_007 is currently running on Friday.

Generated numerical results are stored under results/ and are not versioned.
Full Heidelberg checkpoints are stored under checkpoints/ and are not versioned.
For refined Heidelberg runs, benchmark.inp is versioned while reused
benchmark.op and supporting .dat files remain excluded from Git.

## archive
Legacy validation and smoke-test analyses not intended as primary report results.

## basis_convergence
Historical primitive-basis convergence validation.

## small_direct_benchmark
Direct three-way benchmark on one identical small GAN Hamiltonian:
exact propagation vs converged ML-MCTDH vs GAN first-order Trotter, plus a BCH
effective-Hamiltonian estimator of the Trotter error.
Primary error metric: E_X = max_t max_i |n_i^X(t) - n_i^exact(t)|, over the two
molecular populations. Model: 2 molecular + 4 metal orbitals, 1 nuclear mode,
K=32 FFT grid, t_final = 2000 a.u., 201 output times.

### Nuclear-basis convergence
K = 8 / 16 / 32 / 64 checked. K=32 adopted as canonical (matches the exact and
quantum references point-for-point).
Data: results/small_direct_benchmark/basis_convergence/K*.csv

### ML-MCTDH convergence (Heidelberg)
run_001..run_008, 448 -> 4144 nominal ML coefficients.
Max molecular-population error vs exact:
  run_001 (448)  : 2.386e-05
  run_002 (880)  : 9.950e-06
  run_005 (2368) : 5.356e-06
  run_006 (2928) : 3.946e-06
  run_007 (3520) : 3.655e-06
  run_008 (4144) : 3.645e-06
Smallest natural population underflows to 0 from run_005 on (SPF space saturated);
run_007 -> run_008 changes the error by ~1e-8. Converged tree: run_007.
The residual ~3.6e-6 is a numerical floor (integrator tolerance / operator
representation), not SPF truncation.
Data: results/small_direct_benchmark/comparison/mlmctdh_convergence.csv
Versioned inputs: backend_inputs/small_direct_benchmark/heidelberg/run_001..run_007
(run_008 input not yet versioned).

### Trotter sweep (GAN first-order product formula)
N = 200 / 400 / 600 / 800 / 1000 / 1200 / 1400 / 1600 steps,
dt = 2000/N = 10.0 ... 1.25 a.u.
E_trot(dt): 2.461e-05 (dt=10) -> 2.959e-06 (dt=1.25); slope ~ dt^1.
Data: results/small_direct_benchmark/trotter_sweep/convergence.csv

### BCH effective-Hamiltonian estimator (exact propagation of H_eff)
PennyLane pennylane.labs.trotter_error, BCH order 2. H_eff = (i/dt)*Omega,
propagated exactly on the K=32 space.
  E_eff  (H_eff vs exact)   : 7.584e-05 (dt=10) -> 3.761e-06 (dt=1.25), slope ~ dt^1
  E_model (H_eff vs Trotter): 5.123e-05 (dt=10) -> 8.020e-07 (dt=1.25), slope ~ dt^2
E_eff overestimates E_trot by ~3.1x at dt=10, shrinking to ~1.27x at dt=1.25
(expected: BCH truncation is asymptotic in dt).
H_eff Hermiticity error ~4e-19, norm drift ~1e-15.
Data: results/small_direct_benchmark/effective_hamiltonian/convergence.csv

### Three-way comparison
results/small_direct_benchmark/comparison/benchmark_metrics.json
  ML-MCTDH run_008 (4144 coeff)          : max error 3.645e-06
  GAN Trotter dt=2.5 a.u. (800 steps)    : max error 2.959e-06

### Open item
Validate the estimator by propagating H_eff with ML-MCTDH on the converged
(run_007) tree and forming the difference n_i^{H_eff,ml}(t) - n_i^{H,ml}(t).
Needed because E_trot at small dt (~3e-6) is below the ML-MCTDH error floor
(~3.6e-6); the difference estimator only works via common-mode cancellation of
the ML-MCTDH representation error between the two same-tree propagations. Success
criterion: |E_ml(dt) - E_eff(dt)| << E_trot(dt) across the dt sweep.
