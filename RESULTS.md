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
run_006 currently in progress.
Friday preserves the verified run_005 checkpoint: checkpoints/noau_run005_50fs_raw.tar.gz

## archive
Legacy validation and smoke-test analyses not intended as primary report results.

## basis_convergence
Historical primitive-basis convergence validation.

## small_direct_benchmark
Direct three-way benchmark to be constructed on one identical small GAN Hamiltonian:
Exact propagation vs converged ML-MCTDH vs quantum/Trotter.
Primary error metric: E_X = max_t max_i |n_i^X(t) - n_i^exact(t)|, over the two molecular populations.
