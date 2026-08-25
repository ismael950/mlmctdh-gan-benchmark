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
Direct three-way benchmark to be constructed on one identical small GAN Hamiltonian:
Exact propagation vs converged ML-MCTDH vs quantum/Trotter.
Primary error metric: E_X = max_t max_i |n_i^X(t) - n_i^exact(t)|, over the two molecular populations.
