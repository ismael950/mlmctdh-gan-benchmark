#!/bin/bash
# Run the H_eff ML-MCTDH sweep with Heidelberg MCTDH (WSL / Linux).
#
#   bash scripts/small_benchmark/run_heff_heidelberg.sh
#
# Inputs come from backend_inputs/small_direct_benchmark/heidelberg_heff/<run>/ ;
# raw MCTDH output goes to results/small_direct_benchmark/heidelberg_heff/<run>/raw/
# (set by the `name =` line of each benchmark.inp).
#
# Assumes MCTDH_DIR is set and $MCTDH_DIR/install/mctdh.profile has been sourced,
# or set MCTDH_DIR below.
set -e

: "${MCTDH_DIR:=$HOME/software/MCTDH/mctdh86.10}"
source "$MCTDH_DIR/install/mctdh.profile"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="$REPO/backend_inputs/small_direct_benchmark/heidelberg_heff"

RESULTS="$REPO/results/small_direct_benchmark/heidelberg_heff"

for run in H_ref; do
    d="$BASE/$run"
    [ -d "$d" ] || { echo "skip $run (missing)"; continue; }
    echo "=================== $run ==================="
    mkdir -p "$RESULTS/$run"        # -mnd only creates the final 'raw' component
    cd "$d"
    mctdh86 -mnd -w benchmark.inp
done

echo "done.  Analyse with:"
echo "  python scripts/small_benchmark/analyze_heff_heidelberg.py"
