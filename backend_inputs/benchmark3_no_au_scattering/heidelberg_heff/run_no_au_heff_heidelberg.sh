#!/bin/bash
# Run the NO/Au H_eff sweep with Heidelberg MCTDH (cluster / WSL).
#   bash backend_inputs/.../heidelberg_heff/run_no_au_heff_heidelberg.sh
set -e
: "${MCTDH_DIR:=$HOME/software/MCTDH/mctdh86.10}"
source "$MCTDH_DIR/install/mctdh.profile"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../../results/benchmark3_no_au_scattering/heidelberg_heff"
for run in H_ref run_001 run_002 run_003 run_004; do
    d="$HERE/$run"; [ -d "$d" ] || continue
    echo "=============== $run ==============="
    mkdir -p "$RES/$run"
    cd "$d" && mctdh86 -mnd -w benchmark.inp
done
echo "done -> python scripts/no_au/analyze_no_au_heff.py"
