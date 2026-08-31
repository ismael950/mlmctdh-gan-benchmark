#!/bin/bash
# Run the NO/Au H_eff sweep with Heidelberg MCTDH.
#   bash backend_inputs/.../heidelberg_heff/run_no_au_heff_heidelberg.sh
# Override RUNS to do a subset, e.g.  RUNS='H_ref run_001' bash ...
set -e
: "${MCTDH_DIR:=/data/$USER/software/mctdh86.10}"
: "${MCTDH_BIN:=$MCTDH_DIR/bin/binary/x86_64/mctdh86}"
: "${RUNS:=H_ref run_001 run_002 run_003 run_004}"
source "$MCTDH_DIR/install/mctdh.profile"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../../../results/benchmark3_no_au_scattering/heidelberg_heff"
for run in $RUNS; do
    d="$HERE/$run"; [ -d "$d" ] || continue
    echo "=============== $run  ($(date)) ==============="
    mkdir -p "$RES/$run"
    cd "$d" && "$MCTDH_BIN" -mnd -w benchmark.inp
done
echo "done -> python scripts/no_au/analyze_no_au_heff.py"
