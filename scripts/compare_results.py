from __future__ import annotations

import argparse
import json
from pathlib import Path

from ganbench.heidelberg.analysis import analyze_heidelberg_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one Heidelberg ML-MCTDH run against Exact."
    )
    parser.add_argument(
        "benchmark",
        help="Benchmark/config stem, e.g. benchmark1_newns_anderson.",
    )
    parser.add_argument(
        "--run",
        default="run_001",
        help="Heidelberg run identifier (default: run_001).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    analysis_directory = analyze_heidelberg_run(
        project_root=project_root,
        benchmark=args.benchmark,
        run_id=args.run,
    )

    summary_path = analysis_directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print("Benchmark:", summary["benchmark"])
    print("Run:", summary["run_id"])
    print("ML coefficients:", summary["n_ml_coefficients"])
    print("Maximum molecular-population error:", summary["P_mol_max_abs_error"])
    print("Time of maximum error:", summary["P_mol_time_of_max_abs_error"], "a.u.")
    print("Final-time molecular-population error:", summary["P_mol_final_abs_error"])
    print("Maximum norm error:", summary["max_norm_error"])
    print("Maximum particle-number error:", summary["max_particle_number_error"])
    print("CPU time:", summary["cpu_seconds"], "s")
    print("Wall time:", summary["wall_seconds"], "s")
    print("Largest lowest natural population (all branches):", summary["largest_lowest_natural_population"])
    print(
        "Largest lowest natural population (expandable branches):",
        summary["largest_expandable_lowest_natural_population"],
    )
    print("Saved analysis:", analysis_directory)
    print(
        "Maximum molecular-orbital population error:",
        summary["max_molecular_orbital_population_error"],
    )
    print(
        "Orbital of maximum error:",
        summary["molecular_orbital_of_max_error"],
    )
    print(
        "Time of maximum molecular-orbital error:",
        summary["time_of_max_molecular_orbital_error"],
        "a.u.",
    )
    print(
        "Final maximum molecular-orbital population error:",
        summary[
            "final_max_molecular_orbital_population_error"
        ],
    )

if __name__ == "__main__":
    main()
