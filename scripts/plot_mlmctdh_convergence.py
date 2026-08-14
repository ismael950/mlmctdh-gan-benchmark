from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_PATTERN = re.compile(r"^run_(\d+)$")


def run_number(path: Path) -> int:
    match = RUN_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Invalid run directory name: {path.name}")
    return int(match.group(1))


def collect_runs(project_root: Path, benchmark: str) -> pd.DataFrame:
    
    heidelberg_root = (
        project_root
        / "results"
        / benchmark
        / "heidelberg"
    )

    rows = []

    for run_dir in heidelberg_root.glob("run_*"):
        match = RUN_PATTERN.match(run_dir.name)
        if match is None:
            continue

        summary_file = run_dir / "analysis" / "summary.json"
        if not summary_file.exists():
            print(f"Skipping {run_dir.name}: no analysis/summary.json")
            continue

        with summary_file.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        rows.append(
            {
                "run": run_dir.name,
                "run_number": int(match.group(1)),
                "n_ml_coefficients": summary["n_ml_coefficients"],
                "max_molecular_orbital_population_error": (
                    summary["max_molecular_orbital_population_error"]
                ),
                "final_max_molecular_orbital_population_error": (
                    summary["final_max_molecular_orbital_population_error"]
                ),
                "largest_expandable_lowest_natural_population":
                    summary.get(
                        "largest_expandable_lowest_natural_population"
                    ),
                "cpu_seconds": summary.get("cpu_seconds"),
                "wall_seconds": summary.get("wall_seconds"),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No analyzed Heidelberg runs found in {heidelberg_root}"
        )

    return (
        pd.DataFrame(rows)
        .sort_values("run_number")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ML-MCTDH maximum molecular-population error "
            "against the number of ML coefficients."
        )
    )
    parser.add_argument(
        "benchmark",
        help="Benchmark name, e.g. benchmark1_newns_anderson",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    data = collect_runs(project_root, args.benchmark)

    output_dir = (
        project_root
        / "results"
        / args.benchmark
        / "heidelberg"
        / "convergence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "mlmctdh_convergence.csv"
    figure_file = output_dir / "error_vs_coefficients.png"

    data.to_csv(csv_file, index=False)

    fig, ax = plt.subplots(figsize=(8, 5.2))

    ax.plot(
        data["n_ml_coefficients"],
        data["max_molecular_orbital_population_error"],
        marker="o",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of time-dependent ML coefficients")
    ax.set_ylabel(
    "Maximum molecular-orbital population error"
    )
    ax.set_title(f"ML-MCTDH convergence — {args.benchmark}")
    ax.grid(True, which="both", alpha=0.25)

    for _, row in data.iterrows():
        ax.annotate(
            row["run"],
            (
                row["n_ml_coefficients"],
                row["max_molecular_orbital_population_error"],
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(figure_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Runs included:")
    print(
        data[
            [
                "run",
                "n_ml_coefficients",
                "max_molecular_orbital_population_error",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved table: {csv_file}")
    print(f"Saved figure: {figure_file}")


if __name__ == "__main__":
    main()