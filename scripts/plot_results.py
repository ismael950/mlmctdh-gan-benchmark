from __future__ import annotations

import argparse
from pathlib import Path

from ganbench.plotting import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot observables from a GAN benchmark run."
        )
    )

    parser.add_argument(
        "benchmark",
        help=(
            "Benchmark name, for example "
            "benchmark1_newns_anderson."
        ),
    )

    parser.add_argument(
        "--backend",
        choices=(
            "exact",
            "heidelberg",
            "quantum",
        ),
        default="exact",
        help="Backend whose results should be plotted.",
    )

    args = parser.parse_args()

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    results_directory = (
        project_root
        / "results"
        / args.benchmark
        / args.backend
    )

    if not results_directory.exists():
        raise SystemExit(
            f"Results directory does not exist: "
            f"{results_directory}"
        )

    output_directory = plot_results(
        results_directory
    )

    print(
        "Plots saved to:",
        output_directory,
    )


if __name__ == "__main__":
    main()