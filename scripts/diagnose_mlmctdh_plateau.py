from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_expectation(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    # Current benchmark1_newns_anderson expectation layout:
    # time, norm, nd, nc1..nc6, Qmean, nvib
    if data.shape[1] < 11:
        raise ValueError(
            f"Unexpected expectation format in {path}: "
            f"found {data.shape[1]} columns, expected at least 11."
        )

    return pd.DataFrame(
        {
            "time": data[:, 0],
            "norm": data[:, 1],
            "P_mol": data[:, 2],
            "Q1": data[:, 9],
            "nvib1": data[:, 10],
        }
    )


def aligned_error(method: pd.DataFrame, exact: pd.DataFrame, column: str) -> np.ndarray:
    if len(method) != len(exact) or not np.allclose(method["time"], exact["time"], atol=1e-12, rtol=0):
        raise ValueError("Method and Exact time grids do not match.")
    return np.abs(method[column].to_numpy() - exact[column].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the late-stage ML-MCTDH convergence plateau for benchmark1."
    )
    parser.add_argument("benchmark", nargs="?", default="benchmark1_newns_anderson")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["run_006", "run_007", "run_008", "run_009", "run_010"],
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results" / args.benchmark
    exact_path = results_root / "exact" / "observables.csv"
    if not exact_path.exists():
        raise FileNotFoundError(f"Missing Exact observables: {exact_path}")

    exact = pd.read_csv(exact_path)
    required_exact = {"time", "P_mol", "Q1", "nvib1"}
    missing = required_exact.difference(exact.columns)
    if missing:
        raise ValueError(f"Exact observables are missing columns: {sorted(missing)}")

    run_data: dict[str, pd.DataFrame] = {}
    rows = []

    for run in args.runs:
        expectation_path = results_root / "heidelberg" / run / "raw" / "expectation"
        if not expectation_path.exists():
            print(f"Skipping {run}: missing {expectation_path}")
            continue

        df = read_expectation(expectation_path)
        run_data[run] = df

        row = {"run": run}
        for column in ("P_mol", "Q1", "nvib1"):
            err = aligned_error(df, exact, column)
            idx = int(np.argmax(err))
            row[f"max_error_{column}"] = float(err[idx])
            row[f"time_of_max_{column}"] = float(df["time"].iloc[idx])
            row[f"final_error_{column}"] = float(err[-1])
        rows.append(row)

    if not rows:
        raise RuntimeError("No requested Heidelberg runs were found.")

    output_dir = results_root / "heidelberg" / "plateau_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\nLate-stage error summary")
    print(summary.to_string(index=False))

    # Pairwise checks for the two integrator tests.
    pairs = [("run_007", "run_010"), ("run_008", "run_009")]
    pair_rows = []
    for a, b in pairs:
        if a not in run_data or b not in run_data:
            continue
        da, db = run_data[a], run_data[b]
        if len(da) != len(db) or not np.allclose(da["time"], db["time"], atol=1e-12, rtol=0):
            raise ValueError(f"Time grids do not match for {a} and {b}.")
        pair_row = {"run_a": a, "run_b": b}
        for column in ("P_mol", "Q1", "nvib1"):
            delta = np.abs(da[column].to_numpy() - db[column].to_numpy())
            pair_row[f"max_difference_{column}"] = float(delta.max())
        pair_rows.append(pair_row)

    if pair_rows:
        pairwise = pd.DataFrame(pair_rows)
        pairwise.to_csv(output_dir / "integrator_pairwise.csv", index=False)
        print("\nIntegrator pairwise checks")
        print(pairwise.to_string(index=False))

    # Absolute-error-vs-time plots. One plot per observable.
    for column, ylabel in (
        ("P_mol", r"$|P_{mol}^{ML}-P_{mol}^{Exact}|$"),
        ("Q1", r"$|Q^{ML}-Q^{Exact}|$"),
        ("nvib1", r"$|n_{vib}^{ML}-n_{vib}^{Exact}|$"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for run, df in run_data.items():
            err = aligned_error(df, exact, column)
            ax.plot(df["time"], err, label=run)
        ax.set_xlabel("time [a.u.]")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"absolute_error_{column}.png", dpi=180)
        plt.close(fig)

    # Focus on the early-time region where the P_mol maxima for runs 007/008 occur.
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, df in run_data.items():
        err = aligned_error(df, exact, "P_mol")
        mask = df["time"].to_numpy() <= 6000.0
        ax.plot(df.loc[mask, "time"], err[mask], label=run)
    ax.set_xlabel("time [a.u.]")
    ax.set_ylabel(r"$|P_{mol}^{ML}-P_{mol}^{Exact}|$")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "absolute_error_P_mol_early_time.png", dpi=180)
    plt.close(fig)

    print(f"\nSaved diagnostic outputs to: {output_dir}")


if __name__ == "__main__":
    main()
