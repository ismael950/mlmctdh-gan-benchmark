from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from ganbench.heidelberg.parser import (
    read_expectation,
    read_lowest_natural_populations,
    read_runtime_info,
)
from ganbench.heidelberg.resources import (
    branch_diagnostics_from_input,
    count_ml_coefficients_from_input,
)
from ganbench.model import load_config


def _load_exact_observables(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    return {name: data[name] for name in data.dtype.names or ()}


def _indexed_columns(
    available: dict[str, np.ndarray],
    base: str,
    count: int,
) -> list[str]:
    """Resolve Heidelberg observable names for one indexed family."""

    if count == 1 and base in available:
        return [base]

    candidates = [f"{base}{index}" for index in range(1, count + 1)]
    if all(name in available for name in candidates):
        return candidates

    # As a fallback, accept names like nd1, nd2, ... discovered in the file.
    pattern = re.compile(rf"^{re.escape(base)}(\d+)$")
    discovered = sorted(
        (
            (int(match.group(1)), name)
            for name in available
            if (match := pattern.match(name))
        ),
        key=lambda item: item[0],
    )
    names = [name for _, name in discovered]
    if len(names) >= count:
        return names[:count]

    raise KeyError(
        f"Could not find {count} Heidelberg columns for observable family "
        f"{base!r}. Available columns: {sorted(available)}"
    )


def _error_record(
    name: str,
    times: np.ndarray,
    reference: np.ndarray,
    method: np.ndarray,
) -> dict[str, float | str]:
    absolute_error = np.abs(method - reference)
    max_index = int(np.argmax(absolute_error))
    return {
        "observable": name,
        "max_abs_error": float(absolute_error[max_index]),
        "time_of_max_abs_error": float(times[max_index]),
        "final_abs_error": float(absolute_error[-1]),
    }


def analyze_heidelberg_run(
    project_root: str | Path,
    benchmark: str,
    run_id: str,
) -> Path:
    """Analyze one completed Heidelberg run against the exact reference."""

    project_root = Path(project_root)
    run_directory = (
        project_root
        / "results"
        / benchmark
        / "heidelberg"
        / run_id
    )
    raw_directory = run_directory / "raw"
    backend_input_directory = (
        project_root
        / "backend_inputs"
        / benchmark
        / "heidelberg"
        / run_id
    )
    exact_directory = project_root / "results" / benchmark / "exact"

    required = [
        raw_directory / "expectation",
        raw_directory / "lownatpop",
        raw_directory / "output",
        backend_input_directory / "benchmark.inp",
        exact_directory / "observables.csv",
        project_root / "configs" / f"{benchmark}.yaml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The run cannot be analyzed because these files are missing:\n"
            + "\n".join(missing)
        )

    config = load_config(str(project_root / "configs" / f"{benchmark}.yaml"))
    exact = _load_exact_observables(exact_directory / "observables.csv")
    heidelberg = read_expectation(raw_directory / "expectation")

    exact_times = exact["time"]
    heidelberg_times = heidelberg["time"]
    if exact_times.shape != heidelberg_times.shape or not np.allclose(
        exact_times,
        heidelberg_times,
    ):
        raise ValueError("Exact and Heidelberg time grids do not match.")

    errors: list[dict[str, float | str]] = []

    # Molecular orbital populations and total molecular population.
    h_molecular_names = _indexed_columns(
        heidelberg,
        "nd",
        config.n_molecular_orbitals,
    )
    h_molecular = np.column_stack(
        [heidelberg[name] for name in h_molecular_names]
    )
    e_molecular = np.column_stack(
        [exact[f"d{index}"] for index in range(1, config.n_molecular_orbitals + 1)]
    )

    for index in range(config.n_molecular_orbitals):
        errors.append(
            _error_record(
                f"d{index + 1}",
                exact_times,
                e_molecular[:, index],
                h_molecular[:, index],
            )
        )

    errors.append(
        _error_record(
            "P_mol",
            exact_times,
            np.sum(e_molecular, axis=1),
            np.sum(h_molecular, axis=1),
        )
    )

    # Metal orbital populations.
    h_metal_names = _indexed_columns(
        heidelberg,
        "nc",
        config.n_metal_orbitals,
    )
    h_metal = np.column_stack(
        [heidelberg[name] for name in h_metal_names]
    )
    e_metal = np.column_stack(
        [exact[f"c{index}"] for index in range(1, config.n_metal_orbitals + 1)]
    )

    for index in range(config.n_metal_orbitals):
        errors.append(
            _error_record(
                f"c{index + 1}",
                exact_times,
                e_metal[:, index],
                h_metal[:, index],
            )
        )

    # Nuclear coordinates and vibrational occupations if present in the run.
    for mode in range(1, config.n_vibrational_modes + 1):
        exact_q_name = f"Q{mode}"
        exact_n_name = f"nvib{mode}"

        q_candidates = (
            ["Qmean"] if config.n_vibrational_modes == 1 else []
        ) + [f"Qmean{mode}", f"Q{mode}"]
        n_candidates = (
            ["nvib"] if config.n_vibrational_modes == 1 else []
        ) + [f"nvib{mode}"]

        q_name = next((name for name in q_candidates if name in heidelberg), None)
        n_name = next((name for name in n_candidates if name in heidelberg), None)

        if exact_q_name in exact and q_name is not None:
            errors.append(
                _error_record(
                    exact_q_name,
                    exact_times,
                    exact[exact_q_name],
                    heidelberg[q_name],
                )
            )

        if exact_n_name in exact and n_name is not None:
            errors.append(
                _error_record(
                    exact_n_name,
                    exact_times,
                    exact[exact_n_name],
                    heidelberg[n_name],
                )
            )

    # Conservation diagnostic from electronic populations.
    heidelberg_particle_number = np.sum(h_molecular, axis=1) + np.sum(h_metal, axis=1)
    particle_number_error = float(
        np.max(np.abs(heidelberg_particle_number - config.n_electrons))
    )
    norm_error = float(np.max(np.abs(heidelberg["norm"] - 1.0)))

    # Natural-population diagnostics.
    nat_times, nat_columns = read_lowest_natural_populations(
        raw_directory / "lownatpop"
    )
    branch_info = branch_diagnostics_from_input(backend_input_directory / "benchmark.inp")
    natural_population_rows: list[dict[str, float | int | str | bool]] = []
    for column in nat_columns:
        max_index = int(np.argmax(column.values))
        branch = branch_info.get((column.node, column.mode), {})
        natural_population_rows.append(
            {
                "layer": column.layer,
                "node": column.node,
                "mode": column.mode,
                "child": branch.get("child", "unknown"),
                "rank": branch.get("rank", -1),
                "immediate_capacity": branch.get("immediate_capacity", -1),
                "expandable": branch.get("expandable", False),
                "max_lowest_population": float(column.values[max_index]),
                "time_of_max_lowest_population": float(nat_times[max_index]),
                "final_lowest_population": float(column.values[-1]),
            }
        )

    # ML representation size.
    n_coefficients, coefficient_breakdown = count_ml_coefficients_from_input(
        backend_input_directory / "benchmark.inp"
    )

    runtime = read_runtime_info(raw_directory / "output")

    analysis_directory = run_directory / "analysis"
    analysis_directory.mkdir(parents=True, exist_ok=True)

    with (analysis_directory / "errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0].keys()))
        writer.writeheader()
        writer.writerows(errors)

    with (analysis_directory / "natural_populations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(natural_population_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(natural_population_rows)

    with (analysis_directory / "coefficient_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "node",
            "parent_rank",
            "child_dimensions",
            "n_coefficients",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in coefficient_breakdown:
            writer.writerow(
                {
                    "node": item.node,
                    "parent_rank": item.parent_rank,
                    "child_dimensions": "x".join(map(str, item.child_dimensions)),
                    "n_coefficients": item.n_coefficients,
                }
            )

    error_lookup = {str(row["observable"]): row for row in errors}

    molecular_orbital_errors = [
        error_lookup[f"d{index}"]
        for index in range(
            1,
            config.n_molecular_orbitals + 1,
        )
    ]

    worst_molecular_orbital_error = max(
        molecular_orbital_errors,
        key=lambda row: float(
            row["max_abs_error"]
        ),
    )

    max_molecular_orbital_population_error = float(
        worst_molecular_orbital_error[
            "max_abs_error"
        ]
    )

    molecular_orbital_of_max_error = int(
        str(
            worst_molecular_orbital_error[
                "observable"
            ]
        )[1:]
    )

    time_of_max_molecular_orbital_error = float(
        worst_molecular_orbital_error[
            "time_of_max_abs_error"
        ]
    )

    final_max_molecular_orbital_population_error = max(
        float(row["final_abs_error"])
        for row in molecular_orbital_errors
    )

    expandable_natpops = [
        row for row in natural_population_rows if bool(row["expandable"])
    ]

    summary = {
        "benchmark": benchmark,
        "backend": "heidelberg",
        "run_id": run_id,
        "n_ml_coefficients": n_coefficients,
        "max_norm_error": norm_error,
        "max_particle_number_error": particle_number_error,
        "cpu_seconds": runtime.cpu_seconds,
        "wall_seconds": runtime.wall_seconds,
        "hostname": runtime.hostname,
        "P_mol_max_abs_error": error_lookup["P_mol"]["max_abs_error"],
        "P_mol_final_abs_error": error_lookup["P_mol"]["final_abs_error"],
        "P_mol_time_of_max_abs_error": error_lookup["P_mol"]["time_of_max_abs_error"],
        "largest_lowest_natural_population": max(
            row["max_lowest_population"] for row in natural_population_rows
        ),
        "largest_expandable_lowest_natural_population": (
            max(row["max_lowest_population"] for row in expandable_natpops)
            if expandable_natpops
            else None
        ),
        "max_molecular_orbital_population_error": (
            max_molecular_orbital_population_error
        ),
        "molecular_orbital_of_max_error": (
            molecular_orbital_of_max_error
        ),
        "time_of_max_molecular_orbital_error": (
            time_of_max_molecular_orbital_error
        ),
        "final_max_molecular_orbital_population_error": (
            final_max_molecular_orbital_population_error
        ),
    }

    with (analysis_directory / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return analysis_directory
