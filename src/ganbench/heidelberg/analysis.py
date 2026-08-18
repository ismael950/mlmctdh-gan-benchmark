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


def _load_exact_observables(
    path: Path,
) -> dict[str, np.ndarray]:

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
    )

    if data.shape == ():
        data = np.array(
            [data],
            dtype=data.dtype,
        )

    return {
        name: data[name]
        for name in data.dtype.names or ()
    }


def _indexed_columns(
    available: dict[str, np.ndarray],
    base: str,
    count: int,
) -> list[str]:
    """
    Resolve Heidelberg observable names for one indexed family.

    Examples:
        nd              for one molecular orbital
        nd1, nd2, ...   for several molecular orbitals
        nc1, nc2, ...
    """

    if (
        count == 1
        and base in available
    ):
        return [base]

    candidates = [
        f"{base}{index}"
        for index in range(
            1,
            count + 1,
        )
    ]

    if all(
        name in available
        for name in candidates
    ):
        return candidates

    pattern = re.compile(
        rf"^{re.escape(base)}(\d+)$"
    )

    discovered = sorted(
        (
            (
                int(
                    match.group(1)
                ),
                name,
            )
            for name in available
            if (
                match
                := pattern.match(name)
            )
        ),
        key=lambda item: item[0],
    )

    names = [
        name
        for _, name in discovered
    ]

    if len(names) >= count:
        return names[:count]

    raise KeyError(
        f"Could not find {count} Heidelberg columns "
        f"for observable family {base!r}. "
        f"Available columns: {sorted(available)}"
    )


def _error_record(
    name: str,
    times: np.ndarray,
    reference: np.ndarray,
    method: np.ndarray,
) -> dict[str, float | str]:

    absolute_error = np.abs(
        method
        - reference
    )

    max_index = int(
        np.argmax(
            absolute_error
        )
    )

    return {
        "observable": name,
        "max_abs_error": float(
            absolute_error[
                max_index
            ]
        ),
        "time_of_max_abs_error": float(
            times[
                max_index
            ]
        ),
        "final_abs_error": float(
            absolute_error[-1]
        ),
    }


def _find_physical_coordinate(
    heidelberg: dict[str, np.ndarray],
    name: str,
) -> str | None:

    candidates = [
        f"{name}mean",
        name,
    ]

    return next(
        (
            candidate
            for candidate in candidates
            if candidate in heidelberg
        ),
        None,
    )


def _write_observables_csv(
    path: Path,
    columns: dict[str, np.ndarray],
) -> None:

    names = list(
        columns
    )

    if not names:
        raise ValueError(
            "No observables were supplied."
        )

    lengths = {
        len(
            np.asarray(
                values
            )
        )
        for values in columns.values()
    }

    if len(lengths) != 1:
        raise ValueError(
            "Observable columns have inconsistent lengths."
        )

    n_rows = lengths.pop()

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            names
        )

        for row_index in range(
            n_rows
        ):

            writer.writerow(
                [
                    columns[name][
                        row_index
                    ]
                    for name in names
                ]
            )


def analyze_heidelberg_run(
    project_root: str | Path,
    benchmark: str,
    run_id: str,
    input_run_id: str | None = None,
) -> Path:
    """
    Analyze one completed Heidelberg calculation.

    Two cases are supported automatically:

    1. Exact reference available:
       compute ML-MCTDH vs exact errors.

    2. Exact reference unavailable:
       compute conservation, observables, natural
       populations, resources, and runtime only.

    input_run_id can be used for diagnostic runs such as
    smoke_001 that used the ML tree from run_001.
    """

    project_root = Path(
        project_root
    )

    if input_run_id is None:
        input_run_id = run_id

    run_directory = (
        project_root
        / "results"
        / benchmark
        / "heidelberg"
        / run_id
    )

    raw_directory = (
        run_directory
        / "raw"
    )

    backend_input_directory = (
        project_root
        / "backend_inputs"
        / benchmark
        / "heidelberg"
        / input_run_id
    )

    config_path = (
        project_root
        / "configs"
        / f"{benchmark}.yaml"
    )

    exact_path = (
        project_root
        / "results"
        / benchmark
        / "exact"
        / "observables.csv"
    )

    required = [
        raw_directory
        / "expectation",

        raw_directory
        / "lownatpop",

        raw_directory
        / "output",

        backend_input_directory
        / "benchmark.inp",

        config_path,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "The run cannot be analyzed because "
            "these files are missing:\n"
            + "\n".join(
                missing
            )
        )

    config = load_config(
        str(
            config_path
        )
    )

    heidelberg = read_expectation(
        raw_directory
        / "expectation"
    )

    heidelberg_times = (
        heidelberg[
            "time"
        ]
    )

    # ========================================================
    # Electronic observables
    # ========================================================

    h_molecular_names = (
        _indexed_columns(
            heidelberg,
            "nd",
            config.n_molecular_orbitals,
        )
    )

    h_molecular = np.column_stack(
        [
            heidelberg[name]
            for name
            in h_molecular_names
        ]
    )

    h_metal_names = (
        _indexed_columns(
            heidelberg,
            "nc",
            config.n_metal_orbitals,
        )
    )

    h_metal = np.column_stack(
        [
            heidelberg[name]
            for name
            in h_metal_names
        ]
    )

    total_molecular_population = np.sum(
        h_molecular,
        axis=1,
    )

    total_metal_population = np.sum(
        h_metal,
        axis=1,
    )

    total_particle_number = (
        total_molecular_population
        + total_metal_population
    )

    particle_number_error = float(
        np.max(
            np.abs(
                total_particle_number
                - config.n_electrons
            )
        )
    )

    norm_error = float(
        np.max(
            np.abs(
                heidelberg["norm"]
                - 1.0
            )
        )
    )

    # ========================================================
    # Standardized Heidelberg observables
    # ========================================================

    observables: dict[
        str,
        np.ndarray
    ] = {
        "time": heidelberg_times,
        "norm": heidelberg["norm"],
    }

    for index in range(
        config.n_molecular_orbitals
    ):

        observables[
            f"d{index + 1}"
        ] = (
            h_molecular[
                :,
                index,
            ]
        )

    observables[
        "P_mol"
    ] = total_molecular_population

    for index in range(
        config.n_metal_orbitals
    ):

        observables[
            f"c{index + 1}"
        ] = (
            h_metal[
                :,
                index,
            ]
        )

    observables[
        "N_total"
    ] = total_particle_number

    # ========================================================
    # Nuclear observables
    # ========================================================

    physical_coordinate_summary: dict[
        str,
        dict[str, float]
    ] = {}

    if (
        config.uses_physical_nuclear_coordinates
    ):

        for coordinate in (
            config.nuclear_coordinates
        ):

            column_name = (
                _find_physical_coordinate(
                    heidelberg,
                    coordinate.name,
                )
            )

            if column_name is None:
                continue

            values = heidelberg[
                column_name
            ]

            observables[
                coordinate.name
            ] = values

            physical_coordinate_summary[
                coordinate.name
            ] = {
                "initial": float(
                    values[0]
                ),
                "final": float(
                    values[-1]
                ),
                "minimum": float(
                    np.min(
                        values
                    )
                ),
                "maximum": float(
                    np.max(
                        values
                    )
                ),
            }

    else:

        for mode in range(
            1,
            config.n_vibrational_modes
            + 1,
        ):

            q_candidates = (
                (
                    ["Qmean"]
                    if (
                        config.n_vibrational_modes
                        == 1
                    )
                    else []
                )
                + [
                    f"Qmean{mode}",
                    f"Q{mode}",
                ]
            )

            n_candidates = (
                (
                    ["nvib"]
                    if (
                        config.n_vibrational_modes
                        == 1
                    )
                    else []
                )
                + [
                    f"nvib{mode}",
                ]
            )

            q_name = next(
                (
                    name
                    for name
                    in q_candidates
                    if name in heidelberg
                ),
                None,
            )

            n_name = next(
                (
                    name
                    for name
                    in n_candidates
                    if name in heidelberg
                ),
                None,
            )

            if q_name is not None:

                observables[
                    f"Q{mode}"
                ] = (
                    heidelberg[
                        q_name
                    ]
                )

            if n_name is not None:

                observables[
                    f"nvib{mode}"
                ] = (
                    heidelberg[
                        n_name
                    ]
                )

    # ========================================================
    # Optional exact-reference comparison
    # ========================================================

    has_exact_reference = (
        exact_path.exists()
    )

    errors: list[
        dict[str, float | str]
    ] = []

    if has_exact_reference:

        exact = (
            _load_exact_observables(
                exact_path
            )
        )

        exact_times = (
            exact[
                "time"
            ]
        )

        if (
            exact_times.shape
            != heidelberg_times.shape
            or not np.allclose(
                exact_times,
                heidelberg_times,
            )
        ):

            raise ValueError(
                "Exact and Heidelberg time grids "
                "do not match."
            )

        e_molecular = np.column_stack(
            [
                exact[
                    f"d{index}"
                ]
                for index in range(
                    1,
                    config.n_molecular_orbitals
                    + 1,
                )
            ]
        )

        for index in range(
            config.n_molecular_orbitals
        ):

            errors.append(
                _error_record(
                    f"d{index + 1}",
                    exact_times,
                    e_molecular[
                        :,
                        index,
                    ],
                    h_molecular[
                        :,
                        index,
                    ],
                )
            )

        errors.append(
            _error_record(
                "P_mol",
                exact_times,
                np.sum(
                    e_molecular,
                    axis=1,
                ),
                total_molecular_population,
            )
        )

        e_metal = np.column_stack(
            [
                exact[
                    f"c{index}"
                ]
                for index in range(
                    1,
                    config.n_metal_orbitals
                    + 1,
                )
            ]
        )

        for index in range(
            config.n_metal_orbitals
        ):

            errors.append(
                _error_record(
                    f"c{index + 1}",
                    exact_times,
                    e_metal[
                        :,
                        index,
                    ],
                    h_metal[
                        :,
                        index,
                    ],
                )
            )

        if (
            not config.uses_physical_nuclear_coordinates
        ):

            for mode in range(
                1,
                config.n_vibrational_modes
                + 1,
            ):

                q_name = (
                    f"Q{mode}"
                )

                n_name = (
                    f"nvib{mode}"
                )

                if (
                    q_name in exact
                    and q_name
                    in observables
                ):

                    errors.append(
                        _error_record(
                            q_name,
                            exact_times,
                            exact[q_name],
                            observables[
                                q_name
                            ],
                        )
                    )

                if (
                    n_name in exact
                    and n_name
                    in observables
                ):

                    errors.append(
                        _error_record(
                            n_name,
                            exact_times,
                            exact[n_name],
                            observables[
                                n_name
                            ],
                        )
                    )

    # ========================================================
    # Natural populations
    # ========================================================

    (
        nat_times,
        nat_columns,
    ) = (
        read_lowest_natural_populations(
            raw_directory
            / "lownatpop"
        )
    )

    branch_info = (
        branch_diagnostics_from_input(
            backend_input_directory
            / "benchmark.inp"
        )
    )

    natural_population_rows: list[
        dict[
            str,
            float
            | int
            | str
            | bool
        ]
    ] = []

    for column in nat_columns:

        max_index = int(
            np.argmax(
                column.values
            )
        )

        branch = branch_info.get(
            (
                column.node,
                column.mode,
            ),
            {},
        )

        natural_population_rows.append(
            {
                "layer": column.layer,
                "node": column.node,
                "mode": column.mode,
                "child": branch.get(
                    "child",
                    "unknown",
                ),
                "rank": branch.get(
                    "rank",
                    -1,
                ),
                "immediate_capacity": (
                    branch.get(
                        "immediate_capacity",
                        -1,
                    )
                ),
                "expandable": branch.get(
                    "expandable",
                    False,
                ),
                "max_lowest_population": float(
                    column.values[
                        max_index
                    ]
                ),
                "time_of_max_lowest_population": (
                    float(
                        nat_times[
                            max_index
                        ]
                    )
                ),
                "final_lowest_population": float(
                    column.values[-1]
                ),
            }
        )

    # ========================================================
    # ML representation size
    # ========================================================

    (
        n_coefficients,
        coefficient_breakdown,
    ) = (
        count_ml_coefficients_from_input(
            backend_input_directory
            / "benchmark.inp"
        )
    )

    runtime = read_runtime_info(
        raw_directory
        / "output"
    )

    # ========================================================
    # Analysis directory
    # ========================================================

    analysis_directory = (
        run_directory
        / "analysis"
    )

    analysis_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Standardized observables
    # --------------------------------------------------------

    _write_observables_csv(
        analysis_directory
        / "observables.csv",
        observables,
    )

    # --------------------------------------------------------
    # Exact errors
    #
    # Always create the CSV, even without an exact reference.
    # This gives downstream code a predictable file layout.
    # --------------------------------------------------------

    error_fieldnames = [
        "observable",
        "max_abs_error",
        "time_of_max_abs_error",
        "final_abs_error",
    ]

    with (
        analysis_directory
        / "errors.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=error_fieldnames,
        )

        writer.writeheader()

        if errors:
            writer.writerows(
                errors
            )

    # --------------------------------------------------------
    # Natural populations
    # --------------------------------------------------------

    natural_population_fieldnames = [
        "layer",
        "node",
        "mode",
        "child",
        "rank",
        "immediate_capacity",
        "expandable",
        "max_lowest_population",
        "time_of_max_lowest_population",
        "final_lowest_population",
    ]

    with (
        analysis_directory
        / "natural_populations.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=(
                natural_population_fieldnames
            ),
        )

        writer.writeheader()

        if natural_population_rows:

            writer.writerows(
                natural_population_rows
            )

    # --------------------------------------------------------
    # Coefficient breakdown
    # --------------------------------------------------------

    with (
        analysis_directory
        / "coefficient_breakdown.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        fieldnames = [
            "node",
            "parent_rank",
            "child_dimensions",
            "n_coefficients",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in (
            coefficient_breakdown
        ):

            writer.writerow(
                {
                    "node": item.node,
                    "parent_rank": (
                        item.parent_rank
                    ),
                    "child_dimensions": (
                        "x".join(
                            map(
                                str,
                                item.child_dimensions,
                            )
                        )
                    ),
                    "n_coefficients": (
                        item.n_coefficients
                    ),
                }
            )

    # ========================================================
    # Exact-reference summary fields
    # ========================================================

    P_mol_max_abs_error = None
    P_mol_final_abs_error = None
    P_mol_time_of_max_abs_error = None

    max_molecular_orbital_population_error = None
    molecular_orbital_of_max_error = None
    time_of_max_molecular_orbital_error = None
    final_max_molecular_orbital_population_error = None

    if errors:

        error_lookup = {
            str(
                row[
                    "observable"
                ]
            ): row
            for row in errors
        }

        molecular_orbital_errors = [
            error_lookup[
                f"d{index}"
            ]
            for index in range(
                1,
                config.n_molecular_orbitals
                + 1,
            )
        ]

        worst_molecular_orbital_error = max(
            molecular_orbital_errors,
            key=lambda row: float(
                row[
                    "max_abs_error"
                ]
            ),
        )

        max_molecular_orbital_population_error = (
            float(
                worst_molecular_orbital_error[
                    "max_abs_error"
                ]
            )
        )

        molecular_orbital_of_max_error = int(
            str(
                worst_molecular_orbital_error[
                    "observable"
                ]
            )[1:]
        )

        time_of_max_molecular_orbital_error = (
            float(
                worst_molecular_orbital_error[
                    "time_of_max_abs_error"
                ]
            )
        )

        final_max_molecular_orbital_population_error = max(
            float(
                row[
                    "final_abs_error"
                ]
            )
            for row
            in molecular_orbital_errors
        )

        if (
            "P_mol"
            in error_lookup
        ):

            P_mol_max_abs_error = float(
                error_lookup[
                    "P_mol"
                ][
                    "max_abs_error"
                ]
            )

            P_mol_final_abs_error = float(
                error_lookup[
                    "P_mol"
                ][
                    "final_abs_error"
                ]
            )

            P_mol_time_of_max_abs_error = float(
                error_lookup[
                    "P_mol"
                ][
                    "time_of_max_abs_error"
                ]
            )

    # ========================================================
    # Natural-population summary
    # ========================================================

    expandable_natpops = [
        row
        for row
        in natural_population_rows
        if bool(
            row[
                "expandable"
            ]
        )
    ]

    largest_lowest_natpop = (
        max(
            row[
                "max_lowest_population"
            ]
            for row
            in natural_population_rows
        )
        if natural_population_rows
        else None
    )

    largest_expandable_natpop = (
        max(
            row[
                "max_lowest_population"
            ]
            for row
            in expandable_natpops
        )
        if expandable_natpops
        else None
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "benchmark": benchmark,
        "backend": "heidelberg",
        "run_id": run_id,
        "input_run_id": input_run_id,
        "has_exact_reference": (
            has_exact_reference
        ),

        "n_ml_coefficients": (
            n_coefficients
        ),

        "max_norm_error": (
            norm_error
        ),

        "max_particle_number_error": (
            particle_number_error
        ),

        "initial_molecular_population": float(
            total_molecular_population[
                0
            ]
        ),

        "final_molecular_population": float(
            total_molecular_population[
                -1
            ]
        ),

        "maximum_molecular_population": float(
            np.max(
                total_molecular_population
            )
        ),

        "cpu_seconds": (
            runtime.cpu_seconds
        ),

        "wall_seconds": (
            runtime.wall_seconds
        ),

        "hostname": (
            runtime.hostname
        ),

        "largest_lowest_natural_population": (
            largest_lowest_natpop
        ),

        "largest_expandable_lowest_natural_population": (
            largest_expandable_natpop
        ),

        "physical_coordinates": (
            physical_coordinate_summary
        ),

        # Exact-reference-only fields.
        "P_mol_max_abs_error": (
            P_mol_max_abs_error
        ),

        "P_mol_final_abs_error": (
            P_mol_final_abs_error
        ),

        "P_mol_time_of_max_abs_error": (
            P_mol_time_of_max_abs_error
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

    with (
        analysis_directory
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    return analysis_directory