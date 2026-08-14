from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ganbench.heidelberg.parser import read_expectation

@dataclass(frozen=True)
class BranchState:
    """Convergence information for one ML-MCTDH tree branch."""

    layer: int
    node: int
    mode: int
    child: str
    rank: int
    immediate_capacity: int
    expandable: bool
    max_lowest_population: float
    time_of_max_lowest_population: float
    final_lowest_population: float

    @property
    def remaining_capacity(self) -> int:
        """Number of additional SPFs allowed before this branch saturates."""
        return max(0, self.immediate_capacity - self.rank)


def _read_bool(value: str) -> bool:
    """Read True/False values written to the analysis CSV."""
    return value.strip().lower() in {"true", "1", "yes"}


def read_branch_states(path: str | Path) -> list[BranchState]:
    """Read branch convergence diagnostics from natural_populations.csv."""

    path = Path(path)

    states: list[BranchState] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            states.append(
                BranchState(
                    layer=int(row["layer"]),
                    node=int(row["node"]),
                    mode=int(row["mode"]),
                    child=row["child"],
                    rank=int(row["rank"]),
                    immediate_capacity=int(row["immediate_capacity"]),
                    expandable=_read_bool(row["expandable"]),
                    max_lowest_population=float(row["max_lowest_population"]),
                    time_of_max_lowest_population=float(
                        row["time_of_max_lowest_population"]
                    ),
                    final_lowest_population=float(
                        row["final_lowest_population"]
                    ),
                )
            )

    if not states:
        raise ValueError(f"No branch data found in {path}")

    return states

@dataclass(frozen=True)
class RefinementTarget:
    """Branch or coupled branches selected for the next SPF refinement."""

    branches: tuple[BranchState, ...]
    score: float

    @property
    def is_root(self) -> bool:
        return all(branch.layer == 0 and branch.node == 1 for branch in self.branches)


def select_refinement_target(
    states: list[BranchState],
) -> RefinementTarget | None:
    """
    Select the expandable branch with the largest lowest natural population.

    The two root branches are treated as one coupled Schmidt rank.
    """

    root_branches = tuple(
        state
        for state in states
        if state.layer == 0 and state.node == 1
    )

    candidates = [
        state
        for state in states
        if state.expandable and not (state.layer == 0 and state.node == 1)
    ]

    # Root can only expand while the common rank is below the capacity
    # of both sides of the bipartition.
    if root_branches:
        root_rank = root_branches[0].rank
        root_capacity = min(
            branch.immediate_capacity
            for branch in root_branches
        )

        if root_rank < root_capacity:
            root_score = max(
                branch.max_lowest_population
                for branch in root_branches
            )

            candidates.append(
                max(
                    root_branches,
                    key=lambda state: state.max_lowest_population,
                )
            )
        else:
            root_score = None
    else:
        root_score = None

    if not candidates:
        return None

    worst = max(
        candidates,
        key=lambda state: state.max_lowest_population,
    )

    if worst.layer == 0 and worst.node == 1:
        return RefinementTarget(
            branches=root_branches,
            score=float(root_score),
        )

    return RefinementTarget(
        branches=(worst,),
        score=worst.max_lowest_population,
    )

@dataclass(frozen=True)
class RankUpdate:
    """One proposed SPF-rank change."""

    layer: int
    node: int
    mode: int
    old_rank: int
    new_rank: int


def propose_rank_updates(
    target: RefinementTarget,
    increment: int = 4,
) -> tuple[RankUpdate, ...]:
    """Propose the next rank increase without exceeding branch capacity."""

    if increment <= 0:
        raise ValueError("increment must be positive")

    # Root branches share one Schmidt rank and must remain equal.
    if target.is_root:
        old_ranks = {branch.rank for branch in target.branches}

        if len(old_ranks) != 1:
            raise ValueError(
                "Root branches must currently have the same rank."
            )

        old_rank = next(iter(old_ranks))

        max_common_rank = min(
            branch.immediate_capacity
            for branch in target.branches
        )

        new_rank = min(
            old_rank + increment,
            max_common_rank,
        )

        return tuple(
            RankUpdate(
                layer=branch.layer,
                node=branch.node,
                mode=branch.mode,
                old_rank=old_rank,
                new_rank=new_rank,
            )
            for branch in target.branches
        )

    branch = target.branches[0]

    new_rank = min(
        branch.rank + increment,
        branch.immediate_capacity,
    )

    return (
        RankUpdate(
            layer=branch.layer,
            node=branch.node,
            mode=branch.mode,
            old_rank=branch.rank,
            new_rank=new_rank,
        ),
    )

@dataclass(frozen=True)
class MolecularPopulationChange:
    """Largest molecular-orbital population change between two runs."""

    max_abs_change: float
    orbital_index: int
    time_of_max_change: float
    final_max_abs_change: float


def _molecular_population_columns(
    expectation: dict[str, np.ndarray],
    n_molecular_orbitals: int,
) -> list[str]:
    """Return Heidelberg column names for all molecular orbital populations."""

    if n_molecular_orbitals == 1 and "nd" in expectation:
        return ["nd"]

    names = [
        f"nd{index}"
        for index in range(1, n_molecular_orbitals + 1)
    ]

    missing = [name for name in names if name not in expectation]

    if missing:
        raise KeyError(
            f"Missing molecular population columns: {missing}"
        )

    return names


def compare_molecular_populations(
    previous_expectation: str | Path,
    current_expectation: str | Path,
    n_molecular_orbitals: int,
) -> MolecularPopulationChange:
    """
    Compare all molecular-orbital populations between two ML-MCTDH runs.

    The convergence metric is the maximum absolute population difference
    over every molecular orbital and every propagation time.
    """

    previous = read_expectation(previous_expectation)
    current = read_expectation(current_expectation)

    previous_times = previous["time"]
    current_times = current["time"]

    if (
        previous_times.shape != current_times.shape
        or not np.allclose(previous_times, current_times)
    ):
        raise ValueError(
            "The two Heidelberg runs use different time grids."
        )

    previous_names = _molecular_population_columns(
        previous,
        n_molecular_orbitals,
    )
    current_names = _molecular_population_columns(
        current,
        n_molecular_orbitals,
    )

    previous_populations = np.column_stack(
        [previous[name] for name in previous_names]
    )
    current_populations = np.column_stack(
        [current[name] for name in current_names]
    )

    absolute_change = np.abs(
        current_populations - previous_populations
    )

    time_index, orbital_index = np.unravel_index(
        np.argmax(absolute_change),
        absolute_change.shape,
    )

    final_max_abs_change = float(
        np.max(absolute_change[-1, :])
    )

    return MolecularPopulationChange(
        max_abs_change=float(
            absolute_change[time_index, orbital_index]
        ),
        orbital_index=int(orbital_index + 1),
        time_of_max_change=float(current_times[time_index]),
        final_max_abs_change=final_max_abs_change,
    )

@dataclass(frozen=True)
class PlateauConfig:
    """Rules used to detect and confirm observable convergence."""

    change_tolerance: float = 1.0e-5
    trigger_consecutive_runs: int = 2
    confirmation_runs: int = 3


@dataclass(frozen=True)
class PlateauStatus:
    """Current state of plateau detection."""

    small_change_count: int
    plateau_triggered: bool
    confirmations_completed: int
    plateau_confirmed: bool

def update_plateau_status(
    previous_status: PlateauStatus | None,
    max_abs_change: float,
    config: PlateauConfig = PlateauConfig(),
) -> PlateauStatus:
    """
    Update plateau detection after one new pair of consecutive runs.

    A change above the tolerance resets the entire plateau count.
    """

    if config.change_tolerance <= 0:
        raise ValueError("change_tolerance must be positive")

    if config.trigger_consecutive_runs < 1:
        raise ValueError("trigger_consecutive_runs must be at least 1")

    if config.confirmation_runs < 1:
        raise ValueError("confirmation_runs must be at least 1")

    previous_count = (
        previous_status.small_change_count
        if previous_status is not None
        else 0
    )

    if max_abs_change < config.change_tolerance:
        small_change_count = previous_count + 1
    else:
        # A significant change means the apparent plateau was premature.
        small_change_count = 0

    plateau_triggered = (
        small_change_count >= config.trigger_consecutive_runs
    )

    if plateau_triggered:
        confirmations_completed = max(
            0,
            small_change_count - config.trigger_consecutive_runs,
        )
        confirmations_completed = min(
            confirmations_completed,
            config.confirmation_runs,
        )
    else:
        confirmations_completed = 0

    plateau_confirmed = (
        small_change_count
        >= config.trigger_consecutive_runs + config.confirmation_runs
    )

    return PlateauStatus(
        small_change_count=small_change_count,
        plateau_triggered=plateau_triggered,
        confirmations_completed=confirmations_completed,
        plateau_confirmed=plateau_confirmed,
    )

def evaluate_plateau_history(
    expectation_paths: list[str | Path],
    n_molecular_orbitals: int,
    config: PlateauConfig = PlateauConfig(),
) -> tuple[list[MolecularPopulationChange], PlateauStatus | None]:
    """
    Evaluate convergence over an ordered sequence of ML-MCTDH runs.
    """

    if len(expectation_paths) < 2:
        return [], None

    changes: list[MolecularPopulationChange] = []
    status: PlateauStatus | None = None

    for previous_path, current_path in zip(
        expectation_paths[:-1],
        expectation_paths[1:],
    ):
        change = compare_molecular_populations(
            previous_path,
            current_path,
            n_molecular_orbitals,
        )

        changes.append(change)

        status = update_plateau_status(
            status,
            change.max_abs_change,
            config,
        )

    return changes, status
