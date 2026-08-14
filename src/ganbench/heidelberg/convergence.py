from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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