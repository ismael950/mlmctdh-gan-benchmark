from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ganbench.heidelberg.convergence import RankUpdate


_ML_LINE_PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<layer>\d+)>\s*(?P<body>.*?)\s*$"
)


def write_refined_input(
    source_path: str | Path,
    destination_path: str | Path,
    next_run_number: int,
    rank_updates: Sequence[RankUpdate],
) -> None:
    """
    Create the next Heidelberg input by applying SPF-rank updates.

    The ML tree topology is kept fixed. Only the selected ranks and
    the output run directory are changed.
    """

    source_path = Path(source_path)
    destination_path = Path(destination_path)

    if next_run_number < 1:
        raise ValueError("next_run_number must be positive")

    text = source_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = text.splitlines(keepends=True)

    updates_by_branch = {
        (update.node, update.mode): update
        for update in rank_updates
    }

    if len(updates_by_branch) != len(rank_updates):
        raise ValueError("Duplicate rank updates were supplied.")

    applied_updates: set[tuple[int, int]] = set()

    in_run_section = False
    in_ml_section = False
    node_counter = 0

    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper == "RUN-SECTION":
            in_run_section = True
            output_lines.append(line)
            continue

        if upper == "END-RUN-SECTION":
            in_run_section = False
            output_lines.append(line)
            continue

        if upper == "ML-BASIS-SECTION":
            in_ml_section = True
            output_lines.append(line)
            continue

        if upper == "END-ML-BASIS-SECTION":
            in_ml_section = False
            output_lines.append(line)
            continue

        # Update the Heidelberg output directory.
        if (
            in_run_section
            and line.lstrip().startswith("name =")
        ):
            new_line, replacements = re.subn(
                r"run_\d{3}",
                f"run_{next_run_number:03d}",
                line,
                count=1,
            )

            if replacements != 1:
                raise ValueError(
                    "Could not identify the run number in RUN-SECTION."
                )

            output_lines.append(new_line)
            continue

        # Apply rank changes inside ML-BASIS-SECTION.
        if in_ml_section:
            line_without_newline = line.rstrip("\r\n")
            newline = line[len(line_without_newline):]

            code_part, separator, comment = (
                line_without_newline.partition("#")
            )

            match = _ML_LINE_PATTERN.match(code_part)

            if match:
                node_counter += 1

                layer = int(match.group("layer"))
                body = match.group("body").strip()

                # Primitive groups such as [d c1] contain no SPF ranks.
                if not (
                    body.startswith("[")
                    and body.endswith("]")
                ):
                    ranks = body.split()

                    changed = False

                    for mode_index in range(
                        1,
                        len(ranks) + 1,
                    ):
                        key = (
                            node_counter,
                            mode_index,
                        )

                        update = updates_by_branch.get(key)

                        if update is None:
                            continue

                        if layer != update.layer:
                            raise ValueError(
                                f"Layer mismatch for node "
                                f"{node_counter}."
                            )

                        current_rank = int(
                            ranks[mode_index - 1]
                        )

                        if current_rank != update.old_rank:
                            raise ValueError(
                                f"Expected rank "
                                f"{update.old_rank} at "
                                f"node {node_counter}, "
                                f"mode {mode_index}, "
                                f"but found {current_rank}."
                            )

                        ranks[mode_index - 1] = str(
                            update.new_rank
                        )

                        applied_updates.add(key)
                        changed = True

                    if changed:
                        rebuilt = (
                            f"{match.group('indent')}"
                            f"{layer}> "
                            f"{' '.join(ranks)}"
                        )

                        if separator:
                            rebuilt += f" #{comment}"

                        output_lines.append(
                            rebuilt + newline
                        )
                        continue

        output_lines.append(line)

    missing_updates = (
        set(updates_by_branch) - applied_updates
    )

    if missing_updates:
        raise ValueError(
            f"Could not apply rank updates: "
            f"{sorted(missing_updates)}"
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_path.write_text(
        "".join(output_lines),
        encoding="utf-8",
    )