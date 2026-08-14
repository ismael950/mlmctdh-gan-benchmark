from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NaturalPopulationColumn:
    """Lowest natural-population trace for one ML node mode."""

    layer: int
    node: int
    mode: int
    values: np.ndarray


@dataclass(frozen=True)
class RuntimeInfo:
    """Runtime metadata parsed from a Heidelberg output file."""

    cpu_seconds: float | None
    wall_seconds: float | None
    hostname: str | None


def _hms_to_seconds(hours: int, minutes: int, seconds: float) -> float:
    return 3600.0 * hours + 60.0 * minutes + seconds


def read_expectation(path: str | Path) -> dict[str, np.ndarray]:
    """Read Heidelberg's ``expectation`` file using its own header names."""

    path = Path(path)
    header: list[str] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("# time"):
                tokens = stripped.lstrip("#").split()
                tokens[0] = "time"
                header = tokens
                break

    if header is None:
        raise ValueError(f"Could not find expectation header in {path}.")

    data = np.loadtxt(path, comments="#", ndmin=2)

    if data.shape[1] != len(header):
        raise ValueError(
            f"Expectation file has {data.shape[1]} data columns but "
            f"{len(header)} header entries."
        )

    return {
        name: data[:, index]
        for index, name in enumerate(header)
    }


def read_lowest_natural_populations(
    path: str | Path,
) -> tuple[np.ndarray, list[NaturalPopulationColumn]]:
    """
    Read Heidelberg's ``lownatpop`` file.

    The file stores, for each ML node mode, the *lowest* natural population
    available at every output time. These traces are useful for identifying
    rank bottlenecks.
    """

    path = Path(path)
    layer_tokens: list[str] | None = None
    node_tokens: list[str] | None = None
    mode_tokens: list[str] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("# time"):
                layer_tokens = stripped.lstrip("#").split()[1:]
            elif stripped.startswith("#") and "node:" in stripped:
                node_tokens = stripped.lstrip("#").split()
            elif stripped.startswith("#") and "mode:" in stripped:
                mode_tokens = stripped.lstrip("#").split()
                break

    if layer_tokens is None or node_tokens is None or mode_tokens is None:
        raise ValueError(f"Could not parse lownatpop headers in {path}.")

    if not (len(layer_tokens) == len(node_tokens) == len(mode_tokens)):
        raise ValueError("Inconsistent lownatpop header lengths.")

    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(layer_tokens) + 1:
        raise ValueError("Unexpected number of lownatpop data columns.")

    def parse_tag(token: str, prefix: str) -> int:
        if not token.startswith(prefix):
            raise ValueError(f"Expected {prefix!r} token, got {token!r}.")
        return int(token.split(":", 1)[1])

    columns: list[NaturalPopulationColumn] = []
    for index, (layer, node, mode) in enumerate(
        zip(layer_tokens, node_tokens, mode_tokens),
        start=1,
    ):
        columns.append(
            NaturalPopulationColumn(
                layer=parse_tag(layer, "layer"),
                node=parse_tag(node, "node"),
                mode=parse_tag(mode, "mode"),
                values=data[:, index],
            )
        )

    return data[:, 0], columns


def read_runtime_info(path: str | Path) -> RuntimeInfo:
    """Parse CPU time, wall time, and hostname from Heidelberg ``output``."""

    text = Path(path).read_text(encoding="utf-8", errors="replace")

    cpu_match = re.search(
        r"Total\s+time\s+\[h:m:s\]\s*:\s*(\d+)\s*:\s*(\d+)\s*:\s*([0-9.]+)",
        text,
    )
    wall_match = re.search(
        r"Wall\s+time\s+\[h:m:s\]\s*:\s*(\d+)\s*:\s*(\d+)\s*:\s*([0-9.]+)",
        text,
    )
    host_matches = re.findall(r'Host:\s*"([^"]+)"', text)

    cpu_seconds = None
    if cpu_match:
        cpu_seconds = _hms_to_seconds(
            int(cpu_match.group(1)),
            int(cpu_match.group(2)),
            float(cpu_match.group(3)),
        )

    wall_seconds = None
    if wall_match:
        wall_seconds = _hms_to_seconds(
            int(wall_match.group(1)),
            int(wall_match.group(2)),
            float(wall_match.group(3)),
        )

    hostname = host_matches[-1] if host_matches else None

    return RuntimeInfo(
        cpu_seconds=cpu_seconds,
        wall_seconds=wall_seconds,
        hostname=hostname,
    )
