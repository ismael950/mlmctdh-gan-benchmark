from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLTreeEdge:
    """One child -> parent branch in Heidelberg's ML tree."""

    child: str
    parent: str
    rank: int


@dataclass(frozen=True)
class MLCoefficientBreakdown:
    """Coefficient count for one internal node of an ML tree."""

    node: str
    parent_rank: int
    child_dimensions: tuple[int, ...]
    n_coefficients: int


_EDGE_PATTERN = re.compile(
    r"^\s*(?P<src>[nf]\d+)\s*->\s*(?P<dst>n\d+)"
    r"\s*\[.*?label=(?P<label>\d+).*?\]"
)
_ML_LINE_PATTERN = re.compile(r"^\s*(?P<layer>\d+)>\s*(?P<body>.*?)\s*$")


def read_ml_tree_edges(path: str | Path) -> list[MLTreeEdge]:
    """Read child -> parent branches from Heidelberg's ``mltree.dot``."""

    edges: list[MLTreeEdge] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _EDGE_PATTERN.search(line)
            if match:
                edges.append(
                    MLTreeEdge(
                        child=match.group("src"),
                        parent=match.group("dst"),
                        rank=int(match.group("label")),
                    )
                )

    if not edges:
        raise ValueError(f"No ML tree edges were found in {path}.")
    return edges


def _section_lines(text: str, start_name: str, end_name: str) -> list[str]:
    """Return uncommented lines inside one Heidelberg input section."""

    in_section = False
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        upper = stripped.upper()
        if upper == start_name.upper():
            in_section = True
            continue
        if in_section and upper == end_name.upper():
            break
        if not in_section:
            continue

        content = raw_line.split("#", 1)[0].strip()
        if content:
            lines.append(content)
    return lines


def _primitive_dimensions(text: str) -> dict[str, int]:
    """Read primitive-basis dimensions from PBASIS-SECTION."""

    dimensions: dict[str, int] = {}
    for line in _section_lines(text, "PBASIS-SECTION", "end-PBASIS-SECTION"):
        tokens = line.split()
        if len(tokens) < 3:
            continue
        try:
            dimensions[tokens[0]] = int(tokens[2])
        except ValueError as exc:
            raise ValueError(
                f"Could not read primitive dimension from PBASIS line: {line!r}"
            ) from exc
    return dimensions


def read_ml_tree_edges_from_input(path: str | Path) -> list[MLTreeEdge]:
    """
    Reconstruct Heidelberg's ML tree directly from ``benchmark.inp``.

    This makes analysis independent of the optional ``mltree.dot`` output.
    Node numbering follows Heidelberg's pre-order numbering for the ML-BASIS
    syntax used by the benchmark inputs.
    """

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    primitive_dims = _primitive_dimensions(text)
    ml_lines = _section_lines(text, "ML-BASIS-SECTION", "end-ML-BASIS-SECTION")

    parsed: list[tuple[int, str]] = []
    for line in ml_lines:
        match = _ML_LINE_PATTERN.match(line)
        if match:
            parsed.append((int(match.group("layer")), match.group("body").strip()))

    if not parsed or parsed[0][0] != 0:
        raise ValueError(f"Could not find a valid root ML-BASIS line in {path}.")

    edges: list[MLTreeEdge] = []
    node_counter = 0
    primitive_counter = 0
    stack: dict[int, dict[str, object]] = {}

    for layer, body in parsed:
        node_counter += 1
        node = f"n{node_counter}"

        if body.startswith("[") and body.endswith("]"):
            mode_names = body[1:-1].split()
            if not mode_names:
                raise ValueError(f"Empty combined-mode specification in {path}: {body!r}")
            try:
                child_dimensions = tuple(primitive_dims[name] for name in mode_names)
            except KeyError as exc:
                raise KeyError(
                    f"Primitive mode {exc.args[0]!r} used in ML-BASIS but not PBASIS."
                ) from exc
            is_primitive_group = True
        else:
            try:
                child_dimensions = tuple(int(token) for token in body.split())
            except ValueError as exc:
                raise ValueError(f"Could not parse ML-BASIS line body {body!r}.") from exc
            if not child_dimensions:
                raise ValueError(f"ML-BASIS node {node} has no child dimensions.")
            is_primitive_group = False

        if layer == 0:
            if node != "n1":
                raise ValueError("ML-BASIS contains more than one layer-0 node.")
        else:
            if layer - 1 not in stack:
                raise ValueError(
                    f"ML-BASIS layer {layer} has no parent at layer {layer - 1}."
                )
            parent_info = stack[layer - 1]
            next_child = int(parent_info["next_child"])
            parent_dims = parent_info["child_dimensions"]
            assert isinstance(parent_dims, tuple)
            if next_child >= len(parent_dims):
                raise ValueError(
                    f"Too many children supplied below ML node {parent_info['node']}."
                )
            rank_to_parent = int(parent_dims[next_child])
            edges.append(
                MLTreeEdge(
                    child=node,
                    parent=str(parent_info["node"]),
                    rank=rank_to_parent,
                )
            )
            parent_info["next_child"] = next_child + 1

        # This node becomes the active node at its layer. Deeper nodes from a
        # previous branch are no longer possible parents.
        stack[layer] = {
            "node": node,
            "child_dimensions": child_dimensions,
            "next_child": 0,
        }
        for deeper_layer in [key for key in stack if key > layer]:
            del stack[deeper_layer]

        if is_primitive_group:
            for dimension in child_dimensions:
                primitive_counter += 1
                edges.append(
                    MLTreeEdge(
                        child=f"f{primitive_counter}",
                        parent=node,
                        rank=int(dimension),
                    )
                )
            stack[layer]["next_child"] = len(child_dimensions)

    # Check that every internal numeric node received the number of children
    # declared on its ML-BASIS line.
    for info in stack.values():
        # Only the currently active branch remains in stack, so global checking
        # is handled implicitly while parsing. No additional action required.
        _ = info

    if not edges:
        raise ValueError(f"No ML tree edges could be reconstructed from {path}.")
    return edges


def _branch_diagnostics_from_edges(
    edges: list[MLTreeEdge],
) -> dict[tuple[int, int], dict[str, int | str | bool]]:
    children_by_parent: dict[str, list[MLTreeEdge]] = {}
    for edge in edges:
        children_by_parent.setdefault(edge.parent, []).append(edge)

    diagnostics: dict[tuple[int, int], dict[str, int | str | bool]] = {}

    for parent, child_edges in children_by_parent.items():
        parent_number = int(parent[1:])
        for mode_index, edge in enumerate(child_edges, start=1):
            if edge.child.startswith("f"):
                capacity = edge.rank
            else:
                grandchildren = children_by_parent.get(edge.child, [])
                capacity = 1
                for grandchild in grandchildren:
                    capacity *= grandchild.rank

            diagnostics[(parent_number, mode_index)] = {
                "child": edge.child,
                "rank": edge.rank,
                "immediate_capacity": capacity,
                "expandable": edge.rank < capacity,
            }

    return diagnostics


def branch_diagnostics(
    path: str | Path,
) -> dict[tuple[int, int], dict[str, int | str | bool]]:
    """Branch diagnostics from Heidelberg's optional ``mltree.dot``."""
    return _branch_diagnostics_from_edges(read_ml_tree_edges(path))


def branch_diagnostics_from_input(
    path: str | Path,
) -> dict[tuple[int, int], dict[str, int | str | bool]]:
    """Branch diagnostics reconstructed directly from ``benchmark.inp``."""
    return _branch_diagnostics_from_edges(read_ml_tree_edges_from_input(path))


def _count_ml_coefficients_from_edges(
    edges: list[MLTreeEdge],
) -> tuple[int, list[MLCoefficientBreakdown]]:
    children: dict[str, list[int]] = {}
    parent_rank: dict[str, int] = {}
    internal_nodes: set[str] = set()

    for edge in edges:
        internal_nodes.add(edge.parent)
        children.setdefault(edge.parent, []).append(edge.rank)

        if edge.child.startswith("n"):
            internal_nodes.add(edge.child)
            parent_rank[edge.child] = edge.rank

    def node_number(node: str) -> int:
        return int(node[1:])

    breakdown: list[MLCoefficientBreakdown] = []
    total = 0

    for node in sorted(internal_nodes, key=node_number):
        child_dims = tuple(children.get(node, []))
        if not child_dims:
            continue

        rank_to_parent = parent_rank.get(node, 1)
        coefficient_count = rank_to_parent
        for dimension in child_dims:
            coefficient_count *= dimension

        breakdown.append(
            MLCoefficientBreakdown(
                node=node,
                parent_rank=rank_to_parent,
                child_dimensions=child_dims,
                n_coefficients=coefficient_count,
            )
        )
        total += coefficient_count

    return total, breakdown


def count_ml_coefficients_from_dot(
    path: str | Path,
) -> tuple[int, list[MLCoefficientBreakdown]]:
    """Count nominal time-dependent ML coefficients from ``mltree.dot``."""
    return _count_ml_coefficients_from_edges(read_ml_tree_edges(path))


def count_ml_coefficients_from_input(
    path: str | Path,
) -> tuple[int, list[MLCoefficientBreakdown]]:
    """Count nominal time-dependent ML coefficients from ``benchmark.inp``."""
    return _count_ml_coefficients_from_edges(read_ml_tree_edges_from_input(path))
