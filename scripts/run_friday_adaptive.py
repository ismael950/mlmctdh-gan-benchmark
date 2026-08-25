from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from ganbench.heidelberg.analysis import analyze_heidelberg_run


ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "ganbench_existing_runner",
    ROOT / "scripts/run.py",
)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def execute_heidelberg_run_friday(
    project_root: Path,
    benchmark: str,
    run_id: str,
) -> Path:

    input_directory = (
        project_root
        / "backend_inputs"
        / benchmark
        / "heidelberg"
        / run_id
    )

    raw_directory = (
        project_root
        / "results"
        / benchmark
        / "heidelberg"
        / run_id
        / "raw"
    )

    if (raw_directory / "expectation").exists():
        raise RuntimeError(
            f"{run_id} already completed; refusing to rerun."
        )

    if raw_directory.exists():
        raise RuntimeError(
            f"{raw_directory} already exists; refusing to overwrite."
        )

    raw_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Executing Heidelberg {run_id} on Friday...")

    # Exactly ONE Heidelberg invocation for this run.
    subprocess.run(
        ["mctdh86P", "-mnd", "benchmark.inp"],
        cwd=input_directory,
        check=True,
    )

    if not (raw_directory / "expectation").exists():
        raise RuntimeError(
            f"{run_id} finished without expectation output."
        )

    analysis_directory = analyze_heidelberg_run(
        project_root=project_root,
        benchmark=benchmark,
        run_id=run_id,
    )

    print("Completed and analyzed:", run_id)
    return analysis_directory


# Replace only the WSL execution layer.
# Convergence/refinement logic remains the existing project implementation.
runner.execute_heidelberg_run = execute_heidelberg_run_friday

runner.run_heidelberg(
    str(ROOT / "configs/small_direct_benchmark.yaml")
)
