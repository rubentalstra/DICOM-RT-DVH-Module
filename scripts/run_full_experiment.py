"""End-to-end execution of all experiment phases + report generation.

Runs Phases 1 through 6 sequentially, then generates the complete
experiment report from all CSV data. Fully deterministic and
reproducible from a clean data directory.

Usage:
    python scripts/run_full_experiment.py

Output:
    data/results/*.csv          — Raw experiment data
    data/results/figures/*.png  — Publication figures
    data/results/EXPERIMENT_REPORT.md — Complete report
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_script(name: str) -> None:
    """Run a phase script as a subprocess."""
    script_path = PROJECT_ROOT / "scripts" / name
    print(f"\n{'#' * 70}")
    print(f"# Running: {name}")
    print(f"{'#' * 70}\n")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {name} ({elapsed:.1f}s)")
    if result.returncode != 0:
        print(f"  WARNING: {name} failed. Continuing...")


def main():
    total_t0 = time.time()

    print("=" * 70)
    print("DVH VALIDATION EXPERIMENT — FULL EXECUTION")
    print("=" * 70)

    run_script("run_phase1.py")
    run_script("run_phase2.py")
    run_script("run_phase3.py")
    run_script("run_phase4.py")
    run_script("run_phase5.py")
    run_script("run_phase6.py")
    run_script("generate_report.py")

    elapsed = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"Report: data/results/EXPERIMENT_REPORT.md")
    print(f"Data:   data/results/*.csv")
    print(f"Figures: data/results/figures/*.png")


if __name__ == "__main__":
    main()
