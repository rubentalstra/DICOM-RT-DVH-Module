"""Phase 5: Merge all cache-and-recover results into a master dataset.

Loads Phase 1, 2, and 3 CSVs, normalises column formats, adds phase
and dataset labels, and saves the merged master dataset.

Output:
    data/results/phase5_master_dataset.csv
    data/results/phase5_intersystem.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR


def load_and_merge() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all phase CSVs and merge into a master dataset.

    Returns
    -------
    cache_df : pd.DataFrame
        Cache-layer comparisons (voxel vs cached).
    intersystem_df : pd.DataFrame
        Inter-system comparisons (embedded TPS vs voxel).
    """
    p1 = pd.read_csv(RESULTS_DIR / "phase1_step6_cache_and_recover.csv")
    p2 = pd.read_csv(RESULTS_DIR / "phase2_crossvalidation.csv")
    p3 = pd.read_csv(RESULTS_DIR / "phase3_clinical_validation.csv")

    p1["phase"] = "Phase 1"
    p1["dataset"] = "Nelms phantoms"
    p3["phase"] = "Phase 3"
    p3["dataset"] = "RayStation clinical"

    # Phase 2: split cache-layer from inter-system comparisons
    p2_cache = p2[~p2["parameter_type"].str.startswith("intersystem_")].copy()
    p2_cache["phase"] = "Phase 2"
    p2_cache["dataset"] = "Eclipse + dicompyler"

    p2_inter = p2[p2["parameter_type"].str.startswith("intersystem_")].copy()
    p2_inter["parameter_type"] = p2_inter["parameter_type"].str.replace(
        "intersystem_", "", regex=False,
    )
    p2_inter["phase"] = "Phase 2"
    p2_inter["dataset"] = "Inter-system"

    cache_df = pd.concat([p1, p2_cache, p3], ignore_index=True)
    return cache_df, p2_inter


def run_phase5() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache_df, intersystem_df = load_and_merge()

    master_path = RESULTS_DIR / "phase5_master_dataset.csv"
    cache_df.to_csv(master_path, index=False)
    print(f"Master dataset: {len(cache_df)} rows → {master_path}")

    if len(intersystem_df) > 0:
        inter_path = RESULTS_DIR / "phase5_intersystem.csv"
        intersystem_df.to_csv(inter_path, index=False)
        print(f"Inter-system:   {len(intersystem_df)} rows → {inter_path}")

    # Console summary
    total = len(cache_df)
    passing = cache_df["within_tolerance"].sum()
    print(f"\nTotal: {total}, passing: {passing} ({passing / total * 100:.1f}%)")
    for method in cache_df["interpolation_method"].unique():
        sub = cache_df[cache_df["interpolation_method"] == method]
        pct = sub["within_tolerance"].mean() * 100
        print(f"  {method}: {len(sub)} comparisons, {pct:.1f}% pass")

    return cache_df


if __name__ == "__main__":
    run_phase5()
