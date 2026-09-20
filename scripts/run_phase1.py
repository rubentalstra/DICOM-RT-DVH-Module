"""Phase 1, Step 1.6: Cache-and-recover on Nelms analytical phantoms.

For each phantom structure:
1. Compute voxel DVH using dicompyler-core (reference)
2. Write to DICOM RT DVH at 1 cGy
3. Read back from file
4. Derive all parameters using SAME interpolation method on both sides
5. Compare reference vs cached

Fair comparison: linear(voxel) vs linear(cached), and
cubic_spline(voxel) vs cubic_spline(cached). This isolates caching
precision loss from interpolation method differences.

Output:
    data/results/phase1_step6_cache_and_recover.csv
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DVH_BIN_WIDTH_CGY, INTERPOLATION_METHODS, RESULTS_DIR
from src.comparison import (
    bland_altman_stats,
    compare_parameters,
    results_to_dataframe,
)
from src.dvh_cache import read_dicom_dvh, write_dicom_dvh
from src.dvh_compute import compute_dvh_from_dicom
from src.dvh_parameters import derive_all

NELMS_BASE = PROJECT_ROOT / "data" / "nelms" / "DVH-Analysis-Data-Etc"


def run_phase1():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dose_path = NELMS_BASE / "DOSE GRIDS" / "Linear_SupInf_1mm_Aligned.dcm"
    struct_dir = NELMS_BASE / "STRUCTURES"

    all_results = []

    for shape_dir in ["Spheres", "Cylinders", "Cones"]:
        for struct_path in sorted((struct_dir / shape_dir).glob("*.dcm")):
            if struct_path.stem.startswith("Rt"):
                continue

            try:
                voxel_dvhs = compute_dvh_from_dicom(
                    dose_path, struct_path, bin_width_cGy=DVH_BIN_WIDTH_CGY,
                )
            except Exception as e:
                print(f"  SKIP {struct_path.stem}: {e}")
                continue

            for struct_name, voxel_dvh in voxel_dvhs.items():
                if "POI" in struct_name or voxel_dvh.structure_volume_cc == 0:
                    continue

                with tempfile.TemporaryDirectory() as tmpdir:
                    cache_path = Path(tmpdir) / "cached.dcm"
                    write_dicom_dvh([voxel_dvh], cache_path)
                    cached_dvhs = read_dicom_dvh(cache_path)

                if not cached_dvhs:
                    continue
                cached_dvh = cached_dvhs[0]

                for interp in INTERPOLATION_METHODS:
                    ref_params = derive_all(voxel_dvh, interpolation=interp)
                    cached_params = derive_all(cached_dvh, interpolation=interp)
                    results = compare_parameters(
                        ref_params, cached_params,
                        structure_name=struct_name,
                        structure_volume_cc=voxel_dvh.structure_volume_cc,
                        interpolation_method=interp,
                    )
                    all_results.extend(results)

                print(
                    f"  {struct_name}: vol={voxel_dvh.structure_volume_cc:.1f}cc, "
                    f"mean={voxel_dvh.mean_dose_Gy:.2f}Gy"
                )

    df = results_to_dataframe(all_results)
    csv_path = RESULTS_DIR / "phase1_step6_cache_and_recover.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"PHASE 1 CACHE-AND-RECOVER: {len(df)} comparisons")
    print(f"{'='*60}")
    print(f"Within tolerance: {df['within_tolerance'].sum()} ({df['within_tolerance'].mean()*100:.1f}%)")

    for method in INTERPOLATION_METHODS:
        sub = df[df["interpolation_method"] == method]
        if len(sub) > 0:
            pct = sub["within_tolerance"].mean() * 100
            max_e = sub["abs_error"].max()
            print(f"  {method}: n={len(sub)}, pass={pct:.1f}%, max_err={max_e:.8f}")

    failures = df[~df["within_tolerance"]]
    if len(failures) > 0:
        print(f"\nFAILURES ({len(failures)}):")
        for _, r in failures.head(10).iterrows():
            print(f"  {r['structure_name']}: {r['parameter_name']} err={r['abs_error']:.6f} [{r['interpolation_method']}]")
    else:
        print("\n*** ALL WITHIN TOLERANCE ***")

    return df


if __name__ == "__main__":
    run_phase1()
