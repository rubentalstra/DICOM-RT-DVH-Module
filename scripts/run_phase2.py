"""Phase 2: Cross-validation against commercial TPS DVH.

Part A: dicompyler test data — three-way comparison:
    - Embedded TPS DVH vs our voxel DVH (inter-system check)
    - Cache-and-recover test (same as Phase 1, on different data)

Part B: Eclipse phantoms (SlicerRtData) — cache-and-recover only
    (no embedded DVH available in Eclipse 8.1.20 exports)

Fair comparison: same interpolation method on both sides of
the cache test. Inter-system comparison reported separately.

Output:
    data/results/phase2_crossvalidation.csv
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DVH_BIN_WIDTH_CGY, INTERPOLATION_METHODS, RESULTS_DIR
from src.comparison import compare_parameters, results_to_dataframe
from src.dvh_cache import extract_embedded_dvh, read_dicom_dvh, write_dicom_dvh
from src.dvh_compute import compute_dvh_from_dicom
from src.dvh_parameters import derive_all

DICOMPYLER_DIR = PROJECT_ROOT / "data" / "dicompyler"
SLICERRT_DIR = PROJECT_ROOT / "data" / "slicerrt"


def run_phase2():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # ── Part A: dicompyler ──
    print(f"{'='*60}")
    print("Part A: dicompyler test dataset")
    print(f"{'='*60}")

    dose_path = DICOMPYLER_DIR / "rtdose.dcm"
    struct_path = DICOMPYLER_DIR / "rtss.dcm"

    embedded_dvhs = extract_embedded_dvh(dose_path, struct_path)
    embedded_by_name = {d.structure_name: d for d in embedded_dvhs}
    print(f"  Embedded DVH: {len(embedded_dvhs)} structures")

    voxel_dvhs = compute_dvh_from_dicom(
        dose_path, struct_path, bin_width_cGy=DVH_BIN_WIDTH_CGY,
    )
    print(f"  Voxel DVH: {len(voxel_dvhs)} structures")

    # Published values check
    if "Heart" in embedded_by_name:
        h = embedded_by_name["Heart"]
        print(f"\n  Heart (published check):")
        print(f"    Volume: {h.structure_volume_cc:.2f}cc (expected: 437.46)")
        print(f"    Mean:   {h.mean_dose_Gy:.4f}Gy (expected: 0.643)")

    for struct_name, voxel_dvh in voxel_dvhs.items():
        if voxel_dvh.structure_volume_cc == 0:
            continue

        # Cache-and-recover (fair: same method both sides)
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
                structure_name=f"dicompyler/{struct_name}",
                structure_volume_cc=voxel_dvh.structure_volume_cc,
                interpolation_method=interp,
            )
            all_results.extend(results)

        # Inter-system: embedded vs voxel (separate category)
        if struct_name in embedded_by_name:
            emb_dvh = embedded_by_name[struct_name]
            emb_params = derive_all(emb_dvh, interpolation="linear")
            voxel_params = derive_all(voxel_dvh, interpolation="linear")
            results = compare_parameters(
                emb_params, voxel_params,
                structure_name=f"dicompyler/{struct_name}",
                structure_volume_cc=voxel_dvh.structure_volume_cc,
                interpolation_method="intersystem",
            )
            for r in results:
                r.parameter_type = f"intersystem_{r.parameter_type}"
            all_results.extend(results)

    # ── Part B: Eclipse phantoms ──
    print(f"\n{'='*60}")
    print("Part B: Eclipse phantoms (SlicerRtData)")
    print(f"{'='*60}")

    for phantom in ["prostate", "breast", "ent"]:
        phantom_dir = SLICERRT_DIR / phantom
        rd = sorted(phantom_dir.glob("RD*.dcm"))
        rs = sorted(phantom_dir.glob("RS*.dcm"))
        if not rd or not rs:
            continue

        print(f"\n  --- Eclipse {phantom} ---")
        try:
            voxel_dvhs = compute_dvh_from_dicom(
                rd[0], rs[0], bin_width_cGy=DVH_BIN_WIDTH_CGY,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        for struct_name, voxel_dvh in voxel_dvhs.items():
            if voxel_dvh.structure_volume_cc == 0:
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
                    structure_name=f"Eclipse/{phantom}/{struct_name}",
                    structure_volume_cc=voxel_dvh.structure_volume_cc,
                    interpolation_method=interp,
                )
                all_results.extend(results)

            print(
                f"  {struct_name}: vol={voxel_dvh.structure_volume_cc:.1f}cc, "
                f"mean={voxel_dvh.mean_dose_Gy:.2f}Gy"
            )

    # ── Results ──
    df = results_to_dataframe(all_results)
    csv_path = RESULTS_DIR / "phase2_crossvalidation.csv"
    df.to_csv(csv_path, index=False)

    cache_df = df[~df["parameter_type"].str.startswith("intersystem_")]
    inter_df = df[df["parameter_type"].str.startswith("intersystem_")]

    print(f"\n{'='*60}")
    print(f"PHASE 2 RESULTS: {len(df)} total comparisons")
    print(f"{'='*60}")
    print(f"\nCache test: {len(cache_df)} comparisons")
    print(f"  Within tolerance: {cache_df['within_tolerance'].sum()} ({cache_df['within_tolerance'].mean()*100:.1f}%)")
    for method in INTERPOLATION_METHODS:
        sub = cache_df[cache_df["interpolation_method"] == method]
        if len(sub) > 0:
            pct = sub["within_tolerance"].mean() * 100
            print(f"  {method}: pass={pct:.1f}%, max_err={sub['abs_error'].max():.8f}")

    if len(inter_df) > 0:
        print(f"\nInter-system: {len(inter_df)} comparisons")
        print(f"  Mean abs error: {inter_df['abs_error'].mean():.4f}")
        print(f"  Max abs error: {inter_df['abs_error'].max():.4f}")

    return df


if __name__ == "__main__":
    run_phase2()
