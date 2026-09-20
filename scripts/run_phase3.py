"""Phase 3: Clinical validation on RayStation patient data.

Runs the full cache-and-recover pipeline on all clinical cases:
1. Compute voxel DVH for every structure (dicompyler-core)
2. Write to DICOM RT DVH at 1 cGy
3. Read back
4. Derive all parameters (both interpolation methods)
5. Compare against reference
6. Record treatment type and volume class for stratification

Output:
    data/results/phase3_clinical_validation.csv
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLINICAL_DIR,
    DVH_BIN_WIDTH_CGY,
    INTERPOLATION_METHODS,
    RESULTS_DIR,
    classify_volume,
)
from src.comparison import (
    bland_altman_stats,
    compare_parameters,
    results_to_dataframe,
)
from src.dvh_cache import read_dicom_dvh, write_dicom_dvh
from src.dvh_compute import compute_dvh_from_dicom
from src.dvh_parameters import derive_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Treatment type classification for each patient
TREATMENT_TYPES = {
    "SABR-Long": "SBRT",
    "PAL": "Palliative",
    "MAM": "Conventional",
}


def find_dicom_files(patient_dir: Path) -> dict[str, Path]:
    """Identify RD, RS, RP files in a patient directory."""
    files = {}
    for f in patient_dir.glob("*.dcm"):
        name = f.name
        if name.startswith("RD"):
            files["dose"] = f
        elif name.startswith("RS"):
            files["struct"] = f
        elif name.startswith("RP"):
            files["plan"] = f
    return files


def run_phase3() -> pd.DataFrame:
    """Execute Phase 3 clinical validation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    patient_summaries = []

    patient_dirs = sorted(
        d for d in CLINICAL_DIR.iterdir() if d.is_dir()
    )

    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        treatment_type = TREATMENT_TYPES.get(patient_name, "Unknown")
        dicom_files = find_dicom_files(patient_dir)

        if "dose" not in dicom_files or "struct" not in dicom_files:
            logger.warning("Skipping %s: missing RT Dose or RT Struct", patient_name)
            continue

        print(f"\n{'='*60}")
        print(f"Patient: {patient_name} ({treatment_type})")
        print(f"{'='*60}")

        # Compute voxel DVH for ALL structures
        try:
            voxel_dvhs = compute_dvh_from_dicom(
                dicom_files["dose"],
                dicom_files["struct"],
                bin_width_cGy=DVH_BIN_WIDTH_CGY,
            )
        except Exception as e:
            logger.error("Failed to compute DVH for %s: %s", patient_name, e)
            continue

        print(f"  Structures computed: {len(voxel_dvhs)}")

        structures_processed = 0
        structures_failed = 0

        for struct_name, voxel_dvh in voxel_dvhs.items():
            if voxel_dvh.structure_volume_cc == 0:
                continue

            vol_class = classify_volume(voxel_dvh.structure_volume_cc)

            # Cache-and-recover
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    cache_path = Path(tmpdir) / "cached.dcm"
                    write_dicom_dvh([voxel_dvh], cache_path)
                    cached_dvhs = read_dicom_dvh(cache_path)

                if not cached_dvhs:
                    continue

                cached_dvh = cached_dvhs[0]

                # Fair comparison: same interpolation on both sides
                # This isolates the caching precision loss from
                # interpolation method differences
                for interp in INTERPOLATION_METHODS:
                    ref_params = derive_all(voxel_dvh, interpolation=interp)
                    cached_params = derive_all(cached_dvh, interpolation=interp)
                    results = compare_parameters(
                        ref_params,
                        cached_params,
                        structure_name=f"{patient_name}/{struct_name}",
                        structure_volume_cc=voxel_dvh.structure_volume_cc,
                        interpolation_method=interp,
                    )
                    # Add treatment type metadata
                    for r in results:
                        r.volume_class = vol_class
                    all_results.extend(results)

                structures_processed += 1
                print(
                    f"  {struct_name}: vol={voxel_dvh.structure_volume_cc:.1f}cc "
                    f"({vol_class}), mean={voxel_dvh.mean_dose_Gy:.2f}Gy"
                )

            except Exception as e:
                structures_failed += 1
                logger.warning("  %s: FAILED - %s", struct_name, e)

        patient_summaries.append({
            "patient": patient_name,
            "treatment_type": treatment_type,
            "structures_total": len(voxel_dvhs),
            "structures_processed": structures_processed,
            "structures_failed": structures_failed,
        })

    # Convert to DataFrame
    df = results_to_dataframe(all_results)

    # Save results
    csv_path = RESULTS_DIR / "phase3_clinical_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("PHASE 3 CLINICAL VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total comparisons: {len(df)}")
    if len(df) > 0:
        print(f"Within tolerance: {df['within_tolerance'].sum()} "
              f"({df['within_tolerance'].mean()*100:.1f}%)")

        print("\n--- By Interpolation Method ---")
        for method in INTERPOLATION_METHODS:
            sub = df[df["interpolation_method"] == method]
            if len(sub) > 0:
                pct = sub["within_tolerance"].mean() * 100
                max_e = sub["abs_error"].max()
                print(f"  {method}: {pct:.1f}% pass, max_err={max_e:.6f}")

        print("\n--- By Volume Class ---")
        for vc in ["tiny", "small", "medium", "large"]:
            sub = df[df["volume_class"] == vc]
            if len(sub) > 0:
                pct = sub["within_tolerance"].mean() * 100
                print(f"  {vc}: n={len(sub)}, pass={pct:.1f}%")

        print("\n--- By Patient ---")
        for patient_name in TREATMENT_TYPES:
            sub = df[df["structure_name"].str.startswith(patient_name)]
            if len(sub) > 0:
                pct = sub["within_tolerance"].mean() * 100
                tt = TREATMENT_TYPES[patient_name]
                print(f"  {patient_name} ({tt}): n={len(sub)}, pass={pct:.1f}%")

        # Bland-Altman
        print("\n--- Bland-Altman by Parameter Type ---")
        ba = bland_altman_stats(all_results, group_by="parameter_type")
        print(ba.to_string(index=False))

        # Failures
        failures = df[~df["within_tolerance"]]
        if len(failures) > 0:
            print(f"\n--- FAILURES ({len(failures)}) ---")
            for _, r in failures.head(20).iterrows():
                print(
                    f"  {r['structure_name']}: {r['parameter_name']} "
                    f"ref={r['ref_value']:.4f} cached={r['cached_value']:.4f} "
                    f"err={r['abs_error']:.4f} [{r['interpolation_method']}]"
                )
        else:
            print("\n*** ALL PARAMETERS WITHIN TOLERANCE ***")

    # Patient summary
    print("\n--- Patient Summary ---")
    for ps in patient_summaries:
        print(
            f"  {ps['patient']} ({ps['treatment_type']}): "
            f"{ps['structures_processed']} structures, "
            f"{ps['structures_failed']} failed"
        )

    return df


if __name__ == "__main__":
    run_phase3()
