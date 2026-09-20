"""Phase 6: External validation, sensitivity analysis, and reproducibility checks.

Step 6.1: External validation against dicompyler published reference values
Step 6.2: Bin width sensitivity — how parameters degrade as resolution coarsens
Step 6.3: DS VR precision characterisation
Step 6.4: Pre-publication checklist

Output:
    data/results/phase6_external_validation.csv
    data/results/phase6_sensitivity.csv
    data/results/phase6_findings.md
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import INTERPOLATION_METHODS, RESULTS_DIR, SENSITIVITY_BIN_WIDTHS_CGY
from src.comparison import compare_parameters, results_to_dataframe
from src.dvh_cache import extract_embedded_dvh, read_dicom_dvh, write_dicom_dvh
from src.dvh_compute import compute_dvh_from_dicom
from src.dvh_parameters import (
    compute_dx,
    compute_dxcc,
    compute_mean_dose,
    compute_vx,
    derive_all,
)

NELMS_BASE = PROJECT_ROOT / "data" / "nelms" / "DVH-Analysis-Data-Etc"
DICOMPYLER_DIR = PROJECT_ROOT / "data" / "dicompyler"


# ══════════════════════════════════════════════════════════════
# Step 6.1: External validation against published reference values
# ══════════════════════════════════════════════════════════════

# Published expected values from dicompyler-core documentation
DICOMPYLER_PUBLISHED = {
    "Heart": {
        "volume_cc": 437.46,
        "max_dose_Gy": 3.10,
        "min_dose_Gy": 0.02,
        "mean_dose_Gy": 0.64,
        "D98_Gy": 0.03,
        "D95_Gy": 0.03,
        "D2cc_Gy": 2.93,
    },
}


def run_external_validation() -> pd.DataFrame:
    """Validate cached DVH against external published reference values.

    Pipeline: voxel DVH → DICOM RT DVH cache → read back → derive
    parameters → compare against published values.

    This is the strongest test: an independent reference (published by
    dicompyler-core authors) verifies our entire pipeline end-to-end.
    """
    dose_path = DICOMPYLER_DIR / "rtdose.dcm"
    struct_path = DICOMPYLER_DIR / "rtss.dcm"

    # Also extract embedded TPS DVH for three-way check
    embedded_dvhs = extract_embedded_dvh(dose_path, struct_path)
    embedded_by_name = {d.structure_name: d for d in embedded_dvhs}

    # Compute our voxel DVH
    voxel_dvhs = compute_dvh_from_dicom(dose_path, struct_path, bin_width_cGy=1.0)

    rows = []

    for struct_name, published in DICOMPYLER_PUBLISHED.items():
        if struct_name not in voxel_dvhs:
            continue

        voxel_dvh = voxel_dvhs[struct_name]

        # Cache and recover
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cached.dcm"
            write_dicom_dvh([voxel_dvh], cache_path)
            cached_dvh = read_dicom_dvh(cache_path)[0]

        # Three sources to compare against published values
        sources = {
            "published": published,
            "embedded_tps": {},
            "our_voxel": {},
            "our_cached": {},
        }

        # Embedded TPS values
        if struct_name in embedded_by_name:
            emb = embedded_by_name[struct_name]
            sources["embedded_tps"] = {
                "volume_cc": emb.structure_volume_cc,
                "max_dose_Gy": emb.max_dose_Gy,
                "min_dose_Gy": emb.min_dose_Gy,
                "mean_dose_Gy": emb.mean_dose_Gy,
                "D98_Gy": compute_dx(emb, 98.0),
                "D95_Gy": compute_dx(emb, 95.0),
                "D2cc_Gy": compute_dxcc(emb, 2.0),
            }

        # Our voxel values
        sources["our_voxel"] = {
            "volume_cc": voxel_dvh.structure_volume_cc,
            "max_dose_Gy": voxel_dvh.max_dose_Gy,
            "min_dose_Gy": voxel_dvh.min_dose_Gy,
            "mean_dose_Gy": compute_mean_dose(voxel_dvh),
            "D98_Gy": compute_dx(voxel_dvh, 98.0),
            "D95_Gy": compute_dx(voxel_dvh, 95.0),
            "D2cc_Gy": compute_dxcc(voxel_dvh, 2.0),
        }

        # Our cached values
        sources["our_cached"] = {
            "volume_cc": cached_dvh.structure_volume_cc,
            "max_dose_Gy": cached_dvh.max_dose_Gy,
            "min_dose_Gy": cached_dvh.min_dose_Gy,
            "mean_dose_Gy": compute_mean_dose(cached_dvh),
            "D98_Gy": compute_dx(cached_dvh, 98.0),
            "D95_Gy": compute_dx(cached_dvh, 95.0),
            "D2cc_Gy": compute_dxcc(cached_dvh, 2.0),
        }

        for param_name, pub_value in published.items():
            for source_name in ["embedded_tps", "our_voxel", "our_cached"]:
                src_values = sources[source_name]
                if param_name not in src_values:
                    continue
                our_value = src_values[param_name]
                abs_err = abs(our_value - pub_value)
                rel_err = (abs_err / pub_value * 100) if pub_value != 0 else 0

                rows.append({
                    "structure": struct_name,
                    "parameter": param_name,
                    "published_value": pub_value,
                    "source": source_name,
                    "our_value": our_value,
                    "abs_error": abs_err,
                    "rel_error_pct": rel_err,
                })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Step 6.2: Bin width sensitivity — parameter degradation
# ══════════════════════════════════════════════════════════════


def run_sensitivity_analysis() -> pd.DataFrame:
    """Measure how clinical parameters degrade as bin width coarsens.

    Uses 1 cGy as the reference (dicompyler-core's native resolution).
    Coarser bin widths are produced by rebinning the 1 cGy DVH data.

    NOTE: Sub-cGy resolution (e.g. 0.1 cGy) is NOT available because
    dicompyler-core internally computes at 1 cGy. The sensitivity
    analysis therefore tests only coarser resolutions: 2, 5, 10 cGy.

    This answers: "At what bin width do parameters start to deviate
    beyond clinical tolerance?"
    """
    dose_path = NELMS_BASE / "DOSE GRIDS" / "Linear_SupInf_1mm_Aligned.dcm"

    test_structures = [
        ("Spheres", "Sphere_10_0.dcm"),
        ("Cylinders", "Cylinder_10_0.dcm"),
        ("Cones", "Cone_10_0.dcm"),
    ]

    # Only test coarser bin widths (1 cGy is the native reference)
    coarser_bin_widths = [bw for bw in SENSITIVITY_BIN_WIDTHS_CGY if bw >= 1.0]

    all_rows = []

    for shape_dir, struct_file in test_structures:
        struct_path = NELMS_BASE / "STRUCTURES" / shape_dir / struct_file

        # Reference: native 1 cGy resolution
        ref_dvhs = compute_dvh_from_dicom(dose_path, struct_path, bin_width_cGy=1.0)
        ref_dvh = None
        struct_name = struct_file
        for name, dvh in ref_dvhs.items():
            if "POI" not in name:
                ref_dvh = dvh
                struct_name = name
                break
        if ref_dvh is None:
            continue

        ref_params = derive_all(ref_dvh, interpolation="linear")

        # Compare at each coarser bin width
        for bin_width_cGy in coarser_bin_widths:
            test_dvhs = compute_dvh_from_dicom(
                dose_path, struct_path, bin_width_cGy=bin_width_cGy,
            )
            test_dvh = None
            for name, dvh in test_dvhs.items():
                if "POI" not in name:
                    test_dvh = dvh
                    break
            if test_dvh is None:
                continue

            test_params = derive_all(test_dvh, interpolation="linear")

            for param_name in ref_params:
                if param_name not in test_params:
                    continue
                ref_val = ref_params[param_name]
                test_val = test_params[param_name]
                abs_err = abs(test_val - ref_val)

                all_rows.append({
                    "structure": struct_name,
                    "bin_width_cGy": bin_width_cGy,
                    "parameter": param_name,
                    "ref_value_01cGy": ref_val,
                    "test_value": test_val,
                    "abs_error": abs_err,
                })

    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════
# Step 6.3: DS VR precision characterisation
# ══════════════════════════════════════════════════════════════


def run_ds_precision_test() -> pd.DataFrame:
    """Characterise the actual precision of the DS VR encoding."""
    rows = []

    for magnitude in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8]:
        test_value = magnitude * 1.23456789012345  # 15 sig digits
        formatted = f"{test_value:.8g}"
        recovered = float(formatted)
        abs_err = abs(recovered - test_value)
        rel_err = abs_err / abs(test_value) if test_value != 0 else 0

        rows.append({
            "original": test_value,
            "magnitude": magnitude,
            "formatted": formatted,
            "recovered": recovered,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "sig_digits_preserved": -int(np.floor(np.log10(rel_err))) if rel_err > 0 else 16,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Step 6.4: Pre-publication checklist
# ══════════════════════════════════════════════════════════════


def run_checklist() -> list[tuple[str, bool, str]]:
    """Verify all reproducibility requirements."""
    checks = []

    from config import TOLERANCES
    expected = {"Vx": 1.0, "Dx": 0.5, "Dxcc": 0.5, "mean_dose": 1.0, "gEUD": 1.0, "NTCP_TCP": 1.0}
    tol_ok = all(TOLERANCES[k]["value"] == v for k, v in expected.items())
    checks.append(("Tolerances match proposal Table 4", tol_ok, ""))

    import subprocess
    result = subprocess.run(
        ["git", "log", "--oneline", "--diff-filter=A", "--", "dvh-validation/config.py"],
        capture_output=True, text=True, cwd=PROJECT_ROOT.parent,
    )
    checks.append(("config.py in version control", bool(result.stdout.strip()), result.stdout.strip()))

    for script in ["download_nelms.py", "download_dicompyler.py", "download_slicerrt.py"]:
        checks.append((f"{script} exists", (PROJECT_ROOT / "scripts" / script).exists(), ""))

    for phase in ["run_phase1.py", "run_phase2.py", "run_phase3.py", "run_phase4.py", "run_phase5.py", "run_phase6.py"]:
        checks.append((f"{phase} exists", (PROJECT_ROOT / "scripts" / phase).exists(), ""))

    checks.append(("Nelms dataset present", NELMS_BASE.exists(), ""))
    checks.append(("dicompyler test data present", (DICOMPYLER_DIR / "rtdose.dcm").exists(), ""))
    checks.append(("SlicerRtData present", (PROJECT_ROOT / "data" / "slicerrt" / "prostate").exists(), ""))

    return checks


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════


def run_phase6():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 6.1: External validation ──
    print(f"{'='*60}")
    print("Step 6.1: External validation against published values")
    print(f"{'='*60}")

    ext_df = run_external_validation()
    ext_csv = RESULTS_DIR / "phase6_external_validation.csv"
    ext_df.to_csv(ext_csv, index=False)

    print(f"\n  {'Parameter':<18} {'Published':>10} {'Embedded':>10} {'Voxel':>10} {'Cached':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for param in DICOMPYLER_PUBLISHED["Heart"]:
        pub = DICOMPYLER_PUBLISHED["Heart"][param]
        emb = ext_df[(ext_df["parameter"] == param) & (ext_df["source"] == "embedded_tps")]
        vox = ext_df[(ext_df["parameter"] == param) & (ext_df["source"] == "our_voxel")]
        cac = ext_df[(ext_df["parameter"] == param) & (ext_df["source"] == "our_cached")]
        emb_v = f"{emb['our_value'].iloc[0]:.4f}" if len(emb) > 0 else "—"
        vox_v = f"{vox['our_value'].iloc[0]:.4f}" if len(vox) > 0 else "—"
        cac_v = f"{cac['our_value'].iloc[0]:.4f}" if len(cac) > 0 else "—"
        print(f"  {param:<18} {pub:>10.4f} {emb_v:>10} {vox_v:>10} {cac_v:>10}")

    # ── Step 6.2: Sensitivity analysis ──
    print(f"\n{'='*60}")
    print("Step 6.2: Bin width sensitivity (params vs 0.1 cGy reference)")
    print(f"{'='*60}")

    sens_df = run_sensitivity_analysis()
    sens_csv = RESULTS_DIR / "phase6_sensitivity.csv"
    sens_df.to_csv(sens_csv, index=False)

    print(f"\n  {'Bin Width':>12} {'N':>6} {'Mean Err':>12} {'Max Err':>12} {'Max Err Param':>20}")
    print(f"  {'-'*12} {'-'*6} {'-'*12} {'-'*12} {'-'*20}")
    for bw in SENSITIVITY_BIN_WIDTHS_CGY:
        sub = sens_df[sens_df["bin_width_cGy"] == bw]
        if len(sub) == 0:
            continue
        mean_e = sub["abs_error"].mean()
        max_e = sub["abs_error"].max()
        worst = sub.loc[sub["abs_error"].idxmax()]
        marker = " <-- primary" if bw == 1.0 else ""
        print(f"  {bw:>10.1f} cGy {len(sub):>6} {mean_e:>12.6f} {max_e:>12.6f} {worst['parameter']:>20}{marker}")

    # ── Step 6.3: DS VR precision ──
    print(f"\n{'='*60}")
    print("Step 6.3: DS VR precision characterisation")
    print(f"{'='*60}")

    prec_df = run_ds_precision_test()
    prec_df.to_csv(RESULTS_DIR / "phase6_ds_precision.csv", index=False)
    print(f"\n  {'Magnitude':>12} {'Formatted':>20} {'Rel Error':>14} {'Sig Digits':>12}")
    print(f"  {'-'*12} {'-'*20} {'-'*14} {'-'*12}")
    for _, row in prec_df.iterrows():
        print(f"  {row['magnitude']:>12.0e} {row['formatted']:>20} {row['rel_error']:>14.2e} {row['sig_digits_preserved']:>12}")

    # ── Step 6.4: Checklist ──
    print(f"\n{'='*60}")
    print("Step 6.4: Pre-publication checklist")
    print(f"{'='*60}")

    checks = run_checklist()
    checklist_rows = []
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
        checklist_rows.append({"check": desc, "passed": passed, "detail": detail})

    checklist_df = pd.DataFrame(checklist_rows)
    checklist_df.to_csv(RESULTS_DIR / "phase6_checklist.csv", index=False)

    print(f"\nPhase 6 complete. CSVs saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run_phase6()
