"""Phase 4: Format comparison — DICOM vs JSON.

End-to-end benchmark: measures the complete workflow from "I need DVH
parameters" to "here they are" for each pathway:

1. No cache:     RT Dose + RT Struct → compute DVH → derive params
2. DICOM cache:  Read DICOM RT DVH → derive params
3. JSON cache:   Read JSON flat → derive params

Also compares file sizes (raw + compressed) and metadata completeness.

Output:
    data/results/phase4_format_comparison.csv
    data/results/phase4_metadata_comparison.md
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CLINICAL_DIR, DVH_BIN_WIDTH_CGY, RESULTS_DIR
from src.dvh_cache import read_dicom_dvh, write_dicom_dvh
from src.dvh_compute import compute_dvh_from_dicom
from src.dvh_parameters import derive_all
from src.format_benchmark import (
    generate_metadata_comparison,
    human_readable_size,
    measure_sizes,
    read_json_flat,
    read_json_verbose,
    write_json_flat,
    write_json_verbose,
)


def find_dicom_files(patient_dir: Path) -> dict[str, Path]:
    files = {}
    for f in patient_dir.glob("*.dcm"):
        if f.name.startswith("RD"):
            files["dose"] = f
        elif f.name.startswith("RS"):
            files["struct"] = f
    return files


def run_phase4():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    patient_dirs = sorted(d for d in CLINICAL_DIR.iterdir() if d.is_dir())

    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        dicom_files = find_dicom_files(patient_dir)
        if "dose" not in dicom_files or "struct" not in dicom_files:
            continue

        dose_path = dicom_files["dose"]
        struct_path = dicom_files["struct"]
        dose_size = dose_path.stat().st_size

        print(f"\n{'='*70}")
        print(f"Patient: {patient_name}")
        print(f"RT Dose: {human_readable_size(dose_size)}")
        print(f"{'='*70}")

        # ── Pathway 1: No cache (baseline) ──
        t0 = time.perf_counter()
        voxel_dvhs = compute_dvh_from_dicom(
            dose_path, struct_path, bin_width_cGy=DVH_BIN_WIDTH_CGY,
        )
        dvh_list = [d for d in voxel_dvhs.values() if d.structure_volume_cc > 0]
        for dvh in dvh_list:
            derive_all(dvh, interpolation="linear")
        no_cache_ms = (time.perf_counter() - t0) * 1000

        n_structs = len(dvh_list)
        total_bins = sum(d.n_bins for d in dvh_list)
        print(f"  Structures: {n_structs}, total bins: {total_bins}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Prepare all cache formats
            dcm_path = tmp / "cache.dcm"
            json_flat_path = tmp / "cache_flat.json"
            json_flat_pretty_path = tmp / "cache_flat_pretty.json"
            json_verbose_path = tmp / "cache_verbose.json"
            json_verbose_pretty_path = tmp / "cache_verbose_pretty.json"

            write_dicom_dvh(dvh_list, dcm_path)
            write_json_flat(dvh_list, json_flat_path, pretty=False)
            write_json_flat(dvh_list, json_flat_pretty_path, pretty=True)
            write_json_verbose(dvh_list, json_verbose_path, pretty=False)
            write_json_verbose(dvh_list, json_verbose_pretty_path, pretty=True)

            # ── End-to-end timing: cache → read → derive params ──
            N = 30
            cache_formats = [
                ("dicom", dcm_path, read_dicom_dvh),
                ("json_flat_mini", json_flat_path, read_json_flat),
                ("json_flat_pretty", json_flat_pretty_path, read_json_flat),
                ("json_verbose_mini", json_verbose_path, read_json_verbose),
                ("json_verbose_pretty", json_verbose_pretty_path, read_json_verbose),
            ]

            # Warmup all
            for _, fpath, read_fn in cache_formats:
                recovered = read_fn(fpath)
                for dvh in recovered:
                    derive_all(dvh, interpolation="linear")

            # Time each
            timings = {}
            for fmt_id, fpath, read_fn in cache_formats:
                t0 = time.perf_counter()
                for _ in range(N):
                    recovered = read_fn(fpath)
                    for dvh in recovered:
                        derive_all(dvh, interpolation="linear")
                timings[fmt_id] = (time.perf_counter() - t0) / N * 1000

            # ── File sizes ──
            size_formats = [
                ("dicom", dcm_path),
                ("json_flat_mini", json_flat_path),
                ("json_flat_pretty", json_flat_pretty_path),
                ("json_verbose_mini", json_verbose_path),
                ("json_verbose_pretty", json_verbose_pretty_path),
            ]

            sizes = {}
            for fmt_id, fpath in size_formats:
                sizes[fmt_id] = measure_sizes(fpath, format_id=fmt_id)

            # ── Print results ──
            dicom_raw = sizes["dicom"].raw_bytes
            print(f"\n  END-TO-END: 'I need DVH parameters — how fast?'")
            print(
                f"  {'Pathway':<30} {'Time':>10} {'Speedup':>10} "
                f"{'Raw size':>12} {'Compressed':>12} {'Size ratio':>12}"
            )
            print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
            print(
                f"  {'No cache (compute from dose)':<30} "
                f"{no_cache_ms:>8.0f}ms {'1.0x':>10} "
                f"{human_readable_size(dose_size):>12} {'—':>12} {'—':>12}"
            )

            for fmt_id, fpath, _ in cache_formats:
                t_ms = timings[fmt_id]
                s = sizes[fmt_id]
                speedup = no_cache_ms / t_ms if t_ms > 0 else float("inf")
                ratio = s.raw_bytes / dicom_raw
                print(
                    f"  {fmt_id:<30} "
                    f"{t_ms:>8.1f}ms {speedup:>9.0f}x "
                    f"{human_readable_size(s.raw_bytes):>12} "
                    f"{human_readable_size(s.best_compressed):>12} "
                    f"{ratio:>11.1f}x"
                )

                all_rows.append({
                    "patient": patient_name,
                    "format": fmt_id,
                    "structures": n_structs,
                    "total_bins": total_bins,
                    "no_cache_ms": no_cache_ms,
                    "end_to_end_ms": t_ms,
                    "speedup": speedup,
                    "raw_bytes": s.raw_bytes,
                    "gzip_bytes": s.gzip_bytes,
                    "bz2_bytes": s.bz2_bytes,
                    "zstd_bytes": s.zstd_bytes,
                    "best_compressed_bytes": s.best_compressed,
                    "best_algorithm": s.best_algorithm,
                    "rt_dose_bytes": dose_size,
                })

    # Save results
    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "phase4_format_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Metadata comparison
    meta_rows = generate_metadata_comparison()
    meta_path = RESULTS_DIR / "phase4_metadata_comparison.md"
    with open(meta_path, "w") as f:
        f.write("# DICOM RT DVH vs JSON — Metadata Completeness\n\n")
        f.write("| Field | DICOM Tag | DICOM | JSON |\n")
        f.write("|-------|-----------|-------|------|\n")
        for row in meta_rows:
            f.write(
                f"| {row['field']} | `{row['dicom_tag']}` "
                f"| {row['dicom']} | {row['json']} |\n"
            )
    print(f"Saved metadata comparison to {meta_path}")

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 4 SUMMARY (averages across patients)")
    print(f"{'='*70}")
    for fmt in df["format"].unique():
        sub = df[df["format"] == fmt]
        print(
            f"  {fmt:<30} "
            f"e2e={sub['end_to_end_ms'].mean():>7.1f}ms  "
            f"speedup={sub['speedup'].mean():>6.0f}x  "
            f"raw={human_readable_size(int(sub['raw_bytes'].mean())):>10}  "
            f"compressed={human_readable_size(int(sub['best_compressed_bytes'].mean())):>10}"
        )


if __name__ == "__main__":
    run_phase4()