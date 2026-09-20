"""DICOM vs JSON format size and speed comparison.

Generates multiple JSON representations of the same DVH data and benchmarks
file size (raw and compressed) and read/write speed against native DICOM
RT DVH files. Supports gzip, bz2, and zstd compression.
"""

from __future__ import annotations

import bz2
import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zstandard

from .dvh_cache import read_dicom_dvh, write_dicom_dvh
from .dvh_data import DVHData


# ──────────────────────────────────────────────────────────────────────
# JSON writers
# ──────────────────────────────────────────────────────────────────────


def write_json_flat(
    dvh_list: list[DVHData],
    path: str | Path,
    pretty: bool = True,
) -> Path:
    """Write DVH data as flat-array JSON.

    Format: ``{"structures": [{"name": ..., "bin_width_cGy": ...,
    "volume_cc": ..., "dose_edges_cGy": [...], "diff_volumes_cc": [...]}]}``
    """
    path = Path(path)
    data = {
        "format": "dvh_flat",
        "bin_width_cGy": dvh_list[0].bin_width_cGy if dvh_list else 1.0,
        "structures": [
            {
                "name": dvh.structure_name,
                "volume_cc": dvh.structure_volume_cc,
                "bin_width_cGy": dvh.bin_width_cGy,
                "dose_edges_cGy": dvh.dose_edges_cGy.tolist(),
                "diff_volumes_cc": dvh.diff_volumes_cc.tolist(),
                "min_dose_Gy": dvh.min_dose_Gy,
                "max_dose_Gy": dvh.max_dose_Gy,
                "mean_dose_Gy": dvh.mean_dose_Gy,
            }
            for dvh in dvh_list
        ],
    }
    indent = 2 if pretty else None
    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    return path


def write_json_verbose(
    dvh_list: list[DVHData],
    path: str | Path,
    pretty: bool = True,
) -> Path:
    """Write DVH data as verbose per-bin JSON.

    Format: each bin is an object with dose and volume keys.
    ``{"structures": [{"name": ..., "bins": [{"dose_Gy": ..., "volume_cc": ...}, ...]}]}``
    """
    path = Path(path)
    data = {
        "format": "dvh_verbose",
        "structures": [
            {
                "name": dvh.structure_name,
                "volume_cc": dvh.structure_volume_cc,
                "bin_width_cGy": dvh.bin_width_cGy,
                "min_dose_Gy": dvh.min_dose_Gy,
                "max_dose_Gy": dvh.max_dose_Gy,
                "mean_dose_Gy": dvh.mean_dose_Gy,
                "bins": [
                    {
                        "dose_lower_cGy": float(dvh.dose_edges_cGy[i]),
                        "dose_upper_cGy": float(dvh.dose_edges_cGy[i + 1]),
                        "dose_center_Gy": float(dvh.dose_centers_Gy[i]),
                        "diff_volume_cc": float(dvh.diff_volumes_cc[i]),
                    }
                    for i in range(dvh.n_bins)
                ],
            }
            for dvh in dvh_list
        ],
    }
    indent = 2 if pretty else None
    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────
# JSON readers
# ──────────────────────────────────────────────────────────────────────


def read_json_flat(path: str | Path) -> list[DVHData]:
    """Read flat-array JSON back into DVHData objects."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for s in data["structures"]:
        results.append(DVHData(
            structure_name=s["name"],
            structure_volume_cc=s["volume_cc"],
            dose_edges_cGy=np.array(s["dose_edges_cGy"], dtype=np.float64),
            diff_volumes_cc=np.array(s["diff_volumes_cc"], dtype=np.float64),
            bin_width_cGy=s["bin_width_cGy"],
            source="json_flat",
        ))
    return results


def read_json_verbose(path: str | Path) -> list[DVHData]:
    """Read verbose per-bin JSON back into DVHData objects."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for s in data["structures"]:
        bins = s["bins"]
        dose_edges = [b["dose_lower_cGy"] for b in bins]
        dose_edges.append(bins[-1]["dose_upper_cGy"])
        diff_vols = [b["diff_volume_cc"] for b in bins]
        results.append(DVHData(
            structure_name=s["name"],
            structure_volume_cc=s["volume_cc"],
            dose_edges_cGy=np.array(dose_edges, dtype=np.float64),
            diff_volumes_cc=np.array(diff_vols, dtype=np.float64),
            bin_width_cGy=s["bin_width_cGy"],
            source="json_verbose",
        ))
    return results


# ──────────────────────────────────────────────────────────────────────
# Size measurement
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SizeResult:
    """File size measurements for one format."""
    format_id: str
    raw_bytes: int
    gzip_bytes: int
    bz2_bytes: int
    zstd_bytes: int

    @property
    def best_compressed(self) -> int:
        return min(self.gzip_bytes, self.bz2_bytes, self.zstd_bytes)

    @property
    def best_algorithm(self) -> str:
        sizes = {"gzip": self.gzip_bytes, "bz2": self.bz2_bytes, "zstd": self.zstd_bytes}
        return min(sizes, key=sizes.get)


def measure_sizes(path: str | Path, format_id: str = "") -> SizeResult:
    """Measure raw and compressed file sizes."""
    path = Path(path)
    raw_data = path.read_bytes()
    raw_bytes = len(raw_data)

    gzip_bytes = len(gzip.compress(raw_data, compresslevel=9))
    bz2_bytes = len(bz2.compress(raw_data, compresslevel=9))

    cctx = zstandard.ZstdCompressor(level=19)
    zstd_bytes = len(cctx.compress(raw_data))

    return SizeResult(
        format_id=format_id,
        raw_bytes=raw_bytes,
        gzip_bytes=gzip_bytes,
        bz2_bytes=bz2_bytes,
        zstd_bytes=zstd_bytes,
    )


# ──────────────────────────────────────────────────────────────────────
# Timing benchmark
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TimingResult:
    """Timing measurements for one format."""
    format_id: str
    write_ms: float
    read_ms: float
    n_iterations: int


def benchmark_timing(
    write_fn,
    read_fn,
    write_args: tuple,
    read_args: tuple,
    n_iterations: int = 50,
    n_warmup: int = 5,
) -> TimingResult:
    """Benchmark write and read times for a format.

    Parameters
    ----------
    write_fn : callable
        Function to call for writing.
    read_fn : callable
        Function to call for reading.
    write_args : tuple
        Arguments to pass to write_fn.
    read_args : tuple
        Arguments to pass to read_fn.
    n_iterations : int
        Number of timed iterations.
    n_warmup : int
        Warmup iterations (not timed).

    Returns
    -------
    TimingResult
    """
    # Warmup
    for _ in range(n_warmup):
        write_fn(*write_args)
        read_fn(*read_args)

    # Time writes
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        write_fn(*write_args)
    write_total = time.perf_counter() - t0

    # Time reads
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        read_fn(*read_args)
    read_total = time.perf_counter() - t0

    return TimingResult(
        format_id="",
        write_ms=(write_total / n_iterations) * 1000,
        read_ms=(read_total / n_iterations) * 1000,
        n_iterations=n_iterations,
    )


# ──────────────────────────────────────────────────────────────────────
# Metadata comparison
# ──────────────────────────────────────────────────────────────────────


DICOM_METADATA_FIELDS = [
    ("Patient Name", "PatientName", "DICOM Patient module", "Must define custom field"),
    ("Patient ID", "PatientID", "DICOM Patient module", "Must define custom field"),
    ("Study Instance UID", "StudyInstanceUID", "Globally unique, DICOM-generated", "Must generate or omit"),
    ("Series Instance UID", "SeriesInstanceUID", "Globally unique, DICOM-generated", "Must generate or omit"),
    ("SOP Instance UID", "SOPInstanceUID", "Globally unique, DICOM-generated", "Must generate or omit"),
    ("SOP Class UID", "SOPClassUID", "RT DVH Storage (standardised)", "No equivalent"),
    ("Frame of Reference UID", "FrameOfReferenceUID", "Links to dose/struct spatial frame", "Must define custom field"),
    ("Referenced RT Struct UID", "ReferencedStructureSetSequence", "Direct DICOM reference", "Must store as string field"),
    ("Referenced RT Plan UID", "ReferencedRTPlanSequence", "Direct DICOM reference", "Must store as string field"),
    ("DVH Type", "DVHType", "DIFFERENTIAL or CUMULATIVE (enumerated)", "Custom field"),
    ("Dose Units", "DoseUnits", "GY or RELATIVE (standardised)", "Custom field"),
    ("Dose Type", "DoseType", "PHYSICAL, EFFECTIVE, ERROR (standardised)", "Custom field"),
    ("DVH Volume Units", "DVHVolumeUnits", "CM3 or PERCENT (standardised)", "Custom field"),
    ("DVH Dose Scaling", "DVHDoseScaling", "Standardised tag", "Custom field"),
    ("DVH Number of Bins", "DVHNumberOfBins", "Standardised tag", "Implicit from array length"),
    ("DVH Data", "DVHData", "Interleaved bin_width/volume pairs", "Custom arrays"),
    ("DVH Minimum Dose", "DVHMinimumDose", "Standardised tag", "Custom field"),
    ("DVH Maximum Dose", "DVHMaximumDose", "Standardised tag", "Custom field"),
    ("DVH Mean Dose", "DVHMeanDose", "Standardised tag", "Custom field"),
    ("Referenced ROI Number", "DVHReferencedROISequence", "Links to structure by number", "Custom field"),
    ("Modality", "Modality", "Standard DICOM tag", "Custom field"),
    ("Software Versions", "SoftwareVersions", "Standard DICOM tag", "Custom field"),
    ("Instance Creation Date", "InstanceCreationDate", "Standard DICOM tag", "Custom field"),
    ("Transfer Syntax", "TransferSyntaxUID", "Defines encoding (Implicit/Explicit VR)", "No equivalent"),
    ("Interoperability", "—", "Any DICOM viewer/library can read", "Requires custom parser"),
    ("Standardisation body", "—", "NEMA/ISO (PS3.3)", "No standard (ad hoc schema)"),
    ("Max element size", "—", "~4 GB per element (Implicit VR)", "Unlimited (JSON)"),
]


def generate_metadata_comparison() -> list[dict]:
    """Generate DICOM vs JSON metadata completeness comparison."""
    rows = []
    for field_name, dicom_tag, dicom_notes, json_notes in DICOM_METADATA_FIELDS:
        rows.append({
            "field": field_name,
            "dicom_tag": dicom_tag,
            "dicom": dicom_notes,
            "json": json_notes,
        })
    return rows


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
