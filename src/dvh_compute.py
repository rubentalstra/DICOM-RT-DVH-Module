"""Voxel-level DVH computation from DICOM RT Dose + RT Structure Set.

Uses dicompyler-core for the actual DVH calculation — a well-tested,
published package. This module is a thin wrapper that:

1. Applies compatibility patches for pydicom 3.0 + dicompyler-core 0.5.6
2. Calls dicompyler-core to compute the reference DVH
3. Converts the result into our DVHData dataclass

For the paper: "Reference DVH curves were computed using dicompyler-core
0.5.6 (Panchal & Keyes), which implements slice-by-slice polygon fill
using matplotlib.path.Path.contains_points()."
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

# ── Patch pydicom 3.0 compatibility for dicompyler-core 0.5.6 ──────
# dicompyler-core uses deprecated/removed pydicom APIs.
# These patches restore the expected interfaces without modifying
# the dicompyler-core source code.
import pydicom.dicomio

pydicom.dicomio.read_file = pydicom.dcmread  # type: ignore[attr-defined]

from pydicom.pixels.utils import pixel_dtype  # noqa: E402

import pydicom.pixel_data_handlers.util  # type: ignore[import-untyped] # noqa: E402

pydicom.pixel_data_handlers.util.pixel_dtype = pixel_dtype  # type: ignore[attr-defined]

warnings.filterwarnings("ignore", message=r".*pydicom\.pixel_data_handlers.*")
warnings.filterwarnings(
    "ignore", message=r".*value length.*exceeds the maximum length.*"
)
# ── End patches ─────────────────────────────────────────────────────

import pydicom  # noqa: E402
from dicompylercore import dvhcalc  # noqa: E402

from .dvh_data import DVHData  # noqa: E402

logger = logging.getLogger(__name__)


def compute_dvh_from_dicom(
    rt_dose_path: str | Path,
    rt_struct_path: str | Path,
    bin_width_cGy: float = 1.0,
    structure_names: list[str] | None = None,
) -> dict[str, DVHData]:
    """Compute DVH for all (or selected) structures.

    Parameters
    ----------
    rt_dose_path : str or Path
        Path to the DICOM RT Dose file.
    rt_struct_path : str or Path
        Path to the DICOM RT Structure Set file.
    bin_width_cGy : float
        DVH bin width in cGy (default 1 = 0.01 Gy).
    structure_names : list of str, optional
        If provided, only compute DVH for structures whose name
        contains one of these strings (case-insensitive). If ``None``,
        compute for all structures.

    Returns
    -------
    dict[str, DVHData]
        Mapping from structure name to DVHData object.
    """
    rt_dose_path = Path(rt_dose_path)
    rt_struct_path = Path(rt_struct_path)

    ds_struct = pydicom.dcmread(rt_struct_path)
    ds_dose = pydicom.dcmread(rt_dose_path)

    results: dict[str, DVHData] = {}

    for roi in ds_struct.StructureSetROISequence:
        roi_number = int(roi.ROINumber)
        roi_name = str(roi.ROIName)

        # Filter by structure name if requested
        if structure_names is not None:
            if not any(
                s.lower() in roi_name.lower() for s in structure_names
            ):
                continue

        try:
            dvh_obj = dvhcalc.get_dvh(ds_struct, ds_dose, roi_number)
        except Exception:
            logger.warning("Failed to compute DVH for '%s' (ROI #%d)", roi_name, roi_number)
            continue

        dvh_data = _dicompyler_dvh_to_dvhdata(dvh_obj, roi_name, bin_width_cGy)
        if dvh_data is not None:
            results[roi_name] = dvh_data

    return results


def _dicompyler_dvh_to_dvhdata(
    dvh_obj,
    structure_name: str,
    target_bin_width_cGy: float,
) -> DVHData | None:
    """Convert a dicompyler-core DVH object to our DVHData format.

    dicompyler-core internally computes at 1 cGy (0.01 Gy) resolution.
    If a different bin width is requested, we rebin by summing adjacent
    bins.

    Parameters
    ----------
    dvh_obj : dicompylercore.dvh.DVH
        DVH object from dicompyler-core (cumulative, in Gy and cc).
    structure_name : str
        Name to assign to the DVHData.
    target_bin_width_cGy : float
        Target bin width in cGy.

    Returns
    -------
    DVHData or None
        DVHData object, or None if the DVH has zero volume.
    """
    # Get differential DVH in absolute volume (cc)
    diff_dvh = dvh_obj.differential
    if diff_dvh.volume_units != "cm3":
        diff_dvh = diff_dvh.absolute_volume

    volume_cc = float(dvh_obj.volume)
    if volume_cc == 0:
        return None

    # dicompyler-core bins are in Gy at 1 cGy (0.01 Gy) resolution
    # diff_dvh.bins = bin edges in Gy (N+1 values)
    # diff_dvh.counts = differential volumes in cc (N values)
    native_bins_Gy = np.array(diff_dvh.bins, dtype=np.float64)
    native_counts_cc = np.array(diff_dvh.counts, dtype=np.float64)
    native_bin_width_Gy = float(native_bins_Gy[1] - native_bins_Gy[0])
    native_bin_width_cGy = native_bin_width_Gy * 100.0

    target_bin_width_Gy = target_bin_width_cGy / 100.0

    if abs(native_bin_width_cGy - target_bin_width_cGy) < 0.01:
        # Already at the target resolution
        dose_edges_cGy = native_bins_Gy * 100.0
        diff_volumes_cc = native_counts_cc
    else:
        # Rebin: group adjacent bins
        rebin_factor = round(target_bin_width_Gy / native_bin_width_Gy)
        if rebin_factor < 1:
            rebin_factor = 1

        n_native = len(native_counts_cc)
        n_rebinned = n_native // rebin_factor
        trimmed = n_rebinned * rebin_factor

        rebinned_counts = native_counts_cc[:trimmed].reshape(
            n_rebinned, rebin_factor
        ).sum(axis=1)

        # Handle remainder bins
        if trimmed < n_native:
            remainder = native_counts_cc[trimmed:].sum()
            rebinned_counts = np.append(rebinned_counts, remainder)
            n_rebinned += 1

        # New bin edges
        dose_edges_cGy = np.arange(
            native_bins_Gy[0] * 100.0,
            native_bins_Gy[0] * 100.0 + (n_rebinned + 1) * target_bin_width_cGy,
            target_bin_width_cGy,
        )
        # Ensure exact length match
        dose_edges_cGy = dose_edges_cGy[: n_rebinned + 1]
        diff_volumes_cc = rebinned_counts

    # Trim trailing zero bins for compactness
    last_nonzero = np.nonzero(diff_volumes_cc)[0]
    if len(last_nonzero) > 0:
        trim_idx = last_nonzero[-1] + 1
        diff_volumes_cc = diff_volumes_cc[:trim_idx]
        dose_edges_cGy = dose_edges_cGy[: trim_idx + 1]
    else:
        return None

    return DVHData(
        structure_name=structure_name,
        structure_volume_cc=volume_cc,
        dose_edges_cGy=dose_edges_cGy,
        diff_volumes_cc=diff_volumes_cc,
        bin_width_cGy=target_bin_width_cGy,
        source="voxel",
        metadata={
            "computation_engine": "dicompyler-core 0.5.6",
        },
    )
