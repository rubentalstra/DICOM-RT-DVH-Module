"""Write and read DICOM RT DVH files (the caching layer under test).

This module implements the core functionality being validated: writing
voxel-computed DVH data into DICOM RT DVH objects and reading them back.
The fidelity of the write-then-read round trip at 1 cGy resolution is
the primary research question.

DICOM RT DVH structure (PS3.3 C.4):
    DVH Sequence (3004,0050) contains one item per structure, each with:
    - DVH Type: DIFFERENTIAL or CUMULATIVE
    - DVH Data: interleaved [bin_width, volume, bin_width, volume, ...]
    - DVH Dose Scaling, Volume Units, Number of Bins
    - Referenced ROI Sequence linking back to RT Structure Set
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from .dvh_data import DVHData

# Import config constants
from config import (
    IMPLEMENTATION_CLASS_UID,
    IMPLICIT_VR_LE,
    RT_DVH_STORAGE_SOP_CLASS,
    SOFTWARE_VERSION,
)


def write_dicom_dvh(
    dvh_list: list[DVHData],
    output_path: str | Path,
    referenced_rt_struct_uid: str = "",
    referenced_rt_plan_uid: str = "",
    patient_name: str = "DVH_VALIDATION",
    patient_id: str = "DVH_VAL_001",
) -> Path:
    """Write DVH data to a DICOM RT DVH Storage SOP Instance.

    Parameters
    ----------
    dvh_list : list of DVHData
        DVH data for one or more structures.
    output_path : str or Path
        File path for the output DICOM file.
    referenced_rt_struct_uid : str
        SOP Instance UID of the referenced RT Structure Set.
    referenced_rt_plan_uid : str
        SOP Instance UID of the referenced RT Plan.
    patient_name : str
        Patient name for DICOM header.
    patient_id : str
        Patient ID for DICOM header.

    Returns
    -------
    Path
        Path to the written DICOM file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sop_instance_uid = generate_uid()

    # Create the file dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = RT_DVH_STORAGE_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = IMPLICIT_VR_LE
    file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID

    ds = FileDataset(
        str(output_path),
        {},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    # Patient module
    ds.PatientName = patient_name
    ds.PatientID = patient_id

    # General study module
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")

    # General series module
    ds.Modality = "RTDOSE"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = f"DVH Cache ({SOFTWARE_VERSION})"

    # SOP common module
    ds.SOPClassUID = RT_DVH_STORAGE_SOP_CLASS
    ds.SOPInstanceUID = sop_instance_uid
    ds.InstanceCreationDate = ds.StudyDate
    ds.InstanceCreationTime = ds.StudyTime
    ds.SoftwareVersions = SOFTWARE_VERSION

    # Frame of Reference module
    ds.FrameOfReferenceUID = generate_uid()

    # Referenced Structure Set Sequence
    if referenced_rt_struct_uid:
        ref_struct_item = Dataset()
        ref_struct_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
        ref_struct_item.ReferencedSOPInstanceUID = referenced_rt_struct_uid
        ds.ReferencedStructureSetSequence = Sequence([ref_struct_item])

    # Referenced RT Plan Sequence
    if referenced_rt_plan_uid:
        ref_plan_item = Dataset()
        ref_plan_item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.5"
        ref_plan_item.ReferencedSOPInstanceUID = referenced_rt_plan_uid
        ds.ReferencedRTPlanSequence = Sequence([ref_plan_item])

    # DVH Sequence — one item per structure
    dvh_sequence = []
    for roi_idx, dvh_data in enumerate(dvh_list, start=1):
        dvh_item = _create_dvh_sequence_item(dvh_data, roi_idx)
        dvh_sequence.append(dvh_item)

    ds.DVHSequence = Sequence(dvh_sequence)

    # Write
    ds.save_as(str(output_path), write_like_original=False)
    return output_path


def _create_dvh_sequence_item(dvh_data: DVHData, roi_number: int) -> Dataset:
    """Create a single DVH Sequence item for one structure.

    The DVH Data tag (3004,0058) uses an interleaved format:
    [bin_width_1, volume_1, bin_width_2, volume_2, ...]

    For a uniform-width differential DVH at 1 cGy (= 0.01 Gy),
    every odd-indexed value is 0.01 and every even-indexed value
    is the differential volume in cc.
    """
    item = Dataset()

    # DVH Type
    item.DVHType = "DIFFERENTIAL"
    item.DoseType = "PHYSICAL"
    item.DoseUnits = "GY"
    item.DVHVolumeUnits = "CM3"

    # DVH Dose Scaling — the bin width in dose units (Gy)
    bin_width_Gy = dvh_data.bin_width_cGy / 100.0
    item.DVHDoseScaling = f"{bin_width_Gy:.6f}"

    # Number of bins
    n_bins = dvh_data.n_bins
    item.DVHNumberOfBins = n_bins

    # DVH Data: interleaved [bin_width, volume, ...] pairs
    # Each pair: (dose_bin_width_Gy, differential_volume_cc)
    # Use vectorized construction for performance
    n = dvh_data.n_bins
    bin_width_str = f"{bin_width_Gy:.8g}"
    vol_strs = np.char.mod("%.8g", dvh_data.diff_volumes_cc)
    interleaved = np.empty(2 * n, dtype=object)
    interleaved[0::2] = bin_width_str
    interleaved[1::2] = vol_strs
    item.DVHData = interleaved.tolist()

    # DVH dose statistics
    item.DVHMinimumDose = f"{dvh_data.min_dose_Gy:.6f}"
    item.DVHMaximumDose = f"{dvh_data.max_dose_Gy:.6f}"
    item.DVHMeanDose = f"{dvh_data.mean_dose_Gy:.6f}"

    # Referenced ROI Sequence
    ref_roi_item = Dataset()
    ref_roi_item.ReferencedROINumber = roi_number
    item.DVHReferencedROISequence = Sequence([ref_roi_item])

    return item


def read_dicom_dvh(dvh_path: str | Path) -> list[DVHData]:
    """Read a DICOM RT DVH file and reconstruct DVHData objects.

    Parameters
    ----------
    dvh_path : str or Path
        Path to the DICOM RT DVH file.

    Returns
    -------
    list of DVHData
        One DVHData per structure found in the DVH Sequence.

    Raises
    ------
    ValueError
        If the file does not contain a DVH Sequence.
    """
    dvh_path = Path(dvh_path)
    ds = pydicom.dcmread(dvh_path)

    if not hasattr(ds, "DVHSequence"):
        raise ValueError(f"No DVH Sequence found in {dvh_path}")

    results = []
    for dvh_item in ds.DVHSequence:
        dvh_data = _parse_dvh_sequence_item(dvh_item, source="dicom_cache")
        if dvh_data is not None:
            results.append(dvh_data)

    return results


def extract_embedded_dvh(
    rt_dose_path: str | Path,
    rt_struct_path: str | Path | None = None,
) -> list[DVHData]:
    """Extract embedded DVH from an RT Dose file's DVH Sequence.

    Commercial TPS (Eclipse, Pinnacle, etc.) sometimes embed a DVH
    Sequence (3004,0050) in the RT Dose file. This function extracts
    those embedded DVHs as DVHData objects with source="tps_embedded".

    If an RT Structure Set is provided, structure names are resolved
    from the ROI number references.

    Parameters
    ----------
    rt_dose_path : str or Path
        Path to the DICOM RT Dose file.
    rt_struct_path : str or Path or None
        Optional path to RT Structure Set for resolving structure names.

    Returns
    -------
    list of DVHData
        One DVHData per structure found. Empty list if no DVH Sequence.
    """
    rt_dose_path = Path(rt_dose_path)
    ds = pydicom.dcmread(rt_dose_path)

    if not hasattr(ds, "DVHSequence"):
        return []

    # Build ROI number → name lookup from RT Struct if available
    roi_names: dict[int, str] = {}
    if rt_struct_path is not None:
        ds_struct = pydicom.dcmread(rt_struct_path)
        for roi in ds_struct.StructureSetROISequence:
            roi_names[int(roi.ROINumber)] = str(roi.ROIName)

    results = []
    for dvh_item in ds.DVHSequence:
        dvh_data = _parse_dvh_sequence_item(
            dvh_item,
            source="tps_embedded",
            roi_names=roi_names,
        )
        if dvh_data is not None:
            results.append(dvh_data)

    return results


def _parse_dvh_sequence_item(
    dvh_item: Dataset,
    source: str = "dicom_cache",
    roi_names: dict[int, str] | None = None,
) -> DVHData | None:
    """Parse a single DVH Sequence item into a DVHData object.

    Handles both DIFFERENTIAL and CUMULATIVE DVH types. Cumulative
    DVH is converted to differential form for consistent storage.
    """
    dvh_type = str(getattr(dvh_item, "DVHType", "DIFFERENTIAL")).upper()
    dose_units = str(getattr(dvh_item, "DoseUnits", "GY")).upper()
    volume_units = str(getattr(dvh_item, "DVHVolumeUnits", "CM3")).upper()

    if dvh_type not in ("DIFFERENTIAL", "CUMULATIVE"):
        return None

    # Parse DVH Data: interleaved [bin_width, volume, ...]
    raw_data = dvh_item.DVHData
    values = np.array([float(v) for v in raw_data], dtype=np.float64)

    if len(values) % 2 != 0 or len(values) == 0:
        return None

    n_bins = len(values) // 2
    bin_widths_Gy = values[0::2]
    volumes = values[1::2]

    # Build dose edges from cumulative bin widths
    dose_edges_Gy = np.zeros(n_bins + 1)
    dose_edges_Gy[1:] = np.cumsum(bin_widths_Gy)
    dose_edges_cGy = dose_edges_Gy * 100.0

    # Determine bin width (use median for robustness)
    bin_width_Gy = float(np.median(bin_widths_Gy))
    bin_width_cGy = bin_width_Gy * 100.0

    if dvh_type == "CUMULATIVE":
        # Convert cumulative to differential
        # cumulative[i] = volume receiving >= dose_edge[i]
        # diff[i] = cumulative[i] - cumulative[i+1]
        cum_volumes = volumes
        diff_volumes_cc = np.maximum(-np.diff(cum_volumes, append=0.0), 0.0)
        total_volume_cc = float(cum_volumes[0]) if len(cum_volumes) > 0 else 0.0
    else:
        diff_volumes_cc = volumes
        total_volume_cc = float(diff_volumes_cc.sum())

    # Resolve structure name
    structure_name = "unknown"
    if hasattr(dvh_item, "DVHReferencedROISequence"):
        roi_num = int(dvh_item.DVHReferencedROISequence[0].ReferencedROINumber)
        if roi_names and roi_num in roi_names:
            structure_name = roi_names[roi_num]
        else:
            structure_name = f"ROI_{roi_num}"

    return DVHData(
        structure_name=structure_name,
        structure_volume_cc=total_volume_cc,
        dose_edges_cGy=dose_edges_cGy,
        diff_volumes_cc=diff_volumes_cc,
        bin_width_cGy=bin_width_cGy,
        source=source,
        metadata={
            "dvh_type": dvh_type,
            "dose_units": dose_units,
            "volume_units": volume_units,
            "dvh_min_dose_Gy": float(getattr(dvh_item, "DVHMinimumDose", 0)),
            "dvh_max_dose_Gy": float(getattr(dvh_item, "DVHMaximumDose", 0)),
            "dvh_mean_dose_Gy": float(getattr(dvh_item, "DVHMeanDose", 0)),
        },
    )
