"""Shared DVH data structure used by all pipeline modules.

DVHData is an immutable dataclass representing a dose-volume histogram
for a single structure. It uses the differential form as canonical
representation (matching both the voxel computation output and the
DICOM RT DVH storage format) and derives cumulative form on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DVHData:
    """Immutable DVH representation shared across all pipeline stages.

    Attributes
    ----------
    structure_name : str
        DICOM structure name (e.g. "PTV", "Bladder").
    structure_volume_cc : float
        Total structure volume in cubic centimetres.
    dose_edges_cGy : np.ndarray
        1-D array of bin edges in cGy, shape (N+1,).
        Follows the numpy histogram convention: bin *i* spans
        [dose_edges_cGy[i], dose_edges_cGy[i+1]).
    diff_volumes_cc : np.ndarray
        1-D array of differential volumes per bin in cc, shape (N,).
        diff_volumes_cc[i] is the volume receiving a dose within bin *i*.
    bin_width_cGy : float
        Uniform bin width in cGy (typically 1 for 1 cGy resolution).
    source : str
        Provenance tag: ``"voxel"``, ``"analytical"``, or ``"dicom_cache"``.
    metadata : dict
        Arbitrary key-value pairs for traceability (DICOM UIDs, grid
        resolution, contour spacing, etc.).

    Notes
    -----
    Differential is the canonical form because:

    - It is the natural output of voxel-level DVH computation (histogram).
    - It is the format stored in DICOM RT DVH objects (DVH Data tag).
    - Cumulative DVH is derived from it without loss of information.

    The reverse conversion (cumulative → differential) loses precision
    due to numerical differencing, so we always store differential.
    """

    structure_name: str
    structure_volume_cc: float
    dose_edges_cGy: np.ndarray
    diff_volumes_cc: np.ndarray
    bin_width_cGy: float
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate consistency of the DVH data on construction."""
        n_bins = len(self.diff_volumes_cc)
        n_edges = len(self.dose_edges_cGy)
        if n_edges != n_bins + 1:
            raise ValueError(
                f"dose_edges_cGy has {n_edges} elements but diff_volumes_cc "
                f"has {n_bins} elements; expected {n_bins + 1} edges."
            )
        if n_bins == 0:
            raise ValueError("DVH must have at least one bin.")

    # ── Derived properties ──────────────────────────────────────────

    @property
    def dose_centers_cGy(self) -> np.ndarray:
        """Dose at the centre of each bin, in cGy."""
        return (self.dose_edges_cGy[:-1] + self.dose_edges_cGy[1:]) / 2.0

    @property
    def dose_centers_Gy(self) -> np.ndarray:
        """Dose at the centre of each bin, in Gy."""
        return self.dose_centers_cGy / 100.0

    @property
    def dose_edges_Gy(self) -> np.ndarray:
        """Bin edges in Gy."""
        return self.dose_edges_cGy / 100.0

    @property
    def n_bins(self) -> int:
        """Number of dose bins."""
        return len(self.diff_volumes_cc)

    @property
    def cumulative_volumes_cc(self) -> np.ndarray:
        """Cumulative DVH: volume (cc) receiving >= each bin centre dose.

        Computed as the reverse cumulative sum of the differential volumes.
        Shape (N,), aligned with ``dose_centers_cGy``.
        """
        return np.cumsum(self.diff_volumes_cc[::-1])[::-1]

    @property
    def cumulative_volumes_pct(self) -> np.ndarray:
        """Cumulative DVH as percentage of total structure volume.

        Returns zeros if structure_volume_cc is zero.
        """
        if self.structure_volume_cc == 0:
            return np.zeros_like(self.diff_volumes_cc)
        return self.cumulative_volumes_cc / self.structure_volume_cc * 100.0

    @property
    def max_dose_cGy(self) -> float:
        """Maximum dose in cGy (upper edge of the last non-zero bin)."""
        nonzero = np.nonzero(self.diff_volumes_cc)[0]
        if len(nonzero) == 0:
            return 0.0
        return float(self.dose_edges_cGy[nonzero[-1] + 1])

    @property
    def min_dose_cGy(self) -> float:
        """Minimum dose in cGy (lower edge of the first non-zero bin)."""
        nonzero = np.nonzero(self.diff_volumes_cc)[0]
        if len(nonzero) == 0:
            return 0.0
        return float(self.dose_edges_cGy[nonzero[0]])

    @property
    def max_dose_Gy(self) -> float:
        """Maximum dose in Gy."""
        return self.max_dose_cGy / 100.0

    @property
    def min_dose_Gy(self) -> float:
        """Minimum dose in Gy."""
        return self.min_dose_cGy / 100.0

    @property
    def mean_dose_cGy(self) -> float:
        """Mean dose in cGy, computed from the differential DVH.

        mean = sum(dose_center * volume) / total_volume
        """
        if self.structure_volume_cc == 0:
            return 0.0
        return float(
            np.sum(self.dose_centers_cGy * self.diff_volumes_cc)
            / self.structure_volume_cc
        )

    @property
    def mean_dose_Gy(self) -> float:
        """Mean dose in Gy."""
        return self.mean_dose_cGy / 100.0
