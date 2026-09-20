"""Unit tests for voxel-level DVH computation.

Tests the dicompyler-core wrapper to verify output DVHData structure
correctness, volume conservation, and bin width handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dvh_compute import compute_dvh_from_dicom

# Path to Nelms data (may not be present in CI)
NELMS_BASE = Path(__file__).resolve().parent.parent / "data" / "nelms" / "DVH-Analysis-Data-Etc"
HAS_NELMS = NELMS_BASE.exists() and any(NELMS_BASE.rglob("*.dcm"))


@pytest.mark.skipif(not HAS_NELMS, reason="Nelms dataset not available")
class TestDVHComputeNelms:
    """Integration tests using the Nelms phantom dataset."""

    DOSE_PATH = NELMS_BASE / "DOSE GRIDS" / "Linear_SupInf_1mm_Aligned.dcm"
    STRUCT_PATH = NELMS_BASE / "STRUCTURES" / "Spheres" / "Sphere_10_0.dcm"

    def test_returns_dvhdata(self):
        """Output must be a dict of DVHData objects."""
        results = compute_dvh_from_dicom(self.DOSE_PATH, self.STRUCT_PATH)
        assert len(results) > 0
        for name, dvh in results.items():
            assert isinstance(name, str)
            assert dvh.n_bins > 0
            assert dvh.structure_volume_cc > 0
            assert dvh.source == "voxel"

    def test_sphere_volume_reasonable(self):
        """Sphere volume must be within 5% of analytical value.

        Analytical: V = 4/3 * pi * R^3 where R = 12mm.
        At 1mm grid resolution, voxelisation error up to ~2-3% is expected.
        """
        import math

        results = compute_dvh_from_dicom(self.DOSE_PATH, self.STRUCT_PATH)
        dvh = results["Sphere_10_0"]
        analytical_vol = (4.0 / 3.0) * math.pi * 12.0**3 / 1000.0
        rel_error = abs(dvh.structure_volume_cc - analytical_vol) / analytical_vol
        assert rel_error < 0.05, f"Volume error {rel_error:.2%} exceeds 5%"

    def test_volume_conservation(self):
        """Sum of differential volumes must equal structure volume."""
        results = compute_dvh_from_dicom(self.DOSE_PATH, self.STRUCT_PATH)
        for dvh in results.values():
            np.testing.assert_allclose(
                dvh.diff_volumes_cc.sum(),
                dvh.structure_volume_cc,
                rtol=0.001,
            )

    def test_bin_width_respected(self):
        """Bin edges must use the requested bin width."""
        results = compute_dvh_from_dicom(
            self.DOSE_PATH, self.STRUCT_PATH, bin_width_cGy=1.0,
        )
        for dvh in results.values():
            # Check bin width is uniform at 1 cGy
            widths = np.diff(dvh.dose_edges_cGy)
            np.testing.assert_allclose(widths, 1.0, atol=0.01)

    def test_structure_name_filter(self):
        """Filtering by structure name must work."""
        results = compute_dvh_from_dicom(
            self.DOSE_PATH, self.STRUCT_PATH,
            structure_names=["Sphere"],
        )
        assert len(results) == 1
        assert "Sphere_10_0" in results

    def test_filter_excludes_poi(self):
        """POI structures should be excluded when filtering for shapes."""
        results = compute_dvh_from_dicom(
            self.DOSE_PATH, self.STRUCT_PATH,
            structure_names=["Sphere"],
        )
        assert "POI_1" not in results

    def test_cumulative_monotonically_decreasing(self):
        """Cumulative DVH must be monotonically non-increasing."""
        results = compute_dvh_from_dicom(self.DOSE_PATH, self.STRUCT_PATH)
        for dvh in results.values():
            cum = dvh.cumulative_volumes_cc
            assert np.all(np.diff(cum) <= 1e-10)
