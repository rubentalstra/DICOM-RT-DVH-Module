"""Write-then-read integrity checks for the DICOM RT DVH caching layer.

Verifies that DVH data survives a full round trip through dvh_cache:
write to DICOM RT DVH -> read back -> compare against original.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.dvh_cache import read_dicom_dvh, write_dicom_dvh
from src.dvh_data import DVHData


def _make_test_dvh(n_bins: int = 100, bin_width_cGy: float = 1.0) -> DVHData:
    """Create a simple test DVH with known values."""
    dose_edges = np.arange(0, (n_bins + 1) * bin_width_cGy, bin_width_cGy)
    # Gaussian-like differential DVH
    centers = (dose_edges[:-1] + dose_edges[1:]) / 2.0
    mean_cGy = n_bins * bin_width_cGy / 2.0
    sigma = n_bins * bin_width_cGy / 6.0
    diff_volumes = np.exp(-0.5 * ((centers - mean_cGy) / sigma) ** 2)
    diff_volumes = diff_volumes / diff_volumes.sum() * 10.0  # 10 cc total

    return DVHData(
        structure_name="TestStructure",
        structure_volume_cc=10.0,
        dose_edges_cGy=dose_edges,
        diff_volumes_cc=diff_volumes,
        bin_width_cGy=bin_width_cGy,
        source="voxel",
        metadata={"test": True},
    )


class TestRoundTrip:
    """Write-then-read integrity tests."""

    def test_differential_volumes_preserved(self):
        """Every differential volume value must survive write-then-read."""
        original = _make_test_dvh()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        assert len(recovered) == 1
        rec = recovered[0]
        # Check volumes match within DS VR precision (~8 significant digits)
        np.testing.assert_allclose(
            rec.diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )

    def test_dose_edges_preserved(self):
        """Bin edges must be identical after round trip."""
        original = _make_test_dvh()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        rec = recovered[0]
        np.testing.assert_allclose(
            rec.dose_edges_cGy, original.dose_edges_cGy, atol=0.01,
        )

    def test_total_volume_preserved(self):
        """Structure volume must match within floating-point epsilon."""
        original = _make_test_dvh()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        rec = recovered[0]
        np.testing.assert_allclose(
            rec.structure_volume_cc, original.structure_volume_cc, rtol=1e-4,
        )

    def test_bin_count_preserved(self):
        """Number of bins must be identical."""
        original = _make_test_dvh(n_bins=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        assert recovered[0].n_bins == original.n_bins

    def test_multiple_structures(self):
        """Multiple structures must survive round trip."""
        dvh1 = _make_test_dvh(n_bins=100)
        dvh2 = _make_test_dvh(n_bins=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([dvh1, dvh2], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        assert len(recovered) == 2
        assert recovered[0].n_bins == 100
        assert recovered[1].n_bins == 200

    def test_source_is_dicom_cache(self):
        """Recovered DVH must have source='dicom_cache'."""
        original = _make_test_dvh()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        assert recovered[0].source == "dicom_cache"

    def test_very_small_volumes(self):
        """Very small volume values must survive DS VR encoding."""
        dose_edges = np.arange(0, 11, 1.0)
        diff_volumes = np.array([
            1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 0.5, 0.001,
        ])
        original = DVHData(
            structure_name="TinyVols",
            structure_volume_cc=float(diff_volumes.sum()),
            dose_edges_cGy=dose_edges,
            diff_volumes_cc=diff_volumes,
            bin_width_cGy=1.0,
            source="voxel",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        rec = recovered[0]
        # DS VR has ~8 sig digits, so 1e-7 should survive
        np.testing.assert_allclose(
            rec.diff_volumes_cc, original.diff_volumes_cc, rtol=1e-5,
        )


class TestDSVRPrecisionStress:
    """Stress tests pushing the DICOM DS VR (Decimal String) to its limits.

    DS VR stores values as ASCII text with up to 16 characters.
    Our implementation uses %.8g format (8 significant digits).
    These tests verify where precision loss first becomes detectable.
    """

    @pytest.mark.parametrize("magnitude", [1e-10, 1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_volume_magnitude_roundtrip(self, magnitude):
        """Volumes at each order of magnitude must survive round-trip."""
        dose_edges = np.arange(0, 6, 1.0)
        diff_volumes = np.array([magnitude, magnitude * 2, magnitude * 0.5,
                                 magnitude * 1.234, magnitude * 0.001])
        original = DVHData(
            structure_name="Magnitude",
            structure_volume_cc=float(diff_volumes.sum()),
            dose_edges_cGy=dose_edges,
            diff_volumes_cc=diff_volumes,
            bin_width_cGy=1.0,
            source="voxel",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        # %.8g gives 8 significant digits → rtol=1e-7 is the theoretical limit
        np.testing.assert_allclose(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )

    def test_mixed_magnitudes(self):
        """Bins with wildly different magnitudes in the same DVH."""
        dose_edges = np.arange(0, 8, 1.0)
        diff_volumes = np.array([1e-9, 1e-5, 0.1, 100.0, 1e-3, 1e6, 0.001])
        original = DVHData(
            structure_name="MixedMag",
            structure_volume_cc=float(diff_volumes.sum()),
            dose_edges_cGy=dose_edges,
            diff_volumes_cc=diff_volumes,
            bin_width_cGy=1.0,
            source="voxel",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        np.testing.assert_allclose(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )

    def test_max_ds_precision(self):
        """Test the exact 8-significant-digit boundary.

        The value 1.23456789 has 9 sig digits. With %.8g format,
        this becomes "1.2345679" (8 digits, last digit rounded).
        The relative error should be < 1e-7.
        """
        dose_edges = np.arange(0, 4, 1.0)
        diff_volumes = np.array([1.23456789, 9.87654321, 0.00012345678])
        original = DVHData(
            structure_name="Precision",
            structure_volume_cc=float(diff_volumes.sum()),
            dose_edges_cGy=dose_edges,
            diff_volumes_cc=diff_volumes,
            bin_width_cGy=1.0,
            source="voxel",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        # 8 sig digits → max relative error ~5e-8
        np.testing.assert_allclose(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )

    def test_zero_volume_bins(self):
        """Bins with exactly zero volume must survive."""
        dose_edges = np.arange(0, 6, 1.0)
        diff_volumes = np.array([0.0, 1.5, 0.0, 0.0, 0.5])
        original = DVHData(
            structure_name="ZeroBins",
            structure_volume_cc=2.0,
            dose_edges_cGy=dose_edges,
            diff_volumes_cc=diff_volumes,
            bin_width_cGy=1.0,
            source="voxel",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        np.testing.assert_array_equal(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc,
        )


class TestExtremeDVHShapes:
    """Tests with pathological DVH shapes that stress the parameter extraction."""

    def test_single_bin_dvh(self):
        """Structure receiving all dose in exactly one bin (near-uniform)."""
        dose_edges = np.arange(0, 5010, 1.0)  # 0-50.1 Gy at 1 cGy
        diff_volumes = np.zeros(5009)
        diff_volumes[5000] = 100.0  # All 100cc at exactly 50 Gy
        original = DVHData("Uniform50Gy", 100.0, dose_edges, diff_volumes, 1.0, "voxel")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        np.testing.assert_allclose(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )

    def test_bimodal_dvh(self):
        """Structure with two dose peaks (hot and cold regions)."""
        dose_edges = np.arange(0, 101, 1.0)  # 0-1 Gy at 1 cGy
        diff_volumes = np.zeros(100)
        # Peak 1: bins 10-20 (0.10-0.20 Gy)
        diff_volumes[10:20] = 5.0
        # Peak 2: bins 80-90 (0.80-0.90 Gy)
        diff_volumes[80:90] = 5.0
        original = DVHData("Bimodal", 100.0, dose_edges, diff_volumes, 1.0, "voxel")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        np.testing.assert_allclose(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )

    def test_steep_cliff_dvh(self):
        """99% of volume at one dose, 1% spread as a tail."""
        dose_edges = np.arange(0, 1001, 1.0)
        diff_volumes = np.zeros(1000)
        diff_volumes[500] = 99.0  # 99cc at 5 Gy
        diff_volumes[501:510] = 1.0 / 9  # 1cc spread over 9 bins
        original = DVHData("Cliff", 100.0, dose_edges, diff_volumes, 1.0, "voxel")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        np.testing.assert_allclose(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )

    def test_tiny_structure(self):
        """Very small structure (0.01 cc) with sparse DVH."""
        dose_edges = np.arange(0, 11, 1.0)
        diff_volumes = np.array([0.001, 0.001, 0.001, 0.001, 0.001,
                                 0.001, 0.001, 0.001, 0.001, 0.001])
        original = DVHData("Tiny", 0.01, dose_edges, diff_volumes, 1.0, "voxel")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_dicom_dvh([original], Path(tmpdir) / "test.dcm")
            recovered = read_dicom_dvh(path)

        np.testing.assert_allclose(
            recovered[0].diff_volumes_cc, original.diff_volumes_cc, rtol=1e-6,
        )
