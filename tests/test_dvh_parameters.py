"""Unit tests for DVH parameter derivation.

Tests Vx, Dx, Dxcc, mean dose, gEUD, and NTCP/TCP extraction from
known DVH curves with analytically verifiable parameter values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.dvh_data import DVHData
from src.dvh_parameters import (
    compute_dx,
    compute_dxcc,
    compute_geud,
    compute_mean_dose,
    compute_ntcp_lkb,
    compute_vx,
)


def _uniform_dvh(dose_Gy: float = 50.0, volume_cc: float = 100.0) -> DVHData:
    """Create a DVH for a structure receiving uniform dose.

    All volume is in a single bin at the specified dose.
    """
    dose_cGy = dose_Gy * 100.0
    n_bins = int(dose_cGy) + 10
    dose_edges = np.arange(0, n_bins + 1, 1.0)
    diff_volumes = np.zeros(n_bins)
    bin_idx = int(dose_cGy)
    if bin_idx < n_bins:
        diff_volumes[bin_idx] = volume_cc

    return DVHData(
        structure_name="Uniform",
        structure_volume_cc=volume_cc,
        dose_edges_cGy=dose_edges,
        diff_volumes_cc=diff_volumes,
        bin_width_cGy=1.0,
        source="test",
    )


def _linear_dvh(d_min_Gy: float = 10.0, d_max_Gy: float = 60.0, volume_cc: float = 100.0) -> DVHData:
    """Create a DVH with flat differential (linear cumulative).

    This simulates a cylinder with linear gradient: constant volume
    per dose bin across the dose range.
    """
    d_min_cGy = d_min_Gy * 100.0
    d_max_cGy = d_max_Gy * 100.0
    n_total = int(d_max_cGy) + 10
    dose_edges = np.arange(0, n_total + 1, 1.0)
    diff_volumes = np.zeros(n_total)

    lo = int(d_min_cGy)
    hi = int(d_max_cGy)
    n_filled = hi - lo
    if n_filled > 0:
        vol_per_bin = volume_cc / n_filled
        diff_volumes[lo:hi] = vol_per_bin

    return DVHData(
        structure_name="Linear",
        structure_volume_cc=volume_cc,
        dose_edges_cGy=dose_edges,
        diff_volumes_cc=diff_volumes,
        bin_width_cGy=1.0,
        source="test",
    )


class TestVx:
    """Tests for volume-at-dose (Vx) computation."""

    def test_uniform_below_dose(self):
        """V(x < uniform dose) should return full volume."""
        dvh = _uniform_dvh(dose_Gy=50.0, volume_cc=100.0)
        v = compute_vx(dvh, 49.0)
        assert v == pytest.approx(100.0, abs=1.0)

    def test_uniform_above_dose(self):
        """V(x > uniform dose) should return 0."""
        dvh = _uniform_dvh(dose_Gy=50.0, volume_cc=100.0)
        v = compute_vx(dvh, 51.0)
        assert v == pytest.approx(0.0, abs=0.1)

    def test_linear_midpoint(self):
        """V(mid-dose) of a linear DVH should be ~50% volume."""
        dvh = _linear_dvh(10.0, 60.0, 100.0)
        v = compute_vx(dvh, 35.0)
        assert v == pytest.approx(50.0, abs=2.0)


class TestDx:
    """Tests for dose-at-volume (Dx) computation."""

    def test_uniform_d50(self):
        """D50 of a uniform dose should equal that dose."""
        dvh = _uniform_dvh(dose_Gy=50.0, volume_cc=100.0)
        d = compute_dx(dvh, 50.0)
        assert d == pytest.approx(50.0, abs=0.1)

    def test_linear_d50(self):
        """D50 of a linear DVH should be the midpoint dose."""
        dvh = _linear_dvh(10.0, 60.0, 100.0)
        d = compute_dx(dvh, 50.0)
        assert d == pytest.approx(35.0, abs=0.5)

    def test_linear_d95(self):
        """D95 of a linear DVH: 95% of volume receives >= low end dose."""
        dvh = _linear_dvh(10.0, 60.0, 100.0)
        d = compute_dx(dvh, 95.0)
        # 95% covered => dose at 5% from bottom = 10 + 0.05*50 = 12.5 Gy
        assert d == pytest.approx(12.5, abs=1.0)


class TestDxcc:
    """Tests for dose-at-absolute-volume (Dxcc) computation."""

    def test_linear_d2cc(self):
        """D2cc of a 100cc linear DVH."""
        dvh = _linear_dvh(10.0, 60.0, 100.0)
        d = compute_dxcc(dvh, 2.0)
        # 2cc is in the hottest 2% → dose near max
        assert d == pytest.approx(59.0, abs=1.5)


class TestMeanDose:
    """Tests for mean dose computation."""

    def test_uniform_mean(self):
        """Mean of uniform dose equals that dose."""
        dvh = _uniform_dvh(dose_Gy=50.0)
        assert compute_mean_dose(dvh) == pytest.approx(50.0, abs=0.1)

    def test_linear_mean(self):
        """Mean of linear DVH is the midpoint."""
        dvh = _linear_dvh(10.0, 60.0, 100.0)
        assert compute_mean_dose(dvh) == pytest.approx(35.0, abs=0.5)


class TestGEUD:
    """Tests for generalised equivalent uniform dose."""

    def test_geud_a1_equals_mean(self):
        """gEUD with a=1 should equal mean dose."""
        dvh = _linear_dvh(10.0, 60.0, 100.0)
        geud = compute_geud(dvh, 1.0)
        mean = compute_mean_dose(dvh)
        assert geud == pytest.approx(mean, rel=0.01)

    def test_geud_large_a_approaches_max(self):
        """gEUD with large a should approach max dose."""
        dvh = _linear_dvh(10.0, 60.0, 100.0)
        geud = compute_geud(dvh, 50.0)
        assert geud > 55.0  # Should be close to max (60 Gy)


class TestNTCP:
    """Tests for NTCP LKB model."""

    def test_ntcp_zero_dose(self):
        """NTCP at very low dose should be well below 0.5."""
        dvh = _uniform_dvh(dose_Gy=1.0)
        ntcp = compute_ntcp_lkb(dvh, td50=50.0, m=0.5, n=0.5)
        assert ntcp < 0.1

    def test_ntcp_high_dose(self):
        """NTCP at very high dose should be well above 0.5."""
        dvh = _uniform_dvh(dose_Gy=100.0)
        ntcp = compute_ntcp_lkb(dvh, td50=50.0, m=0.5, n=0.5)
        assert ntcp > 0.9

    def test_ntcp_at_td50(self):
        """NTCP at TD50 should be approximately 0.5 for uniform dose."""
        dvh = _uniform_dvh(dose_Gy=50.0)
        ntcp = compute_ntcp_lkb(dvh, td50=50.0, m=0.5, n=1.0)
        assert ntcp == pytest.approx(0.5, abs=0.05)
