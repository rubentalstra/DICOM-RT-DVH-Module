"""Unit tests for analytical DVH formulas.

Verifies that the closed-form DVH solutions for sphere, cylinder, and cone
produce mathematically correct results by testing against known properties:

- Total volume must match geometric formula
- Dmax/Dmin must correspond to dose at shape extremes
- Symmetry properties (e.g. D50 for symmetric sphere = centre dose)
- Cylinder differential DVH must be approximately flat
- Cone cumulative DVH must follow cubic relationship
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.ground_truth import (
    analytical_dvh,
    cone_cumulative_volume_cc,
    cylinder_cumulative_volume_cc,
    sphere_cumulative_volume_cc,
)


# ── Sphere tests ────────────────────────────────────────────────────


class TestSphereAnalytical:
    """Tests for sphere + linear gradient DVH."""

    R = 12.0  # mm (matching Nelms dataset)
    z_c = 6.0  # mm (matching Nelms dataset)
    slope = 0.68  # Gy/mm (approx for Nelms SupInf grid)
    offset = 17.68  # Gy (so that dose at z=-26 = 0)

    @property
    def total_volume_cc(self) -> float:
        return (4.0 / 3.0) * math.pi * self.R ** 3 / 1000.0

    def test_total_volume(self):
        """V(>=Dmin) must equal 4/3 * pi * R^3."""
        # Dose at bottom of sphere
        d_min = self.slope * (self.z_c - self.R) + self.offset
        vol = sphere_cumulative_volume_cc(
            np.array([d_min - 0.001]),
            self.R, self.z_c, self.slope, self.offset,
        )
        np.testing.assert_allclose(vol[0], self.total_volume_cc, rtol=1e-6)

    def test_zero_volume_above_dmax(self):
        """V(>=d) must be 0 for d > Dmax."""
        d_max = self.slope * (self.z_c + self.R) + self.offset
        vol = sphere_cumulative_volume_cc(
            np.array([d_max + 1.0]),
            self.R, self.z_c, self.slope, self.offset,
        )
        assert vol[0] == pytest.approx(0.0, abs=1e-10)

    def test_d50_equals_centre_dose(self):
        """For any gradient, D50 is the dose at the sphere centre.

        By symmetry of the sphere, exactly half the volume is above
        the equatorial plane (z = z_centre), so V(>= D_centre) = V_total / 2.
        """
        d_centre = self.slope * self.z_c + self.offset
        vol = sphere_cumulative_volume_cc(
            np.array([d_centre]),
            self.R, self.z_c, self.slope, self.offset,
        )
        np.testing.assert_allclose(vol[0], self.total_volume_cc / 2.0, rtol=1e-6)

    def test_monotonically_decreasing(self):
        """Cumulative DVH must be monotonically non-increasing."""
        d_min = self.slope * (self.z_c - self.R) + self.offset
        d_max = self.slope * (self.z_c + self.R) + self.offset
        doses = np.linspace(d_min, d_max, 100)
        vols = sphere_cumulative_volume_cc(
            doses, self.R, self.z_c, self.slope, self.offset,
        )
        assert np.all(np.diff(vols) <= 1e-10)

    def test_negative_slope(self):
        """Formula must work for negative gradient slope."""
        neg_slope = -self.slope
        neg_offset = -neg_slope * (-26)  # keep dose = 0 at z = -26
        d_centre = neg_slope * self.z_c + neg_offset
        vol = sphere_cumulative_volume_cc(
            np.array([d_centre]),
            self.R, self.z_c, neg_slope, neg_offset,
        )
        np.testing.assert_allclose(vol[0], self.total_volume_cc / 2.0, rtol=1e-6)


# ── Cylinder tests ──────────────────────────────────────────────────


class TestCylinderAnalytical:
    """Tests for cylinder + linear gradient DVH."""

    R = 12.0  # mm
    z_bot = -6.0  # mm
    z_top = 18.0  # mm
    slope = 0.68
    offset = 17.68

    @property
    def total_volume_cc(self) -> float:
        H = self.z_top - self.z_bot
        return math.pi * self.R ** 2 * H / 1000.0

    def test_total_volume(self):
        """V(>=Dmin) must equal pi * R^2 * H."""
        d_min = self.slope * self.z_bot + self.offset
        vol = cylinder_cumulative_volume_cc(
            np.array([d_min - 0.001]),
            self.R, self.z_bot, self.z_top, self.slope, self.offset,
        )
        np.testing.assert_allclose(vol[0], self.total_volume_cc, rtol=1e-6)

    def test_linear_cumulative(self):
        """Cumulative DVH must be exactly linear for cylinder + linear gradient."""
        d_min = self.slope * self.z_bot + self.offset
        d_max = self.slope * self.z_top + self.offset
        doses = np.linspace(d_min, d_max, 50)
        vols = cylinder_cumulative_volume_cc(
            doses, self.R, self.z_bot, self.z_top, self.slope, self.offset,
        )
        # Fit a line and check residuals
        coeffs = np.polyfit(doses, vols, 1)
        fitted = np.polyval(coeffs, doses)
        np.testing.assert_allclose(vols, fitted, atol=1e-10)

    def test_flat_differential(self):
        """Differential DVH must be approximately constant."""
        dvh = analytical_dvh(
            "cylinder",
            {"radius_mm": self.R, "z_bottom_mm": self.z_bot, "z_top_mm": self.z_top},
            self.slope, self.offset,
            bin_width_cGy=1.0,
        )
        # Only check bins within the structure dose range (non-zero bins)
        nonzero = dvh.diff_volumes_cc[dvh.diff_volumes_cc > 0]
        if len(nonzero) > 2:
            # Interior bins should be very similar (edges may differ slightly)
            interior = nonzero[1:-1]
            cv = np.std(interior) / np.mean(interior) if np.mean(interior) > 0 else 0
            assert cv < 0.05, f"CV of interior differential bins = {cv:.4f}, expected < 0.05"


# ── Cone tests ──────────────────────────────────────────────────────


class TestConeAnalytical:
    """Tests for cone + linear gradient DVH."""

    R = 12.0  # mm (base radius)
    z_base = 18.0  # mm (wide end)
    z_apex = -6.0  # mm (point)
    slope = 0.68
    offset = 17.68

    @property
    def total_volume_cc(self) -> float:
        H = abs(self.z_base - self.z_apex)
        return (math.pi / 3.0) * self.R ** 2 * H / 1000.0

    def test_total_volume(self):
        """V(>=Dmin) must equal 1/3 * pi * R^2 * H."""
        d_min = self.slope * self.z_apex + self.offset
        vol = cone_cumulative_volume_cc(
            np.array([d_min - 0.001]),
            self.R, self.z_base, self.z_apex, self.slope, self.offset,
        )
        np.testing.assert_allclose(vol[0], self.total_volume_cc, rtol=1e-6)

    def test_near_zero_at_dmax(self):
        """Volume at the maximum dose should be near zero.

        With positive slope, Dmax is at the base (wide end). But only a tiny
        fraction of the cone is at exactly that dose, so V(>=Dmax) ~ 0.
        """
        d_max = self.slope * self.z_base + self.offset
        vol = cone_cumulative_volume_cc(
            np.array([d_max + 0.001]),
            self.R, self.z_base, self.z_apex, self.slope, self.offset,
        )
        assert vol[0] == pytest.approx(0.0, abs=1e-6)

    def test_monotonically_decreasing(self):
        """Cumulative DVH must be monotonically non-increasing."""
        d_min = min(self.slope * self.z_apex + self.offset,
                    self.slope * self.z_base + self.offset)
        d_max = max(self.slope * self.z_apex + self.offset,
                    self.slope * self.z_base + self.offset)
        doses = np.linspace(d_min, d_max, 100)
        vols = cone_cumulative_volume_cc(
            doses, self.R, self.z_base, self.z_apex, self.slope, self.offset,
        )
        assert np.all(np.diff(vols) <= 1e-10)

    def test_cubic_relationship(self):
        """Cumulative DVH must follow a cubic relationship with dose."""
        d_min = self.slope * self.z_apex + self.offset
        d_max = self.slope * self.z_base + self.offset
        doses = np.linspace(d_min + 0.1, d_max - 0.1, 50)
        vols = cone_cumulative_volume_cc(
            doses, self.R, self.z_base, self.z_apex, self.slope, self.offset,
        )
        # Fit cubic polynomial and check quality
        coeffs = np.polyfit(doses, vols, 3)
        fitted = np.polyval(coeffs, doses)
        np.testing.assert_allclose(vols, fitted, atol=1e-6)


# ── DVHData interface tests ─────────────────────────────────────────


class TestAnalyticalDVH:
    """Tests for the analytical_dvh() common interface."""

    def test_sphere_dvhdata_volume(self):
        """DVHData total volume must match analytical formula."""
        dvh = analytical_dvh(
            "sphere",
            {"radius_mm": 12.0, "z_center_mm": 6.0},
            gradient_slope_Gy_per_mm=0.68,
            gradient_offset_Gy=17.68,
        )
        expected = (4.0 / 3.0) * math.pi * 12.0 ** 3 / 1000.0
        assert dvh.structure_volume_cc == pytest.approx(expected, rel=1e-6)
        assert dvh.source == "analytical"

    def test_volume_conservation(self):
        """Sum of differential volumes should approximate total volume."""
        dvh = analytical_dvh(
            "sphere",
            {"radius_mm": 12.0, "z_center_mm": 6.0},
            gradient_slope_Gy_per_mm=0.68,
            gradient_offset_Gy=17.68,
        )
        np.testing.assert_allclose(
            dvh.diff_volumes_cc.sum(),
            dvh.structure_volume_cc,
            rtol=0.001,
        )

    def test_cylinder_volume_conservation(self):
        """Cylinder differential sum must equal total volume."""
        dvh = analytical_dvh(
            "cylinder",
            {"radius_mm": 12.0, "z_bottom_mm": -6.0, "z_top_mm": 18.0},
            gradient_slope_Gy_per_mm=0.68,
            gradient_offset_Gy=17.68,
        )
        np.testing.assert_allclose(
            dvh.diff_volumes_cc.sum(),
            dvh.structure_volume_cc,
            rtol=0.001,
        )

    def test_cone_volume_conservation(self):
        """Cone differential sum must equal total volume."""
        dvh = analytical_dvh(
            "cone",
            {"base_radius_mm": 12.0, "z_base_mm": 18.0, "z_apex_mm": -6.0},
            gradient_slope_Gy_per_mm=0.68,
            gradient_offset_Gy=17.68,
        )
        np.testing.assert_allclose(
            dvh.diff_volumes_cc.sum(),
            dvh.structure_volume_cc,
            rtol=0.001,
        )

    def test_invalid_shape_raises(self):
        """Unknown shape name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown shape"):
            analytical_dvh("tetrahedron", {}, 1.0, 0.0)
