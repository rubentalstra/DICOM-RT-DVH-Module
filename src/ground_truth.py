"""Analytical DVH computation for Nelms phantom shapes.

Implements closed-form DVH solutions for sphere, cylinder, and cone
under a linear dose gradient. These serve as mathematically provable
ground truth for validating the voxel-level DVH computation pipeline.

The formulas compute cumulative DVH V(>=d) analytically, then convert
to differential form for storage in DVHData.

Geometry conventions (matching the Nelms dataset):
    - All shapes have their axis aligned with the z-axis.
    - A linear dose gradient runs along one axis: D(z) = slope * z + offset.
    - "Cumulative DVH" means V(>=d): the volume receiving dose >= d.
"""

from __future__ import annotations

import math

import numpy as np

from .dvh_data import DVHData


# ──────────────────────────────────────────────────────────────────────
# Sphere with linear gradient along z
# ──────────────────────────────────────────────────────────────────────


def sphere_cumulative_volume_cc(
    dose_Gy: np.ndarray,
    radius_mm: float,
    z_center_mm: float,
    gradient_slope_Gy_per_mm: float,
    gradient_offset_Gy: float,
) -> np.ndarray:
    """Analytical cumulative DVH for a sphere in a linear dose gradient.

    For a sphere of radius R centred at z_c in a gradient D(z) = m*z + b,
    the volume receiving dose >= d is the volume of the spherical cap
    above height z_0 = (d - b) / m.

    The spherical cap volume (volume with z >= z_0) is:

        V_cap(z_0) = (pi / 3) * (R - z_0')^2 * (2R + z_0')

    where z_0' = z_0 - z_c is the offset from sphere centre, and
    this formula is valid for -R <= z_0' <= R.

    If m > 0 (dose increases with z), higher dose means smaller cap.
    If m < 0 (dose decreases with z), higher dose means we need z BELOW
    some threshold, so V(>=d) = total_volume - V_cap(z_0).

    Parameters
    ----------
    dose_Gy : array
        Dose values at which to evaluate the cumulative DVH.
    radius_mm : float
        Sphere radius in mm.
    z_center_mm : float
        z-coordinate of sphere centre in mm.
    gradient_slope_Gy_per_mm : float
        Dose gradient slope (Gy per mm).
    gradient_offset_Gy : float
        Dose at z = 0 mm.

    Returns
    -------
    np.ndarray
        Cumulative volumes in cc (same shape as dose_Gy).
    """
    R = radius_mm
    m = gradient_slope_Gy_per_mm
    b = gradient_offset_Gy
    z_c = z_center_mm

    total_vol_mm3 = (4.0 / 3.0) * math.pi * R ** 3

    # z position where dose = d
    # D(z) = m*z + b => z = (d - b) / m
    z_threshold = (np.asarray(dose_Gy, dtype=np.float64) - b) / m

    # Offset from sphere centre
    z_prime = z_threshold - z_c

    # Clamp to sphere bounds
    z_prime = np.clip(z_prime, -R, R)

    # Volume of spherical cap above z_prime (volume with z >= z_prime)
    cap_vol_mm3 = (math.pi / 3.0) * (R - z_prime) ** 2 * (2 * R + z_prime)

    if m > 0:
        # Dose increases with z, so V(>=d) = cap above z_threshold
        result_mm3 = cap_vol_mm3
    else:
        # Dose decreases with z, so V(>=d) = total - cap above z_threshold
        result_mm3 = total_vol_mm3 - cap_vol_mm3

    # Convert mm^3 to cc (1 cc = 1000 mm^3)
    return result_mm3 / 1000.0


# ──────────────────────────────────────────────────────────────────────
# Cylinder with linear gradient along z (along its axis)
# ──────────────────────────────────────────────────────────────────────


def cylinder_cumulative_volume_cc(
    dose_Gy: np.ndarray,
    radius_mm: float,
    z_bottom_mm: float,
    z_top_mm: float,
    gradient_slope_Gy_per_mm: float,
    gradient_offset_Gy: float,
) -> np.ndarray:
    """Analytical cumulative DVH for a cylinder in a linear dose gradient.

    For a cylinder of radius R spanning [z_bot, z_top] in a gradient
    D(z) = m*z + b, the cross-section is constant pi*R^2 at every z.

    V(>=d) = pi * R^2 * (remaining height above z_threshold)

    This produces a LINEAR cumulative DVH.

    Parameters
    ----------
    dose_Gy : array
        Dose values at which to evaluate the cumulative DVH.
    radius_mm : float
        Cylinder radius in mm.
    z_bottom_mm, z_top_mm : float
        z-coordinates of cylinder bottom and top in mm.
    gradient_slope_Gy_per_mm : float
        Dose gradient slope (Gy per mm).
    gradient_offset_Gy : float
        Dose at z = 0 mm.

    Returns
    -------
    np.ndarray
        Cumulative volumes in cc.
    """
    R = radius_mm
    m = gradient_slope_Gy_per_mm
    b = gradient_offset_Gy
    H = z_top_mm - z_bottom_mm  # height in mm
    cross_section_mm2 = math.pi * R ** 2

    z_threshold = (np.asarray(dose_Gy, dtype=np.float64) - b) / m
    z_threshold = np.clip(z_threshold, z_bottom_mm, z_top_mm)

    if m > 0:
        # Dose increases with z; V(>=d) = volume from z_threshold to z_top
        remaining_mm = z_top_mm - z_threshold
    else:
        # Dose decreases with z; V(>=d) = volume from z_bottom to z_threshold
        remaining_mm = z_threshold - z_bottom_mm

    vol_mm3 = cross_section_mm2 * remaining_mm
    return vol_mm3 / 1000.0


# ──────────────────────────────────────────────────────────────────────
# Cone with linear gradient along z (along its axis)
# ──────────────────────────────────────────────────────────────────────


def cone_cumulative_volume_cc(
    dose_Gy: np.ndarray,
    base_radius_mm: float,
    z_base_mm: float,
    z_apex_mm: float,
    gradient_slope_Gy_per_mm: float,
    gradient_offset_Gy: float,
) -> np.ndarray:
    """Analytical cumulative DVH for a cone in a linear dose gradient.

    For a cone with base radius R at z_base and apex at z_apex, the
    cross-sectional radius at height z is:

        r(z) = R * |z - z_apex| / |z_base - z_apex|

    The volume from z_0 to z_base (the portion containing dose >= d) is:

        V = (pi * R^2 / (3 * H^2)) * integral of (z - z_apex)^2 dz

    where H = |z_base - z_apex|.

    Parameters
    ----------
    dose_Gy : array
        Dose values at which to evaluate the cumulative DVH.
    base_radius_mm : float
        Cone base radius in mm.
    z_base_mm : float
        z-coordinate of the cone base (wide end).
    z_apex_mm : float
        z-coordinate of the cone apex (point).
    gradient_slope_Gy_per_mm : float
        Dose gradient slope (Gy per mm).
    gradient_offset_Gy : float
        Dose at z = 0 mm.

    Returns
    -------
    np.ndarray
        Cumulative volumes in cc.
    """
    R = base_radius_mm
    m = gradient_slope_Gy_per_mm
    b = gradient_offset_Gy
    H = abs(z_base_mm - z_apex_mm)

    total_vol_mm3 = (math.pi / 3.0) * R ** 2 * H

    z_threshold = (np.asarray(dose_Gy, dtype=np.float64) - b) / m

    # Clamp to cone z range
    z_lo = min(z_base_mm, z_apex_mm)
    z_hi = max(z_base_mm, z_apex_mm)
    z_threshold = np.clip(z_threshold, z_lo, z_hi)

    # Fractional distance from apex: 0 at apex, 1 at base
    frac_threshold = np.abs(z_threshold - z_apex_mm) / H

    # Volume from apex to z_threshold (cone frustum from apex)
    # V_cone_fraction = total_vol * frac^3
    vol_apex_to_threshold_mm3 = total_vol_mm3 * frac_threshold ** 3

    # Volume from z_threshold to base
    vol_threshold_to_base_mm3 = total_vol_mm3 - vol_apex_to_threshold_mm3

    # Determine which side has higher dose
    dose_at_base = m * z_base_mm + b
    dose_at_apex = m * z_apex_mm + b

    if dose_at_base > dose_at_apex:
        # Higher dose at the base (wide end)
        # V(>=d) = volume from z_threshold towards the base
        result_mm3 = vol_threshold_to_base_mm3
    else:
        # Higher dose at the apex (narrow end)
        # V(>=d) = volume from z_threshold towards the apex
        result_mm3 = vol_apex_to_threshold_mm3

    return result_mm3 / 1000.0


# ──────────────────────────────────────────────────────────────────────
# Common interface: shape name → DVHData
# ──────────────────────────────────────────────────────────────────────


def analytical_dvh(
    shape: str,
    shape_params: dict,
    gradient_slope_Gy_per_mm: float,
    gradient_offset_Gy: float,
    bin_width_cGy: float = 1.0,
) -> DVHData:
    """Compute analytical DVH and return as a DVHData object.

    Parameters
    ----------
    shape : str
        One of ``"sphere"``, ``"cylinder"``, ``"cone"``.
    shape_params : dict
        Shape-specific parameters:

        - sphere: ``radius_mm``, ``z_center_mm``
        - cylinder: ``radius_mm``, ``z_bottom_mm``, ``z_top_mm``
        - cone: ``base_radius_mm``, ``z_base_mm``, ``z_apex_mm``
    gradient_slope_Gy_per_mm : float
        Linear dose gradient slope.
    gradient_offset_Gy : float
        Dose at z = 0.
    bin_width_cGy : float
        DVH bin width in cGy (default 1).

    Returns
    -------
    DVHData
        Analytical DVH in differential form.
    """
    # Determine dose range over the structure
    if shape == "sphere":
        R = shape_params["radius_mm"]
        z_c = shape_params["z_center_mm"]
        z_lo, z_hi = z_c - R, z_c + R
        total_vol_cc = (4.0 / 3.0) * math.pi * R ** 3 / 1000.0
        name = f"sphere_R{R:.0f}mm"
    elif shape == "cylinder":
        R = shape_params["radius_mm"]
        z_lo = shape_params["z_bottom_mm"]
        z_hi = shape_params["z_top_mm"]
        H = z_hi - z_lo
        total_vol_cc = math.pi * R ** 2 * H / 1000.0
        name = f"cylinder_R{R:.0f}mm_H{H:.0f}mm"
    elif shape == "cone":
        R = shape_params["base_radius_mm"]
        z_base = shape_params["z_base_mm"]
        z_apex = shape_params["z_apex_mm"]
        z_lo = min(z_base, z_apex)
        z_hi = max(z_base, z_apex)
        H = z_hi - z_lo
        total_vol_cc = (math.pi / 3.0) * R ** 2 * H / 1000.0
        name = f"cone_R{R:.0f}mm_H{H:.0f}mm"
    else:
        raise ValueError(f"Unknown shape: {shape!r}")

    m = gradient_slope_Gy_per_mm
    b = gradient_offset_Gy

    # Dose range over the structure
    d_at_lo = m * z_lo + b
    d_at_hi = m * z_hi + b
    d_min_Gy = min(d_at_lo, d_at_hi)
    d_max_Gy = max(d_at_lo, d_at_hi)

    # Build bin edges in cGy
    bin_width_Gy = bin_width_cGy / 100.0
    d_min_cGy = math.floor(d_min_Gy / bin_width_Gy) * bin_width_cGy
    d_max_cGy = math.ceil(d_max_Gy / bin_width_Gy) * bin_width_cGy + bin_width_cGy
    d_min_cGy = max(0.0, d_min_cGy)

    dose_edges_cGy = np.arange(d_min_cGy, d_max_cGy + bin_width_cGy, bin_width_cGy)
    dose_centers_Gy = (dose_edges_cGy[:-1] + dose_edges_cGy[1:]) / 2.0 / 100.0

    # Compute cumulative DVH at bin centres
    if shape == "sphere":
        cum_vol = sphere_cumulative_volume_cc(
            dose_centers_Gy,
            shape_params["radius_mm"],
            shape_params["z_center_mm"],
            m, b,
        )
    elif shape == "cylinder":
        cum_vol = cylinder_cumulative_volume_cc(
            dose_centers_Gy,
            shape_params["radius_mm"],
            shape_params["z_bottom_mm"],
            shape_params["z_top_mm"],
            m, b,
        )
    elif shape == "cone":
        cum_vol = cone_cumulative_volume_cc(
            dose_centers_Gy,
            shape_params["base_radius_mm"],
            shape_params["z_base_mm"],
            shape_params["z_apex_mm"],
            m, b,
        )

    # Convert cumulative to differential
    # diff[i] = cum[i] - cum[i+1]  (volume in this bin only)
    # The last bin: diff[-1] = cum[-1] (everything remaining)
    diff_volumes_cc = np.diff(-cum_vol, append=0.0)
    # Ensure non-negative (numerical precision)
    diff_volumes_cc = np.maximum(diff_volumes_cc, 0.0)

    return DVHData(
        structure_name=name,
        structure_volume_cc=total_vol_cc,
        dose_edges_cGy=dose_edges_cGy,
        diff_volumes_cc=diff_volumes_cc,
        bin_width_cGy=bin_width_cGy,
        source="analytical",
        metadata={
            "shape": shape,
            "shape_params": shape_params,
            "gradient_slope_Gy_per_mm": m,
            "gradient_offset_Gy": b,
        },
    )
