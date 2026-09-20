"""Derive clinical DVH parameters: Vx, Dx, Dxcc, mean dose, gEUD, NTCP/TCP.

Extracts dose-volume parameters from DVHData objects using configurable
interpolation methods (linear, cubic spline). Both the voxel-computed
reference and the cached DVH are processed through these same functions
to ensure a fair comparison.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import erf

from .dvh_data import DVHData


def compute_vx(
    dvh: DVHData,
    query_dose_Gy: float,
    interpolation: str = "linear",
) -> float:
    """Volume (cc) receiving >= query_dose_Gy.

    Parameters
    ----------
    dvh : DVHData
        Input DVH.
    query_dose_Gy : float
        Dose threshold in Gy.
    interpolation : str
        ``"linear"`` or ``"cubic_spline"`` (PCHIP).

    Returns
    -------
    float
        Volume in cc receiving >= query_dose_Gy.
    """
    dose_Gy = dvh.dose_centers_Gy
    cum_cc = dvh.cumulative_volumes_cc

    if query_dose_Gy <= dose_Gy[0]:
        return float(cum_cc[0])
    if query_dose_Gy >= dose_Gy[-1]:
        return 0.0

    if interpolation == "linear":
        return float(np.interp(query_dose_Gy, dose_Gy, cum_cc))
    elif interpolation == "cubic_spline":
        interp = PchipInterpolator(dose_Gy, cum_cc)
        result = float(interp(query_dose_Gy))
        return max(0.0, min(result, dvh.structure_volume_cc))
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation!r}")


def compute_dx(
    dvh: DVHData,
    query_volume_pct: float,
    interpolation: str = "linear",
) -> float:
    """Dose (Gy) received by the hottest query_volume_pct% of the structure.

    This is the inverse of the cumulative DVH curve: given a percentage
    of total volume, find the dose threshold.

    Parameters
    ----------
    dvh : DVHData
        Input DVH.
    query_volume_pct : float
        Percentage of total volume (e.g. 95 for D95).
    interpolation : str
        ``"linear"`` or ``"cubic_spline"`` (PCHIP).

    Returns
    -------
    float
        Dose in Gy.
    """
    dose_Gy = dvh.dose_centers_Gy
    cum_pct = dvh.cumulative_volumes_pct

    if dvh.structure_volume_cc == 0:
        return 0.0

    # Cumulative is decreasing, so we need to reverse for interpolation
    # (interp expects x to be increasing)
    dose_rev = dose_Gy[::-1]
    cum_pct_rev = cum_pct[::-1]

    if query_volume_pct <= cum_pct[-1]:
        return float(dose_Gy[-1])
    if query_volume_pct >= cum_pct[0]:
        return float(dose_Gy[0])

    if interpolation == "linear":
        return float(np.interp(query_volume_pct, cum_pct_rev, dose_rev))
    elif interpolation == "cubic_spline":
        # PCHIP requires strictly increasing x; deduplicate flat regions
        unique_mask = np.diff(cum_pct_rev, prepend=-np.inf) > 0
        x_unique = cum_pct_rev[unique_mask]
        y_unique = dose_rev[unique_mask]
        if len(x_unique) < 2:
            return float(np.interp(query_volume_pct, cum_pct_rev, dose_rev))
        interp_fn = PchipInterpolator(x_unique, y_unique)
        result = float(interp_fn(query_volume_pct))
        return max(0.0, result)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation!r}")


def compute_dxcc(
    dvh: DVHData,
    query_volume_cc: float,
    interpolation: str = "linear",
) -> float:
    """Dose (Gy) received by the hottest query_volume_cc of the structure.

    Same as Dx but with absolute volume. This is the hardest test case
    for small structures at coarse DVH resolution.

    Parameters
    ----------
    dvh : DVHData
        Input DVH.
    query_volume_cc : float
        Absolute volume in cc (e.g. 0.03 for D0.03cc).
    interpolation : str
        ``"linear"`` or ``"cubic_spline"`` (PCHIP).

    Returns
    -------
    float
        Dose in Gy.
    """
    dose_Gy = dvh.dose_centers_Gy
    cum_cc = dvh.cumulative_volumes_cc

    if dvh.structure_volume_cc == 0:
        return 0.0

    # Reverse for interpolation (cumulative is decreasing)
    dose_rev = dose_Gy[::-1]
    cum_cc_rev = cum_cc[::-1]

    if query_volume_cc <= cum_cc[-1]:
        return float(dose_Gy[-1])
    if query_volume_cc >= cum_cc[0]:
        return float(dose_Gy[0])

    if interpolation == "linear":
        return float(np.interp(query_volume_cc, cum_cc_rev, dose_rev))
    elif interpolation == "cubic_spline":
        # PCHIP requires strictly increasing x; deduplicate flat regions
        unique_mask = np.diff(cum_cc_rev, prepend=-np.inf) > 0
        x_unique = cum_cc_rev[unique_mask]
        y_unique = dose_rev[unique_mask]
        if len(x_unique) < 2:
            return float(np.interp(query_volume_cc, cum_cc_rev, dose_rev))
        interp_fn = PchipInterpolator(x_unique, y_unique)
        result = float(interp_fn(query_volume_cc))
        return max(0.0, result)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation!r}")


def compute_mean_dose(dvh: DVHData) -> float:
    """Mean dose in Gy from the differential DVH.

    No interpolation needed — this is exact for the given binning.

    mean = sum(dose_center_Gy * diff_volume_cc) / total_volume_cc
    """
    return dvh.mean_dose_Gy


def compute_geud(dvh: DVHData, a: float) -> float:
    """Generalised Equivalent Uniform Dose (gEUD) in Gy.

    gEUD = (sum(v_i * d_i^a) / V_total)^(1/a)

    where v_i = fractional volume of bin i, d_i = dose at bin centre.

    Special cases:
        a = 1  → mean dose
        a → +∞ → max dose
        a → -∞ → min dose

    Parameters
    ----------
    dvh : DVHData
        Input DVH.
    a : float
        gEUD exponent. Positive for serial organs, negative for parallel.

    Returns
    -------
    float
        gEUD in Gy.
    """
    if dvh.structure_volume_cc == 0:
        return 0.0

    dose_Gy = dvh.dose_centers_Gy
    fractional_vol = dvh.diff_volumes_cc / dvh.structure_volume_cc

    # Handle a = 1 specially to avoid numerical issues
    if abs(a - 1.0) < 1e-10:
        return dvh.mean_dose_Gy

    # Mask zero-dose bins to avoid 0^a issues when a < 0
    mask = dose_Gy > 0
    if not np.any(mask):
        return 0.0

    weighted_sum = np.sum(fractional_vol[mask] * dose_Gy[mask] ** a)
    return float(weighted_sum ** (1.0 / a))


def compute_ntcp_lkb(
    dvh: DVHData,
    td50: float,
    m: float,
    n: float,
) -> float:
    """NTCP using the Lyman-Kutcher-Burman (LKB) model.

    Steps:
        1. Compute gEUD with a = 1/n
        2. t = (gEUD - TD50) / (m * TD50)
        3. NTCP = 0.5 * (1 + erf(t / sqrt(2)))

    Parameters
    ----------
    dvh : DVHData
        Input DVH.
    td50 : float
        Dose for 50% complication probability (Gy).
    m : float
        Slope parameter (dimensionless).
    n : float
        Volume effect parameter (dimensionless). a = 1/n.

    Returns
    -------
    float
        NTCP as a fraction [0, 1].
    """
    if dvh.structure_volume_cc == 0 or n == 0 or m == 0 or td50 == 0:
        return 0.0

    geud = compute_geud(dvh, 1.0 / n)
    t = (geud - td50) / (m * td50)
    return float(0.5 * (1.0 + erf(t / math.sqrt(2.0))))


def derive_all(
    dvh: DVHData,
    interpolation: str = "linear",
    vx_doses_Gy: list[float] | None = None,
    dx_percentages: list[float] | None = None,
    dxcc_volumes: list[float] | None = None,
    geud_exponents: list[float] | None = None,
    lkb_params: dict | None = None,
) -> dict[str, float]:
    """Derive all DVH parameters defined in config.py query points.

    Uses config defaults if query points are not provided explicitly.

    Parameters
    ----------
    dvh : DVHData
        Input DVH.
    interpolation : str
        Interpolation method for Vx, Dx, Dxcc.
    vx_doses_Gy, dx_percentages, dxcc_volumes, geud_exponents : lists
        Override config defaults if provided.
    lkb_params : dict
        Override config LKB parameters if provided.

    Returns
    -------
    dict[str, float]
        Parameter name → value mapping. Keys like:
        ``"V10Gy"``, ``"D95"``, ``"D0.03cc"``, ``"mean_dose"``,
        ``"gEUD_a1"``, ``"NTCP_LKB"``.
    """
    # Import config defaults lazily to avoid circular imports
    from config import (
        DX_CC_QUERY_VOLUMES,
        DX_QUERY_PERCENTAGES,
        GEUD_TEST_EXPONENTS,
        LKB_TEST_PARAMS,
        VX_QUERY_DOSES_GY,
    )

    if vx_doses_Gy is None:
        vx_doses_Gy = VX_QUERY_DOSES_GY
    if dx_percentages is None:
        dx_percentages = DX_QUERY_PERCENTAGES
    if dxcc_volumes is None:
        dxcc_volumes = DX_CC_QUERY_VOLUMES
    if geud_exponents is None:
        geud_exponents = GEUD_TEST_EXPONENTS
    if lkb_params is None:
        lkb_params = LKB_TEST_PARAMS

    params: dict[str, float] = {}

    # Vx: volume receiving >= dose
    for dose_gy in vx_doses_Gy:
        params[f"V{dose_gy}Gy"] = compute_vx(dvh, dose_gy, interpolation)

    # Dx: dose to hottest x% of volume
    for pct in dx_percentages:
        params[f"D{pct}"] = compute_dx(dvh, pct, interpolation)

    # Dxcc: dose to hottest x cc
    for vol_cc in dxcc_volumes:
        params[f"D{vol_cc}cc"] = compute_dxcc(dvh, vol_cc, interpolation)

    # Mean dose
    params["mean_dose"] = compute_mean_dose(dvh)

    # gEUD at various exponents
    for a in geud_exponents:
        params[f"gEUD_a{a}"] = compute_geud(dvh, a)

    # NTCP using LKB model
    params["NTCP_LKB"] = compute_ntcp_lkb(
        dvh,
        td50=lkb_params["TD50"],
        m=lkb_params["m"],
        n=lkb_params["n"],
    )

    return params
