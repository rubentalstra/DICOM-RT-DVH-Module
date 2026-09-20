"""Error metrics, Bland-Altman analysis, and tolerance checking.

Computes absolute and relative errors, Bland-Altman statistics (mean bias,
limits of agreement), and checks results against the locked tolerance
thresholds defined in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import TOLERANCES, classify_volume


@dataclass
class ComparisonResult:
    """Result of comparing a single parameter between reference and cached DVH."""

    parameter_name: str
    parameter_type: str
    ref_value: float
    cached_value: float
    abs_error: float
    rel_error_pct: float | None
    tolerance_value: float
    tolerance_unit: str
    within_tolerance: bool
    interpolation_method: str
    structure_name: str
    structure_volume_cc: float
    volume_class: str


def _get_parameter_type(param_name: str) -> str:
    """Map a parameter name to its tolerance category."""
    if param_name.startswith("V") and "Gy" in param_name:
        return "Vx"
    elif param_name.startswith("D") and "cc" in param_name:
        return "Dxcc"
    elif param_name.startswith("D") and param_name[1:].isdigit():
        return "Dx"
    elif param_name == "mean_dose":
        return "mean_dose"
    elif param_name.startswith("gEUD"):
        return "gEUD"
    elif param_name.startswith("NTCP"):
        return "NTCP_TCP"
    else:
        return "unknown"


def check_tolerance(
    param_type: str,
    ref_value: float,
    cached_value: float,
) -> tuple[bool, float, str]:
    """Check if the error is within the locked tolerance.

    Returns
    -------
    tuple of (within_tolerance, tolerance_value, tolerance_unit)
    """
    if param_type not in TOLERANCES:
        return True, 0.0, "N/A"

    tol = TOLERANCES[param_type]
    tol_value = tol["value"]
    tol_unit = tol["unit"]
    abs_err = abs(cached_value - ref_value)

    if tol_unit == "Gy":
        within = abs_err <= tol_value
    elif tol_unit == "% absolute volume":
        within = abs_err <= tol_value
    elif tol_unit == "% relative":
        if ref_value == 0:
            within = abs_err == 0
        else:
            rel_err_pct = (abs_err / abs(ref_value)) * 100.0
            within = rel_err_pct <= tol_value
    elif tol_unit == "% absolute":
        within = abs_err <= tol_value
    else:
        within = True

    return within, tol_value, tol_unit


def compare_parameters(
    ref_params: dict[str, float],
    cached_params: dict[str, float],
    structure_name: str,
    structure_volume_cc: float,
    interpolation_method: str,
) -> list[ComparisonResult]:
    """Compare all parameter pairs and check against locked tolerances.

    Parameters
    ----------
    ref_params : dict
        Reference parameter values (from voxel DVH).
    cached_params : dict
        Cached parameter values (from DICOM RT DVH round-trip).
    structure_name : str
        Name of the structure.
    structure_volume_cc : float
        Total structure volume in cc.
    interpolation_method : str
        Interpolation method used for cached parameters.

    Returns
    -------
    list of ComparisonResult
    """
    volume_class = classify_volume(structure_volume_cc)
    results = []

    for param_name in ref_params:
        if param_name not in cached_params:
            continue

        ref_val = ref_params[param_name]
        cached_val = cached_params[param_name]
        abs_err = abs(cached_val - ref_val)

        rel_err_pct = None
        if ref_val != 0:
            rel_err_pct = (abs_err / abs(ref_val)) * 100.0

        param_type = _get_parameter_type(param_name)
        within_tol, tol_value, tol_unit = check_tolerance(
            param_type, ref_val, cached_val,
        )

        results.append(ComparisonResult(
            parameter_name=param_name,
            parameter_type=param_type,
            ref_value=ref_val,
            cached_value=cached_val,
            abs_error=abs_err,
            rel_error_pct=rel_err_pct,
            tolerance_value=tol_value,
            tolerance_unit=tol_unit,
            within_tolerance=within_tol,
            interpolation_method=interpolation_method,
            structure_name=structure_name,
            structure_volume_cc=structure_volume_cc,
            volume_class=volume_class,
        ))

    return results


def bland_altman_stats(
    results: list[ComparisonResult],
    group_by: str = "parameter_type",
) -> pd.DataFrame:
    """Compute Bland-Altman statistics grouped by a field.

    Parameters
    ----------
    results : list of ComparisonResult
        Comparison results.
    group_by : str
        Field to group by (e.g. ``"parameter_type"``, ``"volume_class"``).

    Returns
    -------
    pd.DataFrame
        Columns: group, n, mean_bias, sd, loa_lower, loa_upper,
        max_abs_error, pct_exceeding_tolerance.
    """
    df = results_to_dataframe(results)

    rows = []
    for group_val, group_df in df.groupby(group_by):
        diffs = group_df["cached_value"] - group_df["ref_value"]
        n = len(diffs)
        mean_bias = float(diffs.mean())
        sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
        loa_lower = mean_bias - 1.96 * sd
        loa_upper = mean_bias + 1.96 * sd
        max_abs = float(group_df["abs_error"].max())
        pct_exceed = float((~group_df["within_tolerance"]).mean() * 100.0)

        rows.append({
            group_by: group_val,
            "n": n,
            "mean_bias": mean_bias,
            "sd": sd,
            "loa_lower": loa_lower,
            "loa_upper": loa_upper,
            "max_abs_error": max_abs,
            "pct_exceeding_tolerance": pct_exceed,
        })

    return pd.DataFrame(rows)


def results_to_dataframe(results: list[ComparisonResult]) -> pd.DataFrame:
    """Convert comparison results to a DataFrame."""
    return pd.DataFrame([
        {
            "structure_name": r.structure_name,
            "structure_volume_cc": r.structure_volume_cc,
            "volume_class": r.volume_class,
            "parameter_type": r.parameter_type,
            "parameter_name": r.parameter_name,
            "ref_value": r.ref_value,
            "cached_value": r.cached_value,
            "abs_error": r.abs_error,
            "rel_error_pct": r.rel_error_pct,
            "tolerance_value": r.tolerance_value,
            "tolerance_unit": r.tolerance_unit,
            "within_tolerance": r.within_tolerance,
            "interpolation_method": r.interpolation_method,
        }
        for r in results
    ])
