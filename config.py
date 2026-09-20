"""
config.py -- LOCKED BEFORE EXPERIMENT STARTS

Tolerance thresholds and experiment constants for the DVH validation study.
These values come from Table 4 of the research proposal and MUST NOT be
modified after Phase 1 begins. Any change invalidates all prior results.

Locked: 2026-03-27
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

NELMS_DIR = DATA_DIR / "nelms"
DICOMPYLER_DIR = DATA_DIR / "dicompyler"
SLICERRT_DIR = DATA_DIR / "slicerrt"
CLINICAL_DIR = DATA_DIR / "clinical"

# ──────────────────────────────────────────────────────────────────────
# DVH resolution under test
# ──────────────────────────────────────────────────────────────────────

DVH_BIN_WIDTH_CGY = 1  # 1 cGy = 0.01 Gy -- the resolution under test

# Sensitivity analysis bin widths (Phase 6)
SENSITIVITY_BIN_WIDTHS_CGY = [0.1, 0.5, 1, 2, 5, 10]

# ──────────────────────────────────────────────────────────────────────
# Tolerance thresholds (from proposal Table 4)
# LOCKED -- do not modify after experiment begins
# ──────────────────────────────────────────────────────────────────────

TOLERANCES = {
    "Vx":        {"value": 1.0, "unit": "% absolute volume"},
    "Dx":        {"value": 0.5, "unit": "Gy"},
    "Dxcc":      {"value": 0.5, "unit": "Gy"},
    "mean_dose": {"value": 1.0, "unit": "% relative"},
    "gEUD":      {"value": 1.0, "unit": "% relative"},
    "NTCP_TCP":  {"value": 1.0, "unit": "% absolute"},
}

# ──────────────────────────────────────────────────────────────────────
# Volume classification (for stratified analysis)
# ──────────────────────────────────────────────────────────────────────

VOLUME_CLASSES = {
    "tiny":   (0, 1),                   # <1 cc  (lens, optic nerve)
    "small":  (1, 10),                  # 1-10 cc  (cochlea, chiasm)
    "medium": (10, 100),                # 10-100 cc  (parotid, cord segment)
    "large":  (100, float("inf")),      # >100 cc  (rectum, bladder, liver)
}

# ──────────────────────────────────────────────────────────────────────
# DVH parameter query points
# ──────────────────────────────────────────────────────────────────────

VX_QUERY_DOSES_GY = [10, 20, 30, 40, 50, 60]
DX_QUERY_PERCENTAGES = [99, 98, 95, 50, 5, 2, 1]
DX_CC_QUERY_VOLUMES = [0.03, 0.1, 1.0, 2.0]

# ──────────────────────────────────────────────────────────────────────
# Interpolation methods to compare
# ──────────────────────────────────────────────────────────────────────

INTERPOLATION_METHODS = ["linear", "cubic_spline"]

# ──────────────────────────────────────────────────────────────────────
# Radiobiology test parameters (for NTCP/TCP computation)
# ──────────────────────────────────────────────────────────────────────

LKB_TEST_PARAMS = {
    "TD50": 50.0,   # Gy
    "m": 0.5,
    "n": 0.5,
}

GEUD_TEST_EXPONENTS = [1, 2, 5, 10, 20]

# ──────────────────────────────────────────────────────────────────────
# DICOM constants
# ──────────────────────────────────────────────────────────────────────

IMPLICIT_VR_LE = "1.2.840.10008.1.2"
RT_DVH_STORAGE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.9"
RT_DOSE_STORAGE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"
RT_STRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
RT_PLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"

IMPLEMENTATION_CLASS_UID = "1.2.826.0.1.3680043.8.498.1"
SOFTWARE_VERSION = "DVH-Validation-1.0"


def classify_volume(volume_cc: float) -> str:
    """Classify a structure volume into a volume class."""
    for name, (lo, hi) in VOLUME_CLASSES.items():
        if lo <= volume_cc < hi:
            return name
    return "large"
