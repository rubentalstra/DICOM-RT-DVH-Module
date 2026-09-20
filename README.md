# DVH Validation Experiment

Validating DICOM RT DVH files at 1 cGy resolution as an intermediate
data layer for dose-volume parameter extraction.

## Hypothesis

DVH parameters (Vx, Dx, Dxcc, mean dose, gEUD, NTCP/TCP) derived from
cached DICOM RT DVH files match those computed from raw dose grids within
clinical tolerance.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiment Phases

| Phase | Description                              | Status  |
|-------|------------------------------------------|---------|
| 0     | Environment setup                        | Done    |
| 1     | Analytical ground truth (Nelms phantoms) | Pending |
| 2     | Cross-validation against commercial TPS  | Pending |
| 3     | Clinical validation (RayStation data)    | Pending |
| 4     | Format comparison (DICOM vs JSON)        | Pending |
| 5     | Statistical analysis and reporting       | Pending |
| 6     | Reproducibility and bias checks          | Pending |

## Tolerances (locked 2026-03-27)

See `config.py` -- these are defined before any experiment runs and
must not change. Values come from Table 4 of the research proposal.

## Data

Clinical data (DICOM files) is not tracked in version control.
Use the scripts in `scripts/` to download public datasets, and
export clinical cases from RayStation separately.
