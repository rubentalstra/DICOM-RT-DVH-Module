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
| 1     | Analytical ground truth (Nelms phantoms) | Done    |
| 2     | Cross-validation against commercial TPS  | Done    |
| 3     | Clinical validation (RayStation data)    | Done    |
| 4     | Format comparison (DICOM vs JSON)        | Done    |
| 5     | Statistical analysis and reporting       | Done    |
| 6     | Reproducibility and bias checks          | Done    |

## Results

All six phases completed. Across 6,864 comparisons spanning 143 structures
from 3 independent datasets, every tested parameter stayed within the
pre-specified tolerances, with zero failures.

Full write-up: `paper/manuscript.md`. Protocol: `dvh_experiment_plan.md`.

## Tolerances (locked 2026-03-27)

See `config.py` -- these are defined before any experiment runs and
must not change. Values come from Table 4 of the research proposal.

## Data

Clinical data (DICOM files) is not tracked in version control.
Use the scripts in `scripts/` to download public datasets, and
export clinical cases from RayStation separately.
