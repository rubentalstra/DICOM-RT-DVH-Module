# Experiment Plan: Validating the DICOM RT DVH Module as an Intermediate Data Layer

## Step-by-Step Execution Guide for Python Implementation

**Version:** 1.0 — March 2026
**Purpose:** Prove (or disprove) that dose–volume parameters derived from cached DICOM RT DVH files at 1 cGy resolution match ground truth values within clinical tolerance — using independently verified reference data, not self-validated outputs.

---

## Critical Principles for Honest Validation

Before writing a single line of code, internalise these rules. They exist to prevent the study from unconsciously defending a conclusion rather than testing a hypothesis.

1. **The hypothesis can fail.** If 1 cGy resolution is not sufficient for Dxcc on small structures, we report that honestly. The paper is valuable either way.
2. **Ground truth must be external.** We never validate our DVH computation by comparing it only against itself. Every comparison has at least one side that comes from an independent, authoritative source.
3. **Tolerances are defined before the experiment runs.** We do not look at results and then pick thresholds that make them pass. Table 4 from the proposal is locked in before Step 1 starts.
4. **All code is deterministic and reproducible.** No random seeds, no manual adjustments, no cherry-picking of "representative" cases. Every result can be regenerated from the same inputs.
5. **Failures get the same reporting depth as successes.** If a parameter type fails tolerance for a specific volume class, we analyse why with the same rigour as the passing cases.

---

## Phase 0: Environment Setup

### Step 0.1 — Create the project structure

```
dvh-validation/
├── README.md
├── requirements.txt
├── config.py                  # All constants, tolerances, paths
├── data/
│   ├── nelms/                 # Nelms analytical phantom files
│   ├── dicompyler/            # dicompyler test dataset
│   ├── slicerrt/              # SlicerRtData Eclipse phantom files
│   ├── clinical/              # Your own anonymised clinical cases
│   └── results/               # All output goes here
├── src/
│   ├── __init__.py
│   ├── ground_truth.py        # Analytical DVH computation (Nelms formulas)
│   ├── dvh_compute.py         # Voxel-level DVH from RT Dose + RT Struct
│   ├── dvh_cache.py           # Write/read DICOM RT DVH files (the caching layer)
│   ├── dvh_parameters.py      # Derive Vx, Dx, Dxcc, mean, gEUD, NTCP/TCP
│   ├── comparison.py          # Bland-Altman, error metrics, tolerance checking
│   ├── format_benchmark.py    # DICOM vs JSON size/speed comparison
│   └── visualisation.py       # Plots, Bland-Altman diagrams
├── tests/
│   ├── test_ground_truth.py   # Unit tests for analytical formulas
│   ├── test_dvh_compute.py    # Unit tests for voxel-level DVH
│   ├── test_dvh_parameters.py # Unit tests for parameter derivation
│   └── test_roundtrip.py      # Write-then-read integrity checks
├── notebooks/
│   ├── 01_nelms_validation.ipynb
│   ├── 02_tps_cross_validation.ipynb
│   ├── 03_clinical_validation.ipynb
│   ├── 04_format_comparison.ipynb
│   └── 05_results_analysis.ipynb
└── scripts/
    ├── download_nelms.py      # Automated dataset download
    ├── download_dicompyler.py
    ├── download_slicerrt.py
    └── run_full_experiment.py  # End-to-end execution
```

### Step 0.2 — Install dependencies

```
# requirements.txt
pydicom>=2.4.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
dicompyler-core>=0.5.6
scikit-image>=0.20.0
SimpleITK>=2.2.0
requests>=2.28.0
tqdm>=4.65.0
```

### Step 0.3 — Lock tolerance thresholds in config.py

These come directly from the proposal (Table 4) and must NOT be changed after seeing results.

```python
# config.py — LOCKED BEFORE EXPERIMENT STARTS

TOLERANCES = {
    "Vx":        {"value": 1.0,  "unit": "% absolute volume"},
    "Dx":        {"value": 0.5,  "unit": "Gy"},
    "Dxcc":      {"value": 0.5,  "unit": "Gy"},
    "mean_dose": {"value": 1.0,  "unit": "% relative"},
    "gEUD":      {"value": 1.0,  "unit": "% relative"},
    "NTCP_TCP":  {"value": 1.0,  "unit": "% absolute"},
}

VOLUME_CLASSES = {
    "tiny":   (0, 1),       # <1 cc (lens, optic nerve)
    "small":  (1, 10),      # 1-10 cc (cochlea, chiasm)
    "medium": (10, 100),    # 10-100 cc (parotid, cord segment)
    "large":  (100, float("inf")),  # >100 cc (rectum, bladder, liver)
}

DVH_BIN_WIDTH_CGY = 1  # 1 cGy = 0.01 Gy — the resolution under test
```

---

## Phase 1: Analytical Ground Truth Validation (Nelms Phantoms)

**Goal:** Prove that our DVH computation pipeline produces correct results by testing against mathematically provable reference values. If this fails, everything downstream is unreliable.

### Step 1.1 — Download the Nelms dataset

**Source:** http://canislupusllc.com/CurveCompare/DVH-Analysis-Data-Etc.zip
**Backup:** https://www.elekta.com/products/radiation-therapy/proknow-news/dvh-calculation-accuracy/

Write `scripts/download_nelms.py` to automate this. Verify checksums if available.

**What you will find:** DICOM RT Structure Set files with sphere, cylinder, and cone contours at various axial spacings (0.2–3 mm). Synthetic DICOM RT Dose files with linear dose gradients at grid resolutions from 0.4–3 mm. Analytical reference values for total volume, Dmax, Dmin, D99, D95, D5, D1, D0.03cc.

### Step 1.2 — Inventory the dataset

Before doing anything, document what you received:

```python
# Pseudocode for inventory
for file in nelms_directory:
    ds = pydicom.dcmread(file)
    log: modality, SOP class, grid resolution, structure names,
         contour spacing, dose gradient parameters
```

Output: `data/results/nelms_inventory.csv` with one row per file.

### Step 1.3 — Implement analytical DVH formulas

In `src/ground_truth.py`, implement the closed-form DVH solutions for each shape + gradient combination from the Nelms 2015 paper:

- **Sphere with linear gradient:** V(d) derived from spherical cap volume as a function of dose threshold
- **Cylinder with linear gradient:** V(d) derived from cross-sectional area as a function of dose
- **Cone with linear gradient:** V(d) derived from conical section volume

Write unit tests (`tests/test_ground_truth.py`) that verify these formulas reproduce the exact values listed in the Nelms paper tables. This validates the math itself before using it as ground truth.

### Step 1.4 — Implement voxel-level DVH computation

In `src/dvh_compute.py`, implement the full DVH computation pipeline:

```python
def compute_dvh_from_dicom(rt_dose_path, rt_struct_path, bin_width_cGy=1):
    """
    Compute differential DVH from raw DICOM RT Dose + RT Structure Set.
    
    This is the REFERENCE computation. The caching layer is validated
    against the output of this function.
    
    Returns: dict of {structure_name: DVH object}
    """
    # 1. Load RT Dose grid: dose_array (3D numpy), grid spacing, origin
    # 2. Load RT Structure Set: contour sequences per structure
    # 3. For each structure:
    #    a. Voxelise contours onto the dose grid (binary mask)
    #    b. Extract dose values at all True voxels
    #    c. Build differential histogram at bin_width_cGy resolution
    #    d. Compute total volume = count(True voxels) * voxel_volume_cc
    # 4. Return DVH data per structure
```

**IMPORTANT — Voxelisation method:** Use a well-documented, standard approach. Document the specific algorithm (polygon fill on each slice, or 3D mesh-based). The voxelisation method is a known source of inter-system variability — we need to state exactly what we used.

### Step 1.5 — Compare voxel-level DVH against analytical solutions

For each Nelms phantom configuration:

| Comparison | Source A | Source B | Expected outcome |
|---|---|---|---|
| Volume accuracy | Our voxelisation | Analytical formula | Match within grid-resolution error |
| Dmax | Our voxel extraction | Analytical (max dose in structure) | Match within 1 bin width |
| D95, D5, etc. | Our DVH interpolation | Analytical closed-form | Match within Nelms-documented tolerance for given grid |

**Decision gate:** If our voxel-level computation disagrees with the analytical solution beyond what the Nelms paper documents for the same grid resolution, STOP AND DEBUG before proceeding.

Output: `data/results/phase1_step5_voxel_vs_analytical.csv`

### Step 1.6 — Cache-and-recover: the core experiment

This is the primary test of the entire research question.

```python
def run_cache_and_recover(voxel_dvh, structure_info):
    """
    The core experiment:
    1. Take a voxel-computed DVH (the reference)
    2. Write it to DICOM RT DVH format at 1 cGy resolution
    3. Read it back
    4. Derive parameters from the cached version
    5. Compare against parameters from the reference
    """
    # Step A: Write DICOM RT DVH file
    cached_path = dvh_cache.write_dicom_dvh(voxel_dvh, structure_info)
    
    # Step B: Read it back (fresh load, not from memory)
    cached_dvh = dvh_cache.read_dicom_dvh(cached_path)
    
    # Step C: Derive parameters from BOTH sources
    ref_params = dvh_parameters.derive_all(voxel_dvh)
    cached_params_linear = dvh_parameters.derive_all(cached_dvh, interp="linear")
    cached_params_spline = dvh_parameters.derive_all(cached_dvh, interp="cubic_spline")
    
    # Step D: Compare
    results = comparison.compare(ref_params, cached_params_linear, cached_params_spline)
    return results
```

**Parameters to derive and compare:**

| Category | Specific parameters | Notes |
|---|---|---|
| Vx | V10Gy, V20Gy, V30Gy, V40Gy, V50Gy, V60Gy | Adjust range to match phantom's dose range |
| Dx | D99, D98, D95, D50, D5, D2, D1 | Percentage of total volume |
| Dxcc | D0.03cc, D0.1cc, D1cc, D2cc | Absolute volume — hardest test |
| Mean dose | Single value per structure | Integral over differential DVH |
| gEUD | a = 1, 2, 5, 10, 20 | Spans parallel to serial organ models |
| NTCP | LKB model, TD50=50Gy, m=0.5, n=0.5 | Standard test parameters |

**Output columns:**
```
phantom_id, structure_name, structure_volume_cc, volume_class,
parameter_type, parameter_name, parameter_value_specific,
ref_value, cached_value_linear, cached_value_spline,
abs_error_linear, abs_error_spline,
rel_error_pct_linear, rel_error_pct_spline,
tolerance_value, tolerance_unit,
within_tolerance_linear, within_tolerance_spline
```

Save to: `data/results/phase1_step6_cache_and_recover.csv`

---

## Phase 2: Cross-Validation Against Commercial TPS DVH

**Goal:** Show consistency with FDA-cleared commercial TPS output. This is a credibility check, not a ground truth validation.

### Step 2.1 — Download SlicerRtData Eclipse phantoms

Source: https://github.com/SlicerRt/SlicerRtData

Target datasets:
- `eclipse-8.1.20-phantom-prostate/`
- `eclipse-8.1.20-phantom-breast/`
- `eclipse-8.1.20-phantom-ent/`

### Step 2.2 — Extract embedded TPS DVH from RT Dose files

```python
def extract_eclipse_dvh(rt_dose_path):
    """
    Check for DVH Sequence (3004,0050) in Eclipse RT Dose files.
    If present, extract the embedded commercial TPS DVH.
    """
    ds = pydicom.dcmread(rt_dose_path)
    if hasattr(ds, 'DVHSequence'):
        for dvh_item in ds.DVHSequence:
            dvh_type = dvh_item.DVHType          # DIFFERENTIAL or CUMULATIVE
            dvh_data = dvh_item.DVHData           # Dose-bin-width, volume pairs
            dvh_units = dvh_item.DVHVolumeUnits   # CM3, PERCENT, etc.
            roi_num = dvh_item.DVHReferencedROISequence[0].ReferencedROINumber
            # Parse and store
    else:
        log("No embedded DVH found — Eclipse may not have exported it")
```

### Step 2.3 — Compute our DVH from the same dose grid

Use `dvh_compute.py` on the Eclipse RT Dose + RT Struct files. Same code as Phase 1.

### Step 2.4 — Three-way comparison

| Comparison | Purpose |
|---|---|
| Our voxel DVH vs. Eclipse embedded DVH | Inter-system consistency check |
| Our cached DVH vs. our voxel DVH | The caching layer test (same as Phase 1 but on clinical data) |
| Our cached DVH vs. Eclipse embedded DVH | End-to-end: does cached output match commercial TPS? |

### Step 2.5 — dicompyler test data validation

Download dicompyler's bundled test dataset. Compare our parameters against their published expected values:

```python
DICOMPYLER_EXPECTED = {
    "Heart": {
        "volume_cc": 437.46,
        "max_dose_Gy": 3.10,
        "min_dose_Gy": 0.02,
        "mean_dose_Gy": 0.64,
        "D100_Gy": 0.00,
        "D98_Gy": 0.03,
        "D95_Gy": 0.03,
        "D2cc_Gy": 2.93,
    }
}
```

Output: `data/results/phase2_crossvalidation.csv`

---

## Phase 3: Clinical Validation (RayStation Data)

**Goal:** Test on real clinical plans with full anatomical complexity.

### Step 3.1 — Case selection criteria

Select at least 20 cases with the following stratification:

| Criterion | Minimum count | Examples |
|---|---|---|
| Conventional fractionation | 10 | Prostate 78Gy/39fx, H&N 70Gy/35fx |
| SBRT | 5 | Lung 54Gy/3fx, Liver 50Gy/5fx |
| Tiny structures (<1 cc) | 5 structures total | Lens, optic nerve |
| Small structures (1–10 cc) | 5 structures total | Cochlea, chiasm |
| Medium structures (10–100 cc) | 10 structures total | Parotid, cord |
| Large structures (>100 cc) | 10 structures total | Rectum, bladder |

### Step 3.2 — Export from RayStation

RayStation scripting (IronPython) to export per case:
1. DICOM RT Dose
2. DICOM RT Structure Set
3. DICOM RT Plan
4. Optionally: RayStation's own DVH point values (via scripting API)

### Step 3.3 — Run the full pipeline

For each clinical case:
1. Compute voxel-level DVH (reference)
2. Write DICOM RT DVH file at 1 cGy
3. Read back
4. Derive all parameters (both interpolation methods)
5. Compare against reference
6. Record volume class and treatment type for stratification

### Step 3.4 — RayStation cross-check

If RayStation DVH values are available, add a secondary comparison:
- Our voxel-level vs. RayStation
- Documents inter-system agreement (strengthens credibility)

Output: `data/results/phase3_clinical_validation.csv`

---

## Phase 4: Format Comparison (DICOM vs JSON)

### Step 4.1 — Generate all format variants

For each cached DICOM RT DVH file, produce:

| Format ID | Description |
|---|---|
| `dicom` | Native DICOM RT DVH (Implicit VR LE) |
| `json_flat_pretty` | `{"bins": [...], "volumes": [...]}`, indented |
| `json_flat_mini` | Same, minified |
| `json_verbose_pretty` | One object per bin with all keys, indented |
| `json_verbose_mini` | Same, minified |

### Step 4.2 — Size and speed benchmarks

```python
def benchmark_format(filepath, format_type, n_iterations=100):
    """
    Measure read time (wall clock, averaged over n_iterations after warmup).
    Measure file size (raw and compressed with gzip, bz2, zstd).
    """
```

### Step 4.3 — Metadata completeness comparison

Create a side-by-side table of all metadata fields available in DICOM RT DVH vs. what a JSON schema would need to define from scratch.

Output: `data/results/phase4_format_comparison.csv` and `data/results/phase4_metadata_comparison.md`

---

## Phase 5: Statistical Analysis and Reporting

### Step 5.1 — Aggregate results

Merge Phase 1, 2, and 3 CSV files into a master analysis dataframe.

### Step 5.2 — Bland-Altman analysis per parameter type

For each (parameter_type, volume_class, treatment_type):

```python
def bland_altman(reference_values, cached_values):
    differences = cached_values - reference_values
    means = (reference_values + cached_values) / 2
    mean_bias = np.mean(differences)
    sd = np.std(differences, ddof=1)
    loa_upper = mean_bias + 1.96 * sd
    loa_lower = mean_bias - 1.96 * sd
    return mean_bias, loa_lower, loa_upper, np.max(np.abs(differences))
```

### Step 5.3 — Summary tables

**Table A: Overall results by parameter type**

| Parameter | N | Mean bias | 95% LoA | Max |Δ| | % > tolerance |
|---|---|---|---|---|---|

**Table B: Results by volume class (within each parameter type)**

**Table C: Conventional vs. SBRT**

**Table D: Linear vs. cubic spline interpolation**

### Step 5.4 — Failure mode analysis

If ANY cell in the tables shows >0% exceeding tolerance:
1. Identify the specific cases
2. Plot the DVH curve at the failure region
3. Show the interpolation error at the exact query point
4. Test whether 0.5 cGy or 0.1 cGy bin width resolves it
5. Report transparently: "At 1 cGy, Dxcc on structures <1 cc in SBRT plans can exceed 0.5 Gy tolerance in X% of cases. Reducing to 0.5 cGy resolves this."

### Step 5.5 — Performance benchmark

```python
# Time: cached DVH → parameter extraction for N patients
# vs: raw DICOM RT Dose + Struct → DVH computation → parameter extraction
for N in [100, 500, 1000]:
    t_cached = time(derive_from_cached, N)
    t_raw = time(derive_from_raw, N)
    speedup = t_raw / t_cached
```

---

## Phase 6: Reproducibility and Bias Checks

### Step 6.1 — Pre-publication checklist

- [ ] Tolerances in config.py are identical to the values in the paper
- [ ] Tolerances were committed to version control BEFORE Phase 1 started
- [ ] All Nelms analytical formulas match the published paper
- [ ] No results were excluded
- [ ] Failed tolerance checks are reported with the same detail as passes
- [ ] Code is in a public repository with a DOI (Zenodo)
- [ ] All external datasets have citations, versions, and download instructions

### Step 6.2 — Sensitivity analysis

Run the full Phase 1 cache-and-recover at multiple bin widths:

| Bin width | Purpose |
|---|---|
| 0.1 cGy | Upper bound on accuracy (10x finer) |
| 0.5 cGy | Intermediate check |
| 1 cGy | **The proposed resolution (primary result)** |
| 2 cGy | Coarser — where does it start to fail? |
| 5 cGy | Much coarser — expected to fail for some parameters |
| 10 cGy | Definitely too coarse — included to show the gradient |

This produces a "bin width vs. accuracy" curve that is one of the most valuable figures in the paper.

### Step 6.3 — Alternative computation engine check

Repeat Phase 1 Step 1.6 using dicompyler-core's DVH computation instead of our own `dvh_compute.py`. If both engines produce the same cache-and-recover accuracy, the result is robust. If they differ, investigate and report.

---

## Execution Timeline Estimate

| Phase | Steps | Estimated effort | Blocking? |
|---|---|---|---|
| Phase 0 | Setup | 1 day | — |
| Phase 1 | Nelms validation | 3–5 days | Yes — gates everything |
| Phase 2 | TPS cross-validation | 2–3 days | No (parallel with Phase 3) |
| Phase 3 | Clinical validation | 3–5 days | Needs RayStation export |
| Phase 4 | Format comparison | 1 day | No (anytime after Phase 1) |
| Phase 5 | Statistical analysis | 2–3 days | After Phases 1–4 |
| Phase 6 | Reproducibility checks | 2–3 days | After Phase 5 |
| **Total** | | **14–20 days** | |

---

## Decision Points (Where to Stop and Evaluate)

**After Step 1.5 (voxel vs. analytical):** If our voxel computation does not match analytical solutions within the expected grid-resolution error, STOP. Fix the computation. Everything downstream depends on this.

**After Step 1.6 (cache-and-recover on phantoms):** If this fails tolerance, that is a genuine research finding. It means 1 cGy is not sufficient for certain combinations. Publishable either way — the conclusion just changes.

**After Phase 2 (TPS cross-validation):** If our values diverge from Eclipse beyond the ~1–3% inter-system range documented by Penoncello et al. (2024), investigate before clinical deployment.

**After Phase 3 (clinical data):** This is where the paper's primary claims stand or fall. The stratified results tell the complete story.
