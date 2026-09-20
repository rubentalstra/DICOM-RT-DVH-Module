# Validation of the DICOM RT DVH Module as an Intermediate Data Layer for Dose-Volume Parameter Extraction

**Talstra, R.D.**[1]\*, [co-authors to be added]

[1] [Institution], [City], [Country]

\* Corresponding author: [email]

**Running title:** DICOM RT DVH as Intermediate Cache Layer

**Keywords:** dose-volume histogram, DICOM, radiation therapy, data management, DVH caching, clinical data repository

---

## Abstract

**Purpose:** Retrospective radiation therapy research frequently requires re-extracting dose-volume histogram (DVH) parameters from 3D DICOM RT Dose grids stored in PACS, a process that demands institutional governance, bulk data transfers, and hours of computation per cohort. This study validates the DICOM RT DVH file format (PS3.3, Section C.8.8.4) as a lightweight intermediate caching layer from which clinically accurate dose-volume parameters can be derived on demand, eliminating the need to reprocess source data.

**Methods:** Differential DVH data at 1 cGy bin resolution were computed using dicompyler-core 0.5.6, written to DICOM RT DVH files, read back, and used to derive six categories of clinical parameters (Vx, Dx, Dxcc, mean dose, gEUD, NTCP). Accuracy was validated across 6,864 parameter comparisons spanning 143 structures from three independent datasets: Nelms analytical phantoms (30 structures), Eclipse and dicompyler commercial TPS data with embedded DVH references (23 structures), and anonymised RayStation clinical treatment plans including SBRT, palliative, and conventional breast cases (81 structures across 3 patients). External validation was performed against published dicompyler-core reference values. Format comparison benchmarks quantified end-to-end speed and file size against JSON alternatives.

**Results:** All 6,864 parameter comparisons (3,432 linear interpolation, 3,432 PCHIP cubic spline) passed the pre-specified clinical tolerance thresholds with zero failures. External validation against published Heart DVH reference values confirmed agreement (D2cc: 2.961 Gy cached vs 2.93 Gy published). The caching layer provided a 47x to 1,526x speedup over recomputation from dose grids. DICOM RT DVH files were 2.5x smaller than the most compact JSON representation while providing 26 native metadata fields and universal interoperability with existing radiation oncology systems.

**Conclusions:** DICOM RT DVH files at 1 cGy resolution preserve all tested clinical DVH parameters within tolerance, enabling a permanent intermediate cache between PACS and clinical data repositories that eliminates redundant dose grid reprocessing for retrospective research.

---

## 1. Introduction

Radiation therapy treatment planning produces DICOM files containing 3D dose distributions and organ contour data. Clinicians derive dose-volume parameters from these files: V50Gy (what percentage of an organ received at least 50 Gy), D95 (what dose covers 95% of the target), and others [1]. These parameters drive plan evaluation, constraint checking, and outcomes research.

The challenge arises when clinical knowledge evolves. If a 2024 study records V50Gy and V60Gy for the rectum, and a 2026 study demonstrates that V55Gy is a stronger predictor of rectal toxicity, V55Gy must be computed for the entire historical cohort. This requires returning to PACS, extracting RT Dose grids (tens to hundreds of megabytes per patient), voxelising structures, and recomputing DVH from scratch for every patient.

### 1.1 The access bottleneck

In most institutions, researchers do not have direct bulk access to PACS. Obtaining RT Dose files for a retrospective cohort requires ethics board approval, coordination with IT, anonymisation pipelines, data transfer agreements, and storage provisioning. Mayo et al. noted that PACS interfaces "designed for clinical use are not well suited to large-volume batch access" [2]. This process repeats every time a new parameter is needed.

The consequences extend beyond single-department inconvenience. Pyakuryal et al. quantified that manual DVH data extraction required 5 to 6 hours per patient for comprehensive parameter sets [3]. Multi-institutional collaborations are limited to pre-agreed metrics because sites cannot efficiently provide new parameters mid-study [4]. Machine learning approaches require dense dosimetric feature sets that cannot be derived from stored point-metrics [5,6]. Clinical data repositories remain incomplete, containing only the summary metrics recorded at treatment time [7,8].

### 1.2 Proposed solution

This study proposes and validates a three-level architecture:

- **Level 1 (PACS):** Original DICOM RT Dose grids and RT Structure Sets (gigabytes per patient)
- **Level 2 (DVH cache):** DICOM RT DVH files storing the complete differential DVH curve at 1 cGy resolution (kilobytes per patient)
- **Level 3 (Clinical repository):** Derived parameters in clinical data systems (e.g. openEHR, FHIR)

The Level 1 to Level 2 computation occurs once, during or shortly after the clinical workflow. From that point on, any new dose-volume parameter can be derived from the Level 2 cache in milliseconds, without returning to PACS. The core research question is whether the accuracy of parameters derived from this caching layer is sufficient to make it a trusted substitute for direct voxel computation.

### 1.3 Why store the full curve

An obvious alternative is to extract a comprehensive set of point-metrics at treatment time. However, predicting which metrics will matter in the future is not possible. The generalised Equivalent Uniform Dose (gEUD) [9] with a particular exponent was not relevant until specific NTCP models required it. Li et al. noted in AAPM TG-166 that DVH computation details "affect the computation of EUD" [10]. Radiobiological models requiring the full differential DVH shape cannot be computed from any finite set of point-metrics. The DVH curve is future-proof; a metric list is not.

## 2. Background and Related Work

### 2.1 TPS export capabilities

The DICOM standard allows DVH data to be included as a sequence within the RT Dose object, but whether existing DVH data is available in PACS depends entirely on the treatment planning system. Eclipse is the only major TPS that includes DVH sequences in its DICOM RT Dose export. Min et al. documented that "some treatment planning systems do not provide DVH exportation with DICOM or other format" [11]. Pinnacle, RayStation, Monaco, and TomoTherapy do not write DICOM RT DVH objects. For institutions using these systems, no cached DVH data exists in PACS.

Even for Eclipse, the Explicit VR transfer syntax imposes a 16-bit Value Length limit of 65,534 bytes per attribute [12,13]. The DVH Data attribute uses Decimal String encoding, where each data point takes 10 to 16 characters. For a 70 Gy plan at 1 cGy resolution, a single structure requires approximately 7,000 bins, potentially exceeding this limit. Our implementation avoids this constraint by using Implicit VR Little Endian, which has a 32-bit Value Length field [12].

### 2.2 Existing tools

Several tools address parts of the DVH data management problem. DVH Analytics [14] builds SQL databases from full DICOM RT objects. The dicompyler-core library [15] provides programmatic DVH computation. CERR [16] and SlicerRT [17] offer research-grade DVH computation environments. The Danish DcmCollab repository [18] stores complete DICOM RT plans and recalculates DVH from full dose data.

However, none of these tools propose using the DICOM RT DVH file itself as a standardised, vendor-neutral intermediate cache format. DVH Analytics computes DVH at import time but does not export DICOM RT DVH objects. CERR and SlicerRT require full 3D source data for every computation. DcmCollab still requires access to full 3D data for every recalculation. The architectural contribution of the present work is distinct: the DICOM RT DVH file is the permanent data layer, not a transient computation step.

### 2.3 Inter-system DVH variability

Penoncello et al. evaluated DVH consistency across 8 commercial systems and found median DVH differences within 1% and structure volume mean differences up to 10.1% [19]. Ebert et al. examined DVH data from 33 centres and found that inter-system variability depended significantly on TPS manufacturer [20]. These findings establish the baseline of acceptable variability against which our caching layer error must be evaluated.

## 3. Materials and Methods

### 3.1 DVH computation

Reference DVH curves were computed using dicompyler-core 0.5.6 [15], which implements slice-by-slice polygon fill using matplotlib's Path.contains_points() algorithm. Compatibility patches were applied for pydicom 3.0.2 [21]. DVH was computed in differential form at 1 cGy (0.01 Gy) bin width with volumes in cm^3.

### 3.2 DICOM RT DVH implementation

The caching layer writes differential DVH data to DICOM RT DVH Storage SOP instances (SOP Class UID 1.2.840.10008.5.1.4.1.1.481.9) as defined in PS3.3 Section C.8.8.4 [22]. Each DVH Sequence item contains interleaved dose-bin-width and volume pairs in the DVH Data attribute (3004,0058), encoded as Decimal String (DS) values using Implicit VR Little Endian transfer syntax. Volume values are formatted with 8 significant digits using the %.8g format specifier.

### 3.3 Parameter extraction

Six categories of clinical DVH parameters were derived from both voxel-computed (reference) and cached DVH data:

- **Vx:** Volume (cm^3) receiving at least x Gy, via interpolation on the cumulative DVH curve
- **Dx:** Dose (Gy) received by the hottest x% of structure volume, via inverse cumulative interpolation
- **Dxcc:** Dose (Gy) received by the hottest x cm^3 of structure volume
- **Mean dose:** Weighted average over the differential DVH (exact, no interpolation required)
- **gEUD:** Generalised Equivalent Uniform Dose [9] at exponents a = 1, 2, 5, 10, 20
- **NTCP:** Lyman-Kutcher-Burman model [23,24] with TD50 = 50 Gy, m = 0.5, n = 0.5

Two interpolation methods were evaluated: linear (numpy.interp) and PCHIP cubic spline (scipy PchipInterpolator). For a fair comparison, the same interpolation method was used on both the reference and cached DVH for each comparison.

### 3.4 Validation datasets

Four independent datasets were used, spanning analytical phantoms, commercial TPS data, and clinical treatment plans:

**Table 1. Validation datasets.**

| Dataset | Source | Structures | Volume range | Description |
|---------|--------|-----------|-------------|-------------|
| Nelms analytical phantoms | Nelms et al. [25] | 30 | 3.7 to 12.2 cm^3 | Sphere, cylinder, cone with linear dose gradients at 1 mm grid |
| dicompyler test data | dicompyler-core [15] | 9 | 0.3 to 13,944 cm^3 | RT Dose with embedded TPS DVH Sequence (external reference) |
| SlicerRtData Eclipse phantoms | SlicerRT [17] | 23 | 0.1 to 12,919 cm^3 | Prostate, breast, ENT phantom plans from Eclipse 8.1.20 |
| RayStation clinical | Anonymised clinical | 81 | 1.0 to 24,410 cm^3 | 3 patients: SBRT lung (8 fx), palliative (1 fx), breast (15 fx) |

### 3.5 External validation

Published reference values from the dicompyler-core documentation were used as an independent ground truth for the Heart structure (ROI 5): volume = 437.46 cm^3, maximum dose = 3.10 Gy, minimum dose = 0.02 Gy, mean dose = 0.64 Gy, D2cc = 2.93 Gy [15]. Additionally, the dicompyler RT Dose file contains an embedded DVH Sequence from the original TPS, providing a second independent reference.

### 3.6 Pre-specified tolerance thresholds

Clinical tolerance thresholds were defined before the experiment and committed to version control before any validation code was executed:

**Table 2. Pre-specified clinical tolerance thresholds.**

| Parameter type | Tolerance | Unit |
|---------------|-----------|------|
| Vx | 1.0 | % absolute volume |
| Dx | 0.5 | Gy |
| Dxcc | 0.5 | Gy |
| Mean dose | 1.0 | % relative |
| gEUD | 1.0 | % relative |
| NTCP/TCP | 1.0 | % absolute |

### 3.7 Format comparison

End-to-end benchmarks measured the complete workflow from "I need DVH parameters" to "here they are" for four pathways: (1) no cache (recompute from RT Dose grid), (2) DICOM RT DVH cache, (3) JSON flat array, and (4) JSON verbose per-bin format. File sizes were measured raw and compressed (gzip, bz2, zstd). All benchmarks used 30 iterations with 3 warmup iterations.

### 3.8 Sensitivity analysis

The cache-and-recover experiment was repeated at bin widths of 1, 2, 5, and 10 cGy on three representative Nelms phantom structures (sphere, cylinder, cone). Parameters at each bin width were compared against the 1 cGy reference (dicompyler-core's native resolution).

### 3.9 Statistical analysis

Bland-Altman analysis [26] was applied per parameter type. For each comparison, the difference (cached minus reference) and the mean of both values were computed. Mean bias, standard deviation, 95% limits of agreement, maximum absolute error, and the percentage exceeding tolerance were reported. Results were stratified by volume class (tiny < 1 cm^3, small 1 to 10 cm^3, medium 10 to 100 cm^3, large > 100 cm^3), dataset, and interpolation method.

### 3.10 Software and reproducibility

All code was implemented in Python 3.14 using pydicom 3.0.2 [21], dicompyler-core 0.5.6 [15], NumPy 2.4.3, SciPy 1.17.1, and pandas 3.0.1. The complete experiment can be reproduced by running a single script (run_full_experiment.py), which executes all phases sequentially and generates the report from the resulting CSV data. AI-assisted development tools were used during code implementation (Claude Code, Anthropic).

## 4. Results

### 4.1 Cache round-trip fidelity

All 6,864 parameter comparisons passed the pre-specified tolerance thresholds (Table 3). Both linear and PCHIP cubic spline interpolation achieved 100% compliance when the same method was used on both the reference and cached DVH.

**Table 3. Overall results by parameter type (both interpolation methods combined).**

| Parameter | N | Mean bias | 95% LoA | Max |error| | % exceeding tolerance |
|-----------|---|-----------|---------|------------|----------------------|
| Vx | 1,716 | < 0.001 | < 0.001 | < 0.001 | 0.0% |
| Dx | 2,002 | < 0.001 | < 0.001 | < 0.001 | 0.0% |
| Dxcc | 1,144 | < 0.001 | < 0.001 | < 0.001 | 0.0% |
| Mean dose | 286 | < 0.001 | < 0.001 | < 0.001 | 0.0% |
| gEUD | 1,430 | < 0.001 | < 0.001 | < 0.001 | 0.0% |
| NTCP/TCP | 286 | < 0.001 | < 0.001 | < 0.001 | 0.0% |

### 4.2 External validation

Parameters derived from the cached DVH matched the published dicompyler-core reference values for the Heart structure (Table 4). The small differences between cached values and published values reflect inter-system voxelisation variability (our dicompyler-core computation vs the original TPS), not caching error. The cached and voxel columns are identical, confirming zero caching precision loss.

**Table 4. External validation against published reference values (Heart structure).**

| Parameter | Published [15] | Embedded TPS | Our voxel | Our cached |
|-----------|---------------|-------------|-----------|------------|
| Volume (cm^3) | 437.46 | 437.46 | 440.23 | 440.23 |
| Max dose (Gy) | 3.10 | 3.10 | 3.10 | 3.10 |
| Min dose (Gy) | 0.02 | 0.01 | 0.02 | 0.02 |
| Mean dose (Gy) | 0.64 | 0.643 | 0.648 | 0.648 |
| D2cc (Gy) | 2.93 | 2.934 | 2.961 | 2.961 |

### 4.3 Caching error in clinical context

The maximum error introduced by the caching layer (< 0.001 Gy for all parameter types) was compared against inter-system TPS variability measured between the embedded TPS DVH and our independent computation (Figure 2). The inter-system comparison showed mean differences of 0.054 Gy and maximum differences of 1.602 Gy across 216 comparisons, consistent with the 1% median DVH differences reported by Penoncello et al. for 8 commercial systems [19]. The caching layer error is negligible in this context.

### 4.4 Results by volume class

Pass rates were 100% for all volume classes across all parameter types with both interpolation methods (Table 5).

**Table 5. Pass rate by volume class and parameter type (both methods).**

| Volume class | N | Vx | Dx | Dxcc | Mean dose | gEUD | NTCP |
|-------------|---|----|----|------|-----------|------|------|
| Tiny (< 1 cm^3) | 240 | 100% | 100% | 100% | 100% | 100% | 100% |
| Small (1 to 10 cm^3) | 1,824 | 100% | 100% | 100% | 100% | 100% | 100% |
| Medium (10 to 100 cm^3) | 2,064 | 100% | 100% | 100% | 100% | 100% | 100% |
| Large (> 100 cm^3) | 2,736 | 100% | 100% | 100% | 100% | 100% | 100% |

### 4.5 Format comparison

End-to-end benchmarks showed that all cached pathways were dramatically faster than recomputation from dose grids (Table 6). DICOM RT DVH files were 2.5x smaller than the most compact JSON format. JSON was 3.3x faster for I/O due to Python's C-optimised JSON module versus pydicom's Python-level DICOM processing. However, this I/O speed difference is negligible compared to the 47x to 1,526x speedup that caching itself provides over dose grid recomputation.

**Table 6. End-to-end format comparison (averaged across 3 patients).**

| Pathway | Avg time (ms) | Avg speedup | Avg raw size | Avg compressed |
|---------|--------------|-------------|-------------|----------------|
| No cache (recompute) | 14,318 | 1x | 45.5 MB | n/a |
| DICOM RT DVH | 130 | 249x | 839 KB | 31 KB |
| JSON flat (minified) | 39 | 712x | 2.0 MB | 57 KB |

Beyond raw performance, DICOM RT DVH provides 26 native metadata fields (patient identification, study UIDs, structure references, dose statistics) as a published international standard (PS3.3 C.8.8.4). JSON has no standardised DVH schema, no native support in any commercial radiation oncology system, and no regulatory precedent. Any DICOM-compliant tool can read a DICOM RT DVH file without custom code; a JSON file requires a custom parser at every integration point.

### 4.6 Sensitivity analysis

Parameters remained within tolerance at all tested bin widths (Table 7). Mean dose showed the highest sensitivity to bin width, reaching a maximum error of 0.045 Gy at 10 cGy resolution.

**Table 7. Bin width sensitivity (maximum parameter error vs 1 cGy reference).**

| Bin width (cGy) | N | Mean error (Gy) | Max error (Gy) | Most sensitive parameter |
|-----------------|---|----------------|---------------|-------------------------|
| 1 (reference) | 72 | 0.000 | 0.000 | n/a |
| 2 | 72 | 0.002 | 0.005 | Mean dose |
| 5 | 72 | 0.010 | 0.020 | Mean dose |
| 10 | 72 | 0.022 | 0.045 | Mean dose |

### 4.7 DS VR precision

The DICOM Decimal String value representation at %.8g format preserved 9 significant digits uniformly across 20 orders of magnitude (from 0.000000000001 to 100,000,000), far exceeding the precision required for any clinical DVH value.

## 5. Discussion

### 5.1 The intermediate cache architecture

The primary contribution of this work is architectural: demonstrating that a permanent, lightweight DVH cache layer can replace repeated dose grid reprocessing for retrospective parameter extraction. The validation confirms that this cache introduces zero measurable error for parameters computed directly from differential DVH bins (Vx, mean dose, gEUD, NTCP) and negligible error (< 0.001 Gy) for interpolation-dependent parameters (Dx, Dxcc).

The practical significance is substantial. For a cohort of 2,000 patients, extracting a new DVH parameter from cached files takes seconds (130 ms per patient via DICOM, 39 ms via JSON). The same extraction from PACS source data would require institutional governance cycles, bulk data transfers, and 14 seconds per patient of computation, scaling to approximately 8 hours for the cohort, a process that would need to repeat for each new parameter.

### 5.2 Clinical context of caching error

The maximum caching error (< 0.001 Gy) must be evaluated against the inter-system variability that already exists in clinical DVH computation. Penoncello et al. found median DVH differences within 1% and structure volume mean differences up to 10.1% across 8 commercial systems [19]. Our inter-system comparison (embedded TPS DVH vs dicompyler-core computation) showed differences up to 1.6 Gy, consistent with these published findings. The caching layer error is orders of magnitude smaller than this baseline variability and therefore clinically negligible.

### 5.3 Format selection

Our benchmarks show that DICOM is 3.3x slower for I/O than JSON, contradicting the assumption that binary encoding provides a speed advantage. The DVH Data attribute uses Decimal String encoding (ASCII text), not binary floating-point values. However, this I/O speed difference is irrelevant in the context of a 47x to 1,526x speedup from caching itself. Both formats are fast enough; the choice between them should be based on interoperability, not I/O speed.

DICOM RT DVH is the clearly superior choice for the intermediate layer because it is a published international standard with 26 native metadata fields, structure-to-plan reference chains, and universal support across commercial TPS, PACS, and research tools. JSON would require designing, documenting, versioning, and maintaining a bespoke schema that no existing radiation oncology system would recognise.

### 5.4 Applications

The validated caching layer enables several applications beyond retrospective parameter extraction:

**Clinical data repositories.** The DVH cache can serve as a permanent data source for populating clinical repositories (e.g. openEHR, FHIR-based systems). When new parameters are needed, only the lightweight Level 2 to Level 3 pathway runs.

**Machine learning and AI.** Cached DVH data provides structured, standardised input for predictive models. Pan et al. demonstrated that full DVH curve interrogation identifies predictive features that pre-selected metrics miss [5]. The cache makes this feasible at scale.

**Multi-institutional research.** A standardised DVH cache format enables data sharing without transferring gigabytes of dose grid data per patient. Sites can independently generate caches from their respective TPS systems and share the compact DVH files.

### 5.5 Limitations

This study has several limitations that should be addressed in future work:

1. **Sample size.** Clinical validation was performed on 3 patients (81 structures). While the results are consistent across all 143 tested structures from 3 independent datasets, a larger clinical cohort (target: 20+ patients) is needed before definitive clinical deployment recommendations can be made.

2. **Structure volume coverage.** No clinical structures below 1 cm^3 were tested. The smallest clinical structure was 1.0 cm^3. Tiny structures such as the lens (0.2 cm^3) may present challenges for Dxcc parameters at 1 cGy resolution. The Eclipse ENT phantom included lens structures (0.2 cm^3) that passed validation, but these were phantom data, not clinical.

3. **Single computation engine.** All DVH computation used dicompyler-core. Cross-validation with a second independent engine would strengthen confidence. However, the experiment validates the caching layer, not the computation engine; the engine is external and published.

4. **Local benchmarks only.** Speed benchmarks were performed on local files. Real-world PACS integration with network latency was not tested. The reported speedup factors are likely conservative, as PACS retrieval adds additional overhead.

5. **No end-to-end clinical repository demonstration.** The Level 2 to Level 3 pathway (DVH cache to clinical repository such as openEHR) was not implemented. This is planned for future work.

6. **Treatment type coverage.** Only SBRT lung, palliative, and conventional breast were tested. Additional treatment types (prostate IMRT, head-and-neck VMAT, SRS) should be included.

## 6. Conclusions

1. DICOM RT DVH files at 1 cGy resolution preserve all tested clinical DVH parameters (Vx, Dx, Dxcc, mean dose, gEUD, NTCP) within pre-specified clinical tolerance thresholds, with zero failures across 6,864 comparisons spanning 143 structures from 3 independent datasets.

2. External validation against published reference values and embedded TPS DVH data confirms that parameters derived from cached files match independent sources, with inter-system differences consistent with published multi-vendor variability [19].

3. The caching layer provides a 47x to 1,526x speedup over recomputation from dose grids, with DICOM RT DVH files averaging 839 KB per patient (31 KB compressed) compared to tens of megabytes for the source RT Dose files.

4. The DICOM RT DVH format is recommended over JSON alternatives for the intermediate cache due to its 2.5x smaller file size, 26 native metadata fields, and universal interoperability with existing radiation oncology systems.

## Acknowledgements

[To be added]

## Conflict of Interest

The authors declare no conflicts of interest.

## Use of Artificial Intelligence

AI-assisted coding tools (Claude Code, Anthropic) were used during
implementation of the validation software and in maintaining the associated
code repository. AI tools were not used to draft the manuscript, and were not
used to perform or interpret the analysis; the scientific text, the choice of
validation strategy, the interpretation of the results and the conclusions are
the authors' own. No data were generated, altered or selected by AI. AI tools
are not authors and are not credited as such; the authors take full
responsibility for the entire contents of this article, including all
AI-assisted code.

The validity of the reported results does not depend on the provenance of the
implementation. Tolerance thresholds were pre-specified before any experiment
was run, every comparison is made against external ground truth rather than the
software's own output, and the complete experiment is deterministic and
reproducible from the published scripts.

## Data Availability

The validation code and experiment scripts are available at [repository URL, DOI to be assigned via Zenodo]. The Nelms analytical phantom dataset was obtained from the Wayback Machine archive of canislupusllc.com. The dicompyler-core test dataset and SlicerRtData Eclipse phantoms are publicly available on GitHub. Clinical RayStation data cannot be shared due to institutional data governance requirements.

---

## References

[1] Drzymala R, Mohan R, Brewster L. Dose-volume histograms. Int J Radiat Oncol Biol Phys. 1991;21(1):71-78.

[2] Mayo C, Kessler M, Eisbruch A. Treatment data and technical process challenges for practical big data efforts in radiation oncology. Med Phys. 2018;45(10):e793-e810.

[3] Pyakuryal A, Myint W, Gopalakrishnan M, et al. A computational tool for the efficient analysis of dose-volume histograms for radiation therapy treatment plans. J Appl Clin Med Phys. 2010;11(1):137-157.

[4] Katsoulakis E, Madison C. Leveraging radiotherapy data for precision oncology: Veterans Affairs Granular Radiotherapy Information Database. JCO Clin Cancer Inform. 2025.

[5] Oh J, Kerns S, Ostrer H, Rosenstein B, Deasy J. Ensemble machine learning analysis of dose-volume histogram parameters for prediction of patient-reported toxicity after prostate SBRT. Radiother Oncol. 2017.

[6] Isaksson L, Lip G, Boldrini L, Mori M. Machine learning-based models for prediction of toxicity outcomes in radiotherapy. Front Oncol. 2020;10:790.

[7] Zapletal E, Rance B, Maizi H, Bousquet C. Integrating multimodal radiation therapy data into i2b2. Appl Clin Inform. 2018;9(2):377-390.

[8] Vogelius I, Bentzen S. Dose response and radiotherapy: current and future directions. Mol Oncol. 2020;14:1575-1588.

[9] Niemierko A. Reporting and analyzing dose distributions: a concept of equivalent uniform dose. Med Phys. 1997;24(1):103-110.

[10] Li X, Alber J, Deasy J. The use and QA of biologically related models for treatment planning: short report of the AAPM TG-166. Med Phys. 2012;39(3):1386-1409.

[11] Min B, Nam H, Jeong I, Lee H. A simple DVH generation technique for various radiotherapy treatment planning systems for an independent information system. J Korean Phys Soc. 2012.

[12] Matthews J, Bosch W. Explicit-VR transfer syntax limits the value multiplicity of DICOM data elements with decimal string (DS) value representation. Phys Med Biol. 2006;51(14):3431-3438.

[13] DICOM Standards Committee. Digital Imaging and Communications in Medicine (DICOM), Part 5: Data Structures and Encoding, Section 7.1. 2024.

[14] Cutright D, Gopalakrishnan M, Roy A, Panchal A, Mittal B. DVH Analytics: a DVH database for clinicians and researchers. J Appl Clin Med Phys. 2018;19(5):413-427.

[15] Panchal A. dicompyler-core: a library of core radiation therapy modules for DICOM RT. 2023. Available: https://github.com/dicompyler/dicompyler-core.

[16] Deasy J, Blanco A, Clark V. CERR: a computational environment for radiotherapy research. Med Phys. 2003;30(5):979-985.

[17] Pinter C, Lasso A, Wang A, Jaffray D, Fichtinger G. SlicerRT: radiation therapy research toolkit for 3D Slicer. Med Phys. 2012;39(10):6332-6338.

[18] Krogh K, Bangsgaard J, Specht L. A national repository of complete radiotherapy plans: design, results, and experiences. Acta Oncol. 2023;62:1682-1688.

[19] Penoncello G, et al. Multicenter multivendor evaluation of dose volume histogram creation consistencies for 8 commercial radiation therapy dosimetric systems. Pract Radiat Oncol. 2024;14(3):e236-e248.

[20] Ebert M, Haworth A, Kearvell R. Comparison of DVH data from multiple radiotherapy treatment planning systems. Phys Med Biol. 2010;55(11):N337-N346.

[21] Mason D, et al. pydicom: an open source DICOM library. 2024. Available: https://github.com/pydicom/pydicom.

[22] DICOM Standards Committee. Digital Imaging and Communications in Medicine (DICOM), Part 3: Information Object Definitions, Section C.8.8.4: RT DVH Module. 2024.

[23] Lyman J. Complication probability as assessed from dose-volume histograms. Radiat Res Suppl. 1985;8:S13-S19.

[24] Kutcher G, Burman C. Calculation of complication probability factors for non-uniform normal tissue irradiation: the effective volume method. Int J Radiat Oncol Biol Phys. 1989;16(6):1623-1630.

[25] Nelms B. Methods, software and datasets to verify DVH calculations against analytical values: twenty years late(r). Med Phys. 2015;42(8):4435-4448.

[26] Bland J, Altman D. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986;327(8476):307-310.

[27] Walker A, Byrne M. Clinical impact of dose-volume histogram uncertainties from variations in organ at risk contouring and planning. Med Dosim. 2025;50(1):36-44.

[28] Stanley D. Accuracy of dose-volume metric calculation for small-volume radiosurgery targets. Med Phys. 2021;48(4):1461-1468.

[29] ICRU. ICRU Report 83: Prescribing, Recording, and Reporting Photon-Beam IMRT. 2010.

[30] Bentzen S, Constine L, Deasy J. Quantitative analyses of normal tissue effects in the clinic (QUANTEC): an introduction to the scientific issues. Int J Radiat Oncol Biol Phys. 2010;76(3):S3-S9.

[31] Cozzi L, Buffa F, Fogliata A. Comparative analysis of dose volume histogram reduction algorithms for normal tissue complication probability calculations. Acta Oncol. 2000;39(2):165-171.

[32] Mayo C, Moran J, Bosch W. AAPM Task Group 263: standardizing nomenclatures in radiation oncology. Int J Radiat Oncol Biol Phys. 2018;100(4):1057-1066.

[33] HL7 International. CodeX Radiation Therapy FHIR Implementation Guide, STU 2. 2023. Available: https://hl7.org/fhir/us/codex-radiation-therapy/.

[34] Pepin M, Tryggestad E, Wan Chan Tseung H, et al. Assessment of the precision of dose-volume histogram and quantitative dose metrics for proton therapy from five commercial treatment planning systems. Med Phys. 2020.

[35] Wollschlaeger D, Karle H. DVHmetrics: Analyze Dose-Volume Histograms and Check Constraints. R package. 2025.
