# Statement on the Use of Artificial Intelligence

This statement describes the use of generative AI tools in the preparation of
this repository and the associated manuscript, in line with ICMJE
recommendations and journal policy on AI disclosure.

## Tool used

Claude Code (Anthropic), a large language model based coding assistant.

## What AI was used for

- **Code implementation.** Assistance in writing the DVH computation, caching
  and parameter-extraction modules under `src/`, the phase runner scripts under
  `scripts/`, and the unit tests under `tests/`.
- **Repository and tooling work.** Repository structure, packaging, the
  separation of this public research repository from the institutional
  deployment code, and documentation maintenance.

## What AI was not used for

- **The manuscript was not drafted by AI.** The scientific text, argument and
  framing in `paper/manuscript.md` are the author's own.
- **The analysis and its interpretation were not performed by AI.** The choice
  of validation strategy, the tolerance thresholds, the reading of the results
  and the conclusions drawn from them are the author's own.
- **No data were generated, imputed, altered or selected by AI.** Every reported
  number derives from the measured datasets described in the manuscript.

## Accountability

AI tools are not authors and are not credited as such. The author takes full
responsibility for the entire contents of this repository and the manuscript,
including all AI-assisted code, and has reviewed it accordingly.

## Why this does not affect the validity of the results

The scientific claims in this work do not rest on trust in the code's
provenance. They rest on agreement with external reference data:

- Tolerance thresholds were pre-specified in `config.py` and locked on
  2026-03-27, before any experiment ran.
- Ground truth is external in every comparison — analytical phantom values,
  published reference data, and DVH data embedded by independent treatment
  planning systems — never the software validating itself.
- The full experiment is deterministic and reproducible. Running
  `scripts/run_full_experiment.py` regenerates every reported result from the
  same inputs.

Code that computes an incorrect DVH fails these comparisons regardless of who or
what wrote it. The validation is therefore independent of the authorship of the
implementation.
