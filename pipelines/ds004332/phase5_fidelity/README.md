# Phase 5 - fidelity analysis workspace

## What this directory is

This directory does **not** contain a completed Phase-5 experiment. It keeps a
small set of reusable tools for understanding the morphometric fidelity of the
methods already evaluated in Phase 4 and, if needed later, for analysing new
JDAC variants with the same measurements.

Phase 4 remains the source of truth for the executed five-condition experiment.
No PMC, BME-X, blinded-QC, automatic gate, or new-model decision is part of the
current workflow.

## Files kept

### `analyze_available_phase4.py`

This is the only script in this directory that has already produced tracked
results. It reads the Phase-4 `recovery_metrics.csv` table and reorganizes it
into:

- a per-subject table;
- a summary by method and motion condition;
- paired method differences;
- one presentation figure and a short report.

It does not create new measurements. Its outputs are a secondary presentation
of Phase-4 results, not an independent experiment and not an automatic decision
about JDAC.

### `build_regional_table.py`

Reusable preparation tool. It can combine raw, preprocessing, JDAC, and future
JDAC-variant FreeSurfer thickness tables into one long regional table. It also
records missing or incomplete scans explicitly.

It has not yet produced the canonical files in this directory. Before using it,
the exact input derivatives and the scientific comparison must be validated.

### `analyze_fidelity.py`

Reusable but unexecuted analysis tool for a complete regional table. It can
calculate subject-level regional errors, paired differences, agreement
statistics, regional tests, and descriptive models.

The presence of this script does not mean that ICC, CCC, mixed models, or FDR
analyses have been approved or completed. Select only the analyses relevant to
the next validated research question.

### `extract_freesurfer_metrics.py`

Generic utility for extracting cortical thickness, surface area, volume, and
completion status from a FreeSurfer `SUBJECTS_DIR`. It can be reused if new JDAC
variants are reconstructed.

### `requirements.txt`

Python dependencies required by these analysis scripts.

## Current source of truth

- Executed experiment: `pipelines/ds004332/phase4_compare_3bras/`
- Executed results: `results/ds004332/phase4_compare_3bras/`
- Secondary Phase-4 summary: `results/ds004332/phase5_fidelity/`

No file in this directory authorizes a new experiment automatically.
