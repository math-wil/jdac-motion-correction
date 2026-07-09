# JDAC Motion Correction

Evaluation of **JDAC** as a motion-artifact correction method for structural brain MRI, using the **ds004332** dataset. The analysis compares cortical thickness estimates from FreeSurfer across raw, preprocessed, and JDAC-corrected T1w images, with motion quantified by the **Agitation** score.

This repository contains the lightweight, versioned part of the project: pipeline scripts, analysis notebooks, CSV summaries, and figures. Large MRI volumes, FreeSurfer outputs, and derived NIfTI files are intentionally kept outside the repository.

Project context: Neuro-iX Lab, ETS. Author: Mathilde Wilfart.

## JDAC Reference

This project evaluates JDAC as introduced in:

- Lintao Zhang, Mengqi Wu, Lihong Wang, David C. Steffens, Guy G. Potter, and Mingxia Liu. "Iterative Learning for Joint Image Denoising and Motion Artifact Correction of 3D Brain MRI." arXiv:2403.08162, 2024. https://arxiv.org/abs/2403.08162
- Original JDAC code repository: https://github.com/goodaycoder/JDAC

The JDAC source code, pretrained models, and original application notebook are not vendored in this repository. Local scripts assume they are available under `~/Documents/jdac/`.

## Research Question

Subject motion during MRI acquisition biases FreeSurfer cortical thickness estimates. This project asks whether JDAC removes that motion-related bias, or whether it mostly smooths images in a way that changes cortical thickness globally.

The main test is:

1. Estimate cortical thickness across several processing conditions.
2. Model thickness as a function of motion, age, and sex.
3. Compare whether the motion-thickness relationship is reduced after JDAC.
4. Check whether any reduction comes with an offset on low-motion scans, which would suggest smoothing rather than targeted correction.

## Processing Conditions

The current analysis uses the **rigid pipeline**, because rigid registration preserves image scale and keeps cortical thickness comparable across conditions.

| Condition | Description | Purpose |
|---|---|---|
| `brut` | Raw T1w image -> FreeSurfer | Baseline reference |
| `preproc` | N4 bias correction + rigid MNI registration + SynthStrip -> FreeSurfer | Effect of preprocessing alone |
| `jdac` | Preprocessed brain -> full JDAC -> FreeSurfer | Main JDAC correction condition |
| `jdac_antiartonly` | JDAC anti-artifact network applied once, without denoiser | Ablation variant |
| `jdac_nodenoise` | JDAC loop without denoiser, anti-artifact step repeated up to four times | Ablation variant |

Earlier native-space and Clinica-affine experiments are kept for reference, but the rigid pipeline is the active analysis path.

## Repository Layout

```text
.
|-- pipelines/
|   `-- ds004332/
|       |-- phase1_RAW/              # FreeSurfer on raw images
|       |-- phase2_PREPROC/          # N4 + rigid registration + SynthStrip
|       |-- phase3_JDAC/             # JDAC inference, variants, and FreeSurfer
|       |-- phase4_compare_3bras/    # Cross-condition thickness analyses
|       |-- agitation/               # Agitation motion score processing
|       |-- utils/                   # Narval transfer and cluster notes
|       `-- _archive_clinica/        # Archived Clinica-affine experiment
`-- results/
    `-- ds004332/
        |-- phase1_RAW/              # Raw thickness and GLM outputs
        |-- phase2_PREPROC/          # Preprocessing QC figures
        |-- phase3_JDAC/             # JDAC figures and GLM outputs
        |-- phase4_compare_3bras/    # Condition-comparison CSVs and figures
        |-- agitation/               # Motion scores
        `-- _archive_clinica/        # Archived Clinica-affine results
```

Each pipeline phase has its own README with more detailed execution notes:

- `pipelines/ds004332/phase1_RAW/README.md`
- `pipelines/ds004332/phase2_PREPROC/README.md`
- `pipelines/ds004332/phase3_JDAC/README.md`
- `pipelines/ds004332/phase4_compare_3bras/README.md`

## External Data

Large files are not tracked in Git.

Expected local layout:

```text
~/Documents/raw_datasets/ds004332/        # BIDS source dataset
~/Documents/derivatives/ds004332/         # Preprocessed, JDAC, and FreeSurfer outputs
~/Documents/jdac/                         # Local JDAC code and pretrained models
```

Lightweight results are tracked under:

```text
results/ds004332/
```

The repository `.gitignore` excludes MRI and FreeSurfer volume formats such as `.nii`, `.nii.gz`, `.mgz`, and `.mgh`.

## Environment and Tools

The project is not packaged as a standalone Python module. Scripts assume the project-specific local and cluster setup used during the analysis.

Main tools used:

- Python analysis stack: `numpy`, `pandas`, `scipy`, `statsmodels`
- MRI tooling: FreeSurfer, SynthStrip, ANTsPy
- JDAC inference: PyTorch, MONAI, nibabel, pretrained JDAC models
- Cluster execution: SLURM on Narval for FreeSurfer batch jobs
- Local environment name used in scripts and notes: `cortical-motion`

Some scripts contain absolute paths for the original workstation or Narval account. Update those paths before running the pipeline on a different machine.

## Pipeline Overview

### Phase 1: Raw FreeSurfer Baseline

Folder: `pipelines/ds004332/phase1_RAW/`

Runs `recon-all` on raw T1w images and fits GLMs of cortical thickness against Agitation, age, and sex.

Tracked outputs:

- `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv`
- `results/ds004332/phase1_RAW/glm_pipeline_b_freesurfer_results.csv`

Current status: 65 of 66 runs are usable. One severe-motion run failed reconstruction.

### Phase 2: Preprocessing-Only Arm

Folder: `pipelines/ds004332/phase2_PREPROC/`

Applies:

1. N4 bias-field correction with ANTsPy.
2. Rigid registration to MNI space.
3. Brain extraction with SynthStrip.
4. FreeSurfer reconstruction on the preprocessed brain.

The rigid transform is used because it rotates and translates the image without scaling it, unlike the earlier affine Clinica experiment.

Current status: 64 of 66 rigid preprocessed runs completed FreeSurfer.

### Phase 3: JDAC Arm and Ablations

Folder: `pipelines/ds004332/phase3_JDAC/`

Runs JDAC on the preprocessed brain images, then runs FreeSurfer on the JDAC outputs. The phase also includes two ablation variants:

- `jdac_antiartonly`
- `jdac_nodenoise`

JDAC internally applies foreground cropping, percentile intensity scaling to `[0, 1]`, and padding. The output is passed directly to FreeSurfer; FreeSurfer performs its own intensity handling.

Current rigid FreeSurfer status:

- `jdac`: 64 of 66
- `jdac_antiartonly`: 66 of 66
- `jdac_nodenoise`: 65 of 66

### Phase 4: Cross-Condition Comparison

Folder: `pipelines/ds004332/phase4_compare_3bras/`

Compares cortical thickness across the five rigid conditions. The main notebook is:

```text
pipelines/ds004332/phase4_compare_3bras/explore_epaisseur_rigide.ipynb
```

The analysis includes:

- thickness offsets on still scans, to detect smoothing;
- within-subject differences between still and motion runs;
- nested model comparisons with and without Agitation;
- image-level metrics such as SSIM and gradient-based similarity.

Useful scripts:

- `compute_image_metrics.py`
- `compare_conditions.py`
- `build_notebook.py`

## Current Findings

The raw condition shows the expected motion bias: higher Agitation is associated with lower cortical thickness.

Full JDAC reduces the motion-thickness relationship more than the other conditions. In the nested model comparison, Agitation no longer clearly predicts thickness after full JDAC. However, full JDAC also lowers thickness on still scans by about 6 percent, indicating a global smoothing or offset effect rather than a purely targeted correction.

The two no-denoiser JDAC variants over-correct. They reverse the motion-thickness slope, with high-motion scans becoming thicker than still scans in the strongest variant.

Preprocessing alone explains a substantial part of the improvement between still and motion scans, so the JDAC effect must be interpreted relative to the `preproc` condition rather than only against raw images.

## Reproducing the Analysis

This repository is mainly an analysis record, not a turn-key pipeline. To reproduce the current analysis on the original setup:

1. Place ds004332 BIDS data under `~/Documents/raw_datasets/ds004332/`.
2. Place or generate derivatives under `~/Documents/derivatives/ds004332/`.
3. Install the MRI and Python dependencies in the `cortical-motion` environment.
4. Run each phase in order, following the phase-specific README files.
5. Use Narval SLURM scripts for FreeSurfer batch jobs where indicated.
6. Run the phase 4 notebook or scripts to regenerate the final comparison tables and figures.

Before running on a new system, review and update absolute paths in the scripts.

## Notes

- The active analysis is the rigid five-condition pipeline.
- Heavy MRI derivatives are intentionally excluded from Git.
- Archived experiments are kept under `_archive_clinica/` for traceability.
- Older unrelated experiments were moved to the archived `motion-analysis` repository.
