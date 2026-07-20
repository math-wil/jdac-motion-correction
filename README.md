# JDAC Motion Correction

Evaluation of **JDAC** as a retrospective motion-artifact correction method for structural brain MRI, using the **ds004332** controlled-motion dataset. The project tests whether JDAC improves downstream **FreeSurfer cortical thickness** measurements, not only whether corrected images look cleaner.

This repository contains the lightweight, versioned part of the project: pipeline scripts, notebooks, CSV summaries, and figures. Large MRI volumes, FreeSurfer subject directories, JDAC outputs, and derived NIfTI files are intentionally kept outside Git.

Project context: Neuro-iX Lab, ETS. Author: Mathilde Wilfart.

## JDAC Reference

This project evaluates JDAC as introduced in:

- Lintao Zhang, Mengqi Wu, Lihong Wang, David C. Steffens, Guy G. Potter, and Mingxia Liu. "Iterative Learning for Joint Image Denoising and Motion Artifact Correction of 3D Brain MRI." arXiv:2403.08162, 2024. https://arxiv.org/abs/2403.08162
- Original JDAC code repository: https://github.com/goodaycoder/JDAC

The JDAC source code, pretrained models, and original application notebook are not vendored here. Local scripts assume they are available separately under the local JDAC checkout used during the analysis.

## Research Question

Subject motion during T1w MRI acquisition biases FreeSurfer cortical thickness estimates. The central question is:

> Does JDAC reduce the motion-related cortical-thickness bias while preserving anatomical fidelity, or does it mainly smooth images and introduce a new thickness offset?

The project separates three effects that can look similar in a simple average:

- **motion bias**: higher Agitation score predicts lower measured thickness;
- **offset / smoothing**: a method lowers thickness even on still scans, where there is no motion to correct;
- **over-correction**: the motion-thickness relationship reverses and high-motion scans become thicker than still scans.

## Current Analysis Path

The active analysis is the **rigid five-condition pipeline** on ds004332. Earlier native-space and Clinica-affine experiments are archived or kept as historical references only.

Rigid registration is used because it rotates/translates the image without scaling the brain. This keeps cortical thickness comparable across conditions, unlike the earlier Clinica-affine experiment.

| Condition | Processing path | Role |
|---|---|---|
| `brut` | raw T1w -> FreeSurfer | baseline motion bias |
| `preproc` | N4 bias correction + rigid MNI registration + SynthStrip -> FreeSurfer | preprocessing-only control |
| `jdac` | preprocessed brain -> full JDAC -> FreeSurfer | main JDAC condition |
| `jdac_antiartonly` | JDAC anti-artifact network once, without denoiser -> FreeSurfer | ablation variant |
| `jdac_nodenoise` | JDAC loop without denoiser, anti-artifact repeated up to four times -> FreeSurfer | ablation variant |

FreeSurfer completion status for the rigid analysis:

| Condition | Successful recon-all runs | Missing / failed runs |
|---|---:|---|
| `brut` | 65 / 66 | `sub-01_run-03` |
| `preproc` | 64 / 66 | `sub-10_run-01`, `sub-11_run-03` |
| `jdac` | 64 / 66 | `sub-10_run-03`, `sub-11_run-03` |
| `jdac_antiartonly` | 66 / 66 | none |
| `jdac_nodenoise` | 65 / 66 | `sub-22_run-01` |

## Repository Layout

```text
.
|-- pipelines/
|   `-- ds004332/
|       |-- phase1_RAW/              # FreeSurfer on raw images
|       |-- phase2_PREPROC/          # N4 + rigid registration + SynthStrip
|       |-- phase3_JDAC/             # JDAC inference, ablations, and FreeSurfer scripts
|       |-- phase4_compare_3bras/    # Historical folder name; now the 5-condition comparison
|       |-- phase5_fidelity/         # aseg.stats volumes and central-structure fidelity
|       |-- agitation/               # Agitation motion-score processing
|       |-- utils/                   # Narval transfer and cluster notes
|       `-- _archive_clinica/        # Archived Clinica-affine experiment
`-- results/
    `-- ds004332/
        |-- phase1_RAW/              # Raw thickness and GLM outputs
        |-- phase2_PREPROC/          # Preprocessing QC figures
        |-- phase3_JDAC/             # JDAC figures and GLM outputs
        |-- phase4_compare_3bras/    # Current 5-condition summaries plus image/recovery metrics
        |-- agitation/               # Motion scores
        `-- _archive_clinica/        # Archived Clinica-affine results
```

Phase-level documentation:

- `pipelines/ds004332/phase1_RAW/README.md`
- `pipelines/ds004332/phase2_PREPROC/README.md`
- `pipelines/ds004332/phase3_JDAC/README.md`
- `pipelines/ds004332/phase4_compare_3bras/README.md`
- `pipelines/ds004332/phase5_fidelity/README.md`

## External Data

Large files are not tracked in Git.

Expected local layout on the analysis workstation:

```text
~/Documents/raw_datasets/ds004332/        # BIDS source dataset
~/Documents/derivatives/ds004332/         # preprocessed, JDAC, thickness, and FreeSurfer-derived outputs
~/Documents/jdac/                         # JDAC source code and pretrained models
```

Tracked lightweight results are under:

```text
results/ds004332/
```

The repository `.gitignore` excludes MRI and FreeSurfer volume formats such as `.nii`, `.nii.gz`, `.mgz`, and `.mgh`.

## Environment and Tools

The project is an analysis record, not a packaged Python module. Scripts assume the workstation and Narval setup used during the project.

Main tools:

- Python analysis stack: `numpy`, `pandas`, `scipy`, `statsmodels`
- MRI tooling: FreeSurfer 8, SynthStrip, ANTsPy
- JDAC inference: PyTorch, MONAI, nibabel, pretrained JDAC models
- Cluster execution: SLURM on Narval for FreeSurfer batch jobs
- Local environment name used in notes and scripts: `cortical-motion`

Some scripts still rely on the original local data layout. Before rerunning on a different machine, review paths and make sure `raw_datasets/` and `derivatives/` are available.

## Pipeline Overview

### Phase 1: Raw FreeSurfer Baseline

Folder: `pipelines/ds004332/phase1_RAW/`

Runs `recon-all` on raw T1w images and fits GLMs of cortical thickness against Agitation, age, and sex.

Tracked outputs:

- `results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv`
- `results/ds004332/phase1_RAW/glm_pipeline_b_freesurfer_results.csv`

Current result: the raw condition shows the expected motion bias. Higher Agitation predicts lower cortical thickness globally (`coef = -0.0662 mm/point`, `p = 6.6e-05` in the current phase 4 M0/M1 comparison).

### Phase 2: Preprocessing-Only Condition

Folder: `pipelines/ds004332/phase2_PREPROC/`

Applies:

1. N4 bias-field correction with ANTsPy.
2. Rigid registration to MNI space on an enlarged grid.
3. Brain extraction with SynthStrip.
4. FreeSurfer reconstruction on the preprocessed brain.

The preprocessing-only condition is essential because part of the apparent improvement already comes from preprocessing, independently of JDAC.

### Phase 3: JDAC and Ablations

Folder: `pipelines/ds004332/phase3_JDAC/`

Runs full JDAC and two no-denoiser ablations on the rigid preprocessed brains, then passes outputs to FreeSurfer.

JDAC internally applies foreground cropping, percentile intensity scaling to `[0, 1]`, and padding. In the current rigid workflow, the JDAC output is passed directly to FreeSurfer; FreeSurfer performs its own intensity handling.

The two ablations test whether removing the denoiser avoids the smoothing seen in full JDAC:

- `jdac_antiartonly`: one anti-artifact pass;
- `jdac_nodenoise`: anti-artifact network repeated in the JDAC loop without the denoiser.

Current result: both no-denoiser variants keep visually sharper images, but they over-correct the cortical-thickness measurements.

### Phase 4: Five-Condition Comparison

Folder: `pipelines/ds004332/phase4_compare_3bras/`

The folder name is historical. It now contains the **five-condition** rigid comparison.

Main executed notebook:

```text
pipelines/ds004332/phase4_compare_3bras/explore_epaisseur_rigide.ipynb
```

Current tracked phase 4 outputs:

| File | Meaning |
|---|---|
| `a_immobiles_offset.csv` | Section A: thickness and offset on still scans |
| `b_ecart_immobile_bouge.csv` | Section B: within-subject still-vs-shaking improvement / over-correction counts |
| `c_m0_vs_m1.csv` | Section C: nested M0 vs M1 models by condition |
| `d_image_metrics_summary.csv` | Section D: image and gradient SSIM summaries |
| `e_recovery_summary.csv` | Section E: regional recovery-to-truth summary |
| `image_metrics.csv` | Detailed per-subject image metrics |
| `recovery_metrics.csv` | Detailed per-subject regional recovery metrics |

The older three-condition `e1/e2/e3` CSVs were removed because they no longer matched the active five-condition analysis. The old `compare_3bras.png` is also obsolete and should not be used.

## Current Findings

The latest coherent results are the executed phase 4 notebook plus the detailed `image_metrics.csv` and `recovery_metrics.csv` files.

### A. Still scans: offset / smoothing

On still scans, there is no motion to correct. Any thickness change is therefore an offset introduced by the processing condition.

| Condition | Still thickness (mm) | Offset vs raw (mm) | Offset vs raw (%) |
|---|---:|---:|---:|
| `brut` | 2.595 | +0.000 | +0.0 |
| `preproc` | 2.518 | -0.077 | -3.0 |
| `jdac` | 2.435 | -0.160 | -6.2 |
| `jdac_antiartonly` | 2.418 | -0.177 | -6.8 |
| `jdac_nodenoise` | 2.378 | -0.216 | -8.3 |

Interpretation: all treatments thin a still brain. Full JDAC and the no-denoiser variants therefore introduce non-trivial smoothing or offset.

### B. Still vs shaking within subject

Preprocessing and full JDAC reduce the still-vs-shaking thickness gap in most usable subjects:

- `preproc`: 16 improved subjects out of 19;
- `jdac`: 15 improved subjects out of 19.

The no-denoiser variants mostly over-correct:

- `jdac_antiartonly`: 14 over-corrected subjects out of 21;
- `jdac_nodenoise`: 19 over-corrected subjects out of 20.

Interpretation: the no-denoiser images can look sharper, but the cortical thickness signal reverses direction.

### C. Does motion still predict thickness?

Nested models compare M0 (`thickness ~ age + sex`) against M1 (`M0 + Agitation`), by condition.

| Condition | n acquisitions | Agitation coefficient (mm/point) | p-value | Reading |
|---|---:|---:|---:|---|
| `brut` | 65 | -0.0662 | 6.6e-05 | motion thins |
| `preproc` | 64 | -0.0308 | 0.028 | motion still thins |
| `jdac` | 64 | -0.0203 | 0.11 | best decoupling |
| `jdac_antiartonly` | 66 | +0.0253 | 0.081 | over-correction trend |
| `jdac_nodenoise` | 65 | +0.0923 | 7.1e-07 | strong over-correction |

Interpretation: full JDAC is the only condition where Agitation no longer clearly improves the model while the coefficient remains close to zero. However, this comes with the still-scan offset above.

### D. Image metrics

Against the clean scan, full JDAC slightly improves image SSIM at high motion (`0.561` vs `0.540` for preproc), but image SSIM alone cannot distinguish correction from smoothing.

On gradient SSIM against the clean scan, no correction clearly beats preproc at high motion:

- `preproc`: 0.489;
- `jdac`: 0.487;
- `jdac_nodenoise`: 0.479.

Using an intra-condition reference to reduce intensity-bias effects, full JDAC improves high-motion gradient similarity (`0.529` vs `0.489` for preproc). `jdac_nodenoise` does not (`0.471`).

Interpretation: full JDAC has a real image-level motion-reduction signal, but the sharper no-denoiser output does not correspond to cleaner anatomical contours.

### E. Regional cortical-thickness fidelity

The decisive question is whether the moved scan gets closer to the subject's true regional thickness, defined here as the raw still scan.

For shaking scans:

| Condition | Remaining motion pattern (mm) | Error to truth (mm) | Offset on still scan (mm) |
|---|---:|---:|---:|
| `brut` | 0.315 | 0.315 | 0.000 |
| `preproc` | 0.271 | 0.311 | 0.087 |
| `jdac` | 0.317 | 0.373 | 0.232 |
| `jdac_antiartonly` | 0.341 | 0.314 | 0.193 |
| `jdac_nodenoise` | 0.415 | 0.336 | 0.255 |

Interpretation: no condition brings the moved scan's regional thickness below the raw error-to-truth baseline. Full JDAC and `jdac_nodenoise` move farther from the truth because of offset. Net of offset, `jdac_nodenoise` worsens the regional pattern.

## Bottom Line

Full JDAC reduces the apparent motion-thickness relationship and has some image-level motion-reduction signal, but it also introduces a substantial negative thickness offset on still scans. The no-denoiser variants do not solve this: they over-correct the thickness signal and do not improve regional fidelity.

The current conclusion is therefore:

> None of the tested JDAC conditions restores cortical thickness to the subject's still-scan regional truth. The best visual or statistical decoupling does not yet translate into anatomically faithful FreeSurfer measurements.

Phase 5 now provides the executed `aseg.stats` analysis. On the almost-still `run-01`, full JDAC shifts major FreeSurfer volumes despite there being little motion to correct: median CortexVol is about 14% lower, TotalGrayVol about 10.8% lower, CerebralWhiteMatterVol about 11% higher, and CSF about 10.2% higher than raw. Across moved scans, none of the tested conditions shows consistent structure-wise recovery toward the subject-specific `raw/run-01` reference. Agitation remains associated with many anatomical volumes after FDR correction.

The reduced-strength pilot described in the former July 15 plan is no longer the active roadmap. The next decision is to verify the central thalamic observation and use the cortical and volumetric evidence to choose a defensible development baseline: JDAC modification or retraining, inclusion of clean images and an identity constraint, motion estimation as an input or auxiliary target, and anatomically sensitive losses. These are research options to compare, not approved experiments.

## Reproducing the Analysis

This repository is mainly an analysis record. To reproduce the current analysis on the original setup:

1. Place ds004332 BIDS data under `~/Documents/raw_datasets/ds004332/`.
2. Place or generate derivatives under `~/Documents/derivatives/ds004332/`.
3. Install the MRI and Python dependencies in the `cortical-motion` environment.
4. Run each phase in order, following the phase-specific README files.
5. Use Narval SLURM scripts for FreeSurfer batch jobs where indicated.
6. Run the phase 4 notebook or scripts to regenerate the cortical A–E comparison.
7. Run `pipelines/ds004332/phase5_fidelity/explore_aseg_rigide.ipynb` for the volumetric analysis; the committed HTML is the executed snapshot.

Before running on a new system, review absolute paths in scripts and update them to the local data layout.

## Notes

- The active analysis is the rigid five-condition pipeline.
- Heavy MRI derivatives are intentionally excluded from Git.
- Archived experiments are kept under `_archive_clinica/` for traceability.
- Older unrelated experiments were moved to the archived `motion-analysis` repository.
