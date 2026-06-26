# Phase 3 — JDAC + FreeSurfer

Bras JDAC : le cerveau prétraité (Phase 2) passe par JDAC, puis recon-all. Sert à comparer au brut (Phase 1) et au prétraité (Phase 2) pour voir si JDAC corrige le biais de mouvement.

## Scripts
- `run_jdac.py` : applique JDAC (reproduit `JDAC_Application.ipynb` des auteurs), env `cortical-motion`, lancé depuis `~/Documents/jdac`.
- `all66_subjects.csv` / `all66_subjects_rigid.csv` : listes d'entrées (natif / rigide).
- `recon_all_jdac.sbatch` / `recon_all_jdac_rigid.sbatch` : SLURM, recon-all 2 passes `-noskullstrip` sur les sorties JDAC.
- `fix_jdac_geometry.py` : recale la sortie JDAC sur la grille du cerveau d'entrée (pour QC superposé ; sans effet sur recon-all).
- `glm_pipeline_AvsB.py` : GLM preproc vs jdac (offset + interaction).
- `fig_jdac_steps.py`, `assemble_jdac_fig.py`, `view_jdac_sample.sh`, `view_jdac_sub01.sh` : figures et QC.

## Entrée / sortie JDAC (vérifié dans le code des auteurs)
JDAC applique lui-même `CropForeground` + `ScaleIntensityRangePercentiles(0, 98 → [0,1])` + `DivisiblePad(k=16)`.
- **Entrée** = cerveau skull-strippé, intensité quelconque. Pas de MNI, pas de 1 mm, pas de recalage requis.
- **Sortie** = [0,1] (affine d'origine ; dimensions modifiées par crop + pad).
- **Pas de dénormalisation avant FreeSurfer** : recon-all conforme et rééchelonne lui-même l'intensité (vérifié). La sortie [0,1] est passée directement à recon-all.
- Détail : `research-notes/02_Experiments/jdac/jdac-entrees-sorties.md`.

## État
- JDAC appliqué sur les 66 cerveaux, en natif et en rigide.
- recon-all JDAC rigide : **64/66** (manquent `sub-10_run-03` et `sub-11_run-03` : topologie de surface sur mouvement sévère).
- Comparaison des 3 bras : `../phase4_compare_3bras/`.
