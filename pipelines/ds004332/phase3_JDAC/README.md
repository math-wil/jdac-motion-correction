# Phase 3 — JDAC + FreeSurfer

Bras JDAC : le cerveau prétraité (Phase 2) passe par JDAC, puis recon-all. Sert à comparer au brut (Phase 1) et au prétraité (Phase 2) pour voir si JDAC corrige le biais de mouvement.

## Scripts

### Pilote actuel : intensité de la correction anti-artefact

- `pilot_jdac_strength.csv` sélectionne 3 sujets couvrant un mouvement faible,
  moyen et fort, avec leur scan immobile `run-01` et shaking `run-03`.
- `run_strength_pilot.sh` lance JDAC avec une force anti-artefact de `0.25` puis
  `0.50` sur ces 6 images. La force `1.00` correspond au JDAC rigide déjà
  calculé et n'est donc pas relancée.
- Les sorties sont séparées dans
  `~/Documents/derivatives/ds004332/jdac_strength_pilot/`; aucune sortie JDAC
  existante n'est écrasée.

- `run_jdac.py` : applique JDAC complet (reproduit `JDAC_Application.ipynb` des auteurs), env `cortical-motion`, lancé depuis `~/Documents/jdac`.
- `run_jdac_nodenoise.py` : variantes sans débruiteur, `jdac_antiartonly` (anti-artefact ×1) et `jdac_nodenoise` (boucle ×4, débruiteur remplacé par l'identité).
- `all66_subjects.csv` / `all66_subjects_rigid.csv` : listes d'entrées (natif / rigide).
- `recon_all_jdac.sbatch` / `recon_all_jdac_rigid.sbatch` : SLURM, recon-all 2 passes `-noskullstrip` sur les sorties JDAC.
- `recon_all_jdac_variant_rigid.sbatch` : même protocole recon-all sur les sorties des deux variantes.
- `fix_jdac_geometry.py` : recale la sortie JDAC sur la grille du cerveau d'entrée (pour QC superposé ; sans effet sur recon-all).
- `glm_pipeline_AvsB.py` : GLM preproc vs jdac (offset + interaction).
- `fig_jdac_steps.py`, `assemble_jdac_fig.py`, `fig_jdac_variants_rigid.py`, `montage_jdac_variants.py`, `view_jdac_sample.sh`, `view_jdac_sub01.sh`, `view_jdac_nodenoise_qc.sh` : figures et QC.

## Entrée / sortie JDAC (vérifié dans le code des auteurs)
JDAC applique lui-même `CropForeground` + `ScaleIntensityRangePercentiles(0, 98 → [0,1])` + `DivisiblePad(k=16)`.
- **Entrée** = cerveau skull-strippé, intensité quelconque. Pas de MNI, pas de 1 mm, pas de recalage requis.
- **Sortie** = [0,1] (affine d'origine ; dimensions modifiées par crop + pad).
- **Pas de dénormalisation avant FreeSurfer** : recon-all conforme et rééchelonne lui-même l'intensité (vérifié). La sortie [0,1] est passée directement à recon-all.
- Détail : `research-notes/02_Experiments/jdac/jdac-entrees-sorties.md`.

## État
- JDAC appliqué sur les 66 cerveaux, en natif et en rigide ; variantes appliquées sur les 66 cerveaux rigides.
- recon-all rigide : jdac **64/66** (manquent `sub-10_run-03`, `sub-11_run-03` : topologie de surface sur mouvement sévère), `jdac_antiartonly` **66/66**, `jdac_nodenoise` **65/66** (manque `sub-22_run-01`, cervelet dans le volume rempli).
- Épaisseur des variantes extraite sur Narval (`aparcstats2table`) et rapatriée dans `derivatives/ds004332/thickness_jdac_{antiartonly,nodenoise}_rigid/`.
- Comparaison des 5 conditions : `../phase4_compare_3bras/`.
