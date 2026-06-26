# pipelines/ds004332

Code du pipeline d'évaluation de JDAC sur ds004332, organisé par phase. Chaque phase a son propre README.

| Dossier | Rôle |
|---|---|
| `phase1_RAW/` | recon-all sur images brutes + GLM épaisseur ~ mouvement (bras RAW). |
| `phase2_PREPROC/` | prétraitement N4 + recalage rigide + SynthStrip, puis recon-all (bras PREPROC). |
| `phase3_JDAC/` | JDAC sur cerveaux prétraités, puis recon-all (bras JDAC). |
| `phase4_compare_3bras/` | comparaison des 3 bras : descriptif, niveaux de mouvement, pentes intra-sujet, modèle mixte. Notebooks `explore_3bras_*`. |
| `agitation/` | calcul du score de mouvement Agitation (covariable des GLM). |
| `utils/` | transfert vers Narval, guide d'exécution sur le cluster. |

Conventions : les scripts SLURM (`.sh` / `.sbatch`) tournent sur **Narval** ; JDAC et les analyses tournent en **local** (env conda `cortical-motion`). Les sorties lourdes (NIfTI, FreeSurfer) restent hors dépôt ; les résultats légers vont dans `results/`.
