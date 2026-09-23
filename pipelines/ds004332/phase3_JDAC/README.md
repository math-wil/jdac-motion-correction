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

### Pilote proposé : entrée minimale de JDAC (non exécuté)

**Question.** N4 et le recalage rigide, ajoutés dans notre chaîne, changent-ils les
résultats obtenus avec les mêmes poids JDAC ? Ce pilote ne cherche pas à classer
des correcteurs. L'article JDAC décrit une extraction du cerveau et une mise à
l'échelle des intensités entre 0 et 1 ; le script d'inférence utilisé ici fait
lui-même le recadrage, la mise à l'échelle et le padding, mais pas l'extraction.

- Six acquisitions pré-sélectionnées, sans regarder le résultat du nouveau bras :
  `sub-17`, `sub-13`, `sub-14`, chacun en `run-01` et `run-03`. Source BIDS :
  `acq-mpragepmcoff_rec-wore`, comme dans `phase2_PREPROC/preproc.py`.
- Bras minimal : T1w brut → SynthStrip seul, sans N4 ni recalage → JDAC complet
  avec les mêmes poids et paramètres que le bras rigide. Référencer le T1w brut
  existant sans le recopier ; conserver
  le masque, le cerveau extrait, l'entrée effectivement remise au réseau et la
  sortie, avec leur géométrie et leur provenance. Les nouveaux dérivés iront
  dans un dossier `jdac_minimal_pilot/` hors Git ; aucun ancien résultat écrasé.
- Contrôle initial, **avant FreeSurfer** : masque (notamment cortex et cervelet),
  grille/affine, remise en place après crop/padding, intensités, montage sur les
  mêmes coupes et même fenêtre. Si un masque ou une géométrie échoue, arrêter et
  documenter le cas au lieu de l'interpréter comme un échec du réseau.
- Si le contrôle passe : FreeSurfer avec la même version et la même procédure
  à deux passes avec `-noskullstrip` pour le cerveau minimal, son entrée
  normalisée sans réseau et la sortie JDAC. Produire également l'entrée
  normalisée **sans réseau** du bras rigide, afin de comparer l'ajout de JDAC
  sur une échelle d'intensité identique dans les deux bras. Les reconstructions
  du cerveau rigide non normalisé et de sa sortie JDAC existent déjà.
  Documenter `-cw256` selon le champ de vue, car le bras rigide utilise une
  grille MNI élargie. Comparer épaisseur (mm), surface (mm²), volume cortical
  et sous-cortical (mm³), échecs et QC. Contraste principal : JDAC moins son
  entrée normalisée, calculé séparément pour le bras minimal et le bras rigide.
  `run-01` vérifie la préservation du scan presque immobile ; `run-03` vérifie
  le scan bougé. Le sujet est l'unité de lecture ; six images ne démontrent ni
  équivalence ni supériorité générale.
- Métriques d'image : définir un support cérébral et une transformation
  d'intensité communs, indépendants de la sortie testée. Ne pas réutiliser
  directement `compute_image_metrics.py` pour conclure : ce script renormalise
  chaque image séparément et choisit une boîte dépendante de la paire.
- Décision après pilote : si minimal et rigide divergent, tester N4 et rigide
  séparément avant de généraliser. L'ancien bras natif N4+SynthStrip est cité
  historiquement, mais ses images n'étaient pas présentes dans les dérivés du
  PC labo lors de la vérification du 23 septembre 2026 ; vérifier Narval avant
  de compter dessus. Aucun calcul n'a été lancé pour ce pilote à cette date.

### Figure de réunion sur un même cerveau

- `fig_sub19_three_runs.py` produit un montage de `sub-19` sur `run-01`,
  `run-02` et `run-03`, pour l'entrée preproc, JDAC complet, anti-artefact 1×
  et sans débruiteur 4×. Les trois lignes représentent le même cerveau avec
  un mouvement croissant.

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
