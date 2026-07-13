# Instructions pour les agents (Claude Code et Codex) — jdac-motion-correction

`CLAUDE.md` (lu par Claude) et `AGENTS.md` (lu par Codex) sont **identiques et doivent le rester**. Toute modification de l'un est reportée sur l'autre.

Dépôt de code de la maîtrise recherche de Mathilde Wilfart (ÉTS, labo Neuro-iX). Évaluation de la correction du mouvement en IRM structurelle (JDAC et variantes) sur **ds004332**, jugée sur l'épaisseur corticale FreeSurfer **en aval**. Objectif du mémoire : **développer une méthode de correction novatrice**. Le vault de notes est le dépôt `research-notes` (état courant dans son `STATUS.md` ; provenance des données dans `03_Concepts/roadmap-donnees-ds004332.md`).

---

## Structure

- `pipelines/ds004332/phaseN_*/` — code par phase : `phase1_RAW`, `phase2_PREPROC`, `phase3_JDAC` (+ variantes sans débruiteur), `phase4_compare_3bras` (comparaison des 5 conditions, cadre A–E).
- `results/ds004332/` — résultats **légers** versionnés (CSV, figures), rangés par phase.
- Données lourdes **hors git** : `~/Documents/derivatives/ds004332/` (preproc, JDAC, épaisseurs), `~/Documents/raw_datasets/ds004332/` (BIDS), `~/Documents/jdac/` (code + poids JDAC). Le `.gitignore` exclut `.nii/.nii.gz/.mgz`.

## Les 5 conditions (pipeline rigide)

brut · preproc (N4 + recalage rigide MNI + SynthStrip) · jdac (complet) · jdac_antiartonly (anti-artefact ×1, sans débruiteur) · jdac_nodenoise (×4, sans débruiteur).

## Analyse (phase 4)

Notebook `explore_epaisseur_rigide.ipynb`, généré par `build_notebook.py`. Scripts : `compute_image_metrics.py` (bloc D, métriques image), `compute_recovery.py` (bloc E, récupération régionale), `compare_conditions.py` (équivalent CLI). Cadre A→E, tout en **mm**, chaque sortie accompagnée d'une analyse chiffrée et traçable.

## Environnements et calcul

- Local : env conda `cortical-motion` (FreeSurfer 8, ANTsPy, MONAI, PyTorch). Lancer Jupyter via `~/miniconda3/envs/cortical-motion/bin/jupyter-lab`.
- Calcul lourd : **Narval** (Compute Canada, compte ctb-sbouix, **MFA obligatoire**). Un agent ne peut pas s'y connecter : fournir les commandes à coller, la personne les exécute ; rapatriement par `rsync` depuis un terminal local.
- Les scripts contiennent des chemins locaux ou Narval ; les ajuster avant de tourner sur une autre machine.

## Registre et méthode

- Français, sobre, concis. **Unités (mm) partout, chaque tableau ou résultat expliqué** (question posée, ce que ça dit) ; aucune valeur non traçable.
- **Pas de nom propre du directeur** dans le code, les commentaires ou les commits (dépôt public).
- Expliquer un script avant de le modifier. Pas de code en trop, pas d'analyse inutile.
- Diagnostiquer de bout en bout avant de lancer ; anticiper les erreurs plutôt que promettre « sans erreur ».

## Git

`git add -A`, commit descriptif en français, `git push`. **Jamais de co-author IA (Claude/Codex) dans les commits.** Branche `main`. Entre deux machines : `git pull` en arrivant, `git push` en partant.
