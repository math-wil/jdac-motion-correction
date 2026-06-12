# Phase 3 — JDAC + FreeSurfer

## Pipeline retenu (Option 2, rigide)

```
T1 brut → FreeSurfer                                       (Phase 1, mesure brut)
T1 brut → N4 + recalage RIGIDE + crop → SynthStrip → FreeSurfer   (Phase 2, mesure prétraité)
        → SynthStrip → JDAC → dénormaliser → FreeSurfer    (Phase 3, mesure corrigée JDAC)
```
Le recalage **rigide** (6 DOF) préserve l'échelle → épaisseurs comparables entre phases (l'affine 12 DOF de Clinica les faussait, +0.14 mm). Le rigide ne sert PAS à JDAC, il sert à la comparabilité FreeSurfer.

## Entrée / sortie JDAC (vérifié dans les notebooks officiels des auteurs)

Source : `jdac/JDAC_Application.ipynb` (cellule 25) + notebooks d'entraînement. JDAC applique lui-même :
`CropForeground` + `ScaleIntensityRangePercentiles(0, 98 → [0,1])` + `DivisiblePad(k=16)`.
- **Entrée = cerveau skull-strippé**, intensité quelconque. **Pas** de MNI, **pas** de 1 mm, **pas** de recalage requis (`Spacingd` importé mais jamais utilisé). Inutile de normaliser/padder à la main.
- **Sortie = [0,1]** (sauvée avec l'affine d'origine ; dimensions modifiées par crop+pad).
- Détail : note `research-notes/02_Experiments/jdac/jdac-entrees-sorties.md`.

## Test rapide JDAC sur sub-01 (skull-strips existants)

But : valider la chaîne JDAC → dénorm → FreeSurfer de bout en bout sur 1 sujet, sans attendre le pipeline rigide final. (Utilise les skull-strips actuels, issus du preprocessing affine — donc pas l'espace final, juste pour tester la machinerie.)

1. **Entrée** : `*_clinica_synthstrip_brain.nii.gz` de sub-01 (run-01/02/03).
   - local : `derivatives/ds004332/clinica_preproc/...`
   - hippocampus : `/project/hippocampus/common/mathilde/ds004332/phase2_preproc/sub-01_run-0X/`
2. **JDAC** : inférence avec les modèles pré-entraînés (`jdac/PretrainedModels`), logique = `JDAC_Application.ipynb` cellule 27. **À VÉRIFIER lundi** : chemins exacts des modèles + où lancer (Narval GPU vs CPU ~5-15 min/image).
3. **Dénormalisation** : la sortie est en [0,1] (percentiles 0–98). Inversion ≈ `img × (p98 − p0) + p0`, avec p0/p98 du cerveau d'entrée. **À VÉRIFIER** : la formule exacte + la **géométrie** de la sortie (CropForeground + DivisiblePad changent les dimensions ; l'affine sauvée est celle d'origine).
4. **FreeSurfer** : recon-all 2 passes `-noskullstrip` sur le cerveau dénormalisé. Réutiliser `../phase2_preproc_freesurfer/phase2_freesurfer.sh` (adapter l'entrée).
5. **Comparer** l'épaisseur Phase 3 (JDAC) vs Phase 2 (prétraité) vs Phase 1 (brut), via `../phase2_preproc_freesurfer/compare_phase1_phase2.py` (à étendre à une 3e colonne JDAC).

## À construire (pipeline final, Option 2)

- Script preprocessing **rigide** : N4 + recalage rigide (ANTsPy) + crop + SynthStrip, en remplacement du Clinica affine. (Décider : recalage rigide vers quoi — gabarit ou image de référence.)
- Rejouer Phase 2 et Phase 3 sur les 22 sujets avec ce preprocessing, puis GLM Pipeline A vs B.

## Points vérifiés vs à confirmer

- ✅ Vérifié (code des auteurs) : entrée skull-strippée, normalisation percentiles interne, pad ÷16, pas de recalage, sortie [0,1].
- ⚠️ À confirmer lundi : commande/chemins JDAC exacts, formule de dénormalisation + géométrie, choix du recalage rigide.
