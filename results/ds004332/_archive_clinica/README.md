# Archive — anciens résultats (Clinica / test M1 préliminaire)

⚠️ **OBSOLÈTE — ne pas utiliser pour le projet en cours.**

Ces résultats viennent de l'**ancien pipeline Clinica** (recalage affine MNI) et du
**test JDAC M1 préliminaire** (sub-01, n=1, sur skull-strips Clinica). Ils ont été
**abandonnés** : le recalage affine MNI redimensionne le cerveau et fausse l'épaisseur
corticale (constat réunion 12/06). Le pipeline retenu est **N4 + SynthStrip en espace
natif** (voir `../phase2_PREPROC/`).

Conservés uniquement pour référence / traçabilité.

Contenu :
- `compare_phase1_phase2_sub-01.csv` — comparaison épaisseur brut vs prétraité **Clinica** (sub-01).
- `preproc_avant_apres_sub-01_run-02.png` — figure avant/après preprocessing **Clinica**.
- `m1_jdac/` — résultats du test JDAC M1 préliminaire (Clinica, sub-01) :
  `m1_thickness_summary.csv`, `m1_jdac_effect_per_region.csv`, `m1_jdac_effect_hist.png`.

Résultats **actifs** :
- `../phase1_RAW/` — bras RAW (baseline)
- `../phase2_PREPROC/` — preprocessing N4 + SynthStrip
- `../phase3_JDAC/` — JDAC (à venir, après recon-all)
- `../agitation/` — covariable Agitation
