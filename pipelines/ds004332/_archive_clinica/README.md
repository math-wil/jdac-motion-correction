# Archive — anciens scripts (Clinica / test M1 préliminaire)

⚠️ **OBSOLÈTE — ne pas relancer.** Gardés pour référence uniquement.

Ces scripts appartiennent à l'**ancien pipeline Clinica** (recalage affine MNI) ou au
**test JDAC M1 préliminaire** (sub-01, n=1). Abandonnés au profit du pipeline
**N4 + SynthStrip natif**. Ils sont sortis des dossiers actifs pour éviter toute
confusion (ex. `recon_all_m1_jdac.sbatch` vs le `recon_all_jdac.sbatch` actif).

Contenu :
- `compare_raw_vs_preproc.py` — comparaison épaisseur brut vs prétraité **Clinica** (chemins codés en dur vers les sorties Clinica, ne tourne plus tel quel).
- `recon_all_m1_jdac.sbatch` — recon-all du test M1 (6 jobs, sub-01).
- `m1_sub01_subjects.csv` — CSV du test M1 (pointe vers les skull-strips Clinica).
- `view_m1_sub01.sh` — visualiseur FSLeyes du test M1.
- `M1_SUBMIT.md` — doc du test M1.

Scripts **actifs** correspondants :
- preprocessing : `../phase2_PREPROC/` (preproc.py, recon_all_preproc.sh, fig/montage/view)
- JDAC 66 : `../phase3_JDAC/` (run_jdac.py, recon_all_jdac.sbatch, all66_subjects.csv)
