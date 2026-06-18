# Test M1 sur Narval — recon-all avant/après JDAC (sub-01)

Objectif : 6 recon-all (3 « before » sans JDAC + 3 « jdac »), pour isoler l'effet de JDAC
sur l'épaisseur corticale. n=1, pipeline affine, préliminaire.

Les 6 images sont prêtes en local dans :
`~/Documents/derivatives/ds004332/jdac_m1_test/narval_inputs/`
Le script : `recon_all_m1_jdac.sbatch` (ce dossier).

## Étapes (à taper sur ton poste local, puis sur Narval)

### 1. Transférer les 6 images + le script vers Narval
(remplace `mathw@narval.alliancecan.ca` par ta connexion habituelle si différente)
```bash
ssh mathw@narval.alliancecan.ca "mkdir -p ~/projects/ctb-sbouix/mathw/m1_jdac_test/inputs"

rsync -avz ~/Documents/derivatives/ds004332/jdac_m1_test/narval_inputs/ \
  mathw@narval.alliancecan.ca:~/projects/ctb-sbouix/mathw/m1_jdac_test/inputs/

rsync -avz ~/Documents/jdac-motion-correction/pipelines/ds004332/phase3_JDAC/recon_all_m1_jdac.sbatch \
  mathw@narval.alliancecan.ca:~/projects/ctb-sbouix/mathw/m1_jdac_test/
```

### 2. Se connecter et soumettre
```bash
ssh mathw@narval.alliancecan.ca
cd ~/projects/ctb-sbouix/mathw/m1_jdac_test
sbatch recon_all_m1_jdac.sbatch
```
6 jobs partent en parallèle (array 0-5), ~2-3 h chacun.

### 3. Suivre
```bash
squeue -u mathw
# logs : recon_m1_jdac_<JOBID>_<0..5>.out / .err
```
Tu recevras un mail BEGIN/END/FAIL par job.

### 4. Quand terminé : vérifier les 6 done
```bash
for s in sub-01_run-01_before sub-01_run-02_before sub-01_run-03_before \
         sub-01_run-01_jdac sub-01_run-02_jdac sub-01_run-03_jdac; do
  d=~/projects/ctb-sbouix/mathw/freesurfer_m1_jdac_test/$s
  [ -f "$d/scripts/recon-all.done" ] && echo "OK   $s" || echo "FAIL $s"
done
```

### 5. Rapatrier les sorties sur hippocampus (pour l'analyse)
Depuis ton poste local (hippocampus est monté) :
```bash
rsync -avz \
  mathw@narval.alliancecan.ca:~/projects/ctb-sbouix/mathw/freesurfer_m1_jdac_test/ \
  /project/hippocampus/common/mathilde/ds004332/freesurfer_m1_jdac_test/
```
Ensuite je calcule la comparaison épaisseur **before vs jdac** (et vs RAW Phase 1) par région.

## Pourquoi ça ne devrait pas planter (correctifs intégrés)
- `--mem=64G` : évite l'OOM de mri_synthseg (vu à 16G).
- 2 passes FS8 : autorecon1 s'arrête à seg2cc (toléré `|| true`), brainmask = T1.mgz, puis autorecon1/2/3.
- skip si `recon-all.done`, nettoyage des locks `IsRunning`.
- entrées dénormalisées (vraies intensités), 1 mm isotrope RAS, before/jdac de mêmes dimensions par run.

## Si un job FAIL
Lire le `.err` correspondant. Pour relancer un seul indice (ex. 3) après avoir nettoyé :
```bash
rm -rf ~/projects/ctb-sbouix/mathw/freesurfer_m1_jdac_test/sub-01_run-01_jdac
sbatch --array=3 recon_all_m1_jdac.sbatch
```
