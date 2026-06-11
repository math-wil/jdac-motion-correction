# FreeSurfer recon-all sur Narval — ds004332

Objectif : lancer `recon-all` sur les 66 images brutes de ds004332 (22 sujets × 3 runs : still, nodding, shaking) pour obtenir les épaisseurs corticales et évaluer l'effet aval du mouvement et de la correction JDAC.

**Important** : utiliser les images brutes (`acq-mpragepmcoff rec-wore`), pas les sorties JDAC (skull-stripped et normalisées [0,1], hors domaine FreeSurfer).

---

## Étape 1 : Transfert des données vers Narval

Depuis la machine locale :

```bash
bash transfer_to_narval.sh av62870
```

Le script copie uniquement les fichiers `sub-XX_acq-mpragepmcoff_rec-wore_run-0X_T1w.nii` vers `$SCRATCH/ds004332/` sur Narval.

---

## Étape 2 : Connexion à Narval et préparation

```bash
ssh av62870@narval.computecanada.ca

# Vérifier que les données sont arrivées
ls $SCRATCH/ds004332/sub-01/anat/

# Copier le script de soumission
# (depuis ~/motion-analysis/pipelines/ds004332/narval/ si le repo est cloné sur Narval)
```

---

## Étape 3 : Soumission du job array

```bash
cd $SCRATCH
sbatch submit_recon_all_ds004332.sh
```

66 jobs sont soumis (array 0-65). Chaque job traite un couple (sujet, run) :
- index // 3 → sujet (sub-01 à sub-22)
- index % 3 → run (run-01, run-02, run-03)

Durée estimée par job : 12-24h (recon-all parallèle, 4 CPUs, ~16 Go RAM).  
Les jobs peuvent tourner simultanément selon la disponibilité du cluster.

---

## Étape 4 : Suivi

```bash
# État des jobs
squeue -u av62870

# Logs (un fichier par job)
tail -f recon_ds004332_<jobid>_<arrayid>.out

# Vérifier les jobs terminés
ls $SCRATCH/freesurfer_ds004332/*/scripts/recon-all.done | wc -l
```

---

## Étape 5 : Résultats

Les surfaces et épaisseurs corticales sont dans `$SCRATCH/freesurfer_ds004332/<sub>_<run>/`. Les fichiers d'intérêt pour l'analyse GLM aval (Pipeline A du projet) :

```
freesurfer_ds004332/
└── sub-01_run-01/
    ├── surf/lh.thickness, rh.thickness
    ├── label/lh.aparc.annot, rh.aparc.annot
    └── stats/lh.aparc.stats, rh.aparc.stats
```

Après la fin des jobs, rapatrier les stats :

```bash
rsync -avz av62870@narval.computecanada.ca:scratch/freesurfer_ds004332/ \
    ~/Documents/Results/ds004332/freesurfer/
```
