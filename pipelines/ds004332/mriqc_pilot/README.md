# Pilote MRIQC `sub-19`

Ce dossier documente le pilote technique exécuté le 14 septembre 2026 avec MRIQC 23.1.0. Il sert à déterminer quelles IQM restent interprétables sur les sorties brain-only de JDAC ; il ne constitue pas une comparaison scientifique sur la cohorte.

## Fichiers

- `run_mriqc_pilot.sh` : reconstitution vérifiée de la préparation BIDS et des deux commandes Apptainer. Le lancement initial avait été fait directement dans le terminal ; ce fichier n'est donc pas présenté comme le script original.
- `extract_iqm.py` : rassemble les 15 JSON produits par MRIQC et qualifie le rôle méthodologique des IQM brain-only. Il ne recalcule pas les métriques.
- `results/ds004332/mriqc_pilot/` : table numérique, qualification et géométrie des entrées. Les images, rapports HTML, journaux et répertoires de travail restent hors Git.

## Pourquoi deux exécutions ?

Les 12 images `preproc`, JDAC, antiartefact ×1 et ×4 sont brain-only sur une grille commune de 1 mm. Les trois bruts sont full-head, dans leur grille native. Les mélanger rendrait notamment les métriques dépendant du fond, du support et de la résolution non comparables.

```text
brain-only : 4 conditions × 3 runs = 12 images
full-head  : brut × 3 runs = 3 images
```

`--no-sub` désactive la soumission des IQM au serveur public MRIQC. Le suffixe `:ro` du montage `/data` rend les entrées en lecture seule dans le conteneur.

## Réextraire les CSV

```bash
python3 extract_iqm.py \
  --brain-json-dir ~/Documents/derivatives/ds004332/mriqc_pilot_20260914/output_brainonly/sub-19/anat \
  --raw-json-dir ~/Documents/derivatives/ds004332/mriqc_pilot_20260914/output_raw/sub-19/anat \
  --output-dir ~/Documents/jdac-motion-correction/results/ds004332/mriqc_pilot
```

La qualification est prudente : « candidate » signifie seulement que l'IQM peut être étudiée à support fixe. Elle ne signifie pas qu'elle mesure correctement le mouvement ni qu'elle prédit la fidélité morphométrique.

## Provenance

- conteneur du laboratoire : `/project/hippocampus/common/containers/mriqc_23.1.0.sif` ;
- image déclarée par ses métadonnées : `docker://nipreps/mriqc:23.1.0` ;
- [exécution MRIQC 23.1](https://mriqc.readthedocs.io/en/23.1.0/running.html) ;
- [commande `apptainer run`](https://apptainer.org/docs/user/main/cli/apptainer_run.html) ;
- [montages Apptainer](https://apptainer.org/docs/user/main/bind_paths_and_mounts.html).
