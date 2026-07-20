# Phase 5 — Volumes sous-corticaux et fidélité anatomique

Cette phase analyse les volumes FreeSurfer de `aseg.stats` séparément de l'épaisseur corticale. Toutes les valeurs analysées sont en **mm³**.

## Questions

1. Les traitements modifient-ils le `run-01` presque immobile ?
2. Les volumes des `run-02/03` se rapprochent-ils du `raw/run-01` du même sujet ?
3. Le volume mesuré reste-t-il associé au score Agitation ?

Le `raw/run-01` est une référence opérationnelle, pas une vérité anatomique parfaite.

## Fichiers

- `build_notebook_aseg.py` : **source reproductible de l'analyse**. C'est ce fichier lisible qui est modifié lorsque les cellules, statistiques ou explications changent; il régénère ensuite le notebook.
- `explore_aseg_rigide.ipynb` : notebook généré et exécuté sur le PC du labo. Il contient le code, les tableaux et les figures interactives.
- `explore_aseg_rigide.html` : instantané exécuté avec toutes les sorties, consultable sans Jupyter.
- `inspect_central_structure.py` : contrôle visuel optionnel qui charge les volumes FreeSurfer `T1.mgz` et `aseg.mgz`. Le notebook CSV ne peut pas remplacer cette vérification voxel par voxel.
- `extract_aseg_stats.py` : outil de secours et de provenance pour reconstruire les tables depuis les `aseg.stats` individuels. Il n'est pas utilisé lorsque les cinq tables existent déjà.

## Tables utilisées sur le PC Linux du labo

Le notebook lit directement les cinq tables déjà présentes dans :

```text
~/Documents/derivatives/ds004332/aseg_stats/
```

Fichiers attendus :

```text
aseg_brut.csv
aseg_preproc.csv
aseg_jdac.csv
aseg_antiartonly.csv
aseg_nodenoise.csv
```

Malgré l'extension `.csv`, ces tables sont lues avec une tabulation comme séparateur. Elles restent hors Git. Aucune copie ni table consolidée n'est nécessaire avant d'exécuter le notebook.

`extract_aseg_stats.py` n'est **pas à exécuter pour l'analyse actuelle** : les cinq tables ont déjà été extraites. Il est conservé dans Git uniquement pour documenter et reproduire l'extraction à partir des fichiers FreeSurfer individuels si les tables sont perdues, incomplètes ou doivent être régénérées. Ses sorties longues et sa matrice de complétude sont alors écrites dans `results/ds004332/phase5_fidelity/`.

## Zone centrale de sub-19

Sur le PC Linux du labo, où se trouvent les reconstructions FreeSurfer :

```bash
python pipelines/ds004332/phase5_fidelity/inspect_central_structure.py \
  --subject sub-19_run-01
```

La figure emploie la même fenêtre pour les quatre conditions et superpose :

- orange : thalamus gauche/droit ;
- rose : troisième ventricule ;
- cyan : ventricules latéraux.

Pour obtenir le label exact d'un voxel repéré dans Freeview :

```bash
python pipelines/ds004332/phase5_fidelity/inspect_central_structure.py \
  --subject sub-19_run-01 --voxel I J K
```

Une adhérence interthalamique éventuelle n'est pas un label autonome standard d'`aseg`. Dans ce cas, la vérification reste visuelle ou nécessite un atlas plus détaillé.

Narval n'est requis que si les reconstructions nécessaires ne sont plus disponibles sur le PC du labo. Le MFA et les transferts restent alors manuels.

## Notebook cortical

Le notebook de Phase 4 conserve les questions A–E, mais la section exploratoire C-bis (non-linéarité et nombreuses comparaisons par strates) n'est plus générée. Le boxplot précise désormais son résultat : amincissement initial des variantes sans débruiteur, puis sur-correction dépendante du mouvement, surtout après quatre passages.

