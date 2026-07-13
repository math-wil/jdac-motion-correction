# Reprise JDAC — décision de travail au 15 juillet 2026

## Périmètre immédiat

La suite immédiate reste **JDAC**, conformément à la discussion avec le directeur.

Ne pas lancer pour l'instant :

- nouvelles analyses PMC ;
- protocole de QC aveugle à 90 cas ;
- BME-X ;
- nouveau modèle ou réentraînement complet.

Ces éléments sont des pistes exploratoires non validées, pas le plan courant.

## Ce qui est déjà établi

Les cinq conditions ont été évaluées : `brut`, `preproc`, `jdac`,
`jdac_antiartonly` et `jdac_nodenoise`.

- JDAC complet réduit l'association Agitation–épaisseur, mais amincit aussi les
  scans immobiles (`-0,160 mm` par rapport au brut).
- Retirer le débruiteur préserve mieux les détails visuels, mais ne restaure pas
  la morphométrie.
- `antiartonly` sur-corrige 14 sujets sur 21.
- `nodenoise` sur-corrige 19 sujets sur 20 et produit le plus grand offset
  immobile (`-0,216 mm`).
- Le QC visuel a déjà montré que de bonnes images ne garantissent pas de bonnes
  reconstructions FreeSurfer. Le répéter seul ne répondrait pas à la question.
- Agitation a déjà été testé sur les acquisitions PMC : le score diminue avec
  PMC + réacquisition. Ce résultat concernait la validation d'Agitation et ne
  justifie pas, à lui seul, une nouvelle branche morphométrique PMC.

## Hypothèse JDAC suivante

Le problème ne vient pas uniquement du débruiteur. Dans le code JDAC, le réseau
anti-artefact calcule un résidu, le limite à `[-0,04 ; +0,04]`, puis l'applique
avec une force fixe `step_lr=1.0`. La boucle peut répéter cette correction.

Hypothèse à tester : **la correction anti-artefact est trop forte**. Une force
réduite pourrait conserver une partie du gain visuel tout en limitant l'offset
et la sur-correction morphométrique.

## Expérience minimale proposée

Modifier le script pour rendre `step_lr` configurable, sans changer les poids
et sans réentraîner le réseau.

Pilote initial :

- forces : `0.25`, `0.50`, `1.00` (référence actuelle) ;
- trois sujets couvrant faible, moyen et fort mouvement ;
- uniquement `run-01` immobile et `run-03` shaking ;
- mêmes coupes et mêmes sujets pour toutes les figures.

Manifest prêt :
`pipelines/ds004332/phase3_JDAC/pilot_jdac_strength.csv`.

Depuis `~/Documents/jdac/` sur le PC du labo, lancer successivement les forces
`0.25`, `0.50` et `1.00` avec `--antiart-step-lr`, en conservant
`--max-iter 4`. Le script génère automatiquement un suffixe distinct pour
chaque force ; aucune sortie JDAC existante n'est écrasée.

Ordre des contrôles :

1. vérifier géométrie, affine, intensités et absence d'erreur technique ;
2. produire images et cartes de différence ;
3. calculer les métriques image déjà utilisées, sans les considérer comme une
   preuve morphométrique ;
4. présenter le pilote et son protocole au directeur ;
5. seulement si la piste est validée, lancer FreeSurfer sur le petit pilote ;
6. ne lancer les 66 acquisitions qu'après un résultat morphométrique favorable.

## Critère de décision

Une force réduite n'est intéressante que si elle :

- diminue l'artefact sur le scan bougé ;
- limite le changement du scan immobile ;
- et, après FreeSurfer, rapproche l'épaisseur régionale du scan bougé de la
  référence immobile sans inverser le signal.

Une amélioration visuelle seule ne suffit pas.

## Réunion du 17 juillet

Présenter :

1. les résultats terminés des deux variantes sans débruiteur ;
2. la conclusion : le débruiteur n'est pas l'unique cause de l'échec ;
3. cette expérience de réduction de la force du résidu comme prochaine étape
   JDAC courte, contrôlée et réversible ;
4. demander validation avant les recon-all du pilote.
