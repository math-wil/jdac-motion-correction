# Sensibilité de JDAC à la préparation d'entrée

**Statut : protocole proposé ; aucun traitement lancé.** Cette expérience est distincte de la phase 3 historique et du benchmark multi-correcteurs. Elle teste si N4 et le recalage rigide de **notre** chaîne modifient l'effet observé de JDAC. Elle ne cherche pas encore à isoler N4 du rigide ni à conclure sur toute la cohorte.

## Images et conditions

Neuf T1w `acq-mpragepmcoff_rec-wore` déjà présents sur le PC labo : `sub-17`, `sub-13` et `sub-14`, chacun en `run-01` (presque immobile), `run-02` (nodding) et `run-03` (shaking). Cette sélection couvre des niveaux de mouvement différents et précède les résultats de la nouvelle condition.

- **Ancienne paire, déjà calculée :** brut → N4 → rigide → SynthStrip (`preproc_rigid`) → JDAC complet (`jdac_rigid`).
- **Nouvelle paire :** le même brut → SynthStrip seul, en géométrie native (`preproc_minimal`) → le même JDAC complet et les mêmes poids (`jdac_minimal`). Pas de N4, de recalage ni de normalisation manuelle ajoutée avant le script JDAC. Ce script recadre, normalise et padde lui-même le cerveau extrait avant le réseau.

Les T1w bruts restent à leur emplacement BIDS. Les nouvelles images, masques et journaux iront dans `~/Documents/derivatives/ds004332/jdac_minimal_pilot/` sur le PC labo, hors Git ; aucune sortie historique ne sera remplacée. Le code d'inférence historique reste dans `../phase3_JDAC/run_jdac.py`. Un éventuel script de préparation du pilote sera rangé **ici**, pas dans la phase 3.

## Comparaisons prévues

1. **Effet ajouté par JDAC dans chaque chaîne :** différence appariée `JDAC − preproc`, d'abord dans la nouvelle paire, puis dans l'ancienne. Cela mesure l'effet de l'outil *tel qu'exécuté*, y compris ses opérations internes de normalisation ; ce n'est pas l'effet du réseau seul.
2. **Récupération morphométrique :** distance de `preproc_minimal` et `jdac_minimal` au `brut/run-01` du même sujet, comme référence opérationnelle commune. Une différence directe entre les deux conditions ne dit pas, à elle seule, laquelle est anatomiquement plus fidèle.
3. **Sensibilité à la préparation :** confronter les changements observés dans la nouvelle paire à ceux de la paire rigide. Si les conclusions divergent, N4 et le rigide devront être testés séparément avant toute généralisation.

Les résultats FreeSurfer à examiner sont l'épaisseur (mm), la surface (mm²), les volumes corticaux et sous-corticaux (mm³), ainsi que les échecs et contrôles qualité. Neuf acquisitions constituent un pilote descriptif, pas un test d'équivalence statistique.

## Ordre d'exécution et contrôle

1. Sur le PC labo : SynthStrip puis inférence JDAC sur les neuf images ; vérifier le masque (cortex, cervelet), les dimensions, l'affine, l'intensité et la superposition des mêmes coupes avec une fenêtre fixe. **Arrêt ici pour examen des images.**
2. Si ce contrôle passe : FreeSurfer sur les neuf `preproc_minimal` et neuf `jdac_minimal` (18 nouvelles reconstructions), avec la même version et la même procédure `-noskullstrip`. Documenter les options liées au champ de vue, notamment `-cw256` utilisé par l'ancienne grille MNI élargie.
3. Les métriques d'image éventuelles devront employer une région et une règle d'intensité fixées indépendamment de la sortie. Le script historique `compute_image_metrics.py` renormalise chaque image séparément et ne doit pas être repris tel quel pour conclure à la fidélité des intensités ou des contours.

**Repère sur l'article JDAC.** Sur MR-ART, les auteurs ont une image bougée et un scan immobile correspondant, tous deux extraits du crâne et normalisés ; ils ajoutent du bruit gaussien à l'image bougée du test. Ils comparent la **sortie** de chaque méthode au **scan immobile correspondant**, pas à son entrée bougée ni au T1w natif avec crâne. Ils rapportent PSNR, RMSE, SSIM et MS-SSIM sur l'image (débruitage) et sur les cartes de gradients (artefacts de mouvement). Pour NBOLD, sans référence immobile, l'évaluation est qualitative. Source : [Zhang et al., sections 4.1–4.4](https://arxiv.org/html/2403.08162v1).
