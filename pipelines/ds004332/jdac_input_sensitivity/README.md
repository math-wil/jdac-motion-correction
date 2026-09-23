# Sensibilité de JDAC à la préparation d'entrée

**Avancement du 23 septembre :** les 9/9 extractions et les 9/9 inférences sont terminées sur le PC labo. Les montages `mask_qc.png` et `jdac_qc.png` ont été examinés ; aucune coupure évidente de l'anatomie ou erreur de grille n'a été constatée. Les 18 entrées ont été transférées sur Narval et le job FreeSurfer `3850218` a été soumis. Aucun résultat morphométrique nouveau n'est encore disponible.

Cette expérience est distincte de la phase 3 historique et du benchmark multi-correcteurs. Elle teste si N4 et le recalage rigide de **notre** chaîne modifient l'effet observé de JDAC. Elle ne cherche pas encore à isoler N4 du rigide ni à conclure sur toute la cohorte. Elle n'est **pas nécessaire** pour répondre à la question factuelle « à quelle référence les auteurs comparent-ils JDAC ? », ni un préalable au protocole de l'article d'évaluation.

## Images et conditions

Neuf T1w `acq-mpragepmcoff_rec-wore` déjà présents sur le PC labo : `sub-17`, `sub-13` et `sub-14`, chacun en `run-01` (presque immobile), `run-02` (nodding) et `run-03` (shaking). Cette sélection couvre des niveaux de mouvement différents et précède les résultats de la nouvelle condition.

- **Ancienne paire, déjà calculée :** brut → N4 → rigide → SynthStrip (`preproc_rigid`) → JDAC complet (`jdac_rigid`).
- **Nouvelle paire :** le même brut → SynthStrip seul, en géométrie native (`preproc_minimal`) → le même JDAC complet et les mêmes poids (`jdac_minimal`). Pas de N4, de recalage ni de normalisation manuelle ajoutée avant le script JDAC. Ce script recadre, normalise et padde lui-même le cerveau extrait avant le réseau.

Les T1w bruts restent à leur emplacement BIDS. Les nouvelles images, masques et journaux iront dans `~/Documents/derivatives/ds004332/jdac_minimal_pilot/` sur le PC labo, hors Git ; aucune sortie historique ne sera remplacée. Le pilote utilise `run_minimal_pilot.py`, qui appelle le SynthStrip du labo puis réutilise le code d'inférence historique `../phase3_JDAC/run_jdac.py` et les mêmes poids. Il n'y a pas de nouveau modèle ou de nouvelle variante JDAC.

## Exécution préparée sur le PC labo

Avec l'environnement `cortical-motion` et depuis `~/Documents/jdac-motion-correction` :

```bash
PY="$HOME/miniconda3/envs/cortical-motion/bin/python"
P="pipelines/ds004332/jdac_input_sensitivity/run_minimal_pilot.py"
"$PY" "$P" --stage check
"$PY" "$P" --stage strip
"$PY" "$P" --stage qc
```

`check` ne crée rien : il vérifie les neuf T1w, SynthStrip, le code JDAC et les deux poids. `strip` crée neuf cerveaux et neuf masques en grille native, puis contrôle forme et affine. `qc` crée `mask_qc.png`, avec les neuf masques superposés aux images brutes dans plusieurs plans, et `qc_manifest.json`, qui donne le volume de chaque masque en mm³. **Arrêt et examen visuel des neuf masques** : tissu cortical préservé, pas de coupe du cervelet, pas de déplacement. Après cet examen seulement :

```bash
"$PY" "$P" --stage jdac --mask-reviewed
"$PY" "$P" --stage qc-jdac
```

L'inférence produit neuf sorties JDAC dans `jdac_minimal/`, vérifie leur grille contre l'entrée brain-only et inscrit l'empreinte SHA-256 des deux poids dans `jdac_manifest.json`. `qc-jdac` vérifie les valeurs finies et produit `jdac_qc.png`, qui montre les mêmes coupes pour l'entrée et la sortie, avec une fenêtre commune `[0,1]` après reproduction de la normalisation d'entrée. Ce montage sert au contrôle technique, pas à conclure à une récupération anatomique. Chaque étape saute une sortie complète déjà présente sans écraser les images historiques. Si une sortie est incomplète ou si la géométrie diffère, le script s'arrête. Les sorties et manifestes restent hors Git.

## Comparaisons prévues

Le fichier `recon_all_minimal.sbatch` définit les 18 reconstructions : une par image (`preproc_minimal` puis `jdac_minimal`), avec trois jobs simultanés au maximum sur Narval. FreeSurfer 8.0.0-1, 8 CPU et 64 Go par job ; même procédure brain-only en deux passes que l'historique, sans `-cw256` (lié à l'ancienne grille MNI élargie). Les entrées, sorties et logs sont séparés sous `/project/ctb-sbouix/mathw/jdac_minimal_pilot/`. Un job soumis n'est pas un résultat FreeSurfer validé : `aseg.stats`, `lh.aparc.stats` et `rh.aparc.stats` doivent exister et leur qualité être vérifiée.

Les références historiques nécessaires ont été vérifiées sur Narval pour ces sujets : 9/9 paires `preproc_rigid`, 9/9 paires `jdac_rigid` et 3/3 `brut/run-01` possèdent `aseg.stats` et les deux `aparc.stats`. Cela ne constitue pas encore un contrôle qualité de leurs segmentations.

1. **Effet ajouté par JDAC dans chaque chaîne :** différence appariée `JDAC − preproc`, d'abord dans la nouvelle paire, puis dans l'ancienne. Cela mesure l'effet de l'outil *tel qu'exécuté*, y compris ses opérations internes de normalisation ; ce n'est pas l'effet du réseau seul.
2. **Récupération morphométrique :** distance de `preproc_minimal` et `jdac_minimal` au `brut/run-01` du même sujet, comme référence opérationnelle commune. Une différence directe entre les deux conditions ne dit pas, à elle seule, laquelle est anatomiquement plus fidèle.
3. **Sensibilité à la préparation :** confronter les changements observés dans la nouvelle paire à ceux de la paire rigide. Si les conclusions divergent, N4 et le rigide devront être testés séparément avant toute généralisation.

Les résultats FreeSurfer à examiner sont l'épaisseur (mm), la surface (mm²), les volumes corticaux et sous-corticaux (mm³), ainsi que les échecs et contrôles qualité. Neuf acquisitions constituent un pilote descriptif, pas un test d'équivalence statistique.

## Ordre d'exécution et contrôle

1. Sur le PC labo : vérifier les neuf sources, créer les neuf cerveaux et masques, puis inspecter les masques avant JDAC. Faire ensuite l'inférence sur ces neuf cerveaux, vérifier les dimensions, l'affine, l'intensité et la superposition des mêmes coupes avec une fenêtre fixe. **Arrêt ici pour examen des images.**
2. Si ce contrôle passe et si le calcul est autorisé : FreeSurfer sur les neuf `preproc_minimal` et neuf `jdac_minimal` (18 nouvelles reconstructions), avec la même version et la même procédure `-noskullstrip`. Documenter les options liées au champ de vue ; ne pas recopier automatiquement `-cw256`, introduit pour l'ancienne grille MNI élargie.
3. Les métriques d'image éventuelles devront employer une région et une règle d'intensité fixées indépendamment de la sortie. Le script historique `compute_image_metrics.py` renormalise chaque image séparément et ne doit pas être repris tel quel pour conclure à la fidélité des intensités ou des contours.

**Repère sur l'article JDAC.** Sur MR-ART, les auteurs ont une image bougée et un scan immobile correspondant, tous deux extraits du crâne et normalisés ; ils ajoutent du bruit gaussien à l'image bougée du test. Ils comparent la **sortie** de chaque méthode au **scan immobile correspondant**, pas à son entrée bougée ni au T1w natif avec crâne. Ils rapportent PSNR, RMSE, SSIM et MS-SSIM sur l'image (débruitage) et sur les cartes de gradients (artefacts de mouvement). Pour NBOLD, sans référence immobile, l'évaluation est qualitative. Source : [Zhang et al., sections 4.1–4.4](https://arxiv.org/html/2403.08162v1).
