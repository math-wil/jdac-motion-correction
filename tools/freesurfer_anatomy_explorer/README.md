# Explorateur anatomique FreeSurfer

Outil 3D purement pédagogique pour apprendre ce que FreeSurfer segmente et
mesure. Il ne lit aucune image, aucun CSV et aucun résultat de ds004332.

L'application permet de :

- tourner et zoomer un cerveau de référence;
- afficher, cacher et rendre transparentes les couches anatomiques;
- déplacer une coupe sagittale, coronale ou axiale;
- cliquer une structure pour voir son nom en français, son nom exact dans
  FreeSurfer et son fichier source;
- suivre un parcours guidé sur le cortex, le sous-cortical, les ventricules et
  les surfaces `white` et `pial`;
- comprendre visuellement `ThickAvg`, `SurfArea`, `GrayVol`, les volumes
  `aseg.stats` et les mesures globales.

## Installation

Dans l'environnement Python de ton choix :

```powershell
cd tools/freesurfer_anatomy_explorer
python -m pip install -r requirements.txt
```

L'environnement `cortical-motion` peut être utilisé s'il contient Python 3.10
ou plus récent. Les dépendances de l'explorateur sont indépendantes des
traitements JDAC.

## Lancement

```powershell
python app.py
```

Le navigateur s'ouvre sur `http://127.0.0.1:8080`. Au premier lancement,
Nilearn télécharge dans le cache utilisateur :

- les surfaces `pial` et `white` de FreeSurfer `fsaverage5`;
- l'atlas sous-cortical Harvard-Oxford.

Si l'atlas Harvard-Oxford ne peut pas être téléchargé, les vraies surfaces
`pial` et `white` de `fsaverage5` restent actives et les structures profondes
passent en schémas spatiaux clairement signalés. Si aucune donnée standard
n'est disponible, l'application démarre entièrement en mode **Schéma
pédagogique**. Dans tous les cas, les formes schématiques ne doivent jamais
servir à faire une mesure. On peut demander ce mode explicitement :

```powershell
python app.py --schematic
```

Pour un lancement sans ouverture automatique du navigateur :

```powershell
python app.py --no-browser
```

## Comment l'utiliser

1. Choisir une **vue intelligente** selon la question : **Extérieur**,
   **Ruban**, **Profond** ou **Ventricules**. Elle règle automatiquement les
   couches, leur opacité, la coupe et la caméra.
2. Tourner le cerveau par glisser-déposer, zoomer avec la molette et cliquer
   une structure pour l'identifier.
3. Ajuster au besoin chaque couche avec son interrupteur et son curseur
   d'opacité.
4. Pour une coupe manuelle, choisir le plan sagittal, coronal ou axial,
   déplacer le curseur, puis utiliser **Centrer** ou **Autre moitié**.
5. Chercher une structure à droite et utiliser **Isoler et centrer la
   structure** pour la dégager du reste du cerveau.
6. Passer de l'onglet **Anatomie** à **Mesures** pour relier la géométrie au
   nom rencontré dans `aparc.stats` ou `aseg.stats`.
## Choix anatomiques

`fsaverage5` fournit de vraies surfaces corticales FreeSurfer légères et
adaptées à une interaction fluide. Harvard-Oxford fournit les principaux noyaux
sous-corticaux sur un cerveau standard. Les rares éléments non présents dans
cet atlas sont affichés comme schémas spatiaux et signalés comme tels dans le
panneau d'information. L'application n'expose aucune valeur de résultat : les
unités servent uniquement à expliquer la nature des mesures.

