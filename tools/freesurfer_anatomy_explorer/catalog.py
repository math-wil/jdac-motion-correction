"""Catalogue pédagogique reliant l'anatomie aux noms exacts de FreeSurfer.

Ce module ne contient aucune donnée de l'étude ds004332. Il décrit seulement
les structures et les mesures que l'explorateur doit expliquer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureInfo:
    """Description courte d'une structure visible dans l'explorateur."""

    identifier: str
    name_fr: str
    family: str
    location: str
    role: str
    freesurfer_names: tuple[str, ...]
    files: tuple[str, ...]
    measures: tuple[str, ...]
    color: str
    atlas_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasureInfo:
    """Définition pédagogique d'une mesure FreeSurfer."""

    identifier: str
    label: str
    source: str
    unit: str
    family: str
    explanation: str
    visual: str


STRUCTURES = (
    StructureInfo(
        "cortex",
        "Cortex (matière grise corticale)",
        "Cortical",
        "Fine nappe plissée qui recouvre les deux hémisphères.",
        "Traite l'information sensorielle, motrice et cognitive selon la région.",
        ("lh.aparc.stats / rh.aparc.stats", "lh.pial / rh.pial"),
        ("stats/lh.aparc.stats", "stats/rh.aparc.stats", "surf/lh.pial", "surf/rh.pial"),
        ("ThickAvg", "SurfArea", "GrayVol"),
        "#d98b5f",
    ),
    StructureInfo(
        "white_matter",
        "Substance blanche cérébrale",
        "Tissu cérébral",
        "Sous le cortex; elle forme l'intérieur des hémisphères.",
        "Relie les régions cérébrales par des faisceaux d'axones.",
        (
            "Left-Cerebral-White-Matter",
            "Right-Cerebral-White-Matter",
            "CerebralWhiteMatterVol",
            "lh.white / rh.white",
        ),
        ("stats/aseg.stats", "surf/lh.white", "surf/rh.white"),
        ("CerebralWhiteMatterVol",),
        "#eee4c8",
        ("Cerebral White Matter",),
    ),
    StructureInfo(
        "csf",
        "LCR (liquide cérébrospinal)",
        "Liquide",
        "Autour du cerveau et dans les ventricules.",
        "Protège le cerveau et participe aux échanges de fluides.",
        ("CSF",),
        ("stats/aseg.stats",),
        ("CSF",),
        "#69b8d6",
    ),
    StructureInfo(
        "ventricles",
        "Ventricules",
        "Liquide",
        "Cavités remplies de LCR au centre du cerveau.",
        "Contiennent et font circuler le liquide cérébrospinal.",
        (
            "Left-Lateral-Ventricle",
            "Right-Lateral-Ventricle",
            "3rd-Ventricle",
            "4th-Ventricle",
        ),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#54c2e8",
        ("Lateral Ventricle", "Third Ventricle", "Fourth Ventricle"),
    ),
    StructureInfo(
        "thalamus",
        "Thalamus",
        "Sous-cortical",
        "Deux noyaux profonds, de part et d'autre du troisième ventricule.",
        "Relais majeur entre de nombreux signaux et le cortex.",
        ("Left-Thalamus-Proper", "Right-Thalamus-Proper"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#8f73c5",
        ("Thalamus",),
    ),
    StructureInfo(
        "caudate",
        "Noyau caudé",
        "Sous-cortical",
        "Structure profonde courbée qui longe les ventricules latéraux.",
        "Participe aux boucles motrices, à l'apprentissage et à la décision.",
        ("Left-Caudate", "Right-Caudate"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#c66d8b",
        ("Caudate",),
    ),
    StructureInfo(
        "putamen",
        "Putamen",
        "Sous-cortical",
        "Noyau profond, latéral au pallidum.",
        "Participe principalement au contrôle et à l'apprentissage moteurs.",
        ("Left-Putamen", "Right-Putamen"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#d28357",
        ("Putamen",),
    ),
    StructureInfo(
        "pallidum",
        "Pallidum",
        "Sous-cortical",
        "Petit noyau profond, médial au putamen.",
        "Module la sortie des circuits des ganglions de la base.",
        ("Left-Pallidum", "Right-Pallidum"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#d7b349",
        ("Pallidum", "Pallidus"),
    ),
    StructureInfo(
        "hippocampus",
        "Hippocampe",
        "Sous-cortical",
        "Structure profonde du lobe temporal médial.",
        "Essentiel à la formation de nouveaux souvenirs et à la navigation.",
        ("Left-Hippocampus", "Right-Hippocampus"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#63a76b",
        ("Hippocampus",),
    ),
    StructureInfo(
        "amygdala",
        "Amygdale",
        "Sous-cortical",
        "Petit noyau en avant de l'hippocampe.",
        "Participe au traitement émotionnel et à la détection de la pertinence.",
        ("Left-Amygdala", "Right-Amygdala"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#e15b64",
        ("Amygdala",),
    ),
    StructureInfo(
        "accumbens",
        "Noyau accumbens",
        "Sous-cortical",
        "Petit noyau antérieur et inférieur, près du caudé et du putamen.",
        "Participe à la motivation, à la récompense et à l'apprentissage.",
        ("Left-Accumbens-area", "Right-Accumbens-area"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#f08a9b",
        ("Accumbens",),
    ),
    StructureInfo(
        "ventraldc",
        "Diencéphale ventral",
        "Sous-cortical",
        "Région profonde sous le thalamus; étiquette composite de FreeSurfer.",
        "Regroupe plusieurs tissus ventraux difficiles à séparer automatiquement.",
        ("Left-VentralDC", "Right-VentralDC"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#a68bc7",
    ),
    StructureInfo(
        "cerebellum",
        "Cervelet",
        "Fosse postérieure",
        "Sous les lobes occipitaux, derrière le tronc cérébral.",
        "Coordonne le mouvement, l'équilibre et certains apprentissages.",
        (
            "Left-Cerebellum-Cortex",
            "Right-Cerebellum-Cortex",
            "Left-Cerebellum-White-Matter",
            "Right-Cerebellum-White-Matter",
        ),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#7db29d",
    ),
    StructureInfo(
        "brainstem",
        "Tronc cérébral",
        "Axe central",
        "Sous le diencéphale, devant le cervelet.",
        "Relie le cerveau à la moelle et soutient des fonctions vitales.",
        ("Brain-Stem",),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#b99674",
        ("Brain-Stem", "Brain Stem"),
    ),
    StructureInfo(
        "corpus_callosum",
        "Corps calleux",
        "Substance blanche",
        "Grand faisceau médian entre les deux hémisphères.",
        "Permet la communication entre les hémisphères.",
        ("CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior"),
        ("stats/aseg.stats",),
        ("Volume_mm3",),
        "#f2d26f",
    ),
)


MEASURES = (
    MeasureInfo(
        "ThickAvg",
        "ThickAvg — épaisseur corticale moyenne",
        "lh.aparc.stats / rh.aparc.stats",
        "mm",
        "Région corticale",
        "Distance moyenne entre la surface blanche et la surface piale dans une région.",
        "Une flèche traverse le ruban gris de white vers pial.",
    ),
    MeasureInfo(
        "SurfArea",
        "SurfArea — aire de surface corticale",
        "lh.aparc.stats / rh.aparc.stats",
        "mm²",
        "Région corticale",
        "Aire de la surface blanche attribuée à la région corticale.",
        "Une parcelle colorée s'étend sur la nappe corticale.",
    ),
    MeasureInfo(
        "GrayVol",
        "GrayVol — volume gris cortical",
        "lh.aparc.stats / rh.aparc.stats",
        "mm³",
        "Région corticale",
        "Volume du ruban de matière grise d'une région, lié à sa surface et à son épaisseur.",
        "La parcelle devient un morceau de ruban ayant une aire et une hauteur.",
    ),
    MeasureInfo(
        "structure_volume",
        "Volume d'une structure aseg",
        "aseg.stats",
        "mm³",
        "Structure segmentée",
        "Nombre de voxels attribués à la structure, multiplié par le volume d'un voxel.",
        "Le bloc sous-cortical entier est rempli: on mesure l'espace qu'il occupe.",
    ),
    MeasureInfo(
        "CortexVol",
        "CortexVol",
        "aseg.stats",
        "mm³",
        "Global",
        "Volume total de matière grise corticale des deux hémisphères.",
        "Tous les rubans corticaux sont réunis.",
    ),
    MeasureInfo(
        "SubCortGrayVol",
        "SubCortGrayVol",
        "aseg.stats",
        "mm³",
        "Global",
        "Somme des volumes des principales structures grises sous-corticales.",
        "Les noyaux gris profonds sont réunis, sans le ruban cortical.",
    ),
    MeasureInfo(
        "TotalGrayVol",
        "TotalGrayVol",
        "aseg.stats",
        "mm³",
        "Global",
        "Volume gris total: cortex, gris sous-cortical et autres composantes incluses par FreeSurfer.",
        "Le ruban cortical et les noyaux gris profonds s'allument ensemble.",
    ),
    MeasureInfo(
        "CerebralWhiteMatterVol",
        "CerebralWhiteMatterVol",
        "aseg.stats",
        "mm³",
        "Global",
        "Volume total de substance blanche des hémisphères cérébraux.",
        "L'intérieur blanc des deux hémisphères est rempli.",
    ),
    MeasureInfo(
        "CSF",
        "CSF",
        "aseg.stats",
        "mm³",
        "Global",
        "Volume de l'étiquette CSF de la segmentation aseg; ce n'est pas tout le LCR intracrânien.",
        "Les espaces classés CSF sont surlignés.",
    ),
    MeasureInfo(
        "BrainSegVol",
        "BrainSegVol",
        "aseg.stats",
        "mm³",
        "Global",
        "Volume total des étiquettes appartenant à la segmentation du cerveau.",
        "Toutes les composantes segmentées du cerveau sont réunies.",
    ),
    MeasureInfo(
        "BrainSegVolNotVent",
        "BrainSegVolNotVent",
        "aseg.stats",
        "mm³",
        "Global",
        "BrainSegVol après exclusion des ventricules et de certaines composantes ventriculaires.",
        "Le cerveau reste rempli tandis que les cavités ventriculaires sont retirées.",
    ),
    MeasureInfo(
        "SupraTentorialVol",
        "SupraTentorialVol",
        "aseg.stats",
        "mm³",
        "Global",
        "Volume situé au-dessus de la tente du cervelet, avec les inclusions définies par FreeSurfer.",
        "Les hémisphères sont montrés sans le cervelet sous-tentoriel.",
    ),
    MeasureInfo(
        "eTIV",
        "eTIV — EstimatedTotalIntraCranialVol",
        "aseg.stats",
        "mm³",
        "Normalisation",
        "Estimation de la taille de la cavité intracrânienne dérivée du recalage; ce n'est pas un tissu.",
        "Une enveloppe crânienne conceptuelle entoure l'ensemble du cerveau.",
    ),
    MeasureInfo(
        "to_eTIV",
        "Ratios -to-eTIV",
        "mesure dérivée à partir de aseg.stats",
        "ratio",
        "Normalisation",
        "Volume d'une structure divisé par eTIV pour tenir compte de la taille de la tête.",
        "Le volume choisi est comparé à l'enveloppe intracrânienne.",
    ),
    MeasureInfo(
        "SurfaceHoles",
        "SurfaceHoles",
        "aseg.stats",
        "compte",
        "Qualité de surface",
        "Nombre de défauts topologiques détectés dans les surfaces avant correction; ce n'est pas un volume.",
        "Des ouvertures conceptuelles sont indiquées sur un maillage cortical.",
    ),
)


STRUCTURES_BY_ID = {item.identifier: item for item in STRUCTURES}
MEASURES_BY_ID = {item.identifier: item for item in MEASURES}


def structure_options() -> list[dict[str, str]]:
    """Options prêtes à afficher dans l'interface."""

    return [
        {"title": f"{item.name_fr} · {item.family}", "value": item.identifier}
        for item in STRUCTURES
    ]


def measure_options() -> list[dict[str, str]]:
    """Options de glossaire prêtes à afficher dans l'interface."""

    return [
        {"title": f"{item.label} [{item.unit}]", "value": item.identifier}
        for item in MEASURES
    ]

