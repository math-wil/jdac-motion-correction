"""Chargement d'un cerveau de référence et génération du mode de secours.

Le modèle principal combine les surfaces fsaverage5 de FreeSurfer avec l'atlas
sous-cortical Harvard-Oxford. Les volumes absents de cet atlas sont représentés
par des formes schématiques clairement identifiées dans l'interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from catalog import STRUCTURES_BY_ID


@dataclass
class MeshPart:
    """Une partie anatomique affichable et sélectionnable."""

    structure_id: str
    mesh: pv.PolyData
    group: str
    source: str
    color: str
    side: str = ""


@dataclass
class ReferenceModel:
    """Ensemble des maillages nécessaires à la scène."""

    parts: list[MeshPart]
    mode: str
    source: str
    warnings: list[str] = field(default_factory=list)


def _polydata(coords: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    vtk_faces = np.column_stack(
        [np.full(len(faces), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(np.asarray(coords, dtype=float), vtk_faces)


def _warp_brain(mesh: pv.PolyData, amount: float = 0.055) -> pv.PolyData:
    """Ajoute des plis doux à un ellipsoïde schématique."""

    result = mesh.copy()
    points = result.points
    radius = np.linalg.norm(points, axis=1)
    radius[radius == 0] = 1
    unit = points / radius[:, None]
    theta = np.arctan2(unit[:, 1], unit[:, 0])
    phi = np.arccos(np.clip(unit[:, 2], -1, 1))
    modulation = 1 + amount * (
        0.55 * np.sin(10 * theta + 2 * np.sin(3 * phi))
        + 0.35 * np.sin(13 * phi)
        + 0.20 * np.sin(17 * theta - 4 * phi)
    )
    result.points = points * modulation[:, None]
    return result


def _ellipsoid(
    center: tuple[float, float, float],
    scale: tuple[float, float, float],
    resolution: int = 48,
) -> pv.PolyData:
    mesh = pv.Sphere(
        radius=1,
        center=center,
        theta_resolution=resolution,
        phi_resolution=max(24, resolution // 2),
    )
    translated = mesh.points - np.asarray(center)
    mesh.points = translated * np.asarray(scale) + np.asarray(center)
    return mesh


def _tube(
    points: list[tuple[float, float, float]], radius: float, sides: int = 32
) -> pv.PolyData:
    spline = pv.Spline(np.asarray(points, dtype=float), n_points=120)
    return spline.tube(radius=radius, n_sides=sides)


def _add_pair(
    parts: list[MeshPart],
    identifier: str,
    centers: tuple[tuple[float, float, float], tuple[float, float, float]],
    scales: tuple[float, float, float],
    group: str,
) -> None:
    info = STRUCTURES_BY_ID[identifier]
    for side, center in zip(("gauche", "droite"), centers):
        parts.append(
            MeshPart(
                identifier,
                _ellipsoid(center, scales),
                group,
                "schéma spatial",
                info.color,
                side,
            )
        )


def _schematic_internal_parts() -> list[MeshPart]:
    """Formes internes utilisées avec ou sans téléchargement de l'atlas."""

    parts: list[MeshPart] = []
    _add_pair(parts, "thalamus", ((-9, -8, 2), (9, -8, 2)), (7, 11, 8), "deep")
    _add_pair(parts, "caudate", ((-13, 3, 9), (13, 3, 9)), (4, 12, 4), "deep")
    _add_pair(parts, "putamen", ((-20, -2, -1), (20, -2, -1)), (7, 12, 7), "deep")
    _add_pair(parts, "pallidum", ((-14, -3, -2), (14, -3, -2)), (4, 8, 5), "deep")
    _add_pair(
        parts,
        "hippocampus",
        ((-22, -20, -14), (22, -20, -14)),
        (6, 15, 5),
        "deep",
    )
    _add_pair(
        parts,
        "amygdala",
        ((-22, -7, -14), (22, -7, -14)),
        (5, 6, 5),
        "deep",
    )
    _add_pair(
        parts,
        "accumbens",
        ((-10, 12, -8), (10, 12, -8)),
        (3.5, 5, 3.5),
        "deep",
    )
    _add_pair(
        parts,
        "ventraldc",
        ((-7, -9, -11), (7, -9, -11)),
        (5, 7, 4),
        "deep",
    )

    ventricle = STRUCTURES_BY_ID["ventricles"]
    for side, sign in (("gauche", -1), ("droite", 1)):
        parts.append(
            MeshPart(
                "ventricles",
                _tube(
                    [
                        (sign * 5, -20, 10),
                        (sign * 8, -8, 17),
                        (sign * 10, 7, 15),
                        (sign * 12, 13, 7),
                    ],
                    radius=3.2,
                ),
                "ventricles",
                "schéma spatial",
                ventricle.color,
                side,
            )
        )

    brainstem = STRUCTURES_BY_ID["brainstem"]
    parts.append(
        MeshPart(
            "brainstem",
            _ellipsoid((0, -20, -31), (8, 10, 20)),
            "brainstem",
            "schéma spatial",
            brainstem.color,
        )
    )

    cerebellum = STRUCTURES_BY_ID["cerebellum"]
    for side, center in (("gauche", (-20, -45, -31)), ("droite", (20, -45, -31))):
        mesh = _warp_brain(_ellipsoid(center, (22, 18, 14), 64), amount=0.035)
        parts.append(
            MeshPart(
                "cerebellum",
                mesh,
                "cerebellum",
                "schéma spatial",
                cerebellum.color,
                side,
            )
        )

    callosum = STRUCTURES_BY_ID["corpus_callosum"]
    parts.append(
        MeshPart(
            "corpus_callosum",
            _tube([(-27, -2, 12), (-12, 1, 19), (0, 3, 21), (12, 1, 19), (27, -2, 12)], 2.4),
            "corpus_callosum",
            "schéma spatial",
            callosum.color,
        )
    )
    return parts


def build_schematic_model(reason: str = "") -> ReferenceModel:
    """Construit un modèle autonome lorsque les données standard sont absentes."""

    parts = _schematic_internal_parts()
    cortex_info = STRUCTURES_BY_ID["cortex"]
    white_info = STRUCTURES_BY_ID["white_matter"]
    csf_info = STRUCTURES_BY_ID["csf"]

    pial = _warp_brain(_ellipsoid((0, -5, 1), (47, 60, 44), 96))
    white = _warp_brain(_ellipsoid((0, -5, 1), (40, 51, 36), 96), amount=0.045)
    csf = _ellipsoid((0, -5, 1), (50, 63, 47), 72)
    parts.extend(
        [
            MeshPart("cortex", pial, "cortex", "schéma pédagogique", cortex_info.color),
            MeshPart(
                "white_matter",
                white,
                "white_matter",
                "schéma pédagogique",
                white_info.color,
            ),
            MeshPart("csf", csf, "csf", "schéma pédagogique", csf_info.color),
        ]
    )
    warnings = []
    if reason:
        warnings.append(reason)
    warnings.append(
        "Le mode schématique conserve les relations spatiales et les noms FreeSurfer, "
        "mais ses formes ne servent jamais à une mesure anatomique."
    )
    return ReferenceModel(
        parts,
        "Schéma pédagogique",
        "Géométrie générée localement, sans donnée personnelle",
        warnings,
    )


def _atlas_mesh(
    image: Any,
    label_indexes: list[int],
    *,
    reduction: float = 0.55,
) -> pv.PolyData | None:
    from nibabel.affines import apply_affine
    from skimage.measure import marching_cubes

    volume = np.asarray(image.dataobj)
    mask = np.isin(volume, label_indexes)
    if mask.sum() < 8:
        return None
    vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5)
    vertices_world = apply_affine(image.affine, vertices)
    mesh = _polydata(vertices_world, faces)
    if mesh.n_cells > 5000:
        mesh = mesh.decimate(reduction)
    return mesh.clean()


def _match_label_indexes(labels: list[str], terms: tuple[str, ...]) -> list[int]:
    indexes: list[int] = []
    for index, label in enumerate(labels):
        normalized = str(label).lower().replace("-", " ")
        if any(term.lower().replace("-", " ") in normalized for term in terms):
            indexes.append(index)
    return indexes


def _load_standard_model(cache_dir: Path) -> ReferenceModel:
    from nibabel import load as load_nifti
    from nilearn import datasets, surface

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    fsaverage = datasets.fetch_surf_fsaverage(
        mesh="fsaverage5", data_dir=str(cache_dir)
    )
    parts: list[MeshPart] = []
    for identifier, group, attribute in (
        ("cortex", "cortex", "pial"),
        ("white_matter", "white_matter", "white"),
    ):
        info = STRUCTURES_BY_ID[identifier]
        for side, suffix in (("gauche", "left"), ("droite", "right")):
            mesh_path = getattr(fsaverage, f"{attribute}_{suffix}")
            loaded = surface.load_surf_mesh(mesh_path)
            parts.append(
                MeshPart(
                    identifier,
                    _polydata(loaded.coordinates, loaded.faces),
                    group,
                    "FreeSurfer fsaverage5",
                    info.color,
                    side,
                )
            )

    atlas_loaded: set[str] = set()
    atlas_error = ""
    try:
        atlas = datasets.fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-1mm", data_dir=str(cache_dir)
        )
        image = load_nifti(atlas.filename)
        labels = [str(label) for label in atlas.labels]
        for info in STRUCTURES_BY_ID.values():
            if not info.atlas_terms or info.identifier in {"white_matter"}:
                continue
            indexes = _match_label_indexes(labels, info.atlas_terms)
            mesh = _atlas_mesh(image, indexes) if indexes else None
            if mesh is not None:
                group = "ventricles" if info.identifier == "ventricles" else "deep"
                if info.identifier == "brainstem":
                    group = "brainstem"
                parts.append(
                    MeshPart(
                        info.identifier,
                        mesh,
                        group,
                        "atlas Harvard-Oxford",
                        info.color,
                    )
                )
                atlas_loaded.add(info.identifier)
    except Exception as exc:
        atlas_error = (
            "L'atlas sous-cortical n'a pas pu être téléchargé "
            f"({exc.__class__.__name__}); les structures profondes utilisent "
            "donc les schémas spatiaux signalés dans leur fiche."
        )
    for part in _schematic_internal_parts():
        if part.structure_id not in atlas_loaded:
            parts.append(part)

    csf_info = STRUCTURES_BY_ID["csf"]
    bounds = pv.MultiBlock([part.mesh for part in parts if part.group == "cortex"]).bounds
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )
    scale = (
        (bounds[1] - bounds[0]) * 0.53,
        (bounds[3] - bounds[2]) * 0.53,
        (bounds[5] - bounds[4]) * 0.53,
    )
    parts.append(
        MeshPart(
            "csf",
            _ellipsoid(center, scale, 72),
            "csf",
            "enveloppe pédagogique",
            csf_info.color,
        )
    )
    warnings = [
        "Les surfaces corticales proviennent de fsaverage5; les noyaux disponibles "
        "proviennent de Harvard-Oxford. Les éléments marqués « schéma spatial » "
        "complètent l'apprentissage mais ne sont pas des segmentations mesurables."
    ]
    if atlas_error:
        warnings.append(atlas_error)
    return ReferenceModel(
        parts,
        "Cerveau standard",
        (
            "FreeSurfer fsaverage5 + compléments schématiques"
            if atlas_error
            else "FreeSurfer fsaverage5 + atlas Harvard-Oxford (Nilearn)"
        ),
        warnings,
    )


def load_reference_model(
    cache_dir: Path | None = None, force_schematic: bool = False
) -> ReferenceModel:
    """Charge le modèle standard; revient proprement au modèle autonome."""

    if force_schematic:
        return build_schematic_model("Mode schématique demandé au lancement.")
    target = cache_dir or Path.home() / ".cache" / "freesurfer-anatomy-explorer"
    try:
        return _load_standard_model(target)
    except Exception as exc:  # le mode hors ligne doit rester utilisable
        return build_schematic_model(
            "Le cerveau standard n'a pas pu être chargé "
            f"({exc.__class__.__name__}). Le mode autonome est actif."
        )

