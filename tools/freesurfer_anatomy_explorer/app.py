#!/usr/bin/env python3
"""Explorateur 3D pédagogique de l'anatomie mesurée par FreeSurfer.

Cette application n'ouvre aucune donnée de l'étude. Elle utilise un cerveau
standard, ou un modèle schématique autonome lorsque le téléchargement n'est pas
possible.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import client, html, vtk
from trame.widgets import vuetify3 as v3

from catalog import (
    MEASURES_BY_ID,
    STRUCTURES_BY_ID,
    measure_options,
    structure_options,
)
from model import MeshPart, ReferenceModel, load_reference_model


GROUP_LABELS = {
    "cortex": "Cortex · surface piale",
    "white_matter": "Substance blanche · surface white",
    "csf": "Enveloppe de LCR",
    "deep": "Noyaux profonds",
    "ventricles": "Ventricules",
    "corpus_callosum": "Corps calleux",
    "brainstem": "Tronc cérébral",
    "cerebellum": "Cervelet",
}

GROUP_DEFAULTS = {
    "cortex": (True, 0.96),
    "white_matter": (False, 0.82),
    "csf": (False, 0.12),
    "deep": (False, 0.95),
    "ventricles": (False, 0.95),
    "corpus_callosum": (False, 0.95),
    "brainstem": (True, 0.95),
    "cerebellum": (True, 0.88),
}

EXPLORATION_MODES = {
    "external": {
        "label": "Extérieur",
        "hint": "La forme générale du cerveau, sans superposition interne.",
        "groups": {
            "cortex": (True, 0.96),
            "brainstem": (True, 0.95),
            "cerebellum": (True, 0.88),
        },
        "clip": (False, "sagittal", 50),
        "view": "lateral",
    },
    "ribbon": {
        "label": "Ruban cortical",
        "hint": "La surface piale et la surface white sont comparées en coupe.",
        "groups": {
            "cortex": (True, 0.48),
            "white_matter": (True, 0.95),
            "brainstem": (True, 0.55),
            "cerebellum": (True, 0.48),
        },
        "clip": (True, "coronal", 50),
        "view": "anterior",
    },
    "deep": {
        "label": "Structures profondes",
        "hint": "Le cortex devient un repère discret pour dégager les noyaux profonds.",
        "groups": {
            "cortex": (True, 0.10),
            "white_matter": (True, 0.06),
            "deep": (True, 0.98),
            "ventricles": (True, 0.86),
            "corpus_callosum": (True, 0.90),
            "brainstem": (True, 0.92),
            "cerebellum": (True, 0.38),
        },
        "clip": (True, "sagittal", 50),
        "view": "lateral",
    },
    "ventricles": {
        "label": "Ventricules",
        "hint": "Le système ventriculaire est isolé dans un contexte anatomique léger.",
        "groups": {
            "cortex": (True, 0.07),
            "white_matter": (True, 0.04),
            "deep": (True, 0.18),
            "ventricles": (True, 1.0),
            "corpus_callosum": (True, 0.20),
            "brainstem": (True, 0.28),
        },
        "clip": (True, "sagittal", 50),
        "view": "lateral",
    },
}

GROUP_ORDER = tuple(GROUP_LABELS)

LESSONS = {
    "orientation": {
        "title": "1 · S'orienter",
        "text": (
            "Le cortex est la nappe plissée externe. Les éléments sous-corticaux "
            "sont des volumes pleins placés en profondeur."
        ),
        "structure": "cortex",
    },
    "peel": {
        "title": "2 · Éplucher le cerveau",
        "text": (
            "Rends le cortex transparent: la substance blanche apparaît, puis les "
            "noyaux profonds et les ventricules deviennent visibles."
        ),
        "structure": "white_matter",
    },
    "ribbon": {
        "title": "3 · Voir le ruban cortical",
        "text": (
            "En coupe, la surface white borde la matière blanche et la surface pial "
            "borde le LCR. L'épaisseur est la distance entre les deux."
        ),
        "structure": "cortex",
    },
    "deep": {
        "title": "4 · Explorer le sous-cortical",
        "text": (
            "Thalamus, caudé, putamen et autres noyaux sont des blocs profonds. "
            "Dans aseg.stats, FreeSurfer rapporte surtout leur volume."
        ),
        "structure": "thalamus",
    },
    "metrics": {
        "title": "5 · Relier surface, épaisseur et volume",
        "text": (
            "Une parcelle corticale possède une aire et une épaisseur locale. "
            "Son volume gris correspond à l'addition de petites colonnes du ruban."
        ),
        "structure": "cortex",
    },
    "files": {
        "title": "6 · Retrouver les noms FreeSurfer",
        "text": (
            "aparc.stats décrit les régions corticales; aseg.stats décrit les "
            "structures segmentées et les mesures globales."
        ),
        "structure": "ventricles",
    },
}


class Explorer:
    """Scène PyVista, état pédagogique et interface Trame."""

    def __init__(self, model: ReferenceModel, server: Any) -> None:
        self.model = model
        self.server = server
        self.state = server.state
        self.ctrl = server.controller
        self.plotter = pv.Plotter(window_size=(1100, 760))
        self.plotter.set_background("#101923", top="#243242")
        self.actors_by_group: dict[str, list[Any]] = defaultdict(list)
        self.parts_by_actor: dict[str, MeshPart] = {}
        self.original_meshes: dict[str, pv.PolyData] = {}
        self.actor_keys_by_structure: dict[str, list[str]] = defaultdict(list)
        self.highlight_actors: list[Any] = []
        self._build_scene()
        self._initialize_state()
        self._bind_state()

    def _build_scene(self) -> None:

        for index, part in enumerate(self.model.parts):
            key = f"{part.structure_id}:{part.side or 'centre'}:{index}"
            default_visible, default_opacity = GROUP_DEFAULTS[part.group]
            actor = self.plotter.add_mesh(
                part.mesh,
                name=key,
                color=part.color,
                opacity=default_opacity,
                smooth_shading=True,
                specular=0.18,
                pickable=True,
                show_edges=False,
            )
            actor.visibility = default_visible
            self.actors_by_group[part.group].append(actor)
            self.parts_by_actor[key] = part
            self.original_meshes[key] = part.mesh
            self.actor_keys_by_structure[part.structure_id].append(key)

        mesh_bounds = np.asarray(
            [mesh.bounds for mesh in self.original_meshes.values()], dtype=float
        )
        self.scene_bounds = (
            float(mesh_bounds[:, 0].min()),
            float(mesh_bounds[:, 1].max()),
            float(mesh_bounds[:, 2].min()),
            float(mesh_bounds[:, 3].max()),
            float(mesh_bounds[:, 4].min()),
            float(mesh_bounds[:, 5].max()),
        )

        self.plotter.camera_position = "xz"
        self.plotter.camera.azimuth = 20
        self.plotter.camera.elevation = 8
        self.plotter.reset_camera()
        self.plotter.enable_mesh_picking(
            callback=self._on_pick,
            use_actor=True,
            show=False,
            left_clicking=True,
            picker="hardware",
        )

    def _initialize_state(self) -> None:
        state = self.state
        state.trame__title = "Comprendre FreeSurfer"
        state.structure_items = structure_options()
        state.measure_items = measure_options()
        state.lesson_items = [
            {"title": data["title"], "value": identifier}
            for identifier, data in LESSONS.items()
        ]
        state.selected_structure = "cortex"
        state.selected_measure = "ThickAvg"
        state.lesson_step = "orientation"
        state.clip_enabled = False
        state.clip_axis = "sagittal"
        state.clip_position = 50
        state.clip_invert = False
        state.explore_mode = "external"
        state.explore_hint = EXPLORATION_MODES["external"]["hint"]
        state.info_tab = "anatomy"
        state.model_mode = self.model.mode
        state.model_source = self.model.source
        state.model_warning = " ".join(self.model.warnings)
        state.pick_data = None
        for group, (visible, opacity) in GROUP_DEFAULTS.items():
            setattr(state, f"show_{group}", visible)
            setattr(state, f"opacity_{group}", round(opacity * 100))
        self._update_structure_panel("cortex")
        self._update_measure_panel("ThickAvg")
        self._update_lesson_panel("orientation")

    def _bind_state(self) -> None:
        state = self.state

        @state.change("selected_structure")
        def _selected_structure(selected_structure: str, **_: Any) -> None:
            if selected_structure in STRUCTURES_BY_ID:
                self.select_structure(selected_structure)

        @state.change("selected_measure")
        def _selected_measure(selected_measure: str, **_: Any) -> None:
            if selected_measure in MEASURES_BY_ID:
                self._update_measure_panel(selected_measure)

        @state.change("lesson_step")
        def _lesson_step(lesson_step: str, **_: Any) -> None:
            if lesson_step in LESSONS:
                self.apply_lesson(lesson_step)

        @state.change("explore_mode")
        def _explore_mode(explore_mode: str, **_: Any) -> None:
            if explore_mode in EXPLORATION_MODES:
                self.apply_exploration_mode(explore_mode)

        @state.change("clip_enabled", "clip_axis", "clip_position", "clip_invert")
        def _clip(**_: Any) -> None:
            self.apply_clip()

        @state.change("pick_data")
        def _pick_data(pick_data: dict[str, Any] | None, **_: Any) -> None:
            if pick_data and pick_data.get("worldPosition"):
                self._on_world_pick(pick_data["worldPosition"])

        for group in GROUP_ORDER:
            state.change(f"show_{group}", f"opacity_{group}")(
                self._make_group_callback(group)
            )

        self.ctrl.reset_camera = self.reset_camera
        self.ctrl.show_anterior = lambda: self.set_view("anterior")
        self.ctrl.show_lateral = lambda: self.set_view("lateral")
        self.ctrl.show_superior = lambda: self.set_view("superior")
        self.ctrl.reset_layers = self.reset_layers
        self.ctrl.center_clip = self.center_clip
        self.ctrl.invert_clip = self.invert_clip
        self.ctrl.focus_selected = self.focus_selected
    def _make_group_callback(self, group: str) -> Any:
        def _callback(**_: Any) -> None:
            visible = bool(getattr(self.state, f"show_{group}"))
            opacity = float(getattr(self.state, f"opacity_{group}")) / 100
            for actor in self.actors_by_group[group]:
                actor.visibility = visible
                actor.prop.opacity = opacity
            self._render()

        return _callback

    def _actor_key(self, actor: Any) -> str | None:
        for key in self.parts_by_actor:
            if self.plotter.renderer.actors.get(key) is actor:
                return key
        return None

    def _on_pick(self, actor: Any) -> None:
        key = self._actor_key(actor)
        if key is None:
            return
        part = self.parts_by_actor[key]
        self.state.selected_structure = part.structure_id

    def _on_world_pick(self, world_position: list[float]) -> None:
        """Associe un clic 3D à la structure visible la plus proche."""

        point = world_position[:3]
        nearest: tuple[float, str] | None = None
        for key, part in self.parts_by_actor.items():
            actor = self.plotter.renderer.actors.get(key)
            if actor is None or not actor.visibility or actor.prop.opacity < 0.03:
                continue
            mesh = actor.mapper.dataset
            if mesh is None or mesh.n_points == 0:
                continue
            index = mesh.find_closest_point(point)
            distance = float(np.linalg.norm(mesh.points[index] - point))
            candidate = (distance, part.structure_id)
            if nearest is None or candidate[0] < nearest[0]:
                nearest = candidate
        if nearest is not None:
            self.state.selected_structure = nearest[1]
    def _update_structure_panel(self, identifier: str) -> None:
        info = STRUCTURES_BY_ID[identifier]
        sources = sorted(
            {
                self.parts_by_actor[key].source
                for key in self.actor_keys_by_structure.get(identifier, [])
            }
        )
        self.state.selected_name = info.name_fr
        self.state.selected_family = info.family
        self.state.selected_location = info.location
        self.state.selected_role = info.role
        self.state.selected_fs_names = " · ".join(info.freesurfer_names)
        self.state.selected_files = " · ".join(info.files)
        self.state.selected_measures = " · ".join(info.measures)
        self.state.selected_source = " + ".join(sources) or "description pédagogique"

    def _update_measure_panel(self, identifier: str) -> None:
        measure = MEASURES_BY_ID[identifier]
        self.state.measure_mode = identifier
        self.state.measure_label = measure.label
        self.state.measure_source = measure.source
        self.state.measure_unit = measure.unit
        self.state.measure_family = measure.family
        self.state.measure_explanation = measure.explanation
        self.state.measure_visual = measure.visual

    def _update_lesson_panel(self, identifier: str) -> None:
        lesson = LESSONS[identifier]
        self.state.lesson_title = lesson["title"]
        self.state.lesson_text = lesson["text"]

    def _replace_highlight(self, identifier: str) -> None:
        for actor in self.highlight_actors:
            self.plotter.remove_actor(actor, render=False)
        self.highlight_actors.clear()
        for key in self.actor_keys_by_structure.get(identifier, []):
            part = self.parts_by_actor[key]
            highlighted_mesh = self._clipped_mesh(part.mesh)
            actor = self.plotter.add_mesh(
                highlighted_mesh,
                name=f"highlight:{key}",
                color="#ffd166",
                opacity=0.58,
                smooth_shading=True,
                specular=0.05,
                pickable=False,
                show_edges=False,
            )
            self.highlight_actors.append(actor)

    def select_structure(self, identifier: str) -> None:
        self._update_structure_panel(identifier)
        self._replace_highlight(identifier)
        self._render()

    def apply_exploration_mode(self, identifier: str) -> None:
        mode = EXPLORATION_MODES[identifier]
        selected_by_mode = {
            "external": "cortex",
            "ribbon": "white_matter",
            "deep": "thalamus",
            "ventricles": "ventricles",
        }
        configured_groups = mode["groups"]
        for group in GROUP_ORDER:
            visible, opacity = configured_groups.get(
                group, (False, GROUP_DEFAULTS[group][1])
            )
            setattr(self.state, f"show_{group}", visible)
            setattr(self.state, f"opacity_{group}", round(opacity * 100))
        clip_enabled, clip_axis, clip_position = mode["clip"]
        self.state.explore_hint = mode["hint"]
        self.state.clip_enabled = clip_enabled
        self.state.clip_axis = clip_axis
        self.state.clip_position = clip_position
        self.state.clip_invert = False
        self.state.selected_structure = selected_by_mode[identifier]
        self.set_view(mode["view"])
        self.apply_clip()

    def focus_selected(self) -> None:
        identifier = self.state.selected_structure
        keys = self.actor_keys_by_structure.get(identifier, [])
        if not keys:
            return
        target_groups = {self.parts_by_actor[key].group for key in keys}
        for group in GROUP_ORDER:
            setattr(self.state, f"show_{group}", False)
        if target_groups & {"deep", "ventricles", "corpus_callosum"}:
            self.state.show_cortex = True
            self.state.opacity_cortex = 8
            self.state.show_white_matter = True
            self.state.opacity_white_matter = 4
        elif "white_matter" in target_groups:
            self.state.show_cortex = True
            self.state.opacity_cortex = 32
        for group in target_groups:
            setattr(self.state, f"show_{group}", True)
            setattr(self.state, f"opacity_{group}", 100)
        self.state.clip_enabled = False
        self.apply_clip()
        selected_bounds = np.asarray(
            [self.original_meshes[key].bounds for key in keys], dtype=float
        )
        bounds = (
            float(selected_bounds[:, 0].min()),
            float(selected_bounds[:, 1].max()),
            float(selected_bounds[:, 2].min()),
            float(selected_bounds[:, 3].max()),
            float(selected_bounds[:, 4].min()),
            float(selected_bounds[:, 5].max()),
        )
        self.plotter.reset_camera(bounds=bounds)
        self._render()

    def apply_lesson(self, identifier: str) -> None:
        self._update_lesson_panel(identifier)
        lesson = LESSONS[identifier]
        if identifier == "orientation":
            self.reset_layers()
            self.state.clip_enabled = False
            self.set_view("lateral")
        elif identifier == "peel":
            self.state.show_cortex = True
            self.state.opacity_cortex = 12
            self.state.show_white_matter = True
            self.state.opacity_white_matter = 20
            self.state.show_deep = True
            self.state.show_ventricles = True
            self.state.clip_enabled = False
            self.set_view("anterior")
        elif identifier == "ribbon":
            self.state.show_cortex = True
            self.state.opacity_cortex = 55
            self.state.show_white_matter = True
            self.state.opacity_white_matter = 75
            self.state.show_deep = False
            self.state.clip_enabled = True
            self.state.clip_axis = "coronal"
            self.state.clip_position = 49
            self.set_view("anterior")
        elif identifier == "deep":
            self.state.show_cortex = True
            self.state.opacity_cortex = 10
            self.state.show_white_matter = True
            self.state.opacity_white_matter = 8
            self.state.show_deep = True
            self.state.show_ventricles = True
            self.state.clip_enabled = True
            self.state.clip_axis = "sagittal"
            self.state.clip_position = 52
            self.set_view("lateral")
        elif identifier == "metrics":
            self.state.selected_measure = "ThickAvg"
            self.state.show_cortex = True
            self.state.opacity_cortex = 58
            self.state.show_white_matter = True
            self.state.opacity_white_matter = 90
            self.state.clip_enabled = True
            self.state.clip_axis = "coronal"
            self.state.clip_position = 52
            self.set_view("anterior")
        elif identifier == "files":
            self.state.clip_enabled = True
            self.state.clip_axis = "sagittal"
            self.state.clip_position = 54
        self.state.selected_structure = lesson["structure"]
        self.apply_clip()

    def _clip_parameters(self) -> tuple[tuple[int, int, int], list[float]]:
        axis = self.state.clip_axis
        axis_index = {"sagittal": 0, "coronal": 1, "axial": 2}[axis]
        normal_by_axis = {
            "sagittal": (1, 0, 0),
            "coronal": (0, -1, 0),
            "axial": (0, 0, 1),
        }
        position = float(self.state.clip_position) / 100
        low = self.scene_bounds[axis_index * 2]
        high = self.scene_bounds[axis_index * 2 + 1]
        coordinate = low + (high - low) * position
        origin = [
            (self.scene_bounds[0] + self.scene_bounds[1]) / 2,
            (self.scene_bounds[2] + self.scene_bounds[3]) / 2,
            (self.scene_bounds[4] + self.scene_bounds[5]) / 2,
        ]
        origin[axis_index] = coordinate
        return normal_by_axis[axis], origin

    def _clipped_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        if not bool(self.state.clip_enabled):
            return mesh
        normal, origin = self._clip_parameters()
        try:
            return mesh.clip(
                normal=normal,
                origin=origin,
                invert=bool(self.state.clip_invert),
            )
        except Exception:
            return mesh

    def apply_clip(self) -> None:
        for key, mesh in self.original_meshes.items():
            actor = self.plotter.renderer.actors.get(key)
            if actor is not None:
                actor.mapper.dataset = self._clipped_mesh(mesh)
        identifier = getattr(self.state, "selected_structure", None)
        if identifier in STRUCTURES_BY_ID:
            self._replace_highlight(identifier)
        self._render()

    def center_clip(self) -> None:
        self.state.clip_position = 50
        self.apply_clip()

    def invert_clip(self) -> None:
        self.state.clip_invert = not bool(self.state.clip_invert)
        self.apply_clip()

    def reset_layers(self) -> None:
        self.state.explore_mode = "external"
        self.apply_exploration_mode("external")

    def reset_camera(self) -> None:
        self.plotter.reset_camera()
        self._render()

    def set_view(self, view: str) -> None:
        if view == "anterior":
            self.plotter.view_xz()
        elif view == "lateral":
            self.plotter.view_yz()
        elif view == "superior":
            self.plotter.view_xy()
        self.plotter.reset_camera()
        self._render()

    def _render(self) -> None:
        self.plotter.render()
        update = getattr(self.ctrl, "view_update", None)
        update_exists = getattr(update, "exists", None)
        if callable(update) and (not callable(update_exists) or update_exists()):
            update()

    def build_ui(self) -> None:
        ctrl = self.ctrl
        with SinglePageLayout(self.server) as layout:
            layout.icon.hide()
            layout.title.set_text("Comprendre FreeSurfer")
            with layout.toolbar:
                v3.VChip(
                    "{{ model_mode }}",
                    size="small",
                    variant="tonal",
                    classes="ml-3",
                )
                v3.VSpacer()
                v3.VBtn("Antérieur", variant="text", size="small", click=ctrl.show_anterior)
                v3.VBtn("Latéral", variant="text", size="small", click=ctrl.show_lateral)
                v3.VBtn("Supérieur", variant="text", size="small", click=ctrl.show_superior)
                v3.VBtn("Recentrer", variant="text", size="small", click=ctrl.reset_camera)

            with layout.content:
                client.Style(
                    """
                    :root {
                      --ink: #17232d;
                      --muted: #62717d;
                      --line: #d9e1e7;
                      --panel: #ffffff;
                      --canvas: #142230;
                      --accent: #0f6f7c;
                      --accent-soft: #dceff1;
                    }
                    html, body, #app { overflow: hidden; }
                    .explorer-grid {
                      display: grid;
                      grid-template-columns: minmax(300px, 340px) minmax(500px, 1fr) minmax(330px, 390px);
                      height: calc(100vh - 64px);
                      min-height: 640px;
                      background: #edf1f4;
                      color: var(--ink);
                    }
                    .side-panel {
                      min-width: 0;
                      overflow-y: auto;
                      background: var(--panel);
                      padding: 18px;
                      scrollbar-color: #aab6bf transparent;
                      scrollbar-width: thin;
                    }
                    .left-panel { border-right: 1px solid var(--line); }
                    .right-panel { border-left: 1px solid var(--line); }
                    .section-kicker {
                      color: var(--accent);
                      font-size: .72rem;
                      font-weight: 800;
                      letter-spacing: .08em;
                      text-transform: uppercase;
                    }
                    .panel-title { font-size: 1.25rem; line-height: 1.2; margin: 3px 0 14px; }
                    .section-title { font-size: .95rem; margin: 20px 0 8px; }
                    .helper-text { color: var(--muted); font-size: .82rem; line-height: 1.4; }
                    .mode-toggle {
                      display: grid !important;
                      grid-template-columns: 1fr 1fr;
                      width: 100%;
                      height: auto !important;
                      gap: 7px;
                      background: transparent !important;
                      box-shadow: none !important;
                    }
                    .mode-toggle .v-btn {
                      min-width: 0 !important;
                      border: 1px solid #b9c7d0 !important;
                      border-radius: 8px !important;
                      text-transform: none;
                      letter-spacing: 0;
                    }
                    .layer-control {
                      margin-bottom: 7px;
                      padding: 8px 10px 5px;
                      border: 1px solid var(--line);
                      border-radius: 9px;
                      background: #fbfcfd;
                    }
                    .layer-heading { display: flex; align-items: center; gap: 6px; min-width: 0; }
                    .layer-heading .v-input { flex: 1 1 auto; min-width: 0; }
                    .opacity-value {
                      flex: 0 0 auto;
                      color: var(--muted);
                      font-size: .72rem;
                      font-variant-numeric: tabular-nums;
                    }
                    .layer-control .v-slider-track__background,
                    .clip-controls .v-slider-track__background {
                      opacity: 1 !important;
                      background: #c3ced6 !important;
                    }
                    .layer-control .v-slider-track__fill,
                    .clip-controls .v-slider-track__fill,
                    .layer-control .v-slider-thumb__surface,
                    .clip-controls .v-slider-thumb__surface {
                      background: var(--accent) !important;
                    }
                    .layer-control .v-input--disabled { opacity: .48 !important; }
                    .cut-card {
                      border: 1px solid #b9d7db;
                      border-radius: 10px;
                      padding: 10px 12px 12px;
                      background: #f3fafb;
                    }
                    .axis-toggle { width: 100%; }
                    .axis-toggle .v-btn { flex: 1 1 0; min-width: 0; padding: 0 7px; }
                    .viewer-shell {
                      position: relative;
                      min-width: 0;
                      height: calc(100vh - 64px);
                      min-height: 640px;
                      overflow: hidden;
                      background: var(--canvas);
                    }
                    .vtk-container { width: 100%; height: 100%; }
                    .viewer-instructions {
                      position: absolute;
                      z-index: 3;
                      top: 14px;
                      left: 14px;
                      padding: 7px 10px;
                      border: 1px solid rgba(255,255,255,.16);
                      border-radius: 8px;
                      color: #e9f2f6;
                      background: rgba(8,18,28,.76);
                      font-size: .78rem;
                      pointer-events: none;
                    }
                    .viewer-mode {
                      position: absolute;
                      z-index: 3;
                      left: 14px;
                      bottom: 14px;
                      max-width: 420px;
                      padding: 9px 11px;
                      border-left: 3px solid #5fc1cb;
                      color: #e9f2f6;
                      background: rgba(8,18,28,.80);
                      font-size: .8rem;
                      line-height: 1.35;
                      pointer-events: none;
                    }
                    .info-card {
                      border: 1px solid var(--line);
                      border-left: 4px solid #d98b5f;
                      border-radius: 10px;
                      padding: 15px;
                      background: #fff;
                    }
                    .info-tabs { width: 100%; }
                    .info-tabs .v-btn { flex: 1 1 0; }
                    code {
                      display: block;
                      white-space: normal;
                      overflow-wrap: anywhere;
                      color: #33444f;
                      background: #f3f5f7;
                      border-radius: 5px;
                      padding: 4px 6px;
                    }
                    @media (max-width: 1180px) {
                      html, body, #app { overflow: auto; }
                      .explorer-grid { grid-template-columns: 300px minmax(500px, 1fr); height: auto; }
                      .right-panel { grid-column: 1 / -1; border-left: 0; border-top: 1px solid var(--line); }
                      .side-panel { max-height: none; }
                    }
                    @media (max-width: 820px) {
                      .explorer-grid { display: flex; flex-direction: column; min-height: 0; }
                      .viewer-shell { order: -1; height: 62vh; min-height: 460px; }
                    }
                    """
                )
                with html.Div(classes="explorer-grid"):
                    with html.Div(classes="side-panel left-panel"):
                        html.Div("Exploration guidée", classes="section-kicker")
                        html.H2("Choisir ce que l’on veut comprendre", classes="panel-title")
                        v3.VSelect(
                            v_model=("lesson_step", "orientation"),
                            items=("lesson_items",),
                            item_title="title",
                            item_value="value",
                            label="Parcours pédagogique",
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                            classes="mb-3",
                        )
                        with v3.VAlert(type="info", variant="tonal", density="compact"):
                            html.Strong("{{ lesson_title }}")
                            html.Div("{{ lesson_text }}", classes="mt-1 text-body-2")

                        html.H3("Vues intelligentes", classes="section-title")
                        with v3.VBtnToggle(
                            v_model=("explore_mode", "external"),
                            mandatory=True,
                            color="primary",
                            variant="outlined",
                            density="compact",
                            classes="mode-toggle",
                        ):
                            v3.VBtn("Extérieur", value="external", size="small")
                            v3.VBtn("Ruban", value="ribbon", size="small")
                            v3.VBtn("Profond", value="deep", size="small")
                            v3.VBtn("Ventricules", value="ventricles", size="small")
                        html.P("{{ explore_hint }}", classes="helper-text mt-2 mb-0")

                        html.H3("Couches", classes="section-title")
                        html.P(
                            "Active une couche puis règle sa transparence. Les vues ci-dessus font les réglages utiles automatiquement.",
                            classes="helper-text mb-2",
                        )
                        for group in GROUP_ORDER:
                            with html.Div(classes="layer-control"):
                                with html.Div(classes="layer-heading"):
                                    v3.VSwitch(
                                        v_model=(f"show_{group}", GROUP_DEFAULTS[group][0]),
                                        label=GROUP_LABELS[group],
                                        density="compact",
                                        hide_details=True,
                                        color="primary",
                                        inset=True,
                                    )
                                    html.Span(
                                        f"{{{{ opacity_{group} }}}} %",
                                        classes="opacity-value",
                                    )
                                v3.VSlider(
                                    v_model=(
                                        f"opacity_{group}",
                                        round(GROUP_DEFAULTS[group][1] * 100),
                                    ),
                                    min=0,
                                    max=100,
                                    step=1,
                                    density="compact",
                                    hide_details=True,
                                    color="#0f6f7c",
                                    track_color="#c3ced6",
                                    thumb_label=True,
                                    disabled=(f"!show_{group}",),
                                )

                        html.H3("Coupe anatomique", classes="section-title")
                        with html.Div(classes="cut-card clip-controls"):
                            v3.VSwitch(
                                v_model=("clip_enabled", False),
                                label="Afficher une coupe",
                                density="compact",
                                hide_details=True,
                                color="primary",
                                inset=True,
                            )
                            with v3.VBtnToggle(
                                v_model=("clip_axis", "sagittal"),
                                mandatory=True,
                                divided=True,
                                color="primary",
                                density="compact",
                                variant="outlined",
                                disabled=("!clip_enabled",),
                                classes="axis-toggle mt-3",
                            ):
                                v3.VBtn("Sagittale", value="sagittal", size="x-small")
                                v3.VBtn("Coronale", value="coronal", size="x-small")
                                v3.VBtn("Axiale", value="axial", size="x-small")
                            html.Div(
                                "Position du plan · {{ clip_position }} %",
                                classes="helper-text mt-3",
                            )
                            v3.VSlider(
                                v_model=("clip_position", 50),
                                min=2,
                                max=98,
                                step=1,
                                density="compact",
                                hide_details=True,
                                color="#0f6f7c",
                                track_color="#c3ced6",
                                thumb_label=True,
                                disabled=("!clip_enabled",),
                            )
                            with v3.VRow(dense=True, classes="mt-1"):
                                with v3.VCol(cols=6):
                                    v3.VBtn(
                                        "Centrer",
                                        block=True,
                                        size="small",
                                        variant="outlined",
                                        disabled=("!clip_enabled",),
                                        click=ctrl.center_clip,
                                    )
                                with v3.VCol(cols=6):
                                    v3.VBtn(
                                        "Autre moitié",
                                        block=True,
                                        size="small",
                                        variant="outlined",
                                        disabled=("!clip_enabled",),
                                        click=ctrl.invert_clip,
                                    )
                        v3.VBtn(
                            "Revenir à la vue extérieure",
                            variant="text",
                            block=True,
                            classes="mt-3",
                            click=ctrl.reset_layers,
                        )

                    with html.Div(classes="viewer-shell"):
                        html.Div(
                            "Glisser : tourner · Molette : zoomer · Clic : identifier",
                            classes="viewer-instructions",
                        )
                        html.Div("{{ explore_hint }}", classes="viewer-mode")
                        with html.Div(classes="vtk-container"):
                            view = vtk.VtkLocalView(
                                self.plotter.ren_win,
                                picking_modes=("['click']",),
                                click="pick_data = $event",
                                style="width: 100%; height: 100%; min-height: 640px;",
                            )
                            ctrl.view_update = view.update
                            ctrl.view_reset_camera = view.reset_camera

                    with html.Div(classes="side-panel right-panel"):
                        html.Div("Comprendre la sélection", classes="section-kicker")
                        html.H2("Structure et mesure FreeSurfer", classes="panel-title")
                        with v3.VBtnToggle(
                            v_model=("info_tab", "anatomy"),
                            mandatory=True,
                            divided=True,
                            color="primary",
                            density="compact",
                            variant="outlined",
                            classes="info-tabs mb-4",
                        ):
                            v3.VBtn("Anatomie", value="anatomy", size="small")
                            v3.VBtn("Mesures", value="measure", size="small")

                        with html.Div(v_if="info_tab === 'anatomy'"):
                            v3.VSelect(
                                v_model=("selected_structure", "cortex"),
                                items=("structure_items",),
                                item_title="title",
                                item_value="value",
                                label="Chercher une structure",
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                classes="mb-3",
                            )
                            v3.VBtn(
                                "Isoler et centrer la structure",
                                block=True,
                                color="primary",
                                variant="tonal",
                                classes="mb-3",
                                click=ctrl.focus_selected,
                            )
                            with html.Div(classes="info-card"):
                                html.H2("{{ selected_name }}", classes="text-h5 mb-1")
                                v3.VChip(
                                    "{{ selected_family }}",
                                    size="small",
                                    variant="tonal",
                                    classes="mb-3",
                                )
                                with v3.VAlert(
                                    v_if="selected_family === 'Cortical'",
                                    type="info",
                                    variant="tonal",
                                    density="compact",
                                    classes="mb-3",
                                ):
                                    html.Span("Nappe corticale : surface, épaisseur et volume gris.")
                                with v3.VAlert(
                                    v_if="selected_family === 'Sous-cortical'",
                                    type="info",
                                    variant="tonal",
                                    density="compact",
                                    classes="mb-3",
                                ):
                                    html.Span("Structure profonde : FreeSurfer rapporte principalement son volume.")
                                html.P("{{ selected_location }}", classes="mb-2")
                                html.P("{{ selected_role }}", classes="mb-3")
                                html.Div("Nom exact FreeSurfer", classes="text-caption")
                                html.Code("{{ selected_fs_names }}")
                                html.Div("Fichier(s)", classes="text-caption mt-3")
                                html.Code("{{ selected_files }}")
                                html.Div("Mesure(s)", classes="text-caption mt-3")
                                html.Strong("{{ selected_measures }}")
                                html.Div("Géométrie affichée", classes="text-caption mt-3")
                                html.Span("{{ selected_source }}")

                        with html.Div(v_if="info_tab === 'measure'"):
                            v3.VSelect(
                                v_model=("selected_measure", "ThickAvg"),
                                items=("measure_items",),
                                item_title="title",
                                item_value="value",
                                label="Choisir une mesure",
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                classes="mb-3",
                            )
                            with html.Div(classes="info-card"):
                                html.H2("{{ measure_label }}", classes="text-h5 mb-2")
                                with v3.VChipGroup(classes="mb-3"):
                                    v3.VChip("{{ measure_unit }}", size="small", variant="tonal")
                                    v3.VChip("{{ measure_family }}", size="small", variant="tonal")
                                html.P("{{ measure_explanation }}", classes="mb-3")
                                html.Div("Dans le fichier", classes="text-caption")
                                html.Code("{{ measure_source }}")
                                html.Div("Interprétation visuelle", classes="text-caption mt-3")
                                html.Span("{{ measure_visual }}")

                        with v3.VAlert(
                            type="info",
                            variant="tonal",
                            density="compact",
                            classes="mt-4",
                        ):
                            html.Div("{{ model_source }}", classes="font-weight-medium")
                            html.Div("{{ model_warning }}", classes="text-caption mt-1")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance l'explorateur 3D éducatif de FreeSurfer."
    )
    parser.add_argument(
        "--schematic",
        action="store_true",
        help="N'utilise aucun téléchargement; charge le modèle pédagogique local.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Dossier du cache Nilearn (par défaut: cache utilisateur).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Ne pas ouvrir automatiquement le navigateur.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYVISTA_TRAME_SERVER_PROXY_PREFIX", "")
    model = load_reference_model(args.cache_dir, force_schematic=args.schematic)
    server = get_server("freesurfer-anatomy-explorer", client_type="vue3")
    explorer = Explorer(model, server)
    explorer.build_ui()
    server.start(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

