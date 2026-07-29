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
    "cortex": (True, 0.42),
    "white_matter": (True, 0.28),
    "csf": (False, 0.10),
    "deep": (True, 0.95),
    "ventricles": (True, 0.85),
    "corpus_callosum": (True, 0.95),
    "brainstem": (True, 0.95),
    "cerebellum": (True, 0.75),
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
        self.plotter.add_axes(
            xlabel="Droite",
            ylabel="Antérieur",
            zlabel="Supérieur",
            line_width=2,
        )
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

        @state.change("clip_enabled", "clip_axis", "clip_position")
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

    def select_structure(self, identifier: str) -> None:
        self._update_structure_panel(identifier)
        for actor in self.highlight_actors:
            self.plotter.remove_actor(actor, render=False)
        self.highlight_actors.clear()
        for key in self.actor_keys_by_structure.get(identifier, []):
            part = self.parts_by_actor[key]
            actor = self.plotter.add_mesh(
                part.mesh,
                name=f"highlight:{key}",
                style="wireframe",
                color="#fff4ad",
                line_width=4,
                opacity=0.9,
                pickable=False,
            )
            self.highlight_actors.append(actor)
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

    def apply_clip(self) -> None:
        enabled = bool(self.state.clip_enabled)
        axis = self.state.clip_axis
        position = float(self.state.clip_position) / 100
        normal_by_axis = {
            "sagittal": (1, 0, 0),
            "coronal": (0, -1, 0),
            "axial": (0, 0, 1),
        }
        axis_index = {"sagittal": 0, "coronal": 1, "axial": 2}[axis]
        for key, mesh in self.original_meshes.items():
            actor = self.plotter.renderer.actors.get(key)
            if actor is None:
                continue
            if not enabled:
                actor.mapper.dataset = mesh
                continue
            bounds = mesh.bounds
            low = bounds[axis_index * 2]
            high = bounds[axis_index * 2 + 1]
            coordinate = low + (high - low) * position
            center = list(mesh.center)
            center[axis_index] = coordinate
            try:
                clipped = mesh.clip(
                    normal=normal_by_axis[axis],
                    origin=center,
                    invert=False,
                )
                actor.mapper.dataset = clipped
            except Exception:
                actor.mapper.dataset = mesh
        self._render()

    def reset_layers(self) -> None:
        for group, (visible, opacity) in GROUP_DEFAULTS.items():
            setattr(self.state, f"show_{group}", visible)
            setattr(self.state, f"opacity_{group}", round(opacity * 100))
        self._render()

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
        if callable(update):
            update()

    def build_ui(self) -> None:
        ctrl = self.ctrl
        with SinglePageLayout(self.server) as layout:
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
                    .explorer-root { background: #f4f6f8; min-height: calc(100vh - 64px); }
                    .control-column { max-height: calc(100vh - 64px); overflow-y: auto; }
                    .viewer-column { background: #101923; min-height: 620px; }
                    .info-block { border-left: 4px solid #d98b5f; }
                    .measure-diagram {
                      position: relative; height: 150px; overflow: hidden;
                      background: linear-gradient(180deg, #dceef7 0 26%, #d99972 26% 56%,
                        #eee4c8 56% 100%);
                      border-radius: 10px;
                    }
                    .measure-diagram .pial, .measure-diagram .white {
                      position: absolute; left: 8%; width: 84%; height: 28px;
                      border-top: 4px solid #8c4e32; border-radius: 50%;
                    }
                    .measure-diagram .pial { top: 38px; }
                    .measure-diagram .white { top: 83px; border-color: #b9aa81; }
                    .measure-diagram .arrow {
                      position: absolute; left: 50%; top: 53px; height: 43px;
                      border-left: 3px solid #26384a;
                    }
                    .measure-diagram .arrow::before, .measure-diagram .arrow::after {
                      content: ""; position: absolute; left: -6px; width: 9px; height: 9px;
                      border-left: 3px solid #26384a; border-top: 3px solid #26384a;
                    }
                    .measure-diagram .arrow::before { top: 0; transform: rotate(45deg); }
                    .measure-diagram .arrow::after { bottom: 0; transform: rotate(225deg); }
                    .diagram-label { position: absolute; font-weight: 500; color: #26384a; }
                    .label-pial { top: 12px; left: 10%; }
                    .label-white { bottom: 8px; left: 10%; }
                    .label-thickness { top: 66px; left: 54%; }
                    .area-patch {
                      position: absolute; left: 20%; top: 34px; width: 46%; height: 18px;
                      background: rgba(255, 224, 92, .78); border: 2px solid #705f13;
                      border-radius: 50%; transform: rotate(-3deg);
                    }
                    .volume-patch {
                      position: absolute; left: 25%; top: 52px; width: 38%; height: 39px;
                      background: rgba(217, 96, 62, .58); border: 2px solid #7b3526;
                      transform: skewX(-12deg);
                    }
                    .deep-block {
                      position: absolute; left: 37%; top: 49px; width: 78px; height: 58px;
                      background: #8f73c5; border: 3px solid #48366f;
                      border-radius: 46% 54% 49% 51%; box-shadow: inset -10px -8px 0 rgba(0,0,0,.12);
                    }
                    .etiv-envelope {
                      position: absolute; left: 11%; top: 18px; width: 78%; height: 112px;
                      border: 4px dashed #26384a; border-radius: 48%;
                    }
                    .global-brain {
                      position: absolute; left: 24%; top: 39px; width: 52%; height: 75px;
                      background: rgba(217, 139, 95, .66); border: 3px solid #8c4e32;
                      border-radius: 48% 52% 44% 56%;
                    }
                    .vtk-container { min-height: 620px; height: calc(100vh - 64px); }
                    @media (max-width: 960px) {
                      .control-column { max-height: none; overflow: visible; }
                      .viewer-column, .vtk-container { min-height: 520px; height: 520px; }
                    }
                    """
                )
                with v3.VContainer(fluid=True, classes="pa-0 explorer-root"):
                    with v3.VRow(no_gutters=True):
                        with v3.VCol(cols=12, md=3, classes="pa-3 control-column"):
                            v3.VSelect(
                                v_model=("lesson_step", "orientation"),
                                items=("lesson_items",),
                                item_title="title",
                                item_value="value",
                                label="Parcours guidé",
                                density="compact",
                                variant="outlined",
                            )
                            with v3.VAlert(
                                type="info",
                                variant="tonal",
                                density="compact",
                                classes="mb-3",
                            ):
                                html.Strong("{{ lesson_title }}")
                                html.Div("{{ lesson_text }}", classes="mt-1")

                            html.H3("Couches", classes="mb-2")
                            for group in GROUP_ORDER:
                                with v3.VCard(
                                    variant="flat",
                                    classes="mb-2 pa-2",
                                    color="surface-variant",
                                ):
                                    v3.VSwitch(
                                        v_model=(f"show_{group}", GROUP_DEFAULTS[group][0]),
                                        label=GROUP_LABELS[group],
                                        density="compact",
                                        hide_details=True,
                                        color="primary",
                                    )
                                    html.Div(
                                        "Opacité: " f"{{{{ opacity_{group} }}}} %",
                                        classes="text-caption mt-1",
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
                                        disabled=(f"!show_{group}",),
                                    )

                            html.H3("Plan de coupe", classes="mt-4 mb-2")
                            v3.VSwitch(
                                v_model=("clip_enabled", False),
                                label="Activer la coupe",
                                density="compact",
                                hide_details=True,
                                color="primary",
                            )
                            v3.VSelect(
                                v_model=("clip_axis", "sagittal"),
                                items=(
                                    [
                                        {"title": "Sagittale · gauche/droite", "value": "sagittal"},
                                        {"title": "Coronale · avant/arrière", "value": "coronal"},
                                        {"title": "Axiale · haut/bas", "value": "axial"},
                                    ],
                                ),
                                item_title="title",
                                item_value="value",
                                density="compact",
                                variant="outlined",
                                label="Orientation",
                                classes="mt-3",
                                disabled=("!clip_enabled",),
                            )
                            html.Div("Position: {{ clip_position }} %", classes="text-caption mt-2")
                            v3.VSlider(
                                v_model=("clip_position", 50),
                                min=2,
                                max=98,
                                step=1,
                                density="compact",
                                hide_details=True,
                                disabled=("!clip_enabled",),
                            )
                            v3.VBtn(
                                "Réinitialiser les couches",
                                variant="outlined",
                                block=True,
                                classes="mt-4",
                                click=ctrl.reset_layers,
                            )

                        with v3.VCol(cols=12, md=6, classes="viewer-column"):
                            with html.Div(
                                classes="vtk-container",
                                style="height: calc(100vh - 64px); min-height: 620px;",
                            ):
                                view = vtk.VtkLocalView(
                                    self.plotter.ren_win,
                                    picking_modes=("['click']",),
                                    click="pick_data = $event",
                                    style="width: 100%; height: 100%; min-height: 620px;",
                                )
                                ctrl.view_update = view.update
                                ctrl.view_reset_camera = view.reset_camera

                        with v3.VCol(cols=12, md=3, classes="pa-3 control-column"):
                            html.H3("Structure sélectionnée", classes="mb-2")
                            v3.VSelect(
                                v_model=("selected_structure", "cortex"),
                                items=("structure_items",),
                                item_title="title",
                                item_value="value",
                                label="Chercher une structure",
                                density="compact",
                                variant="outlined",
                            )
                            with v3.VCard(classes="pa-3 mb-4 info-block"):
                                html.H2("{{ selected_name }}", classes="text-h6")
                                v3.VChip(
                                    "{{ selected_family }}",
                                    size="small",
                                    variant="tonal",
                                    classes="my-2",
                                )
                                with v3.VAlert(
                                    v_if="selected_family === 'Cortical'",
                                    type="info",
                                    variant="tonal",
                                    density="compact",
                                    classes="mb-2",
                                ):
                                    html.Span("NAPPE fine : surface + épaisseur + volume gris.")
                                with v3.VAlert(
                                    v_if="selected_family === 'Sous-cortical'",
                                    type="warning",
                                    variant="tonal",
                                    density="compact",
                                    classes="mb-2",
                                ):
                                    html.Span("BLOC profond : FreeSurfer rapporte son volume.")
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

                            html.H3("Glossaire des mesures", classes="mb-2")
                            v3.VSelect(
                                v_model=("selected_measure", "ThickAvg"),
                                items=("measure_items",),
                                item_title="title",
                                item_value="value",
                                label="Choisir une mesure",
                                density="compact",
                                variant="outlined",
                            )
                            with v3.VCard(classes="pa-3 mb-3"):
                                html.H2("{{ measure_label }}", classes="text-h6")
                                with v3.VChipGroup(classes="my-2"):
                                    v3.VChip("{{ measure_unit }}", size="small", variant="tonal")
                                    v3.VChip("{{ measure_family }}", size="small", variant="tonal")
                                html.P("{{ measure_explanation }}", classes="mb-2")
                                html.Div("Dans le fichier", classes="text-caption")
                                html.Code("{{ measure_source }}")
                                html.Div("Ce que le dessin montre", classes="text-caption mt-3")
                                html.Span("{{ measure_visual }}")

                            with html.Div(classes="measure-diagram"):
                                html.Span(
                                    "surface pial",
                                    v_if="['ThickAvg', 'SurfArea', 'GrayVol'].includes(measure_mode)",
                                    classes="diagram-label label-pial",
                                )
                                html.Div(
                                    v_if="['ThickAvg', 'SurfArea', 'GrayVol'].includes(measure_mode)",
                                    classes="pial",
                                )
                                html.Div(v_if="measure_mode === 'ThickAvg'", classes="arrow")
                                html.Span(
                                    "épaisseur",
                                    v_if="measure_mode === 'ThickAvg'",
                                    classes="diagram-label label-thickness",
                                )
                                html.Div(v_if="measure_mode === 'SurfArea'", classes="area-patch")
                                html.Span(
                                    "parcelle = aire",
                                    v_if="measure_mode === 'SurfArea'",
                                    classes="diagram-label label-thickness",
                                )
                                html.Div(v_if="measure_mode === 'GrayVol'", classes="volume-patch")
                                html.Span(
                                    "aire × hauteur",
                                    v_if="measure_mode === 'GrayVol'",
                                    classes="diagram-label label-thickness",
                                )
                                html.Div(
                                    v_if="['ThickAvg', 'SurfArea', 'GrayVol'].includes(measure_mode)",
                                    classes="white",
                                )
                                html.Span(
                                    "surface white",
                                    v_if="['ThickAvg', 'SurfArea', 'GrayVol'].includes(measure_mode)",
                                    classes="diagram-label label-white",
                                )
                                html.Div(
                                    v_if="measure_mode === 'structure_volume'",
                                    classes="deep-block",
                                )
                                html.Span(
                                    "bloc plein = volume",
                                    v_if="measure_mode === 'structure_volume'",
                                    classes="diagram-label label-pial",
                                )
                                html.Div(
                                    v_if="measure_mode === 'eTIV' || measure_mode === 'to_eTIV'",
                                    classes="etiv-envelope",
                                )
                                html.Div(
                                    v_if="!['ThickAvg', 'SurfArea', 'GrayVol', 'structure_volume'].includes(measure_mode)",
                                    classes="global-brain",
                                )

                            with v3.VAlert(
                                type="warning",
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

