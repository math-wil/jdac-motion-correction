"""Tests de la logique 3D qui ne nécessitent pas de navigateur."""

from trame.app import get_server

from app import Explorer
from model import load_reference_model


def test_sagittal_clip_uses_one_plane_for_every_mesh() -> None:
    model = load_reference_model(force_schematic=True)
    explorer = Explorer(
        model,
        get_server("clip-plane-test", client_type="vue3"),
    )
    explorer.state.clip_enabled = True
    explorer.state.clip_axis = "sagittal"
    explorer.state.clip_position = 50

    _, origin = explorer._clip_parameters()
    coordinate = origin[0]
    explorer.apply_clip()

    crossing_meshes = 0
    aligned_meshes = 0
    for key, original in explorer.original_meshes.items():
        if original.bounds[0] < coordinate < original.bounds[1]:
            clipped = explorer.plotter.renderer.actors[key].mapper.dataset
            if clipped.n_points:
                crossing_meshes += 1
                clipped_limits = (clipped.bounds[0], clipped.bounds[1])
                if min(abs(limit - coordinate) for limit in clipped_limits) < 1e-4:
                    aligned_meshes += 1

    assert crossing_meshes > 0
    assert aligned_meshes == crossing_meshes
