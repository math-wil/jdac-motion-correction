"""Tests légers du contenu pédagogique, sans lancer l'interface 3D."""

from catalog import MEASURES_BY_ID, STRUCTURES_BY_ID


def test_requested_deep_structures_are_catalogued() -> None:
    expected = {
        "thalamus",
        "caudate",
        "putamen",
        "pallidum",
        "hippocampus",
        "amygdala",
        "accumbens",
        "ventraldc",
    }
    assert expected <= STRUCTURES_BY_ID.keys()


def test_cortical_measure_units() -> None:
    assert MEASURES_BY_ID["ThickAvg"].unit == "mm"
    assert MEASURES_BY_ID["SurfArea"].unit == "mm²"
    assert MEASURES_BY_ID["GrayVol"].unit == "mm³"


def test_global_freesurfer_measures_are_explained() -> None:
    expected = {
        "CortexVol",
        "SubCortGrayVol",
        "TotalGrayVol",
        "CerebralWhiteMatterVol",
        "CSF",
        "BrainSegVol",
        "BrainSegVolNotVent",
        "SupraTentorialVol",
        "eTIV",
        "to_eTIV",
        "SurfaceHoles",
    }
    assert expected <= MEASURES_BY_ID.keys()

