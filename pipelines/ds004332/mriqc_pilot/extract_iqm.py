#!/usr/bin/env python3
"""Extraire et qualifier les IQM du pilote MRIQC de ``sub-19``.

Ce script ne recalcule aucune métrique. Il rassemble les JSON produits par
MRIQC, puis attribue à chaque IQM un rôle méthodologique explicite pour les
sorties brain-only. Une valeur finie et variable n'est pas automatiquement
considérée comme comparable ou scientifiquement valide.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


META = ("group", "acq", "run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brain-json-dir",
        type=Path,
        required=True,
        help="Dossier contenant les 12 JSON MRIQC brain-only.",
    )
    parser.add_argument(
        "--raw-json-dir",
        type=Path,
        required=True,
        help="Dossier contenant les trois JSON MRIQC full-head.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dossier des deux CSV produits.",
    )
    return parser.parse_args()


def read_rows(group: str, directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*_T1w.json")):
        match = re.search(r"acq-([^_]+)_run-(\d+)", path.name)
        if match is None:
            raise ValueError(f"Nom BIDS non reconnu : {path.name}")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        iqm = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows.append(
            {
                "group": group,
                "acq": match.group(1),
                "run": match.group(2),
                **iqm,
            }
        )
    return rows


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def scientific_role(iqm: str) -> tuple[str, str]:
    """Retourner une décision prudente et sa justification."""
    if iqm in {"fber", "qi_1", "qi_2"} or iqm.startswith(
        ("snrd_", "summary_bg_")
    ):
        return (
            "invalide_brainonly",
            "dépend du fond ou de l'air supprimé par le skull-stripping",
        )
    if iqm.startswith(("size_", "spacing_")):
        return (
            "controle_geometrie",
            "décrit la grille commune et non la qualité ou le mouvement",
        )
    if iqm == "cjv" or iqm.startswith("snr_"):
        return (
            "candidate_tissulaire",
            "calculable à support fixe ; association à la morphométrie à tester",
        )
    if iqm == "cnr":
        return (
            "candidate_conditionnelle",
            "calculée malgré un bruit de l'air nul ; ne pas comparer au full-head",
        )
    if iqm == "efc":
        return (
            "diagnostic_support",
            "sensible au support, à la grille et au zero-padding",
        )
    if iqm.startswith("fwhm_"):
        return (
            "diagnostic_lissage",
            "décrit le lissage ; ne prouve pas la fidélité anatomique",
        )
    if iqm.startswith(("rpve_", "icvs_", "tpm_overlap_")):
        return (
            "secondaire_segmentation",
            "dépend de la segmentation MRIQC ; ne remplace pas FreeSurfer",
        )
    if iqm.startswith(("inu_", "summary_csf_", "summary_gm_", "summary_wm_")) or iqm == "wm2max":
        return (
            "secondaire_intensite",
            "diagnostic de tissus ou de preprocessing, pas mesure directe du mouvement",
        )
    return ("a_revoir", "rôle non défini dans le protocole du pilote")


def write_table(rows: list[dict[str, object]], columns: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(META) + columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows("brainonly", args.brain_json_dir) + read_rows(
        "raw", args.raw_json_dir
    )
    brain_rows = [row for row in rows if row["group"] == "brainonly"]
    raw_rows = [row for row in rows if row["group"] == "raw"]
    if len(brain_rows) != 12 or len(raw_rows) != 3:
        raise RuntimeError(
            f"Pilote incomplet : {len(brain_rows)} brain-only et {len(raw_rows)} raw"
        )

    iqm_columns = sorted({key for row in rows for key in row if key not in META})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(rows, iqm_columns, args.output_dir / "iqm_table.csv")

    qualification_path = args.output_dir / "iqm_qualification_brainonly.csv"
    with qualification_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "iqm",
            "decision_brainonly",
            "raison",
            "n_finite",
            "n_total",
            "n_unique_finite",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for iqm in iqm_columns:
            values = finite_values(brain_rows, iqm)
            decision, reason = scientific_role(iqm)
            writer.writerow(
                {
                    "iqm": iqm,
                    "decision_brainonly": decision,
                    "raison": reason,
                    "n_finite": len(values),
                    "n_total": len(brain_rows),
                    "n_unique_finite": len(set(values)),
                }
            )

    print(
        f"{len(rows)} volumes, {len(iqm_columns)} IQM -> "
        f"{args.output_dir / 'iqm_table.csv'}"
    )
    print(f"Qualification prudente -> {qualification_path}")


if __name__ == "__main__":
    main()
