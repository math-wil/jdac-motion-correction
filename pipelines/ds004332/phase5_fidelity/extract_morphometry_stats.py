#!/usr/bin/env python3
"""Extrait les mesures morphométriques FreeSurfer des cinq conditions.

Le fichier produit contient, pour chaque sujet, run et condition :
- l'épaisseur, la surface et le volume gris des régions corticales aparc ;
- les volumes globaux et régionaux de aseg.stats.

Aucune comparaison statistique n'est effectuée ici. Ce script ne fait que
centraliser les mesures sources avec leur unité et leur fichier de provenance.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CONDITIONS = (
    "brut",
    "preproc",
    "jdac",
    "jdac_antiartonly",
    "jdac_nodenoise",
)

DEFAULT_ROOTS = {
    "brut": "~/projects/def-sbouix/mathw/freesurfer_ds004332",
    "preproc": "~/projects/ctb-sbouix/mathw/freesurfer_preproc_rigid_ds004332",
    "jdac": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_ds004332",
    "jdac_antiartonly": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_antiartonly_ds004332",
    "jdac_nodenoise": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_nodenoise_ds004332",
}

ID_RE = re.compile(r"^(sub-[^_]+)_(run-[^_]+)$")
APARC_METRICS = {
    "ThickAvg": ("thickness", "mm"),
    "SurfArea": ("surface_area", "mm2"),
    "GrayVol": ("cortical_gray_volume", "mm3"),
}

OUTPUT_FIELDS = [
    "subject",
    "run",
    "condition",
    "family",
    "hemi",
    "region",
    "metric",
    "value",
    "unit",
    "source_file",
]


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("format attendu : condition=/chemin/SUBJECTS_DIR")
    condition, path = value.split("=", 1)
    if condition not in CONDITIONS:
        raise argparse.ArgumentTypeError(f"condition inconnue : {condition}")
    return condition, Path(path).expanduser()


def split_identifier(identifier: str) -> tuple[str, str]:
    match = ID_RE.match(identifier)
    if not match:
        raise ValueError(f"identifiant sujet/run inattendu : {identifier}")
    return match.groups()


def parse_aparc(
    path: Path,
    condition: str,
    subject: str,
    run: str,
    hemi: str,
) -> list[dict[str, object]]:
    """Lit lh.aparc.stats ou rh.aparc.stats en respectant ColHeaders."""
    headers: list[str] | None = None
    rows: list[dict[str, object]] = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("# ColHeaders "):
            headers = raw.removeprefix("# ColHeaders ").split()
            continue
        if not raw or raw.startswith("#"):
            continue
        if headers is None:
            raise ValueError(f"# ColHeaders absent dans {path}")

        fields = raw.split()
        if len(fields) != len(headers):
            raise ValueError(
                f"nombre de colonnes inattendu dans {path}: "
                f"{len(fields)} au lieu de {len(headers)}"
            )
        record = dict(zip(headers, fields))
        region = record["StructName"]
        for source_name, (metric, unit) in APARC_METRICS.items():
            rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "condition": condition,
                    "family": "cortical_region",
                    "hemi": hemi,
                    "region": region,
                    "metric": metric,
                    "value": float(record[source_name]),
                    "unit": unit,
                    "source_file": str(path),
                }
            )
    return rows


def parse_aseg(
    path: Path,
    condition: str,
    subject: str,
    run: str,
) -> list[dict[str, object]]:
    """Lit les mesures globales et la table régionale de aseg.stats."""
    rows: list[dict[str, object]] = []
    headers: list[str] | None = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("# Measure "):
            fields = [part.strip() for part in raw.removeprefix("# Measure ").split(",")]
            if len(fields) >= 5 and fields[-1].replace("^", "") in {"mm3", "mm³"}:
                rows.append(
                    {
                        "subject": subject,
                        "run": run,
                        "condition": condition,
                        "family": "aseg_global",
                        "hemi": "",
                        "region": fields[1],
                        "metric": "volume",
                        "value": float(fields[-2]),
                        "unit": "mm3",
                        "source_file": str(path),
                    }
                )
            continue

        if raw.startswith("# ColHeaders "):
            headers = raw.removeprefix("# ColHeaders ").split()
            continue
        if not raw or raw.startswith("#") or headers is None:
            continue

        fields = raw.split()
        if len(fields) != len(headers):
            raise ValueError(
                f"nombre de colonnes inattendu dans {path}: "
                f"{len(fields)} au lieu de {len(headers)}"
            )
        record = dict(zip(headers, fields))
        rows.append(
            {
                "subject": subject,
                "run": run,
                "condition": condition,
                "family": "aseg_region",
                "hemi": "",
                "region": record["StructName"],
                "metric": "volume",
                "value": float(record["Volume_mm3"]),
                "unit": "mm3",
                "source_file": str(path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction unifiée des mesures morphométriques FreeSurfer."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        type=parse_root,
        help="condition=/chemin/SUBJECTS_DIR (répétable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/ds004332/phase5_fidelity/morphometry_long.csv"),
    )
    parser.add_argument(
        "--completeness-out",
        type=Path,
        default=Path("results/ds004332/phase5_fidelity/morphometry_completeness.csv"),
    )
    args = parser.parse_args()

    roots = {name: Path(path).expanduser() for name, path in DEFAULT_ROOTS.items()}
    roots.update(dict(args.root))
    rows: list[dict[str, object]] = []
    completeness: list[dict[str, object]] = []

    for condition in CONDITIONS:
        root = roots[condition]
        for subject_number in range(1, 23):
            for run_number in range(1, 4):
                identifier = f"sub-{subject_number:02d}_run-{run_number:02d}"
                subject, run = split_identifier(identifier)
                stats_dir = root / identifier / "stats"
                paths = {
                    "aseg": stats_dir / "aseg.stats",
                    "lh_aparc": stats_dir / "lh.aparc.stats",
                    "rh_aparc": stats_dir / "rh.aparc.stats",
                }
                completeness.append(
                    {
                        "condition": condition,
                        "identifier": identifier,
                        "aseg_found": paths["aseg"].is_file(),
                        "lh_aparc_found": paths["lh_aparc"].is_file(),
                        "rh_aparc_found": paths["rh_aparc"].is_file(),
                        "stats_dir": str(stats_dir),
                    }
                )
                if paths["aseg"].is_file():
                    rows.extend(parse_aseg(paths["aseg"], condition, subject, run))
                if paths["lh_aparc"].is_file():
                    rows.extend(
                        parse_aparc(
                            paths["lh_aparc"], condition, subject, run, "lh"
                        )
                    )
                if paths["rh_aparc"].is_file():
                    rows.extend(
                        parse_aparc(
                            paths["rh_aparc"], condition, subject, run, "rh"
                        )
                    )

    if not rows:
        roots_text = "\n".join(f"  {name}: {path}" for name, path in roots.items())
        raise SystemExit(
            "Aucun fichier FreeSurfer trouvé. Racines inspectées :\n" + roots_text
        )

    counts: dict[tuple[object, ...], int] = {}
    for row in rows:
        key = tuple(row[field] for field in OUTPUT_FIELDS[:-1])
        counts[key] = counts.get(key, 0) + 1
    repeated = [key for key, count in counts.items() if count > 1]
    if repeated:
        raise ValueError(f"{len(repeated)} mesures dupliquées ; premier cas : {repeated[0]}")

    write_csv(args.out, rows, OUTPUT_FIELDS)
    write_csv(
        args.completeness_out,
        completeness,
        [
            "condition",
            "identifier",
            "aseg_found",
            "lh_aparc_found",
            "rh_aparc_found",
            "stats_dir",
        ],
    )
    print(f"Mesures : {args.out} ({len(rows)} lignes)")
    print(f"Complétude : {args.completeness_out} ({len(completeness)} acquisitions)")


if __name__ == "__main__":
    main()
