#!/usr/bin/env python3
"""Extrait tous les volumes en mm3 des fichiers FreeSurfer stats/aseg.stats."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

CONDITIONS = ("brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise")
DEFAULT_ROOTS = {
    "brut": "~/projects/def-sbouix/mathw/freesurfer_ds004332",
    "preproc": "~/projects/ctb-sbouix/mathw/freesurfer_preproc_rigid_ds004332",
    "jdac": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_ds004332",
    "jdac_antiartonly": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_antiartonly_ds004332",
    "jdac_nodenoise": "~/projects/ctb-sbouix/mathw/freesurfer_jdac_rigid_nodenoise_ds004332",
}
ID_RE = re.compile(r"^(sub-[^_]+)_(run-[^_]+)$")


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("format attendu: condition=/chemin/SUBJECTS_DIR")
    condition, path = value.split("=", 1)
    if condition not in CONDITIONS:
        raise argparse.ArgumentTypeError(f"condition inconnue: {condition}")
    return condition, Path(path).expanduser()


def parse_aseg(path: Path, condition: str) -> list[dict[str, object]]:
    match = ID_RE.match(path.parents[1].name)
    if not match:
        raise ValueError(f"identifiant sujet/run inattendu: {path.parents[1].name}")
    subject, run = match.groups()
    common = {"subject": subject, "run": run, "condition": condition, "source_file": str(path)}
    rows: list[dict[str, object]] = []
    in_table = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("# Measure "):
            fields = [part.strip() for part in raw[len("# Measure "):].split(",")]
            if len(fields) >= 5 and fields[-1].replace("^", "") in {"mm3", "mm³"}:
                rows.append({**common, "measure_type": "global", "measure": fields[1],
                             "seg_id": "", "value_mm3": float(fields[-2])})
        elif raw.startswith("# ColHeaders"):
            in_table = True
        elif in_table and raw and not raw.startswith("#"):
            fields = raw.split()
            if len(fields) >= 5:
                rows.append({**common, "measure_type": "structure", "measure": fields[4],
                             "seg_id": int(fields[1]), "value_mm3": float(fields[3])})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction longue des volumes aseg.stats.")
    parser.add_argument("--root", action="append", default=[], type=parse_root,
                        help="condition=/chemin/SUBJECTS_DIR (répétable)")
    parser.add_argument("--out", type=Path,
                        default=Path("results/ds004332/phase5_fidelity/aseg_volumes_long.csv"))
    parser.add_argument("--completeness-out", type=Path,
                        default=Path("results/ds004332/phase5_fidelity/aseg_completeness.csv"))
    args = parser.parse_args()
    roots = {name: Path(path).expanduser() for name, path in DEFAULT_ROOTS.items()}
    roots.update(dict(args.root))
    all_rows: list[dict[str, object]] = []
    completeness: list[dict[str, object]] = []
    for condition in CONDITIONS:
        root = roots[condition]
        found = {path.parents[1].name: path
                 for path in sorted(root.glob("sub-*_run-*/stats/aseg.stats"))}
        for subject_number in range(1, 23):
            for run_number in range(1, 4):
                identifier = f"sub-{subject_number:02d}_run-{run_number:02d}"
                path = found.get(identifier)
                completeness.append({
                    "condition": condition,
                    "identifier": identifier,
                    "aseg_stats_found": bool(path),
                    "source_file": str(path) if path else str(root / identifier / "stats/aseg.stats"),
                })
                if path:
                    all_rows.extend(parse_aseg(path, condition))
    if not all_rows:
        roots_text = "\n".join(f"  {name}: {path}" for name, path in roots.items())
        raise SystemExit("Aucun aseg.stats trouvé. Racines inspectées:\n" + roots_text)
    write_csv(args.out, all_rows, ["subject", "run", "condition", "measure_type",
              "measure", "seg_id", "value_mm3", "source_file"])
    write_csv(args.completeness_out, completeness,
              ["condition", "identifier", "aseg_stats_found", "source_file"])
    print(f"Volumes: {args.out} ({len(all_rows)} lignes)")
    print(f"Complétude: {args.completeness_out}")


if __name__ == "__main__":
    main()
