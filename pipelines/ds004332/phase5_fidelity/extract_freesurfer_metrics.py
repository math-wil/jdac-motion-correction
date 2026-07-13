#!/usr/bin/env python3
"""Extract cortical metrics and explicit completion status from a SUBJECTS_DIR."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REGIONS_EXCLUDE = {"unknown", "corpuscallosum"}
EXPECTED_REGIONS = 68


def parse_subject_id(subject_id: str) -> tuple[str, str]:
    match = re.search(r"(sub-[A-Za-z0-9]+).*?(run-[0-9]+)", subject_id)
    if not match:
        raise ValueError(f"Cannot parse subject/run from {subject_id!r}")
    return match.group(1), match.group(2)


def parse_aparc_stats(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            region = fields[0]
            if region in REGIONS_EXCLUDE:
                continue
            rows.append(
                {
                    "region": region,
                    "n_vertices": int(fields[1]),
                    "surface_area_mm2": float(fields[2]),
                    "gray_volume_mm3": float(fields[3]),
                    "thickness_mm": float(fields[4]),
                    "thickness_sd_mm": float(fields[5]),
                }
            )
    return rows


def parse_euler_output(output: str, holes_text: str = "") -> tuple[int | None, int | None]:
    euler_match = re.search(r"euler\s*#?\s*=\s*(-?\d+)", output, flags=re.IGNORECASE)
    if not euler_match:
        euler_match = re.search(r"(?<![\d.])-?\d+(?![\d.])", output)
    holes_match = re.search(r"-?\d+", holes_text)
    if not holes_match:
        holes_match = re.search(r"(?:-->|:)\s*(\d+)\s+holes?", output, flags=re.IGNORECASE)
    euler = (
        int(euler_match.group(1) if euler_match.lastindex else euler_match.group(0))
        if euler_match
        else None
    )
    holes = (
        int(holes_match.group(1) if holes_match.lastindex else holes_match.group(0))
        if holes_match
        else None
    )
    return euler, holes


def euler_metrics(surface: Path) -> tuple[int | None, int | None, str]:
    command = shutil.which("mris_euler_number")
    if not command or not surface.is_file():
        return None, None, ""
    with tempfile.TemporaryDirectory() as directory:
        holes_path = Path(directory) / "holes.txt"
        completed = subprocess.run(
            [command, "-o", str(holes_path), str(surface)],
            capture_output=True,
            text=True,
            check=False,
        )
        holes_text = (
            holes_path.read_text(encoding="utf-8", errors="replace")
            if holes_path.is_file()
            else ""
        )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    euler, holes = parse_euler_output(output, holes_text)
    return euler, holes, output.replace("\n", " | ")


def expected_ids(manifest: Path | None, subjects_dir: Path) -> list[str]:
    if manifest:
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "output_subject_id" not in rows[0]:
            raise ValueError(f"{manifest}: expected output_subject_id column")
        return [row["output_subject_id"] for row in rows]
    return sorted(path.name for path in subjects_dir.iterdir() if path.is_dir())


def extract(
    subjects_dir: Path, method: str, ids: list[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    regional_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for output_id in ids:
        subject, run = parse_subject_id(output_id)
        root = subjects_dir / output_id
        done = (root / "scripts/recon-all.done").is_file()
        scan_rows = []
        euler: dict[str, int | str | None] = {}
        for hemi in ["lh", "rh"]:
            stats_path = root / "stats" / f"{hemi}.aparc.stats"
            if stats_path.is_file():
                for row in parse_aparc_stats(stats_path):
                    scan_rows.append(
                        {
                            "subject": subject,
                            "run": run,
                            "method": method,
                            "hemi": hemi,
                            **row,
                            "reference_type": "processed_observation",
                        }
                    )
            value, holes, raw_output = euler_metrics(root / "surf" / f"{hemi}.orig.nofix")
            euler[f"{hemi}_euler_number"] = value
            euler[f"{hemi}_topological_holes"] = holes
            euler[f"{hemi}_euler_output"] = raw_output
        n_regions = len(scan_rows)
        if done and n_regions == EXPECTED_REGIONS:
            status = "complete"
        elif not root.exists():
            status = "missing"
        elif n_regions:
            status = "partial_or_invalid"
        else:
            status = "failed_or_incomplete"
        for row in scan_rows:
            row["fs_status"] = status
        regional_rows.extend(scan_rows)
        status_rows.append(
            {
                "subject": subject,
                "run": run,
                "method": method,
                "output_subject_id": output_id,
                "fs_status": status,
                "recon_all_done": done,
                "n_regions": n_regions,
                **euler,
            }
        )
    return regional_rows, status_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-regional", type=Path, required=True)
    parser.add_argument("--output-status", type=Path, required=True)
    args = parser.parse_args()

    ids = expected_ids(args.manifest, args.subjects_dir)
    regional, status = extract(args.subjects_dir, args.method, ids)
    if regional:
        write_csv(args.output_regional, regional)
    write_csv(args.output_status, status)
    complete = sum(row["fs_status"] == "complete" for row in status)
    print(f"{complete}/{len(status)} complete; {len(regional)} regional rows")


if __name__ == "__main__":
    main()
