#!/usr/bin/env python3
"""Build the canonical regional table for the post-JDAC fidelity benchmark.

The table uses raw/run-01 as an *operational still reference*, never as a
perfect ground truth. Missing FreeSurfer runs remain explicit in a companion
scan-status table instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUN_TO_MOTION = {
    "run-01": "still",
    "run-02": "nodding",
    "run-03": "shaking",
}

WIDE_METHODS = {
    "preproc": "thickness_preproc_rigid_{hemi}.csv",
    "jdac": "thickness_jdac_rigid_{hemi}.csv",
    "jdac_antiartonly": (
        "thickness_jdac_antiartonly_rigid/"
        "thickness_jdac_antiartonly_rigid_{hemi}.csv"
    ),
    "jdac_nodenoise": (
        "thickness_jdac_nodenoise_rigid/"
        "thickness_jdac_nodenoise_rigid_{hemi}.csv"
    ),
}

KEY = ["subject", "run", "method", "hemi", "region"]


def split_scan_id(value: str) -> tuple[str, str]:
    match = re.search(r"(sub-[A-Za-z0-9]+).*?(run-[0-9]+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse subject/run from {value!r}")
    return match.group(1), match.group(2)


def load_raw(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"subject", "hemi", "region", "ThickAvg"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    parsed = data["subject"].map(split_scan_id)
    data["subject"] = parsed.map(lambda item: item[0])
    data["run"] = parsed.map(lambda item: item[1])
    data["method"] = "raw"
    data = data.rename(
        columns={
            "ThickAvg": "thickness_mm",
            "SurfArea": "surface_area_mm2",
            "GrayVol": "gray_volume_mm3",
        }
    )
    for column in ["surface_area_mm2", "gray_volume_mm3"]:
        if column not in data:
            data[column] = np.nan
    return data[KEY + ["thickness_mm", "surface_area_mm2", "gray_volume_mm3"]]


def load_wide_method(method: str, derivatives: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pattern = WIDE_METHODS[method]
    for hemi in ["lh", "rh"]:
        path = derivatives / pattern.format(hemi=hemi)
        if not path.is_file():
            raise FileNotFoundError(path)
        wide = pd.read_csv(path, sep=None, engine="python")
        id_column = wide.columns[0]
        wide = wide.rename(columns={id_column: "scan_id"})
        regional = [
            column
            for column in wide.columns
            if column.endswith("_thickness") and "MeanThickness" not in column
        ]
        if len(regional) != 34:
            raise ValueError(
                f"{path}: expected 34 cortical thickness columns, found {len(regional)}"
            )
        long = wide.melt(
            id_vars="scan_id",
            value_vars=regional,
            var_name="region_raw",
            value_name="thickness_mm",
        )
        parsed = long["scan_id"].map(split_scan_id)
        long["subject"] = parsed.map(lambda item: item[0])
        long["run"] = parsed.map(lambda item: item[1])
        long["method"] = method
        long["hemi"] = hemi
        long["region"] = (
            long["region_raw"]
            .str.replace(f"{hemi}_", "", regex=False)
            .str.replace("_thickness", "", regex=False)
        )
        long["surface_area_mm2"] = np.nan
        long["gray_volume_mm3"] = np.nan
        frames.append(
            long[KEY + ["thickness_mm", "surface_area_mm2", "gray_volume_mm3"]]
        )
    return pd.concat(frames, ignore_index=True)


def load_standard_long(method: str, path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"subject", "run", "hemi", "region", "thickness_mm"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "method" in data and set(data["method"].dropna().unique()) - {method}:
        raise ValueError(f"{path}: method column does not match {method!r}")
    data["method"] = method
    for column in ["surface_area_mm2", "gray_volume_mm3"]:
        if column not in data:
            data[column] = np.nan
    return data[KEY + ["thickness_mm", "surface_area_mm2", "gray_volume_mm3"]]


def parse_extra(values: list[str]) -> dict[str, Path]:
    extras: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--extra must be METHOD=CSV, got {value!r}")
        method, raw_path = value.split("=", 1)
        extras[method.strip()] = Path(raw_path).expanduser().resolve()
    return extras


def build_table(
    raw_csv: Path,
    agitation_csv: Path,
    derivatives: Path,
    participants_tsv: Path | None,
    extras: dict[str, Path],
    strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames = [load_raw(raw_csv)]
    loaded_methods = ["raw"]
    warnings: list[str] = []

    for method in WIDE_METHODS:
        try:
            frames.append(load_wide_method(method, derivatives))
            loaded_methods.append(method)
        except FileNotFoundError as error:
            message = f"{method}: missing derivative file {error.filename or error}"
            if strict:
                raise FileNotFoundError(message) from error
            warnings.append(message)

    for method, path in extras.items():
        frames.append(load_standard_long(method, path))
        loaded_methods.append(method)

    table = pd.concat(frames, ignore_index=True)
    table["thickness_mm"] = pd.to_numeric(table["thickness_mm"], errors="coerce")
    table = table[table["thickness_mm"].notna() & (table["thickness_mm"] > 0)].copy()
    table["motion_label"] = table["run"].map(RUN_TO_MOTION)
    if table["motion_label"].isna().any():
        bad = sorted(table.loc[table["motion_label"].isna(), "run"].unique())
        raise ValueError(f"Unknown runs: {bad}")

    duplicate = table.duplicated(KEY, keep=False)
    if duplicate.any():
        sample = table.loc[duplicate, KEY].head().to_dict("records")
        raise ValueError(f"Duplicate regional keys, for example: {sample}")

    agitation = pd.read_csv(agitation_csv).rename(
        columns={"condition": "run", "sub": "subject", "motion": "agitation"}
    )
    table = table.merge(
        agitation[["subject", "run", "agitation"]],
        on=["subject", "run"],
        how="left",
        validate="many_to_one",
    )

    table["age"] = np.nan
    table["sex"] = pd.NA
    table["sex_bin"] = np.nan
    if participants_tsv and participants_tsv.is_file():
        participants = pd.read_csv(participants_tsv, sep="\t").rename(
            columns={"participant_id": "subject"}
        )
        keep = [column for column in ["subject", "age", "sex"] if column in participants]
        table = table.drop(columns=["age", "sex", "sex_bin"]).merge(
            participants[keep], on="subject", how="left", validate="many_to_one"
        )
        if "age" not in table:
            table["age"] = np.nan
        if "sex" not in table:
            table["sex"] = pd.NA
        table["sex_bin"] = table["sex"].map({"F": 1, "M": 0})
    elif participants_tsv:
        warnings.append(f"participants file missing: {participants_tsv}")

    reference_area = (
        table[(table["method"] == "raw") & (table["run"] == "run-01")][
            ["subject", "hemi", "region", "surface_area_mm2"]
        ]
        .drop_duplicates(["subject", "hemi", "region"])
        .rename(columns={"surface_area_mm2": "reference_surface_area_mm2"})
    )
    table = table.merge(
        reference_area,
        on=["subject", "hemi", "region"],
        how="left",
        validate="many_to_one",
    )
    table["reference_type"] = np.where(
        (table["method"] == "raw") & (table["run"] == "run-01"),
        "operational_still_reference",
        "processed_observation",
    )

    expected_subjects = sorted(agitation["subject"].dropna().unique())
    expected = pd.MultiIndex.from_product(
        [expected_subjects, RUN_TO_MOTION, loaded_methods],
        names=["subject", "run", "method"],
    ).to_frame(index=False)
    counts = (
        table.groupby(["subject", "run", "method"], observed=True)
        .agg(n_regions=("region", "size"), n_hemis=("hemi", "nunique"))
        .reset_index()
    )
    status = expected.merge(counts, on=["subject", "run", "method"], how="left")
    status[["n_regions", "n_hemis"]] = status[["n_regions", "n_hemis"]].fillna(0).astype(int)
    status["fs_status"] = np.select(
        [status["n_regions"].eq(68) & status["n_hemis"].eq(2), status["n_regions"].eq(0)],
        ["complete", "missing"],
        default="partial_or_invalid",
    )
    table = table.merge(
        status[["subject", "run", "method", "fs_status"]],
        on=["subject", "run", "method"],
        how="left",
        validate="many_to_one",
    )
    table = table.sort_values(KEY).reset_index(drop=True)
    status = status.sort_values(["method", "subject", "run"]).reset_index(drop=True)
    return table, status, warnings


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=repo / "results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv",
    )
    parser.add_argument(
        "--agitation-csv",
        type=Path,
        default=repo / "results/ds004332/agitation/ds004332_agitation_clinica.csv",
    )
    parser.add_argument(
        "--derivatives",
        type=Path,
        default=Path.home() / "Documents/derivatives/ds004332",
    )
    parser.add_argument(
        "--participants-tsv",
        type=Path,
        default=Path.home() / "Documents/raw_datasets/ds004332/participants.tsv",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="METHOD=CSV",
        help="Add a standardized long regional CSV, for example jdac_pilot=pilot_regional.csv",
    )
    completeness = parser.add_mutually_exclusive_group()
    completeness.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail when any current derivative is missing (default)",
    )
    completeness.add_argument(
        "--allow-missing",
        dest="strict",
        action="store_false",
        help="Diagnostic mode only; allow a deliberately incomplete table",
    )
    parser.set_defaults(strict=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "results/ds004332/phase5_fidelity",
    )
    args = parser.parse_args()

    try:
        table, status, warnings = build_table(
            raw_csv=args.raw_csv.resolve(),
            agitation_csv=args.agitation_csv.resolve(),
            derivatives=args.derivatives.expanduser().resolve(),
            participants_tsv=args.participants_tsv.expanduser().resolve(),
            extras=parse_extra(args.extra),
            strict=args.strict,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "regional_metrics_long.csv", index=False)
    status.to_csv(args.output_dir / "scan_status.csv", index=False)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"{len(table)} regional rows -> {args.output_dir / 'regional_metrics_long.csv'}")
    print(status.groupby(["method", "fs_status"]).size().to_string())


if __name__ == "__main__":
    main()
