#!/usr/bin/env python3
"""Analyze the tracked Phase-4 subject summaries when regional derivatives are absent.

This is deliberately labelled a legacy/available-data analysis.  It uses the
mean regional absolute error already stored in recovery_metrics.csv. It is a
secondary presentation of Phase-4 measurements, not a new experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
RUN_LABELS = {"run-02": "nodding", "run-03": "shaking"}


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy().tolist()]
    return "\n".join([header, divider, *rows])


def bootstrap_median(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    estimates = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(n_bootstrap)]
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def analyze(recovery: pd.DataFrame, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    recovery = recovery.rename(columns={"sub": "subject", "condition": "method"}).copy()
    summary_rows = []
    for (method, run), group in recovery.groupby(["method", "run"]):
        rng = np.random.default_rng(seed + sum(map(ord, method + run)))
        low, high = bootstrap_median(group["mae_truth"].to_numpy(), n_bootstrap, rng)
        summary_rows.append(
            {
                "method": method,
                "run": run,
                "motion_label": RUN_LABELS[run],
                "n_subjects": len(group),
                "median_legacy_mean_mae_mm": group["mae_truth"].median(),
                "mean_legacy_mean_mae_mm": group["mae_truth"].mean(),
                "ci95_low": low,
                "ci95_high": high,
                "median_within_method_mean_mae_mm": group["mae_within"].median(),
                "median_clean_offset_mean_mae_mm": group["offset"].median(),
            }
        )

    difference_rows = []
    for method in [value for value in METHOD_ORDER if value != "preproc"]:
        for run in RUN_LABELS:
            candidate = recovery[(recovery["method"] == method) & (recovery["run"] == run)][
                ["subject", "mae_truth"]
            ].rename(columns={"mae_truth": "candidate"})
            comparator = recovery[(recovery["method"] == "preproc") & (recovery["run"] == run)][
                ["subject", "mae_truth"]
            ].rename(columns={"mae_truth": "preproc"})
            pair = candidate.merge(comparator, on="subject", how="inner")
            differences = (pair["candidate"] - pair["preproc"]).to_numpy()
            rng = np.random.default_rng(seed + sum(map(ord, method + run + "preproc")))
            low, high = bootstrap_median(differences, n_bootstrap, rng)
            difference_rows.append(
                {
                    "method": method,
                    "comparator": "preproc",
                    "run": run,
                    "n_subjects": len(pair),
                    "median_paired_difference_mm": np.median(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(difference_rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(summary["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, run in zip(axes, RUN_LABELS):
        block = summary[summary["run"] == run].set_index("method").reindex(methods)
        values = block["median_legacy_mean_mae_mm"].to_numpy(float)
        lower = values - block["ci95_low"].to_numpy(float)
        upper = block["ci95_high"].to_numpy(float) - values
        positions = np.arange(len(methods))
        axis.errorbar(positions, values, yerr=[lower, upper], fmt="o", capsize=4)
        axis.set_xticks(positions, methods, rotation=35, ha="right")
        axis.set_title(RUN_LABELS[run].title())
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Median subject mean regional MAE (mm)")
    fig.suptitle("Available Phase-4 fidelity checkpoint (provisional)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery",
        type=Path,
        default=repo / "results/ds004332/phase4_compare_3bras/recovery_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "results/ds004332/phase5_fidelity",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    recovery = pd.read_csv(args.recovery)
    summary, differences = analyze(recovery, args.bootstrap, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recovery.rename(columns={"sub": "subject", "condition": "method"}).to_csv(
        args.output_dir / "p1_available_subject_endpoints.csv", index=False
    )
    summary.to_csv(args.output_dir / "p1_available_method_summary.csv", index=False)
    differences.to_csv(args.output_dir / "p1_available_paired_differences.csv", index=False)
    plot_summary(summary, args.output_dir / "P1_AVAILABLE_fidelity.png")

    display = summary[
        [
            "method",
            "motion_label",
            "n_subjects",
            "median_legacy_mean_mae_mm",
            "ci95_low",
            "ci95_high",
        ]
    ].round(4)
    report = [
        "# Secondary fidelity summary from Phase 4",
        "",
        "This is a secondary presentation of the already executed Phase-4 results. It is not a new experiment and does not impose a go/no-go decision.",
        "",
        "The table uses the per-subject **mean regional MAE** inherited from `phase4_compare_3bras/recovery_metrics.csv`.",
        "",
        "`raw/run-01` remains an operational still reference, not perfect ground truth.",
        "",
        markdown_table(display),
        "",
        "## Interpretation boundary",
        "",
        "These descriptive summaries show how the existing methods compare. The choice of the next JDAC modification or experiment remains a scientific decision and is not encoded as an automatic gate.",
    ]
    (args.output_dir / "P1_AVAILABLE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("Secondary Phase-4 fidelity summary generated")
    print(f"Outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
