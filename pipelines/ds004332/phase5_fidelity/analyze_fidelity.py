#!/usr/bin/env python3
"""Describe morphometric fidelity from a complete regional table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


RUN_ORDER = ["run-01", "run-02", "run-03"]
MOTION_ORDER = ["still", "nodding", "shaking"]
DEFAULT_PRIMARY = ["raw", "preproc", "jdac"]


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.fillna("").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy().tolist()]
    return "\n".join([header, divider, *rows])


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if vx == 0 and vy == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    covariance = np.cov(x, y, ddof=1)[0, 1]
    denominator = vx + vy + (np.mean(x) - np.mean(y)) ** 2
    return float(2 * covariance / denominator) if denominator else float("nan")


def icc_absolute_agreement(matrix: np.ndarray) -> float:
    """ICC(A,1), equivalent to a two-way absolute-agreement single-measure ICC."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        return float("nan")
    if not np.isfinite(values).all():
        return float("nan")
    n, k = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    col_means = values.mean(axis=0)
    ms_subject = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_run = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = values - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_subject + (k - 1) * ms_error + k * (ms_run - ms_error) / n
    return float((ms_subject - ms_error) / denominator) if denominator else float("nan")


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    estimates = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        estimates[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def bootstrap_icc(matrix: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if matrix.shape[0] < 3:
        return float("nan"), float("nan")
    estimates = []
    for _ in range(n_bootstrap):
        sample = matrix[rng.integers(0, matrix.shape[0], matrix.shape[0]), :]
        value = icc_absolute_agreement(sample)
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def attach_references(table: pd.DataFrame) -> pd.DataFrame:
    reference = table[(table["method"] == "raw") & (table["run"] == "run-01")][
        ["subject", "hemi", "region", "thickness_mm"]
    ].rename(columns={"thickness_mm": "reference_thickness_mm"})
    reference = reference.drop_duplicates(["subject", "hemi", "region"])
    data = table.merge(reference, on=["subject", "hemi", "region"], how="left", validate="many_to_one")

    still = table[table["run"] == "run-01"][
        ["subject", "method", "hemi", "region", "thickness_mm"]
    ].rename(columns={"thickness_mm": "method_still_thickness_mm"})
    still = still.drop_duplicates(["subject", "method", "hemi", "region"])
    data = data.merge(
        still,
        on=["subject", "method", "hemi", "region"],
        how="left",
        validate="many_to_one",
    )
    data["signed_error_mm"] = data["thickness_mm"] - data["reference_thickness_mm"]
    data["abs_error_mm"] = data["signed_error_mm"].abs()
    data["within_method_abs_error_mm"] = (
        data["thickness_mm"] - data["method_still_thickness_mm"]
    ).abs()
    return data


def _weighted_average(group: pd.DataFrame, value: str) -> float:
    weights = pd.to_numeric(group["reference_surface_area_mm2"], errors="coerce")
    values = pd.to_numeric(group[value], errors="coerce")
    valid = weights.notna() & values.notna() & weights.gt(0)
    if not valid.any():
        return float(values.mean())
    return float(np.average(values[valid], weights=weights[valid]))


def compute_subject_endpoints(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["subject", "method", "run", "motion_label"]
    for keys, group in data.groupby(group_columns, observed=True):
        subject, method, run, motion_label = keys
        rows.append(
            {
                "subject": subject,
                "method": method,
                "run": run,
                "motion_label": motion_label,
                "n_regions": int(group["abs_error_mm"].notna().sum()),
                "median_abs_error_mm": float(group["abs_error_mm"].median()),
                "mean_abs_error_mm": float(group["abs_error_mm"].mean()),
                "median_signed_error_mm": float(group["signed_error_mm"].median()),
                "weighted_signed_error_mm": _weighted_average(group, "signed_error_mm"),
                "weighted_global_thickness_mm": _weighted_average(group, "thickness_mm"),
                "within_method_median_abs_error_mm": float(
                    group["within_method_abs_error_mm"].median()
                ),
                "agitation": float(group["agitation"].dropna().iloc[0])
                if group["agitation"].notna().any()
                else np.nan,
                "age": float(group["age"].dropna().iloc[0])
                if "age" in group and group["age"].notna().any()
                else np.nan,
                "sex_bin": float(group["sex_bin"].dropna().iloc[0])
                if "sex_bin" in group and group["sex_bin"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "subject", "run"]).reset_index(drop=True)


def common_complete_subjects(endpoints: pd.DataFrame, methods: list[str]) -> list[str]:
    required = {(method, run) for method in methods for run in RUN_ORDER}
    complete = []
    for subject, group in endpoints.groupby("subject"):
        observed = set(
            group.loc[group["n_regions"].eq(68), ["method", "run"]].itertuples(index=False, name=None)
        )
        if required <= observed:
            complete.append(subject)
    return sorted(complete)


def summarize_endpoints(
    endpoints: pd.DataFrame,
    common: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for analysis_set, subset in [
        ("common_complete", endpoints[endpoints["subject"].isin(common)]),
        ("all_available", endpoints[endpoints["n_regions"].eq(68)]),
    ]:
        for (method, run, motion), group in subset.groupby(
            ["method", "run", "motion_label"], observed=True
        ):
            values = group["median_abs_error_mm"].to_numpy(float)
            rng = np.random.default_rng(seed + sum(map(ord, method + run + analysis_set)))
            low, high = bootstrap_ci(values, n_bootstrap, rng)
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "method": method,
                    "run": run,
                    "motion_label": motion,
                    "n_subjects": len(group),
                    "median_primary_mae_mm": float(np.median(values)) if len(values) else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "median_signed_error_mm": float(group["median_signed_error_mm"].median()),
                    "median_weighted_signed_error_mm": float(
                        group["weighted_signed_error_mm"].median()
                    ),
                    "median_within_method_mae_mm": float(
                        group["within_method_median_abs_error_mm"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_differences(
    endpoints: pd.DataFrame,
    common: list[str],
    comparator: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    subset = endpoints[
        endpoints["subject"].isin(common) & endpoints["n_regions"].eq(68)
    ]
    rows = []
    methods = sorted(set(subset["method"]) - {comparator})
    for method in methods:
        for run in RUN_ORDER:
            left = subset[(subset["method"] == method) & (subset["run"] == run)][
                ["subject", "median_abs_error_mm"]
            ].rename(columns={"median_abs_error_mm": "candidate"})
            right = subset[(subset["method"] == comparator) & (subset["run"] == run)][
                ["subject", "median_abs_error_mm"]
            ].rename(columns={"median_abs_error_mm": "comparator"})
            pair = left.merge(right, on="subject", how="inner")
            differences = (pair["candidate"] - pair["comparator"]).to_numpy(float)
            rng = np.random.default_rng(seed + sum(map(ord, method + run + comparator)))
            low, high = bootstrap_ci(differences, n_bootstrap, rng)
            rows.append(
                {
                    "method": method,
                    "comparator": comparator,
                    "run": run,
                    "n_subjects": len(pair),
                    "median_difference_mm": float(np.median(differences)) if len(pair) else np.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def compute_agreement(
    table: pd.DataFrame, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (method, hemi, region), group in table.groupby(["method", "hemi", "region"]):
        wide = group.pivot_table(index="subject", columns="run", values="thickness_mm", aggfunc="first")
        wide = wide.reindex(columns=RUN_ORDER).dropna()
        matrix = wide.to_numpy(float)
        rng = np.random.default_rng(seed + sum(map(ord, method + hemi + region)))
        low, high = bootstrap_icc(matrix, n_bootstrap, rng)
        rows.append(
            {
                "method": method,
                "hemi": hemi,
                "region": region,
                "n_subjects": len(wide),
                "icc_a1": icc_absolute_agreement(matrix),
                "icc_ci95_low": low,
                "icc_ci95_high": high,
                "ccc_run02_vs_run01": lin_ccc(wide["run-01"], wide["run-02"]),
                "ccc_run03_vs_run01": lin_ccc(wide["run-01"], wide["run-03"]),
            }
        )
    regional = pd.DataFrame(rows)
    summaries = []
    for method, group in regional.groupby("method"):
        valid = group["icc_a1"].dropna()
        summaries.append(
            {
                "method": method,
                "n_regions": len(valid),
                "median_icc_a1": float(valid.median()),
                "icc_q1": float(valid.quantile(0.25)),
                "icc_q3": float(valid.quantile(0.75)),
                "regions_icc_gt_075": int((valid > 0.75).sum()),
                "median_ccc_run02_vs_run01": float(group["ccc_run02_vs_run01"].median()),
                "median_ccc_run03_vs_run01": float(group["ccc_run03_vs_run01"].median()),
            }
        )
    return regional, pd.DataFrame(summaries)


def regional_tests(data: pd.DataFrame, comparator: str = "preproc") -> pd.DataFrame:
    rows = []
    for run in ["run-02", "run-03"]:
        reference = data[(data["method"] == comparator) & (data["run"] == run)]
        for method in sorted(set(data["method"]) - {comparator}):
            candidate = data[(data["method"] == method) & (data["run"] == run)]
            pair = candidate.merge(
                reference,
                on=["subject", "hemi", "region"],
                suffixes=("_candidate", "_comparator"),
            )
            for (hemi, region), group in pair.groupby(["hemi", "region"]):
                differences = group["abs_error_mm_candidate"] - group["abs_error_mm_comparator"]
                try:
                    p_value = stats.wilcoxon(differences).pvalue if len(group) >= 5 else np.nan
                except ValueError:
                    p_value = 1.0
                rows.append(
                    {
                        "run": run,
                        "method": method,
                        "comparator": comparator,
                        "hemi": hemi,
                        "region": region,
                        "n_subjects": len(group),
                        "median_difference_mm": float(differences.median()),
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_fdr_bh"] = np.nan
    for run, indexes in result.groupby("run").groups.items():
        valid = result.loc[indexes, "p_value"].notna()
        selected = np.asarray(indexes)[valid.to_numpy()]
        if len(selected):
            result.loc[selected, "p_fdr_bh"] = multipletests(
                result.loc[selected, "p_value"], method="fdr_bh"
            )[1]
    return result


def fit_mixed_model(data: pd.DataFrame, methods: list[str], common: list[str]) -> pd.DataFrame:
    subset = data[
        data["method"].isin(methods)
        & data["subject"].isin(common)
        & data["run"].isin(["run-02", "run-03"])
    ].dropna(subset=["signed_error_mm"])
    subset = subset.copy()
    subset["region_id"] = subset["hemi"].astype(str) + "_" + subset["region"].astype(str)
    subset["method"] = pd.Categorical(subset["method"], categories=methods, ordered=True)
    subset["motion_label"] = pd.Categorical(
        subset["motion_label"], categories=["nodding", "shaking"], ordered=True
    )
    if subset.empty:
        return pd.DataFrame(columns=["term", "estimate", "std_error", "p_value"])
    formula = (
        "signed_error_mm ~ C(method, Treatment('raw')) "
        "* C(motion_label, Treatment('nodding'))"
    )
    model = smf.mixedlm(
        formula,
        subset,
        groups=subset["subject"],
        re_formula="1",
        vc_formula={"region": "0 + C(region_id)"},
    )
    try:
        fit = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
    except Exception:
        fit = model.fit(reml=False, method="powell", maxiter=500, disp=False)
    terms = fit.fe_params.index
    return pd.DataFrame(
        {
            "term": terms,
            "estimate": fit.fe_params[terms].to_numpy(),
            "std_error": fit.bse_fe[terms].to_numpy(),
            "p_value": fit.pvalues[terms].to_numpy(),
            "model_converged": bool(fit.converged),
            "n_observations": int(fit.nobs),
        }
    )


def quadratic_sensitivity(endpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in endpoints.groupby("method"):
        subset = group.dropna(
            subset=["weighted_global_thickness_mm", "agitation", "age", "sex_bin"]
        )
        if subset["subject"].nunique() < 10:
            rows.append({"method": method, "status": "skipped_missing_demographics"})
            continue
        linear = smf.mixedlm(
            "weighted_global_thickness_mm ~ age + sex_bin + agitation",
            subset,
            groups=subset["subject"],
        ).fit(reml=False, method="powell", disp=False)
        quadratic = smf.mixedlm(
            "weighted_global_thickness_mm ~ age + sex_bin + agitation + I(agitation ** 2)",
            subset,
            groups=subset["subject"],
        ).fit(reml=False, method="powell", disp=False)
        lr = max(0.0, 2 * (quadratic.llf - linear.llf))
        rows.append(
            {
                "method": method,
                "status": "fit",
                "n_scans": len(subset),
                "quadratic_coefficient": quadratic.params.get("I(agitation ** 2)", np.nan),
                "lrt_chi2": lr,
                "p_lrt": stats.chi2.sf(lr, 1),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["analysis_set"] == "common_complete"].copy()
    methods = sorted(data["method"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for axis, run, title in zip(axes, RUN_ORDER, ["Still", "Nodding", "Shaking"]):
        block = data[data["run"] == run].set_index("method").reindex(methods)
        positions = np.arange(len(methods))
        values = block["median_primary_mae_mm"].to_numpy(float)
        lower = values - block["ci95_low"].to_numpy(float)
        upper = block["ci95_high"].to_numpy(float) - values
        axis.errorbar(positions, values, yerr=[lower, upper], fmt="o", capsize=4)
        axis.set_xticks(positions, methods, rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Median regional absolute error (mm)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Fidelity to raw/run-01 operational reference")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    common: list[str],
) -> None:
    main = summary[summary["analysis_set"] == "common_complete"].copy()
    columns = ["method", "motion_label", "n_subjects", "median_primary_mae_mm", "ci95_low", "ci95_high"]
    lines = [
        "# Regional fidelity analysis",
        "",
        "`raw/run-01` is treated as an operational still reference, not perfect ground truth.",
        "",
        f"Common complete subjects: **{len(common)}**.",
        "## Primary endpoint",
        "",
        markdown_table(main[columns].round(4)),
        "",
        "## Agreement under induced motion",
        "",
        markdown_table(agreement.round(4)),
        "",
        "The ICC is absolute agreement across deliberately different motion conditions; it is not labelled conventional test-retest reliability.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "results/ds004332/phase5_fidelity/regional_metrics_long.csv",
    )
    parser.add_argument(
        "--scan-status",
        type=Path,
        default=repo / "results/ds004332/phase5_fidelity/scan_status.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "results/ds004332/phase5_fidelity",
    )
    parser.add_argument("--primary-methods", nargs="+", default=DEFAULT_PRIMARY)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    required = {
        "subject",
        "run",
        "motion_label",
        "method",
        "hemi",
        "region",
        "thickness_mm",
        "reference_surface_area_mm2",
        "agitation",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")
    absent_methods = set(args.primary_methods) - set(table["method"].unique())
    if absent_methods:
        raise ValueError(f"Primary methods absent from table: {sorted(absent_methods)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = attach_references(table)
    endpoints = compute_subject_endpoints(data)
    common = common_complete_subjects(endpoints, args.primary_methods)
    summary = summarize_endpoints(endpoints, common, args.bootstrap, args.seed)
    differences = paired_differences(endpoints, common, "preproc", args.bootstrap, args.seed)
    regional_agreement, agreement_summary = compute_agreement(table, args.bootstrap, args.seed)
    tests = regional_tests(data)

    endpoints.to_csv(args.output_dir / "subject_endpoints.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    differences.to_csv(args.output_dir / "paired_method_differences.csv", index=False)
    regional_agreement.to_csv(args.output_dir / "regional_agreement.csv", index=False)
    agreement_summary.to_csv(args.output_dir / "agreement_summary.csv", index=False)
    tests.to_csv(args.output_dir / "regional_pairwise_tests.csv", index=False)

    if args.scan_status.is_file():
        status = pd.read_csv(args.scan_status)
        failures = (
            status.groupby(["method", "run", "fs_status"], observed=True)
            .size()
            .rename("n_scans")
            .reset_index()
        )
        failures.to_csv(args.output_dir / "failure_rates.csv", index=False)

    if not args.skip_models:
        fit_mixed_model(data, args.primary_methods, common).to_csv(
            args.output_dir / "mixed_model_coefficients.csv", index=False
        )
        quadratic_sensitivity(endpoints).to_csv(
            args.output_dir / "quadratic_sensitivity.csv", index=False
        )

    plot_summary(summary, args.output_dir / "fidelity_summary.png")
    write_report(args.output_dir / "FIDELITY_REPORT.md", summary, agreement_summary, common)
    print(f"Common complete subjects: {len(common)}")
    print(f"Outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
