from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "pipelines/ds004332/phase5_fidelity"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PHASE5 / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analysis = load_module("phase5_analysis", "analyze_fidelity.py")
extractor = load_module("phase5_extractor", "extract_freesurfer_metrics.py")


class AgreementTests(unittest.TestCase):
    def test_ccc_identity_is_one(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(analysis.lin_ccc(values, values), 1.0)

    def test_icc_identity_is_one_and_absolute_offset_is_penalized(self):
        baseline = np.array([1.8, 2.0, 2.2, 2.5, 2.8])
        perfect = np.column_stack([baseline, baseline, baseline])
        shifted = np.column_stack([baseline, baseline + 0.2, baseline + 0.4])
        self.assertAlmostEqual(analysis.icc_absolute_agreement(perfect), 1.0)
        self.assertLess(analysis.icc_absolute_agreement(shifted), 1.0)

    def test_mixed_model_uses_raw_as_reference(self):
        rows = []
        rng = np.random.default_rng(42)
        subjects = [f"sub-{index:02d}" for index in range(1, 13)]
        for subject_index, subject in enumerate(subjects):
            for method_index, method in enumerate(["raw", "preproc", "jdac"]):
                for run, motion in [("run-02", "nodding"), ("run-03", "shaking")]:
                    for region_index in range(8):
                        rows.append(
                            {
                                "subject": subject,
                                "method": method,
                                "run": run,
                                "motion_label": motion,
                                "hemi": "lh" if region_index < 4 else "rh",
                                "region": f"region-{region_index}",
                                "signed_error_mm": 0.02 * method_index
                                + (0.03 if run == "run-03" else 0.0)
                                + 0.002 * subject_index
                                + rng.normal(0, 0.003),
                            }
                        )
        coefficients = analysis.fit_mixed_model(
            pd.DataFrame(rows), ["raw", "preproc", "jdac"], subjects
        )
        self.assertFalse(coefficients.empty)
        self.assertTrue(coefficients["term"].str.contains("Treatment\\('raw'\\)").any())


class EndpointTests(unittest.TestCase):
    def test_operational_reference_and_subject_median(self):
        rows = []
        for run, delta in [("run-01", 0.0), ("run-02", -0.2), ("run-03", -0.4)]:
            for region, thickness in [("a", 2.0 + delta), ("b", 3.0 + delta)]:
                rows.append(
                    {
                        "subject": "sub-01",
                        "run": run,
                        "motion_label": {"run-01": "still", "run-02": "nodding", "run-03": "shaking"}[run],
                        "method": "raw",
                        "hemi": "lh",
                        "region": region,
                        "thickness_mm": thickness,
                        "reference_surface_area_mm2": 1.0,
                        "agitation": 0.1,
                        "age": np.nan,
                        "sex_bin": np.nan,
                    }
                )
        table = pd.DataFrame(rows)
        attached = analysis.attach_references(table)
        endpoints = analysis.compute_subject_endpoints(attached).set_index("run")
        self.assertAlmostEqual(endpoints.loc["run-01", "median_abs_error_mm"], 0.0)
        self.assertAlmostEqual(endpoints.loc["run-02", "median_abs_error_mm"], 0.2)
        self.assertAlmostEqual(endpoints.loc["run-03", "median_abs_error_mm"], 0.4)

    def test_end_to_end_core_statistics_on_complete_synthetic_table(self):
        rows = []
        motion_delta = {
            "raw": {"run-01": 0.0, "run-02": -0.10, "run-03": -0.20},
            "preproc": {"run-01": -0.05, "run-02": -0.10, "run-03": -0.15},
            "jdac": {"run-01": -0.15, "run-02": -0.17, "run-03": -0.19},
        }
        for subject_index in range(10):
            subject = f"sub-{subject_index + 1:02d}"
            for method in ["raw", "preproc", "jdac"]:
                for run, motion in [("run-01", "still"), ("run-02", "nodding"), ("run-03", "shaking")]:
                    for region_index in range(68):
                        rows.append(
                            {
                                "subject": subject,
                                "run": run,
                                "motion_label": motion,
                                "method": method,
                                "hemi": "lh" if region_index < 34 else "rh",
                                "region": f"region-{region_index % 34:02d}",
                                "thickness_mm": 2.0
                                + subject_index * 0.01
                                + region_index * 0.001
                                + motion_delta[method][run],
                                "reference_surface_area_mm2": 100.0 + region_index,
                                "agitation": {"run-01": 0.2, "run-02": 1.0, "run-03": 2.0}[run],
                                "age": 20 + subject_index,
                                "sex_bin": float(subject_index % 2),
                            }
                        )
        table = pd.DataFrame(rows)
        attached = analysis.attach_references(table)
        endpoints = analysis.compute_subject_endpoints(attached)
        common = analysis.common_complete_subjects(endpoints, ["raw", "preproc", "jdac"])
        self.assertEqual(len(common), 10)
        summary = analysis.summarize_endpoints(endpoints, common, 20, 123)
        regional, agreement = analysis.compute_agreement(table, 20, 123)
        self.assertEqual(len(regional), 3 * 68)
        self.assertEqual(len(agreement), 3)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            analysis.plot_summary(summary, directory / "summary.png")
            analysis.write_report(directory / "report.md", summary, agreement, common)
            self.assertTrue((directory / "summary.png").is_file())
            report = (directory / "report.md").read_text(encoding="utf-8")
            self.assertIn("Primary endpoint", report)
            self.assertNotIn("gate", report.lower())


class FreeSurferParserTests(unittest.TestCase):
    def test_aparc_parser(self):
        content = (
            "# header\n"
            "bankssts 10 654 2014 2.679 0.1 0 0 0 0\n"
            "unknown 5 100 100 1.000 0.1 0 0 0 0\n"
            "cuneus 12 1489 3150 1.945 0.1 0 0 0 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lh.aparc.stats"
            path.write_text(content, encoding="utf-8")
            rows = extractor.parse_aparc_stats(path)
        self.assertEqual([row["region"] for row in rows], ["bankssts", "cuneus"])
        self.assertAlmostEqual(rows[0]["thickness_mm"], 2.679)

    def test_euler_output_parser_accepts_plain_and_verbose_formats(self):
        self.assertEqual(extractor.parse_euler_output("-334\n", "168\n"), (-334, 168))
        self.assertEqual(
            extractor.parse_euler_output("euler # = -334 --> 168 holes", ""),
            (-334, 168),
        )


if __name__ == "__main__":
    unittest.main()
