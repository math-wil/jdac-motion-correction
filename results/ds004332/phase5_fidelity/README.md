# Secondary fidelity summary derived from Phase 4

This directory contains **no new MRI processing and no new FreeSurfer
reconstruction**. The files below were generated from the executed Phase-4
tables and should be interpreted only as an alternative summary of those
existing results.

## Files

- `p1_available_subject_endpoints.csv`: Phase-4 recovery measurements arranged
  by subject, method, and moved run. Despite the historical `p1_available`
  prefix, this is derived Phase-4 data.
- `p1_available_method_summary.csv`: number of available subjects and median
  subject-level mean regional MAE for each method and movement condition, with
  bootstrap confidence intervals.
- `p1_available_paired_differences.csv`: within-subject difference between each
  method and preprocessing. A positive difference means a larger error than
  preprocessing; a negative difference means a smaller error.
- `P1_AVAILABLE_fidelity.png`: presentation figure generated from the method
  summary.
- `P1_AVAILABLE_REPORT.md`: human-readable version of the same descriptive
  results.

## Interpretation limits

The reference is each subject's raw still scan (`run-01`). It is an operational
reference, not perfect anatomical ground truth. The input Phase-4 table contains
mean regional errors; therefore these files do not establish a new regional
endpoint and do not constitute a pass/fail gate.

The authoritative executed results remain in
`results/ds004332/phase4_compare_3bras/`.
