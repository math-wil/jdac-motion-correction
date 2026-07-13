# Secondary fidelity summary from Phase 4

This is a secondary presentation of the already executed Phase-4 results. It is
not a new experiment and does not impose a go/no-go decision.

The values come from the per-subject mean regional MAE in
`phase4_compare_3bras/recovery_metrics.csv`. The raw still scan (`run-01`) is an
operational reference, not perfect anatomical ground truth.

| method | motion | n | median mean regional MAE (mm) | 95% bootstrap CI |
| --- | --- | ---: | ---: | --- |
| raw | nodding | 22 | 0.2408 | 0.2200 to 0.2922 |
| raw | shaking | 21 | 0.3072 | 0.2657 to 0.3393 |
| preproc | nodding | 22 | 0.2466 | 0.2325 to 0.2834 |
| preproc | shaking | 21 | 0.2976 | 0.2625 to 0.3317 |
| jdac | nodding | 22 | 0.3589 | 0.3206 to 0.3743 |
| jdac | shaking | 20 | 0.3591 | 0.3365 to 0.4058 |
| jdac anti-artifact only | nodding | 22 | 0.2797 | 0.2436 to 0.3154 |
| jdac anti-artifact only | shaking | 22 | 0.3070 | 0.2696 to 0.3409 |
| jdac without denoising | nodding | 22 | 0.2832 | 0.2550 to 0.3594 |
| jdac without denoising | shaking | 22 | 0.3096 | 0.2861 to 0.3292 |

## What this says

In the existing Phase-4 measurements, full JDAC has a larger morphometric error
than preprocessing for both movement conditions. The two ablations are closer
to preprocessing than full JDAC. This supports investigating which JDAC
component or strength causes the degradation, but it does not by itself choose
the next modification or authorize a new experiment.
