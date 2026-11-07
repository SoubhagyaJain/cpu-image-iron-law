# First run — retained for methodological transparency

The initial benchmark used identical configurations, 20 warmups,
100 requests per configuration per round, and 3 rounds, but measured each
case in contiguous blocks rather than interleaving all four cases per group.

The recorded console summary was:

| Configuration | Preprocess mean (ms) | Inference mean (ms) | Postprocess mean (ms) | E2E mean (ms) | Improvement |
|---|---:|---:|---:|---:|---:|
| baseline | 1.773 | 0.282 | 0.031 | 2.086 | — |
| inference_only | 1.773 | 0.235 | 0.034 | 2.042 | +2.10% |
| preprocess_only | 0.305 | 0.223 | 0.021 | 0.549 | +73.68% |
| both | 0.290 | 0.206 | 0.025 | 0.521 | +75.04% |

After revising the harness to randomize configuration order for every group
of four requests (a fairer design against time-related drift), all raw timing
samples, stage shares, Amdahl projections and the chart were regenerated.
Those **final** raw measurements are in `timings.csv`, and the final summary
is in `REPORT.md` and `benchmark.json`.

Only the console-stage aggregates above were retained from the first run;
there is no claim to have archived its original raw per-request samples.
This is a useful demonstration of why careful benchmarking and reruns matter.
