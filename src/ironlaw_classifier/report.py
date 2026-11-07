"""Aggregate timings and derive Amdahl speedup from measured fractions."""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from .pipeline import Sample

STAGES = ("preprocessing_ms", "inference_ms", "postprocessing_ms")
METRICS = (*STAGES, "end_to_end_ms")
CASES = ("baseline", "inference_only", "preprocess_only", "both")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records: list[Sample]) -> dict:
    summary = {}
    for case in CASES:
        subset = [r for r in records if r.case == case]
        if not subset:
            raise ValueError(f"no records for {case}")
        values = {key: [getattr(sample, key) for sample in subset] for key in METRICS}
        stage_mean = {key: mean(values[key]) for key in METRICS}
        summary[case] = {
            "samples": len(subset),
            "mean_ms": stage_mean,
            "median_ms": {key: median(values[key]) for key in METRICS},
            "p95_ms": {key: percentile(values[key], 0.95) for key in METRICS},
            "stage_share_percent": {
                key: stage_mean[key] / stage_mean["end_to_end_ms"] * 100 for key in STAGES
            },
        }
    baseline = summary["baseline"]["mean_ms"]["end_to_end_ms"]
    summary["improvement_percent_vs_baseline"] = {
        case: (baseline - summary[case]["mean_ms"]["end_to_end_ms"]) / baseline * 100
        for case in CASES
    }
    summary["speedup_vs_baseline"] = {
        case: baseline / summary[case]["mean_ms"]["end_to_end_ms"]
        for case in CASES
    }
    shares = summary["baseline"]["stage_share_percent"]
    summary["baseline_dominant_stage"] = max(STAGES, key=lambda k: shares[k])
    # Amdahl projections assume the OTHER stage times stay identical.
    expected = {}
    for stage, case in (("inference_ms", "inference_only"), ("preprocessing_ms", "preprocess_only")):
        original = summary["baseline"]["mean_ms"][stage]
        replaced = summary[case]["mean_ms"][stage]
        f = original / baseline
        local_speedup = original / replaced
        expected[case] = {
            "baseline_fraction": f,
            "local_speedup": local_speedup,
            "predicted_end_to_end_speedup": 1 / ((1 - f) + f / local_speedup),
            "predicted_improvement_percent": 100 * (1 - ((1 - f) + f / local_speedup)),
        }
    summary["amdahl_projections"] = expected
    return summary


def save_records_csv(records: list[Sample], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", *METRICS])
        writer.writeheader()
        for item in records:
            writer.writerow(asdict(item))


def save_summary_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "samples", *[f"mean_{k}" for k in METRICS],
                         "e2e_median_ms", "e2e_p95_ms", "improvement_percent", "speedup_x"])
        for case in CASES:
            item = summary[case]
            writer.writerow([case, item["samples"], *[item["mean_ms"][key] for key in METRICS],
                             item["median_ms"]["end_to_end_ms"],
                             item["p95_ms"]["end_to_end_ms"],
                             summary["improvement_percent_vs_baseline"][case],
                             summary["speedup_vs_baseline"][case]])


def save_plot(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    labels = ["Baseline", "Inference only", "Preprocessing only", "Both"]
    bottoms = [0.0] * len(CASES)
    for stage, title in (("preprocessing_ms", "Preprocessing"),
                         ("inference_ms", "Inference"),
                         ("postprocessing_ms", "Postprocessing")):
        times = [summary[case]["mean_ms"][stage] for case in CASES]
        ax.bar(labels, times, bottom=bottoms, label=title)
        bottoms = [x + y for x, y in zip(bottoms, times)]
    ax.set_ylabel("Mean latency per request (ms)")
    ax.set_title("CPU image classification: stage-by-stage latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_report(summary: dict, meta: dict, prediction: dict) -> str:
    baseline = summary["baseline"]["mean_ms"]["end_to_end_ms"]
    rows = []
    for case in CASES:
        s = summary[case]
        m = s["mean_ms"]
        rows.append(
            f"| {case} | {m['preprocessing_ms']:.3f} | {m['inference_ms']:.3f} | "
            f"{m['postprocessing_ms']:.3f} | {m['end_to_end_ms']:.3f} | "
            f"{s['median_ms']['end_to_end_ms']:.3f} | {s['p95_ms']['end_to_end_ms']:.3f} | "
            f"{summary['improvement_percent_vs_baseline'][case]:+.2f}% |"
        )
    shares = summary["baseline"]["stage_share_percent"]
    dominant = summary["baseline_dominant_stage"]
    inf = summary["amdahl_projections"]["inference_only"]
    pre = summary["amdahl_projections"]["preprocess_only"]
    win = "preprocess_only" if summary["preprocess_only"]["mean_ms"]["end_to_end_ms"] < summary["inference_only"]["mean_ms"]["end_to_end_ms"] else "inference_only"
    text = f"""# Measured CPU classifier latency report

> Real measurements on **this benchmark host**, not predictions for a user's laptop.
> A generated, easy three-class pattern dataset is used; do not interpret its
> accuracy as evidence of real-world vision model quality.

## Experiment metadata

- Platform: `{meta['platform']}`
- Python: `{meta['python']}`; PyTorch: `{meta['torch']}`
- CPU threads used by PyTorch: `{meta['threads']}`; logical CPUs reported: `{meta['cpu_count']}`
- Input: {meta['input_name']} ({meta['input_bytes']} bytes, loaded once into RAM)
- Warmup iterations per variant: {meta['warmup']}; measured iterations per variant: {meta['iterations']}; rounds: {meta['rounds']}
- Checkpoint training accuracy: {meta['train_accuracy']:.1%}; validation accuracy: {meta['val_accuracy']:.1%}
- Optimized model engine: `{meta['optimized_engine']}`
- Prediction: `{prediction['label']}` (confidence {prediction['confidence']:.4f})
- **Excluded:** file I/O, training, model loading, trace compilation, and service/network costs.

## Results (milliseconds per request)

All stage columns show **means**. End-to-end median and p95 are also reported.
The stage means sum to the corresponding end-to-end mean by construction.
A negative improvement means the change made things slower.

| Case | Preprocess mean | Inference mean | Postprocess mean | E2E mean | E2E median | E2E p95 | Mean E2E improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![Latency breakdown](latency_breakdown.png)

## Where was the time spent?

At baseline, **{dominant}** accounted for **{shares[dominant]:.2f}%** of mean end-to-end latency.
Preprocessing: {shares['preprocessing_ms']:.2f}%; inference: {shares['inference_ms']:.2f}%; postprocessing: {shares['postprocessing_ms']:.2f}%.

The larger E2E improvement among the single-stage experiments was **{win}**.
Actual changes can deviate from Amdahl projections because CPU caching,
allocator activity, operating-system scheduling and measurement noise affect the
other stages too.

## Amdahl's Law

For stage baseline fraction `f` and stage-only local speedup `s`,

`S_total = 1 / ((1 - f) + f / s)`

- Inference-only: `f = {inf['baseline_fraction']:.4f}`, `s = {inf['local_speedup']:.3f}x`.
  Predicted E2E speedup **{inf['predicted_end_to_end_speedup']:.3f}x**, predicted improvement **{inf['predicted_improvement_percent']:+.2f}%**.
- Preprocess-only: `f = {pre['baseline_fraction']:.4f}`, `s = {pre['local_speedup']:.3f}x`.
  Predicted E2E speedup **{pre['predicted_end_to_end_speedup']:.3f}x**, predicted improvement **{pre['predicted_improvement_percent']:+.2f}%**.

Amdahl's **hard upper bound** from making inference infinitely fast is:
`S_max_inference = 1 / (1 - f_inference) = {1 / (1 - inf['baseline_fraction']):.3f}x`.
No inference implementation alone can beat this bound while other stages stay unchanged.

## Iron Law (simple, serial-pipeline engineering approximation)

`T_end_to_end = T_pre + T_inference + T_post`

Each stage consumes some combination of: moving data (`D / bandwidth`),
arithmetic (`FLOPs / effective_compute_rate`), and overhead (`L`).
Thus `T_stage ≈ T_data + T_compute + T_overhead` is an explanatory,
non-overlapping *approximation*, not an exact hardware identity.

- Python's per-pixel preprocessing performs thousands of interpreted operations,
  repeated scalar conversions, and temporary-list work, so overhead dominates.
- NumPy vectorization moves the same pixels in optimized native loops, reducing
  interpreter overhead and often improving locality.
- Optimizing the model execution path reduces some dispatcher overhead, but has limited impact
  if model inference is a small share of end-to-end time.

## How to read this experiment correctly

This intentionally contrasts a **correct but beginner-level preprocessing
implementation** with a vectorized implementation. It demonstrates the
consequences of choosing the wrong optimization target; it is not a realistic
estimate of the preprocessing cost in an already well-optimized production
service. `torch.jit.trace` is used for the educational inference-only comparison;
TorchScript is deprecated. For a modern alternative, rerun with
`--optimized-engine compiled`, and measure `torch.compile` cold-start costs
separately. Compiler support and results depend on your platform.

Re-run on your laptop and commit **your own** `benchmark.json`,
`timings.csv`, and `latency_breakdown.png` to document your hardware.
"""
    return text
