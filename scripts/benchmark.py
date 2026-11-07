"""Measure four equivalent CPU inference pipelines after warmup."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from ironlaw_classifier.model import load_model, trace_model
from ironlaw_classifier.pipeline import CASE_CONFIGS, PREPROCESSORS, run_once
from ironlaw_classifier.report import CASES, markdown_report, save_plot, save_records_csv, save_summary_csv, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("data/red_horizontal.png"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tinycnn.pt"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--optimized-engine", choices=("traced", "compiled"), default="traced",
                        help="Use TorchScript for a quick reference comparison, or torch.compile for a modern optional experiment")
    args = parser.parse_args()
    if min(args.iterations, args.rounds, args.threads) < 1 or args.warmup < 0:
        parser.error("iterations, rounds and threads must be >= 1, warmup >= 0")

    # Control CPU parallelism across ALL scenarios. Use taskset/process affinity
    # externally if needed. Do not force CUDA even if it is available.
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    eager = load_model(str(args.checkpoint)).cpu().eval()
    optimized = (trace_model(eager) if args.optimized_engine == "traced"
                 else torch.compile(eager, backend="inductor", mode="default"))
    engines = {"eager": eager, "traced": optimized}
    image_bytes = args.image.read_bytes()  # disk I/O excluded from benchmark

    # Fairness: same image/tensors and close logits across implementations.
    py = PREPROCESSORS["python"](image_bytes)
    vec = PREPROCESSORS["numpy"](image_bytes)
    torch.testing.assert_close(py, vec, rtol=1e-6, atol=1e-6)
    with torch.inference_mode():
        torch.testing.assert_close(eager(py), optimized(vec), rtol=1e-4, atol=1e-5)

    cases = {}
    for case, (preprocessor_id, engine_id) in CASE_CONFIGS.items():
        cases[case] = (PREPROCESSORS[preprocessor_id], engines[engine_id])

    # Warm each combination to avoid first-call/JIT/init costs skewing results.
    for case in CASES:
        preprocessor, engine = cases[case]
        for _ in range(args.warmup):
            run_once(case, image_bytes, engine, preprocessor)

    rng = random.Random(2026)
    records = []
    predictions = {}
    for round_id in range(args.rounds):
        # Pair variants closely in time so OS/CPU drift has less chance to favor
        # one whole block. Shuffle once per four-way request group, not inside
        # the timed function.
        for _ in range(args.iterations):
            order = list(CASES)
            rng.shuffle(order)
            for case in order:
                preprocessor, engine = cases[case]
                sample, prediction = run_once(case, image_bytes, engine, preprocessor)
                records.append(sample)
                predictions[case] = prediction
        print(f"Completed interleaved round {round_id + 1}/{args.rounds}", flush=True)

    labels = {p.label for p in predictions.values()}
    if len(labels) != 1:
        raise RuntimeError(f"different predicted labels across cases: {predictions}")

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    meta = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "input_name": args.image.name,
        "input_bytes": len(image_bytes),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "train_accuracy": float(checkpoint["train_accuracy"]),
        "val_accuracy": float(checkpoint["val_accuracy"]),
        "optimized_engine": args.optimized_engine,
        "engine_build_excluded": True,
        "file_io_excluded": True,
    }
    summary = summarize(records)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps({
        "metadata": meta,
        "predictions": {key: asdict(value) for key, value in predictions.items()},
        "summary": summary,
    }, indent=2), encoding="utf-8")
    save_records_csv(records, args.out / "timings.csv")
    save_summary_csv(summary, args.out / "summary.csv")
    save_plot(summary, args.out / "latency_breakdown.png")
    (args.out / "REPORT.md").write_text(markdown_report(summary, meta, asdict(predictions["baseline"])), encoding="utf-8")
    print("\nMean latency by stage (ms; % improvement vs baseline):")
    for case in CASES:
        s = summary[case]["mean_ms"]
        print(f"{case:16} pre={s['preprocessing_ms']:7.3f}  infer={s['inference_ms']:7.3f}  "
              f"post={s['postprocessing_ms']:6.3f}  E2E={s['end_to_end_ms']:7.3f}  "
              f"change={summary['improvement_percent_vs_baseline'][case]:+7.2f}%")
    print(f"\nDominant baseline stage: {summary['baseline_dominant_stage']}")
    print(f"Saved benchmark outputs under {args.out}/")


if __name__ == "__main__":
    main()
