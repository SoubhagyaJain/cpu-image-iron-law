<img width="1852" height="998" alt="Screenshot 2026-11-07 150650" src="https://github.com/user-attachments/assets/f27f44a0-89d8-4ee0-ab28-bc1c435e9b1d" />



# The Iron Law of ML Systems — CPU Image Classifier Lab

**Question:** If an image classifier is slow, should we optimize its neural network first?
This repository measures the entire synchronous request path and uses **Amdahl's Law**
to decide which stage deserves optimization.

### Measured results

On a CPU-based PyTorch image classification pipeline, baseline mean end-to-end latency was 1.299 ms. Preprocessing accounted for 86.74% of this latency. Optimizing model inference alone produced only a 1.25% end-to-end improvement, while vectorizing image preprocessing reduced latency by 73.33%. Combining both optimizations reduced latency to 0.326 ms, achieving approximately 3.98× overall speedup. The results illustrate Amdahl's Law and the importance of identifying the dominant system cost before optimization.

**Verification:** The latest test run passed all 6 tests. It reported three
PyTorch `FutureWarning`s from the intentionally retained TorchScript
`trace`/`freeze` benchmark path; see the inference-only optimization caveat below.

### What it does

Train a small **PyTorch CPU CNN** on three labeled synthetic image classes, then
benchmark the same image and trained weights under four scenarios:

| Scenario | Preprocessing | Inference | Postprocessing |
|---|---|---|---|
| `baseline` | Python pixel loop | PyTorch eager | Same softmax/argmax |
| `inference_only` | Python pixel loop | Traced/frozen model | Same softmax/argmax |
| `preprocess_only` | Vectorized NumPy | PyTorch eager | Same softmax/argmax |
| `both` | Vectorized NumPy | Traced/frozen model | Same softmax/argmax |

The preprocessing methods **decode the same PNG, resize the same way, and produce
numerically equivalent float32 RGB tensors**. The two inference engines use the
same trained weights. Equality checks run **before** the benchmark.

**Important:** The naive Python baseline is deliberately inefficient but correct:
the educational goal is to show a bottleneck. This is not a claim that a
production PyTorch pipeline should use Python pixel loops.

## Repository map

```text
cpu-image-iron-law/
├── README.md                      # experiment, installation, methodology
├── pyproject.toml                 # installable Python package and dependencies
├── .gitignore
├── .github/workflows/ci.yml      # unit tests and smoke test
├── src/ironlaw_classifier/
│   ├── __init__.py
│   ├── data.py                    # offline 3-class pattern dataset + sample PNGs
│   ├── model.py                   # TinyCNN, load checkpoint, trace+freeze engine
│   ├── preprocess.py              # slow Python loop vs vectorized NumPy
│   ├── pipeline.py                # three timed stages + prediction + 4 scenarios
│   └── report.py                  # means, medians, p95, Amdahl, CSV/plot/Markdown
├── scripts/
│   ├── train.py                   # reproducible training, checkpoint creation
│   └── benchmark.py               # CPU runner, validation, warmup, 4-way benchmark
├── docs/
│   └── ENGINEERING_NOTES.md         # architecture, pitfalls, formulas, interview Q&A
├── tests/
│   ├── test_preprocess.py         # numerically equivalent input tensors
│   ├── test_pipeline.py           # traced logits, prediction, timing accounting
│   └── test_reporting.py          # exact 10ms toy Amdahl computation
├── data/
│   ├── red_horizontal.png
│   ├── green_vertical.png
│   └── blue_circle.png
├── artifacts/
│   └── tinycnn.pt                 # generated CPU checkpoint
└── results/
    ├── benchmark.json             # raw summary, hardware, stage share, Amdahl
    ├── INITIAL_RUN.md              # initial block-run numbers and methodology change
    ├── timings.csv                # one sample per request per scenario
    ├── summary.csv                # per-scenario mean/median/p95 and improvement
    ├── latency_breakdown.png      # stacked latency chart
    └── REPORT.md                  # narrative with measured numbers
```

## Setup (Windows PowerShell, CPU-only)

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Prefer a CPU-only PyTorch wheel even on a laptop with an NVIDIA GPU:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

If your system uses a different compatible Python, replace `py -3.12` with
`python`. On Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`.
For current PyTorch platform-specific installation, see https://pytorch.org/get-started/locally/.

## Run it

```powershell
# 1. Train a 3-class classifier. Checkpoint also ships with the repository.
python scripts/train.py --epochs 10

# 2. Verify all implementations agree.
python -m pytest

# 3. Measure on YOUR CPU (300 timing samples per scenario).
python scripts/benchmark.py --warmup 20 --iterations 100 --rounds 3 --threads 1

# 4. Inspect these outputs:
# results/REPORT.md, results/summary.csv, results/benchmark.json,
# results/timings.csv, results/latency_breakdown.png
```

Small sanity run first: `python scripts/benchmark.py --warmup 5 --iterations 10 --rounds 1`.
To repeat on a different input: `--image data/green_vertical.png`.

## Benchmark rules

1. **CPU only:** model always runs on CPU; no CUDA calls; use `--threads 1` to
   control variability. Report the number of CPU threads used.
2. **Same workload:** the same image bytes and checkpoint in every scenario.
3. **Disk reads excluded:** input PNG bytes are loaded into memory once; decoding
   and resizing occur inside preprocessing. Disk I/O is a separate concern.
4. **Training, initialization and tracing excluded:** do these before warmup;
   otherwise a one-time compilation cost would distort steady-state latency.
5. **Warmup:** run every case before taking measurements.
6. **Timer:** `time.perf_counter_ns()` measures consecutive stage boundaries.
7. **Randomized interleaving:** every group of four requests runs all four
   scenarios in random order, reducing confounding from time/CPU drift.
8. **Statistics:** mean stage durations show how time is apportioned; end-to-end
   **mean, median and p95** are all stored. Only mean is used for Amdahl maths.
9. **Accuracy held constant:** matching tensors and logits validated; all
   scenarios must return the same label. Compare model accuracy separately.
10. **Noise:** expect jitter from CPU boost, OS scheduling, background programs,
    and memory allocator behavior. Run multiple times to verify small effects.
11. **Revised harness:** `results/INITIAL_RUN.md` records the earlier block-run
    aggregates; the final raw CSV/report use randomized per-group interleaving.

The experiment measures **latency for one synchronous image request at a time**,
**not** GPU efficiency, multithreaded throughput, service-level latency, or
batch inference. Warmup also cannot eliminate every source of variability.

## Analysis: Iron Law

For this sequential request path:

`T_e2e = T_preprocessing + T_inference + T_postprocessing`.

A useful physical-cost lens for any stage is:

`T_stage ≈ T_data_movement + T_compute + T_overhead`

or, as a simple serial resource-cost model:

`T_stage ≈ bytes_moved/effective_bandwidth + FLOPs/effective_compute_rate + overhead`.

The terms can **overlap** on modern hardware: a complete roofline model uses a
`max()` for overlapping compute and memory costs plus overhead. The simple sum
is therefore an educational approximation, not a universal exact identity.

Here, the beginner Python preprocessing loop has substantial *interpreter*
overhead. Vectorization moves work into native code over contiguous arrays,
without reducing the number of classes or changing the image content.

## Analysis: Amdahl's Law

For the fraction `f` of total baseline time spent in the optimized stage and
local speedup `s` within that stage:

`S_total = 1 / ((1 - f) + f/s)`

For example, a purely **illustrative** 10ms baseline that spends 8ms in
preprocessing, 1ms in inference and 1ms in postprocessing:

- If inference becomes **2x faster**: `10 / (8 + 0.5 + 1) = 1.0526x`,
  or just **5% end-to-end latency reduction**.
- If preprocessing becomes **4x faster**: `10 / (2 + 1 + 1) = 2.5x`,
  or **60% end-to-end latency reduction**.
- If inference becomes infinitely fast, the best possible overall speedup is
  `10 / 9 = 1.111x` while other stages are unchanged.

This toy example is *not* a benchmark measurement. The actual measurements are
written under `results/` by `scripts/benchmark.py`.

## Inference-only optimization caveat

This project uses `torch.jit.trace` + `torch.jit.freeze` for a lightweight,
repeatable *educational* comparison of eager vs a traced CPU model.
TorchScript is **deprecated** in current PyTorch releases; it may speed up **or
slow down** inference on your CPU. The benchmark reports the measured effect,
including negative improvements. For new production code, consider
`torch.compile` instead, and benchmark compilation time, cold-start effects,
thread counts, and steady-state latency independently. You can try it with:

```powershell
python scripts/benchmark.py --optimized-engine compiled --warmup 20 --iterations 100 --rounds 3 --out results_compiled
```

This optional path requires compiler support on your platform and may take
substantial startup time. The included saved measurements use **traced** mode.

An experiment that turns out slower is still valid evidence: it means your
particular intervention was ineffective, not that the law was wrong.

## Limitations / next engineering steps

- Move from synthetic shapes to real images and an accuracy benchmark.
- Compare CPU thread-count sweeps: `--threads 1`, `--threads 2`, `--threads 4`.
- Add repeated runs across input sizes 64, 128, 224 pixels and batch sizes.
- Add instrumentation for decoding, resize, normalization, and tensor copying
  *inside preprocessing* to see the exact culprit.
- Measure distributions under concurrent requests rather than only serial p95.
- Use `torch.profiler` to investigate kernel costs after the stage-level bottleneck
  is established.

## References

- PyTorch installation: https://pytorch.org/get-started/locally/
- PyTorch inference mode: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
- PyTorch `torch.jit.freeze`: https://docs.pytorch.org/docs/stable/generated/torch.jit.freeze.html
- Amdahl's law: https://en.wikipedia.org/wiki/Amdahl%27s_law
