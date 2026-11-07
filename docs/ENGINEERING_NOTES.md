# Engineering Notes — How to Explain This in an AI/ML Systems Interview

## 1. The end-to-end request path

```text
             PNG bytes (read before benchmark)
                        |
                        v
   [preprocessing: PNG decode, RGB, resize, normalize, tensor]
                        |
                    (1,3,64,64)
                        |
                        v
       [inference: TinyCNN CPU, outputs 3 logits]
                        |
                     (1,3)
                        |
                        v
      [postprocessing: softmax, argmax, label lookup]
                        |
              class label + confidence
```

Three timestamps are gathered at boundaries using a monotonic high-resolution
clock (`perf_counter_ns`). Four timestamps delimit exactly three durations.
This proves `T_total = T_pre + T_infer + T_post` for a **serial request**,
up to exact integer timestamp arithmetic.

Why `@torch.inference_mode()` and `model.eval()`? The first disables autograd
tracking costs; the second sets proper evaluation behavior for layers such as
Dropout and BatchNorm. They are *not* interchangeable.

## 2. The model and sample data

- Three classes: red horizontal bar, green vertical bar, blue circle.
- 64x64 RGB image -> Conv(3->8,stride 2) -> ReLU -> Conv(8->16,stride 2)
  -> ReLU -> global average pooling -> Linear(16->3).
- Offline generated labels + random noise/translation ensure model is genuinely
  trained instead of relying on random weights. It is **not** ImageNet accuracy.
- Sample checkpoint is included; retrain with `python scripts/train.py`.

## 3. What is deliberately kept constant?

- Same checkpoint and all model parameters.
- Same RGB conversion, interpolation and input dimensions.
- Same images and outputs (tensor/logit close checks before measurement).
- Same postprocessor and same CPU thread count.
- Same units, warmup procedure and measured request count.

The **only** experimental levers are the preprocessing implementation and
inference execution engine. This prevents a common benchmarking error:
changing preprocessing semantics or model size while claiming a pure speedup.

## 4. What does the Iron Law mean here?

A serial approximation for each stage is:

`T_stage ≈ (D_vol / BW_eff) + (O / (R_peak * eta_hw)) + L_lat`

- `D_vol`: bytes of data moved (e.g., pixels and model tensors)
- `BW_eff`: effective memory/cache bandwidth (not an advertised peak)
- `O`: number of arithmetic operations
- `R_peak`: nominal arithmetic rate of the CPU
- `eta_hw`: realized utilization/efficiency
- `L_lat`: overhead such as Python loops, memory allocations, dispatch, decode

This equation is *not* an exact lower-bound identity for all modern hardware.
Memory and compute often overlap. More generally, a roofline-like lower bound
is `max(D_vol / BW_eff, O / R_eff) + unoverlapped overhead`.

Here, the baseline preprocessing implementation performs Python-level per-pixel
work. Its dominant cost is *overhead*, not the mathematical complexity of
normalization. The NumPy path uses native vectorized loops to avoid that cost.
No change to the number of CNN operations is necessary for that win.

## 5. Amdahl's Law and interpreting a speedup

Let `f = original stage time / original total time` and
`s = original stage time / changed stage time`.
Then `S_total = 1 / ((1-f) + f/s)` and
`percent_latency_reduction = (1 - 1/S_total)*100`.

Example **illustrative**, not benchmarked:

```
Baseline 10ms = 8ms preprocessing + 1ms inference + 1ms post

Inference 2x faster: 8 + 0.5 + 1 = 9.5ms
  E2E speedup = 10/9.5 = 1.0526x
  latency reduction = (10-9.5)/10 * 100 = 5%

Preprocessing 4x faster: 2 + 1 + 1 = 4ms
  E2E speedup = 10/4 = 2.5x
  latency reduction = (10-4)/10 * 100 = 60%
```

Pay attention: a **2.5x speedup** is a **60% latency reduction**, not a 150%
latency reduction. `speedup` and `% improvement` are different quantities.

Amdahl's projection assumes unchanged stage times for the parts not modified.
Real measurements may differ because processing in one stage changes cache
state, allocator pressure and OS scheduling for another stage.

## 6. Why report mean, median and p95?

- **Mean** conserves the sum of stage durations across requests and is most
  convenient for a time-budget breakdown and Amdahl calculation.
- **Median** describes the typical request and is less sensitive to outliers.
- **p95** describes the tail of the observed single-request latency distribution.
- `median(T_pre)+median(T_infer)+median(T_post)` is not guaranteed to equal
  `median(T_e2e)` — which is why the report uses **means** for decomposition.

## 7. Interview questions with strong responses

**Q: Why not optimize the CNN immediately?**
A: We measured the request path, found an end-to-end bottleneck, computed an
Amdahl bound, and picked the stage with the largest avoidable cost. Optimizing
a small fraction of latency has little end-to-end leverage.

**Q: Does `torch.compile` always help?**
A: No. It introduces compilation/cold-start costs, can add memory overhead,
and CPU speedups depend on the model, data shapes, CPU library and OS.
We isolate construction and warmup from steady-state timing and keep correctness
checks in place. TorchScript here is a deprecated but lightweight educational
reference, not a default production recommendation.

**Q: Why no CUDA sync?**
A: This is deliberately CPU-only. Operations are synchronous at the Python
boundary. A GPU variant must synchronize the device or use device events to
avoid reporting only asynchronous launch costs.

**Q: What if disk I/O is the bottleneck?**
A: It's excluded here by reading PNG bytes once. A realistic serving benchmark
should measure additional stages: file/network input, queueing, decoding,
model time, result serialization and transport.

**Q: What does the speedup say about classifier quality?**
A: Nothing by itself. Same weights, equivalent inputs and outputs are verified,
but a synthetic classifier's validation accuracy does not transfer to real images.
A production optimization must preserve accuracy, calibration and safety guardrails.

**Q: Why not report throughput?**
A: This measures serial latency, not concurrent throughput. A 1/mean-latency
reciprocal is only a rough unloaded capacity estimate; queueing, contention,
CPU scheduling and batching change throughput-versus-latency curves.

## 8. Acceptance criteria / credible README claims

- [x] CPU model trained on reproducible, labeled images.
- [x] All three stages instrumented independently.
- [x] Baseline, inference-only, dominant-stage-only, both.
- [x] Same input, model weights, output label, near-equal logits.
- [x] Warmup + repeated measurements, stage means, median, p95.
- [x] Measured percentage improvements + Amdahl projections.
- [x] Saved raw timings and reproducible run commands.
- [x] Unit tests and CI smoke test.
- [ ] Repeat 3+ times on user's CPU and characterize run-to-run variance.
- [ ] Production data/accuracy evaluation and concurrent-load benchmark.

## 9. What to claim on your resume

> Built a reproducible CPU inference benchmark in PyTorch, profiling preprocessing,
> CNN inference and postprocessing; validated output equivalence and applied
> Amdahl's Law to prioritize the true latency bottleneck. Include your actual
> hardware, measured stage-level results and percent improvements.

Do not claim production-grade real-image accuracy, GPU inference performance,
or throughput improvements that were not measured.
