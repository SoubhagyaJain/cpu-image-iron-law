"""Same prediction function and exactly three time-measured pipeline stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Callable

import torch

from .data import CLASS_NAMES
from .preprocess import preprocess_numpy, preprocess_python


@dataclass(frozen=True)
class Prediction:
    class_index: int
    label: str
    confidence: float


@dataclass(frozen=True)
class Sample:
    case: str
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    end_to_end_ms: float


def postprocess(logits: torch.Tensor) -> Prediction:
    probabilities = torch.softmax(logits, dim=1)[0]
    class_index = int(torch.argmax(probabilities).item())
    return Prediction(
        class_index=class_index,
        label=CLASS_NAMES[class_index],
        confidence=float(probabilities[class_index].item()),
    )


@torch.inference_mode()
def run_once(
    case: str,
    png_bytes: bytes,
    model: Callable[[torch.Tensor], torch.Tensor],
    preprocessing: Callable[[bytes], torch.Tensor],
) -> tuple[Sample, Prediction]:
    t0 = perf_counter_ns()
    x = preprocessing(png_bytes)
    t1 = perf_counter_ns()
    logits = model(x)
    t2 = perf_counter_ns()
    prediction = postprocess(logits)
    t3 = perf_counter_ns()
    scale = 1e-6  # nanoseconds -> milliseconds
    record = Sample(
        case=case,
        preprocessing_ms=(t1 - t0) * scale,
        inference_ms=(t2 - t1) * scale,
        postprocessing_ms=(t3 - t2) * scale,
        end_to_end_ms=(t3 - t0) * scale,
    )
    return record, prediction


CASE_CONFIGS = {
    "baseline": ("python", "eager"),
    "inference_only": ("python", "traced"),
    "preprocess_only": ("numpy", "eager"),
    "both": ("numpy", "traced"),
}
PREPROCESSORS = {"python": preprocess_python, "numpy": preprocess_numpy}
