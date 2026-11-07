from pathlib import Path

import torch

from ironlaw_classifier.data import CLASS_NAMES
from ironlaw_classifier.model import load_model
from ironlaw_classifier.preprocess import preprocess_numpy


def test_checkpoint_classifies_all_three_examples():
    root = Path(__file__).resolve().parents[1]
    model = load_model(str(root / "artifacts" / "tinycnn.pt"))
    with torch.inference_mode():
        for index, name in enumerate(CLASS_NAMES):
            png = (root / "data" / f"{name}.png").read_bytes()
            prediction_index = int(model(preprocess_numpy(png)).argmax(dim=1).item())
            assert prediction_index == index, f"incorrect prediction for {name}"
