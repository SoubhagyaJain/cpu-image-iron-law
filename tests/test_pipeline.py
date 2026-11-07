from io import BytesIO
import numpy as np
from PIL import Image
import torch

from ironlaw_classifier.data import CLASS_NAMES, make_pattern
from ironlaw_classifier.model import TinyCNN, trace_model
from ironlaw_classifier.pipeline import postprocess, run_once
from ironlaw_classifier.preprocess import preprocess_numpy, preprocess_python


def sample_bytes():
    buf = BytesIO()
    Image.fromarray(make_pattern(2, np.random.default_rng(3))).save(buf, format="PNG")
    return buf.getvalue()


def test_prediction_label_and_confidence():
    result = postprocess(torch.tensor([[0.1, 0.2, 5.0]]))
    assert result.label == CLASS_NAMES[2]
    assert result.class_index == 2
    assert 0.98 < result.confidence <= 1


def test_eager_traced_equivalent():
    torch.manual_seed(1)
    model = TinyCNN().eval()
    traced = trace_model(model)
    x = preprocess_numpy(sample_bytes())
    with torch.inference_mode():
        torch.testing.assert_close(model(x), traced(x), rtol=1e-4, atol=1e-5)


def test_stage_times_sum_to_e2e():
    model = TinyCNN().eval()
    png = sample_bytes()
    for fn in (preprocess_python, preprocess_numpy):
        result, prediction = run_once("test", png, model, fn)
        assert abs(result.preprocessing_ms + result.inference_ms + result.postprocessing_ms - result.end_to_end_ms) < 1e-8
        assert result.end_to_end_ms > 0
        assert prediction.label in CLASS_NAMES
