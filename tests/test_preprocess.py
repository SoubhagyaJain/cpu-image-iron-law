from io import BytesIO
import numpy as np
from PIL import Image
import torch

from ironlaw_classifier.data import make_pattern
from ironlaw_classifier.preprocess import preprocess_numpy, preprocess_python


def test_preprocess_equivalent():
    rng = np.random.default_rng(99)
    for index in (0, 1, 2):
        buf = BytesIO()
        Image.fromarray(make_pattern(index, rng)).save(buf, format="PNG")
        slow = preprocess_python(buf.getvalue())
        fast = preprocess_numpy(buf.getvalue())
        assert slow.shape == fast.shape == (1, 3, 64, 64)
        assert slow.dtype == fast.dtype == torch.float32
        torch.testing.assert_close(slow, fast, rtol=0, atol=1e-6)
