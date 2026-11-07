"""Small convolutional classifier with exactly the same weights in all cases."""
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def load_model(checkpoint_path: str) -> TinyCNN:
    # weights_only avoids arbitrary pickle code from untrusted checkpoints.
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TinyCNN()
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model


def trace_model(model: TinyCNN) -> torch.jit.ScriptModule:
    """TorchScript tracing used for a measurable educational CPU comparison.

    TorchScript is deprecated for new production systems; see README for the
    modern torch.compile alternative and its compilation/warmup tradeoffs.
    """
    example = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model.eval(), example)
        return torch.jit.freeze(traced.eval())
