"""Train TinyCNN offline on 3 easy labeled synthetic patterns, CPU-only."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from ironlaw_classifier.data import CLASS_NAMES, create_sample_images, make_dataset
from ironlaw_classifier.model import TinyCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tinycnn.pt"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    X, y = make_dataset(600, seed=42)
    X_val, y_val = make_dataset(150, seed=43)
    model = TinyCNN().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(y))
        for batch in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_accuracy = (model(X).argmax(1) == y).float().mean().item()
            val_accuracy = (model(X_val).argmax(1) == y_val).float().mean().item()
        print(f"Epoch {epoch + 1:02d}: loss={loss.item():.4f}, train={train_accuracy:.1%}, val={val_accuracy:.1%}")
    args.checkpoint.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "seed": 42,
        "epochs": args.epochs,
    }, args.checkpoint)
    create_sample_images(Path("data"))
    print(f"Saved checkpoint to {args.checkpoint} and 3 example PNGs to data/")


if __name__ == "__main__":
    main()
