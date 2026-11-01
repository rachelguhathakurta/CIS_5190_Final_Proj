from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from model import NormalizedRegressor
from preprocess import prepare_data, compute_norm_stats


def haversine_distance_meters(a: Iterable[float], b: Iterable[float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(h))


def batch_iterator(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    count = inputs.size(0)
    indices = torch.randperm(count) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch_idx = indices[start:stop]
        yield inputs[batch_idx], targets[batch_idx]


def train_one_epoch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    batches = 0

    for batch_inputs, batch_targets in batch_iterator(
        inputs, targets, batch_size, shuffle
    ):
        images = batch_inputs.to(device)
        labels_deg = batch_targets.to(device)  # raw degrees

        optimizer.zero_grad()
        outputs_norm = model.backbone(images)  # training compares in normalized space
        labels_norm = model.normalize_targets(labels_deg)
        loss = criterion(outputs_norm, labels_norm)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches += 1

    return running_loss / max(1, batches)


def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    preds, actuals, distances = [], [], []

    with torch.no_grad():
        for batch_inputs, batch_targets in batch_iterator(
            inputs, targets, batch_size, shuffle=False
        ):
            images = batch_inputs.to(device)
            labels_deg = batch_targets.to(device)  # raw degrees

            outputs_deg = model(images)  # model returns degrees in eval mode
            outputs_np = outputs_deg.cpu().numpy()
            labels_np = labels_deg.cpu().numpy()

            preds.append(outputs_np)
            actuals.append(labels_np)
            for pred, actual in zip(outputs_np, labels_np):
                distances.append(haversine_distance_meters(actual, pred))

    preds_np = np.concatenate(preds, axis=0) if preds else np.empty((0, 2))
    actuals_np = np.concatenate(actuals, axis=0) if actuals else np.empty((0, 2))

    mae = mean_absolute_error(actuals_np, preds_np) if len(preds_np) else float("nan")
    rmse = (
        mean_squared_error(actuals_np, preds_np) ** 0.5
        if len(preds_np)
        else float("nan")
    )
    avg_distance = float(np.mean(distances)) if distances else float("nan")

    return {"mae": mae, "rmse": rmse, "avg_distance_m": avg_distance}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Image-to-GPS model.")
    parser.add_argument("--train-dir", default="dataset/train_split.csv")
    parser.add_argument("--val-dir", default="dataset/val_split.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--shuffle", action="store_true")
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        "--output",
        default="model.pth",
        help="Checkpoint path.",
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device()

    train_inputs, train_targets = prepare_data(args.train_dir)
    val_inputs, val_targets = prepare_data(args.val_dir)

    stats = compute_norm_stats(args.train_dir)
    model = NormalizedRegressor(pretrained=True)
    model.set_normalization(stats)
    model = model.to(device)
    print(stats.lat_mean, stats.lat_std, stats.lon_mean, stats.lon_std)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_inputs,
            train_targets,
            args.batch_size,
            args.shuffle,
            optimizer,
            criterion,
            device,
        )
        scheduler.step()
        metrics = evaluate(
            model,
            val_inputs,
            val_targets,
            args.batch_size,
            device,
        )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_mae: {metrics['mae']:.4f} "
            f"- val_rmse: {metrics['rmse']:.4f} "
            f"- val_avg_distance_m: {metrics['avg_distance_m']:.2f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),output_path,)
    print(f"Model weights saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
