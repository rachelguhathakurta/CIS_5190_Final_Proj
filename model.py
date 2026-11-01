
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)
DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("model.pth")
def get_inference_transform() -> transforms.Compose:
    """Deterministic preprocessing used at inference time."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


@dataclass(frozen=True)
class NormalizationStats:
    lat_mean: float = 0.0
    lat_std: float = 1.0
    lon_mean: float = 0.0
    lon_std: float = 1.0


class ResNetGPSRegressor(nn.Module):
    """ResNet18 regressor that predicts latitude and longitude."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except AttributeError:
                backbone = models.resnet18(pretrained=True)
        else:
            backbone = models.resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 2)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class NormalizedRegressor(nn.Module):
    """
    Wraps ResNetGPSRegressor and applies denormalization automatically in eval mode.
    Provides helpers to normalize/denormalize targets consistently for training.
    """
    def __init__(self, pretrained: bool = True, normalization: NormalizationStats | None = None) -> None:
        super().__init__()
        self.backbone = ResNetGPSRegressor(pretrained=pretrained)
        self.normalization = normalization or NormalizationStats()

    def set_normalization(self, stats: NormalizationStats) -> None:
        self.normalization = stats

    def _denormalize(self, preds: torch.Tensor) -> torch.Tensor:
        stats = self.normalization
        scales = torch.tensor([stats.lat_std, stats.lon_std], device=preds.device, dtype=preds.dtype)
        shifts = torch.tensor([stats.lat_mean, stats.lon_mean], device=preds.device, dtype=preds.dtype)
        return preds * scales + shifts

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        stats = self.normalization
        means = torch.tensor([stats.lat_mean, stats.lon_mean], device=targets.device, dtype=targets.dtype)
        stds = torch.tensor([stats.lat_std, stats.lon_std], device=targets.device, dtype=targets.dtype)
        return (targets - means) / stds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.backbone(x)
        if not self.training:
            return self._denormalize(preds)
        return preds


class Model(nn.Module):
    """
    Submission-ready model that mirrors the template requirements.
    """

    def __init__(self, weights_path: str | Path | None = None, device: str | torch.device | None = None, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = self._resolve_device(device)
        self.transform = get_inference_transform()
        self.normalization = NormalizationStats(
            lat_mean=39.951938629150390,
            lat_std=0.0006589124677702785,
            lon_mean=-75.19157409667969,
            lon_std=0.00034724801662378013,
        )
        self.model = ResNetGPSRegressor(pretrained=False).to(self.device)
        self._load_weights(weights_path)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_weights(self, weights_path: str | Path | None) -> None:
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        if not path.exists():
            # Leave randomly initialized weights so that predict still runs.
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
            while key.startswith("backbone.backbone."):
                key = key[len("backbone."):]
            normalized[key] = value
        self.model.load_state_dict(normalized, strict=True)
        self.normalization = NormalizationStats(
            lat_mean=39.951938629150390,
            lat_std=0.0006589124677702785,
            lon_mean=-75.19157409667969,
            lon_std=0.00034724801662378013,
            )

    def eval(self) -> Model:
        return super().eval()

    def _prepare_tensor(self, sample: Any) -> torch.Tensor:
        if isinstance(sample, torch.Tensor):
            return sample
        if isinstance(sample, (str, Path)):
            image = Image.open(sample).convert("RGB")
            return self.transform(image)
        if isinstance(sample, Image.Image):
            return self.transform(sample)
        if isinstance(sample, dict):
            if "image" in sample:
                return self._prepare_tensor(sample["image"])
            if "input" in sample:
                return self._prepare_tensor(sample["input"])
        if isinstance(sample, Sequence) and sample:
            # Some pipelines pass tuples like (tensor, metadata)
            return self._prepare_tensor(sample[0])
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    def _denormalize(self, preds: torch.Tensor) -> torch.Tensor:
        stats = self.normalization
        scales = torch.tensor([stats.lat_std, stats.lon_std], device=preds.device)
        shifts = torch.tensor([stats.lat_mean, stats.lon_mean], device=preds.device)
        return preds * scales + shifts

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        tensors = torch.stack([self._prepare_tensor(sample) for sample in batch])
        tensors = tensors.to(self.device)

        with torch.no_grad():
            preds = self.model(tensors)
            preds = self._denormalize(preds).cpu().tolist()

        return preds


def get_model() -> Model:
    return Model()
