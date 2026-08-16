"""Baseline classifiers for single-epoch sleep staging."""

from __future__ import annotations

import torch
from torch import nn


class RawCNN1D(nn.Module):
    """Lightweight 1D CNN for raw waveforms ``(B, C, 3000)`` → ``(B, 5)``."""

    def __init__(self, *, in_channels: int = 1, n_classes: int = 5) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=25, stride=5, padding=12),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 32, kernel_size=15, stride=3, padding=7),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"RawCNN1D expects (B, C, T), got shape {tuple(x.shape)}")
        h = self.features(x).squeeze(-1)
        return self.classifier(h)


class BandPowerMLP(nn.Module):
    """Small MLP for band-power features ``(B, C, 10)`` → ``(B, 5)``."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        n_features: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or n_features <= 0:
            raise ValueError("in_channels and n_features must be positive")
        self.in_channels = in_channels
        self.n_features = n_features
        self.n_classes = n_classes
        in_dim = in_channels * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"BandPowerMLP expects (B, C, F), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels or x.shape[2] != self.n_features:
            raise ValueError(
                f"BandPowerMLP expected (B, {self.in_channels}, {self.n_features}), "
                f"got {tuple(x.shape)}"
            )
        return self.net(x)


class STFTCNN2D(nn.Module):
    """Lightweight 2D CNN for STFT log-power ``(B, C, F, T)`` → ``(B, 5)``."""

    def __init__(self, *, in_channels: int = 1, n_classes: int = 5) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"STFTCNN2D expects (B, C, F, T), got shape {tuple(x.shape)}")
        h = self.features(x).flatten(1)
        return self.classifier(h)
