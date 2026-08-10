"""Model factory for Phase 4a baselines."""

from __future__ import annotations

from torch import nn

from sleep_staging.models.baselines import BandPowerMLP, RawCNN1D, STFTCNN2D


def build_baseline_model(
    representation: str,
    *,
    in_channels: int = 1,
    n_classes: int = 5,
    n_band_features: int = 10,
) -> nn.Module:
    """Build the Phase 4a baseline matching an encoding representation."""
    if representation == "raw":
        return RawCNN1D(in_channels=in_channels, n_classes=n_classes)
    if representation == "bandpower":
        return BandPowerMLP(
            in_channels=in_channels,
            n_features=n_band_features,
            n_classes=n_classes,
        )
    if representation == "time_frequency":
        return STFTCNN2D(in_channels=in_channels, n_classes=n_classes)
    raise ValueError(
        f"Unknown representation {representation!r}; "
        "expected 'raw', 'bandpower', or 'time_frequency'"
    )
