from __future__ import annotations

import pytest
import torch

from sleep_staging.models import BandPowerMLP, RawCNN1D, STFTCNN2D, build_baseline_model


def test_raw_cnn_shape() -> None:
    model = RawCNN1D(in_channels=1)
    assert model(torch.randn(2, 1, 3000)).shape == (2, 5)


def test_bandpower_mlp_shape_and_validation() -> None:
    model = BandPowerMLP(in_channels=1, n_features=10)
    assert model(torch.randn(2, 1, 10)).shape == (2, 5)
    with pytest.raises(ValueError):
        model(torch.randn(2, 2, 10))


def test_stft_cnn_shape() -> None:
    model = STFTCNN2D(in_channels=1)
    assert model(torch.randn(2, 1, 75, 28)).shape == (2, 5)


def test_factory_uses_frozen_representations() -> None:
    assert isinstance(build_baseline_model("raw"), RawCNN1D)
    assert isinstance(build_baseline_model("bandpower"), BandPowerMLP)
    assert isinstance(build_baseline_model("time_frequency"), STFTCNN2D)
    with pytest.raises(ValueError):
        build_baseline_model("lstm")
