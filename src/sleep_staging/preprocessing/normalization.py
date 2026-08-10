"""Per-recording signal normalization.

Applies channel-wise statistics over the current recording window (after
optional wake cropping). Dataset-wide normalization is intentionally not
supported — subject/recording scale differences are handled per recording.

MNE stores EEG in Volts; ``eps=1e-8`` (0.01 µV) is below typical noise floors
while protecting flat / disconnected channels from division by zero.
"""

from __future__ import annotations

import mne
import numpy as np
from numpy.typing import NDArray

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)

SUPPORTED_METHODS = frozenset({"zscore", "robust", "center"})


class RecordingNormalizer(Transform):
    """Normalize each channel independently over the full recording.

    Parameters
    ----------
    method:
        ``"zscore"`` (mean/std), ``"robust"`` (median/IQR), or ``"center"``
        (mean only).
    eps:
        Numerical floor for scale denominators.
    """

    name = "recording_normalizer"

    def __init__(self, *, method: str = "zscore", eps: float = 1e-8) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {sorted(SUPPORTED_METHODS)}, got {method!r}"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.method = method
        self.eps = float(eps)

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        if not state.raw.preload:
            state.raw.load_data()
        data = state.raw.get_data()
        if data.size == 0:
            raise TransformError("Cannot normalize empty signal data")

        normalized, stats = _normalize_channels(data, method=self.method, eps=self.eps)
        state.raw = _replace_raw_data(state.raw, normalized)
        state.extras["normalization"] = {
            "method": self.method,
            **stats,
        }
        logger.info(
            "Normalized recording with method=%s across %d channel(s)",
            self.method,
            data.shape[0],
        )
        return state


def _replace_raw_data(
    raw: mne.io.BaseRaw,
    data: NDArray[np.floating],
) -> mne.io.BaseRaw:
    """Return a new ``RawArray`` with ``data``, preserving info/annotations.

    Avoids mutating the private ``raw._data`` buffer directly.
    """
    annotations = raw.annotations.copy() if raw.annotations is not None else None
    replacement = mne.io.RawArray(
        data.astype(np.float64, copy=False),
        raw.info.copy(),
        verbose="ERROR",
    )
    if annotations is not None and len(annotations) > 0:
        replacement.set_annotations(annotations, emit_warning=False)
    return replacement


def _normalize_channels(
    data: NDArray[np.floating],
    *,
    method: str,
    eps: float,
) -> tuple[NDArray[np.floating], dict[str, list[float] | None]]:
    """Return normalized data and per-channel summary statistics."""
    if method == "center":
        means = data.mean(axis=1, keepdims=True)
        return data - means, {
            "channel_means": means.ravel().tolist(),
            "channel_stds": None,
            "channel_medians": None,
            "channel_iqrs": None,
        }

    if method == "zscore":
        means = data.mean(axis=1, keepdims=True)
        stds = np.maximum(data.std(axis=1, keepdims=True), eps)
        return (data - means) / stds, {
            "channel_means": means.ravel().tolist(),
            "channel_stds": stds.ravel().tolist(),
            "channel_medians": None,
            "channel_iqrs": None,
        }

    # Robust scaler: (x - median) / IQR, with IQR floored by eps.
    medians = np.median(data, axis=1, keepdims=True)
    q75 = np.percentile(data, 75, axis=1, keepdims=True)
    q25 = np.percentile(data, 25, axis=1, keepdims=True)
    iqrs = np.maximum(q75 - q25, eps)
    return (data - medians) / iqrs, {
        "channel_means": None,
        "channel_stds": None,
        "channel_medians": medians.ravel().tolist(),
        "channel_iqrs": iqrs.ravel().tolist(),
    }
