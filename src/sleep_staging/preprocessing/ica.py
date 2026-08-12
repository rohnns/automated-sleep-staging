"""MNE ICA artifact cleaning for selected EEG channels.

Fits ICA on EEG only, optionally marks EOG/blink-related components via the
available EOG channel (``find_bads_eog``), excludes flagged components, and
applies the cleaned reconstruction to EEG. Non-EEG channels (EOG/EMG/aux) are
left unchanged. Channel names, order, sampling rate, and epoch labels are
preserved.

Sleep-EDF SC typically provides only two EEG derivations, so ICA rank is low
(at most two components). With fewer than two usable EEG channels, fitting is
skipped gracefully. EOG detection defaults to correlation scoring because
z-score detection is unreliable with only two components.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
from mne.preprocessing import ICA

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)

_MIN_EEG_CHANNELS = 2


class ICATransform(Transform):
    """Fit/apply MNE ICA on EEG with optional EOG-guided exclusion.

    Parameters
    ----------
    n_components:
        Number of ICA components. ``None`` uses all selected EEG channels
        (capped by channel count). Values larger than the EEG count are capped.
    random_state:
        Seed forwarded to MNE ICA for reproducibility.
    method:
        ICA algorithm (``fastica``, ``infomax``, ``picard``).
    max_iter:
        Maximum ICA iterations (int or MNE ``\"auto\"``).
    detect_eog:
        If true and an EOG channel is present, run ``find_bads_eog``.
    eog_threshold:
        Threshold for ``find_bads_eog`` (interpretation depends on measure).
    eog_measure:
        Scoring measure for EOG detection (``correlation`` or ``zscore``).
    verbose:
        MNE verbosity.
    """

    name = "ica"

    def __init__(
        self,
        *,
        n_components: int | None = None,
        random_state: int = 42,
        method: str = "fastica",
        max_iter: int | str = 500,
        detect_eog: bool = True,
        eog_threshold: float = 0.8,
        eog_measure: str = "correlation",
        verbose: str | bool | None = "ERROR",
    ) -> None:
        if method not in {"fastica", "infomax", "picard"}:
            raise ValueError("method must be 'fastica', 'infomax', or 'picard'")
        if eog_measure not in {"correlation", "zscore"}:
            raise ValueError("eog_measure must be 'correlation' or 'zscore'")
        if n_components is not None and int(n_components) < 1:
            raise ValueError("n_components must be >= 1 when set")
        self.n_components = None if n_components is None else int(n_components)
        self.random_state = int(random_state)
        self.method = str(method)
        self.max_iter = max_iter
        self.detect_eog = bool(detect_eog)
        self.eog_threshold = float(eog_threshold)
        self.eog_measure = str(eog_measure)
        self.verbose = verbose

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        raw = state.raw
        if not raw.preload:
            raw.load_data()

        ch_types = [t.lower() for t in raw.get_channel_types()]
        eeg_names = [
            name
            for name, ch_type in zip(raw.ch_names, ch_types, strict=True)
            if ch_type == "eeg" and name not in raw.info["bads"]
        ]
        eog_names = [
            name
            for name, ch_type in zip(raw.ch_names, ch_types, strict=True)
            if ch_type == "eog"
        ]

        base_extras: dict[str, Any] = {
            "ran": False,
            "skipped_reason": None,
            "eeg_channels": list(eeg_names),
            "eog_channels": list(eog_names),
            "n_components_requested": self.n_components,
            "n_components_fitted": None,
            "excluded_components": [],
            "eog_detection": {
                "performed": False,
                "ch_name": None,
                "threshold": self.eog_threshold,
                "measure": self.eog_measure,
                "scores": [],
                "detected_components": [],
            },
            "random_state": self.random_state,
            "method": self.method,
            "max_iter": self.max_iter,
        }

        if len(eeg_names) < _MIN_EEG_CHANNELS:
            reason = (
                f"fewer than {_MIN_EEG_CHANNELS} usable EEG channels "
                f"(have {len(eeg_names)})"
            )
            base_extras["skipped_reason"] = reason
            state.extras["ica"] = base_extras
            logger.warning("ICA skipped: %s", reason)
            return state

        n_comp = self.n_components
        if n_comp is None:
            n_comp = len(eeg_names)
        else:
            n_comp = min(n_comp, len(eeg_names))

        # MNE rejects fitting a single-component ICA.
        if n_comp < _MIN_EEG_CHANNELS:
            reason = f"n_components={n_comp} is below ICA minimum {_MIN_EEG_CHANNELS}"
            base_extras["skipped_reason"] = reason
            state.extras["ica"] = base_extras
            logger.warning("ICA skipped: %s", reason)
            return state

        names_before = list(raw.ch_names)
        sfreq_before = float(raw.info["sfreq"])
        n_times_before = int(raw.n_times)
        labels_before = None if state.epoch_labels is None else state.epoch_labels.labels

        ica = ICA(
            n_components=n_comp,
            random_state=self.random_state,
            method=self.method,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )
        ica.fit(raw, picks=eeg_names, verbose=self.verbose)

        excluded: list[int] = []
        eog_info = dict(base_extras["eog_detection"])
        if self.detect_eog and eog_names:
            eog_ch = eog_names[0]
            eog_info["performed"] = True
            eog_info["ch_name"] = eog_ch
            try:
                bads, scores = ica.find_bads_eog(
                    raw,
                    ch_name=eog_ch,
                    threshold=self.eog_threshold,
                    measure=self.eog_measure,
                    verbose=self.verbose,
                )
                excluded = [int(x) for x in bads]
                eog_info["detected_components"] = list(excluded)
                eog_info["scores"] = _scores_to_list(scores)
            except Exception as exc:  # noqa: BLE001 - record and continue without exclusion
                eog_info["error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("ICA EOG detection failed (%s); continuing without exclusion", exc)
        elif self.detect_eog and not eog_names:
            eog_info["performed"] = False
            eog_info["skipped_reason"] = "no EOG channel available"

        ica.exclude = list(excluded)
        # Apply only to fitted EEG channels; leave EOG/EMG untouched.
        ica.apply(raw, exclude=ica.exclude, verbose=self.verbose)

        if list(raw.ch_names) != names_before:
            raise RuntimeError("ICA changed channel names/order")
        if float(raw.info["sfreq"]) != sfreq_before:
            raise RuntimeError("ICA changed sampling frequency")
        if int(raw.n_times) != n_times_before:
            raise RuntimeError("ICA changed sample count")
        if labels_before is not None and state.epoch_labels is not None:
            if state.epoch_labels.labels != labels_before:
                raise RuntimeError("ICA changed epoch labels")

        state.extras["ica"] = {
            **base_extras,
            "ran": True,
            "skipped_reason": None,
            "n_components_fitted": int(ica.n_components_),
            "excluded_components": list(excluded),
            "eog_detection": eog_info,
        }
        logger.info(
            "ICA applied on %s (n_components=%s, excluded=%s)",
            eeg_names,
            ica.n_components_,
            excluded,
        )
        return state


def _scores_to_list(scores: Any) -> list[float]:
    """Flatten MNE EOG scores into a JSON-friendly float list."""
    arr = np.asarray(scores, dtype=float).ravel()
    return [float(x) for x in arr]
