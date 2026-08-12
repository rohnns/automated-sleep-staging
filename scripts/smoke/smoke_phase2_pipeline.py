"""Full default-pipeline smoke on five Sleep-EDF SC recordings."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from sleep_staging.acquisition import load_recording
from sleep_staging.config.settings import load_settings
from sleep_staging.preprocessing.pipeline import preprocess_recording

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    captured: list[str] = []

    def _capture(message, category, filename, lineno, file=None, line=None):
        captured.append(f"{category.__name__}: {message}")

    warnings.showwarning = _capture

    settings = load_settings(CONFIG_PATH, project_root=PROJECT_ROOT)
    prep = settings.preprocessing
    print("CONFIG reference.mode =", prep.reference.mode)
    print(
        "CONFIG filters EEG",
        prep.filter.eeg_l_freq,
        prep.filter.eeg_h_freq,
        "EOG",
        prep.filter.eog_l_freq,
        prep.filter.eog_h_freq,
    )
    print("CONFIG normalize", prep.normalize.method, "enabled", prep.normalize.enabled)

    cassette = Path("D:/SleepEDFX/sleep-cassette")
    files = [
        cassette / f"{rid}-PSG.edf"
        for rid in ("SC4001E0", "SC4002E0", "SC4011E0", "SC4012E0", "SC4021E0")
    ]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PSG file(s): {missing}")

    for psg in files:
        recording = load_recording(psg, preload=True, settings=settings.acquisition)
        result = preprocess_recording(recording, prep, copy=True)
        data = result.raw.get_data()
        sfreq = float(result.sampling_frequency)
        n_samples = int(result.raw.n_times)
        duration = n_samples / sfreq
        labels = result.epoch_labels
        assert labels is not None
        n_epochs = labels.n_epochs
        expected_samples = int(round(n_epochs * labels.duration_sec * sfreq))
        onsets = np.asarray(labels.onsets_sec, dtype=float)
        grid_ok = bool(np.allclose(np.mod(onsets, 30.0), 0.0, atol=1e-9))
        means = data.mean(axis=1)
        stds = data.std(axis=1)
        nonfinite = bool(np.any(~np.isfinite(data)))
        filt = result.extras.get("filter", {}).get("per_channel", {})
        wake = result.extras.get("wake_crop", {})
        norm = result.extras.get("normalize", {})

        print("===", psg.stem)
        print("  channels", list(result.raw.ch_names), "types", result.raw.get_channel_types())
        for ch in result.raw.ch_names:
            cfg = filt.get(ch, {})
            print(
                f"  filter {ch}: {cfg.get('l_freq')}-{cfg.get('h_freq')} Hz "
                f"type={cfg.get('mne_type')}"
            )
        print("  sfreq", sfreq, "n_samples", n_samples, "duration_s", round(duration, 3))
        print(
            "  n_epochs",
            n_epochs,
            "epoch_dur",
            labels.duration_sec,
            "expected_samples",
            expected_samples,
            "sample_epoch_align",
            n_samples == expected_samples,
        )
        print(
            "  onset_first_last",
            (float(onsets[0]), float(onsets[-1])) if n_epochs else None,
            "grid_aligned_30s",
            grid_ok,
        )
        print(
            "  wake_crop tmin/tmax",
            wake.get("tmin_sec"),
            wake.get("tmax_sec"),
            "align",
            wake.get("align_to_epoch_grid"),
        )
        print("  NaN/Inf", nonfinite)
        print("  normalize extras keys", sorted(norm.keys()))
        print(
            "  channel mean/std",
            [
                (ch, round(float(m), 6), round(float(s), 6))
                for ch, m, s in zip(result.raw.ch_names, means, stds)
            ],
        )
        print("  applied", result.applied_transforms)

    print("WARNINGS_CAPTURED", len(captured))
    for item in captured[:30]:
        print("WARN:", item)


if __name__ == "__main__":
    main()
