from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from sleep_staging.acquisition import load_recording
from sleep_staging.config.settings import load_settings
from sleep_staging.preprocessing import (
    BadChannelDetector,
    ChannelSelector,
    ReferenceTransform,
    SignalFilter,
)
from sleep_staging.preprocessing.types import PreprocessedRecording


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _logical_name_to_actual(ch_names: list[str], logical: str) -> str | None:
    for name in ch_names:
        if logical in name:
            return name
    return None


def _band_power_ratios(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> dict[str, float]:
    total = float(np.sum(psd))
    if total <= 0.0:
        return {"in_band": 0.0, "below": 0.0, "above": 0.0}
    lo, hi = band
    in_mask = (freqs >= lo) & (freqs <= hi)
    below_mask = freqs < lo
    above_mask = freqs > hi
    return {
        "in_band": float(np.sum(psd[in_mask]) / total),
        "below": float(np.sum(psd[below_mask]) / total),
        "above": float(np.sum(psd[above_mask]) / total),
    }


def _run_one(psg_path: Path, settings) -> dict:
    recording = load_recording(psg_path, preload=True, settings=settings.acquisition)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)

    sfreq = float(state.sampling_frequency)
    n_before = int(state.raw.n_times)
    ch_before = list(state.raw.ch_names)
    pre_data = state.raw.get_data().copy()
    pre_types = state.raw.get_channel_types()

    # Required chain: ChannelSelector -> BadChannelDetector -> Reference(original) -> SignalFilter
    state = ChannelSelector(
        names=settings.preprocessing.channels.names,
        types=settings.preprocessing.channels.types,
        require_all_names=settings.preprocessing.channels.require_all_names,
        include_emg=settings.preprocessing.channels.include_emg,
    )(state)
    state = BadChannelDetector(
        flat_std_threshold=settings.preprocessing.bad_channel.flat_std_threshold,
        nan_frac_threshold=settings.preprocessing.bad_channel.nan_frac_threshold,
        saturation_frac_threshold=settings.preprocessing.bad_channel.saturation_frac_threshold,
        eeg_high_std_threshold=settings.preprocessing.bad_channel.eeg_high_std_threshold,
        eeg_peak_to_peak_threshold=settings.preprocessing.bad_channel.eeg_peak_to_peak_threshold,
        eog_high_std_threshold=settings.preprocessing.bad_channel.eog_high_std_threshold,
        eog_peak_to_peak_threshold=settings.preprocessing.bad_channel.eog_peak_to_peak_threshold,
        emg_high_std_threshold=settings.preprocessing.bad_channel.emg_high_std_threshold,
        emg_peak_to_peak_threshold=settings.preprocessing.bad_channel.emg_peak_to_peak_threshold,
        mark_mne_bads=settings.preprocessing.bad_channel.mark_mne_bads,
    )(state)
    state = ReferenceTransform(mode="original")(state)
    state = SignalFilter(
        l_freq=settings.preprocessing.filter.l_freq,
        h_freq=settings.preprocessing.filter.h_freq,
        eeg_l_freq=settings.preprocessing.filter.eeg_l_freq,
        eeg_h_freq=settings.preprocessing.filter.eeg_h_freq,
        eog_l_freq=settings.preprocessing.filter.eog_l_freq,
        eog_h_freq=settings.preprocessing.filter.eog_h_freq,
        emg_l_freq=settings.preprocessing.filter.emg_l_freq,
        emg_h_freq=settings.preprocessing.filter.emg_h_freq,
        notch_freqs=settings.preprocessing.filter.notch_freqs,
    )(state)

    n_after = int(state.raw.n_times)
    ch_after = list(state.raw.ch_names)
    post_data = state.raw.get_data()
    post_types = state.raw.get_channel_types()

    logical_bands = {
        "Fpz-Cz": (0.5, 30.0),
        "Pz-Oz": (0.5, 30.0),
        "horizontal": (0.5, 15.0),
        "submental": (10.0, 30.0),
    }
    report_channels = ["Fpz-Cz", "Pz-Oz", "horizontal", "submental"]

    channel_results: dict[str, dict | None] = {}
    for logical in report_channels:
        actual_before = _logical_name_to_actual(ch_before, logical)
        actual_after = _logical_name_to_actual(ch_after, logical)
        if actual_before is None or actual_after is None:
            channel_results[logical] = None
            continue

        ib = ch_before.index(actual_before)
        ia = ch_after.index(actual_after)
        pre = pre_data[ib]
        post = post_data[ia]
        freqs_pre, psd_pre = welch(pre, fs=sfreq, nperseg=min(4096, pre.size))
        freqs_post, psd_post = welch(post, fs=sfreq, nperseg=min(4096, post.size))
        assert np.allclose(freqs_pre, freqs_post)

        band = logical_bands[logical]
        ratio_pre = _band_power_ratios(freqs_pre, psd_pre, band)
        ratio_post = _band_power_ratios(freqs_post, psd_post, band)

        filter_cfg = state.extras.get("filter", {}).get("per_channel", {}).get(actual_after, {})

        channel_results[logical] = {
            "actual_name": actual_after,
            "channel_type": post_types[ia],
            "sfreq_before_hz": sfreq,
            "sfreq_after_hz": sfreq,
            "samples_before": n_before,
            "samples_after": n_after,
            "applied_filter": {
                "l_freq": filter_cfg.get("l_freq"),
                "h_freq": filter_cfg.get("h_freq"),
                "notch_freqs": filter_cfg.get("notch_freqs"),
            },
            "std_pre": float(np.std(pre)),
            "std_post": float(np.std(post)),
            "ptp_pre": float(np.ptp(pre)),
            "ptp_post": float(np.ptp(post)),
            "has_nonfinite_pre": bool(np.any(~np.isfinite(pre))),
            "has_nonfinite_post": bool(np.any(~np.isfinite(post))),
            "psd_band_ratio_pre": ratio_pre,
            "psd_band_ratio_post": ratio_post,
            "psd_attenuation_ok": {
                "below_band_decreased": ratio_post["below"] < ratio_pre["below"],
                "above_band_decreased": ratio_post["above"] < ratio_pre["above"],
            },
        }

    return {
        "recording_id": psg_path.name,
        "channels_before": ch_before,
        "channels_after": ch_after,
        "sample_count_unchanged": n_before == n_after,
        "channel_order_unchanged": ch_after == [c for c in ch_before if c in ch_after],
        "channel_results": channel_results,
        "filter_extras": state.extras.get("filter", {}),
    }


def main() -> None:
    settings = load_settings(CONFIG_PATH, project_root=PROJECT_ROOT)
    cassette = Path("D:/SleepEDFX/sleep-cassette")
    files = [
        cassette / "SC4001E0-PSG.edf",
        cassette / "SC4002E0-PSG.edf",
        cassette / "SC4011E0-PSG.edf",
        cassette / "SC4012E0-PSG.edf",
        cassette / "SC4021E0-PSG.edf",
    ]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected SC PSG file(s): {missing}")

    out = [_run_one(psg_path, settings) for psg_path in files]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
