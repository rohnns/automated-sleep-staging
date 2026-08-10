"""Config loading includes preprocessing settings."""

from __future__ import annotations

from pathlib import Path

import yaml

from sleep_staging.config import load_settings


def test_load_preprocessing_settings() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = load_settings(project_root / "configs" / "default.yaml")
    prep = settings.preprocessing
    assert prep.epoch_duration_sec == 30.0
    assert prep.wake_crop.enabled is True
    assert prep.wake_crop.buffer_sec == 1800.0
    assert prep.wake_crop.minutes == 30.0
    assert prep.stage_map.unmapped_policy == "ignore"
    assert prep.stage_map.ignore_label == "IGNORE"
    assert prep.channels.names == ("Fpz-Cz", "Pz-Oz", "horizontal", "submental")
    assert prep.filter.notch_freqs == (50.0,)
    assert prep.normalize.method == "zscore"
    enc = settings.encodings
    assert enc.representation == "raw"
    assert enc.bandpower.bands[0][0] == "delta"
    assert len(enc.bandpower.bands) == 5
    assert enc.bandpower.include_log_absolute is True
    assert enc.bandpower.include_relative is True
    assert enc.bandpower.include_ratios is False
    assert enc.bandpower.welch.nperseg == 400
    assert enc.bandpower.welch.noverlap == 200
    assert enc.bandpower.welch.window == "hamming"
    assert enc.bandpower.welch.average == "median"
    assert enc.bandpower.expected_sfreq == 100.0
    assert enc.time_frequency.method == "stft"
    assert enc.time_frequency.stft.hop_length == 100
    assert enc.time_frequency.stft.n_fft == 256
    assert enc.time_frequency.stft.log_scale is True
    assert enc.time_frequency.stft.expected_sfreq == 100.0


def test_channels_mapping_form(tmp_path: Path) -> None:
    config = {
        "acquisition": {"data_root": "D:/SleepEDFX"},
        "preprocessing": {
            "channels": {
                "names": ["Fpz-Cz"],
                "require_all_names": True,
            },
            "normalize": {"method": "robust"},
        },
        "encodings": {
            "representation": "time_frequency",
            "time_frequency": {"method": "cwt"},
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = load_settings(path, project_root=tmp_path)
    assert settings.preprocessing.channels.names == ("Fpz-Cz",)
    assert settings.preprocessing.normalize.method == "robust"
    assert settings.encodings.representation == "time_frequency"
    assert settings.encodings.time_frequency.method == "cwt"
