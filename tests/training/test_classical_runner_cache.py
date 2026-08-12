from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.phase4_helpers import make_collection
from sleep_staging.config import load_settings
from sleep_staging.representations.types import EncodedDataset, RepresentationMetadata
from sleep_staging.training import classical_runner as runner


def _dummy_psg(name: str = "SC4001E0-PSG.edf") -> Path:
    return Path(name)


def test_classical_cache_write_read_and_invalid_incompatible(tmp_path: Path, monkeypatch) -> None:
    settings = load_settings(Path("configs/default.yaml"))
    collection = make_collection("bandpower", subjects=("S1",), n_epochs=4)
    encoded = collection.items[0]

    monkeypatch.setattr(runner, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "out" / "result.json")

    meta = runner._build_cache_metadata(
        settings=settings,
        encoded=encoded,
        source_psg=_dummy_psg(),
        source_hypnogram=_dummy_psg("SC4001E0-Hypnogram.edf"),
    )
    data_path = runner._save_cache(encoded, meta)
    meta_path = data_path.with_suffix(".json")
    loaded = runner._load_cache(data_path, meta_path, expected_fingerprint=meta["encoder_config_fingerprint"])
    assert loaded is not None
    np.testing.assert_allclose(loaded.features, encoded.features)
    np.testing.assert_array_equal(loaded.labels, encoded.labels)
    assert loaded.subject_id == encoded.subject_id
    assert loaded.recording_id == encoded.recording_id

    bad_meta = dict(meta)
    bad_meta["encoder_config_fingerprint"] = "incompatible"
    meta_path.write_text(runner.json.dumps(bad_meta, indent=2, sort_keys=True, default=runner._json_default), encoding="utf-8")
    assert runner._load_cache(data_path, meta_path, expected_fingerprint=meta["encoder_config_fingerprint"]) is None


def test_classical_cache_resume_skips_reprocessing(tmp_path: Path, monkeypatch) -> None:
    settings = load_settings(Path("configs/default.yaml"))
    collection = make_collection("bandpower", subjects=("S1", "S2"), n_epochs=4)
    base_encoded = collection.items[0]
    encoded = EncodedDataset(
        features=base_encoded.features,
        labels=base_encoded.labels,
        metadata=RepresentationMetadata(
            representation="bandpower",
            channel_names=("Fpz-Cz",),
            sfreq=100.0,
            epoch_duration_sec=30.0,
            feature_shape=base_encoded.metadata.feature_shape,
            algorithm="welch",
            band_names=("delta", "theta", "alpha", "sigma", "beta"),
            extras=dict(base_encoded.metadata.extras),
        ),
        subject_id="SC400",
        recording_id="SC4001E0-PSG",
        onsets_sec=base_encoded.onsets_sec,
        ignore_index=-100,
    )

    monkeypatch.setattr(runner, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(runner, "RESULT_PATH", tmp_path / "out" / "result.json")

    psg = _dummy_psg()
    calls = {"n": 0}

    def fake_build(*args, **kwargs):
        calls["n"] += 1
        return encoded

    monkeypatch.setattr(runner, "_build_bandpower_dataset_for_recording", fake_build)

    # First call builds + caches.
    out1, status1 = runner.load_or_build_bandpower_cache(
        psg,
        settings=settings,
        encoder=None,
        vocabulary=None,
    )
    assert status1.loaded_from_cache is False
    assert calls["n"] == 1
    assert out1.recording_id == encoded.recording_id

    # Second call should reuse cache and not invoke the builder again.
    out2, status2 = runner.load_or_build_bandpower_cache(
        psg,
        settings=settings,
        encoder=None,
        vocabulary=None,
    )
    assert status2.loaded_from_cache is True
    assert calls["n"] == 1
    np.testing.assert_array_equal(out2.features, encoded.features)
