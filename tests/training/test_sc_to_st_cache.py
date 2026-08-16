from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from tests.phase4_helpers import make_collection
from sleep_staging.config import load_settings
from sleep_staging.training import sc_to_st_cache as cache


def _settings():
    return load_settings(Path("configs/default.yaml"))


def _fake_encoded(representation: str):
    collection = make_collection(representation, subjects=("SC400",), n_epochs=4)
    return collection.items[0]


def _install_fake_builder(monkeypatch, encoded, calls):
    def fake_build(psg_path, *, settings, encoder, vocabulary):
        calls["n"] += 1
        return encoded

    monkeypatch.setattr(cache, "_preprocess_and_encode_one", fake_build)


# ---------------------------------------------------------------------------
# 1. cache miss -> preprocess/encode -> cache write
# ---------------------------------------------------------------------------
def test_cache_miss_preprocesses_and_writes_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    settings = _settings()
    encoded = _fake_encoded("bandpower")
    calls = {"n": 0}
    _install_fake_builder(monkeypatch, encoded, calls)

    psg = Path("SC4001E0-PSG.edf")
    result, status = cache.load_or_build_encoded_dataset(
        psg, settings=settings, representation="bandpower", encoder=None, vocabulary=None, verbose=False
    )

    assert status.loaded_from_cache is False
    assert calls["n"] == 1
    data_path, meta_path = cache.cache_paths("bandpower", psg.stem)
    assert data_path.exists()
    assert meta_path.exists()
    np.testing.assert_allclose(result.features, encoded.features)


# ---------------------------------------------------------------------------
# 2. cache hit -> load without preprocessing
# ---------------------------------------------------------------------------
def test_cache_hit_loads_without_reprocessing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    settings = _settings()
    encoded = _fake_encoded("bandpower")
    calls = {"n": 0}
    _install_fake_builder(monkeypatch, encoded, calls)

    psg = Path("SC4001E0-PSG.edf")
    cache.load_or_build_encoded_dataset(
        psg, settings=settings, representation="bandpower", encoder=None, vocabulary=None, verbose=False
    )
    assert calls["n"] == 1

    result2, status2 = cache.load_or_build_encoded_dataset(
        psg, settings=settings, representation="bandpower", encoder=None, vocabulary=None, verbose=False
    )
    assert status2.loaded_from_cache is True
    assert calls["n"] == 1  # builder not invoked again
    np.testing.assert_array_equal(result2.labels, encoded.labels)
    np.testing.assert_allclose(result2.features, encoded.features)


# ---------------------------------------------------------------------------
# 3. stale/incompatible cache rejection
# ---------------------------------------------------------------------------
def test_stale_cache_is_rejected_and_rebuilt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    settings = _settings()
    encoded = _fake_encoded("bandpower")
    calls = {"n": 0}
    _install_fake_builder(monkeypatch, encoded, calls)

    psg = Path("SC4001E0-PSG.edf")
    cache.load_or_build_encoded_dataset(
        psg, settings=settings, representation="bandpower", encoder=None, vocabulary=None, verbose=False
    )
    assert calls["n"] == 1

    # Simulate a changed preprocessing config (different fingerprint).
    changed_settings = dataclasses.replace(
        settings,
        preprocessing=dataclasses.replace(
            settings.preprocessing,
            filter=dataclasses.replace(settings.preprocessing.filter, eeg_h_freq=25.0),
        ),
    )
    _, status = cache.load_or_build_encoded_dataset(
        psg, settings=changed_settings, representation="bandpower", encoder=None, vocabulary=None, verbose=False
    )
    assert status.loaded_from_cache is False
    assert calls["n"] == 2  # rebuilt because the cached fingerprint no longer matches


def test_load_cache_rejects_wrong_format_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    settings = _settings()
    encoded = _fake_encoded("raw")
    metadata = cache.build_cache_metadata(
        settings=settings,
        encoded=encoded,
        source_psg=Path("SC4001E0-PSG.edf"),
        cohort="SC",
        representation="raw",
    )
    data_path, meta_path = cache.save_cache(encoded, metadata)

    fingerprint = metadata["encoder_fingerprint"]
    good = cache.load_cache(
        data_path, meta_path, expected_fingerprint=fingerprint, representation="raw", cohort="SC", source_stem="SC4001E0-PSG"
    )
    assert good is not None

    import json

    bad_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bad_meta["cache_format_version"] = 999
    meta_path.write_text(json.dumps(bad_meta), encoding="utf-8")
    assert (
        cache.load_cache(
            data_path, meta_path, expected_fingerprint=fingerprint, representation="raw", cohort="SC", source_stem="SC4001E0-PSG"
        )
        is None
    )


# ---------------------------------------------------------------------------
# 4. unique cache keys across different PSG stems
# ---------------------------------------------------------------------------
def test_cache_keys_unique_across_psg_stems(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    data_a, meta_a = cache.cache_paths("raw", "SC4001E0-PSG")
    data_b, meta_b = cache.cache_paths("raw", "SC4001E1-PSG")
    data_c, meta_c = cache.cache_paths("raw", "ST7011J0-PSG")

    assert len({data_a, data_b, data_c}) == 3
    assert len({meta_a, meta_b, meta_c}) == 3
    assert data_a.name == "SC4001E0-PSG.npz"
    assert data_c.name == "ST7011J0-PSG.npz"


# ---------------------------------------------------------------------------
# 5. SC/ST cache separation
# ---------------------------------------------------------------------------
def test_sc_and_st_cache_entries_are_separated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    settings = _settings()
    sc_encoded = _fake_encoded("bandpower")
    st_encoded = _fake_encoded("bandpower")
    calls = {"n": 0}

    def fake_build(psg_path, *, settings, encoder, vocabulary):
        calls["n"] += 1
        return sc_encoded if "SC" in psg_path.stem else st_encoded

    monkeypatch.setattr(cache, "_preprocess_and_encode_one", fake_build)

    sc_psg = Path("SC4001E0-PSG.edf")
    st_psg = Path("ST7011J0-PSG.edf")

    _, sc_status = cache.load_or_build_encoded_dataset(
        sc_psg, settings=settings, representation="bandpower", encoder=None, vocabulary=None, verbose=False
    )
    _, st_status = cache.load_or_build_encoded_dataset(
        st_psg, settings=settings, representation="bandpower", encoder=None, vocabulary=None, verbose=False
    )

    assert sc_status.cohort == "SC"
    assert st_status.cohort == "ST"
    assert sc_status.data_path != st_status.data_path
    assert calls["n"] == 2

    # An ST cache entry must never satisfy a lookup for the "same stem" under
    # a different claimed cohort (cohort is part of the validity check).
    fingerprint = cache.encoder_fingerprint(settings, "bandpower")
    mismatched = cache.load_cache(
        st_status.data_path,
        st_status.meta_path,
        expected_fingerprint=fingerprint,
        representation="bandpower",
        cohort="SC",
        source_stem=st_psg.stem,
    )
    assert mismatched is None


# ---------------------------------------------------------------------------
# 6. Raw/BandPower/STFT cache loading
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("representation", ["raw", "bandpower", "time_frequency"])
def test_cache_roundtrip_for_each_representation(tmp_path, monkeypatch, representation) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    settings = _settings()
    encoded = _fake_encoded(representation)
    calls = {"n": 0}
    _install_fake_builder(monkeypatch, encoded, calls)

    psg = Path("SC4001E0-PSG.edf")
    first, status1 = cache.load_or_build_encoded_dataset(
        psg, settings=settings, representation=representation, encoder=None, vocabulary=None, verbose=False
    )
    second, status2 = cache.load_or_build_encoded_dataset(
        psg, settings=settings, representation=representation, encoder=None, vocabulary=None, verbose=False
    )

    assert status1.loaded_from_cache is False
    assert status2.loaded_from_cache is True
    assert calls["n"] == 1
    assert second.metadata.representation == representation
    assert tuple(second.features.shape[1:]) == encoded.metadata.feature_shape
    np.testing.assert_allclose(second.features, encoded.features)


# ---------------------------------------------------------------------------
# Representation directory layout
# ---------------------------------------------------------------------------
def test_cache_directory_layout_matches_spec(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    for rep in ("raw", "bandpower", "time_frequency"):
        data_path, meta_path = cache.cache_paths(rep, "SC4001E0-PSG")
        assert data_path.parent == tmp_path / "sc_to_st" / rep
        assert meta_path.parent == tmp_path / "sc_to_st" / rep


# ---------------------------------------------------------------------------
# 7. Existing time_frequency caches remain a CACHE HIT after the STFT
#    chunking fix (backends.py). The chunk_size knob lives only on
#    STFTBackend and is never part of settings.encodings.time_frequency, so
#    it must not appear in encoder_fingerprint() and must not affect an
#    entry written by a genuine (real, non-mocked) encode pass.
# ---------------------------------------------------------------------------
def _real_time_frequency_dataset(settings, *, n_epochs: int = 4):
    from sleep_staging.representations import EpochTensorBatch, LabelVocabulary, build_encoder

    enc_settings = dataclasses.replace(settings.encodings, representation="time_frequency")
    encoder = build_encoder(enc_settings)

    sfreq = 100.0
    duration_sec = 30.0
    n_times = int(round(duration_sec * sfreq))
    t = np.arange(n_times, dtype=np.float64) / sfreq
    signals = np.stack(
        [np.sin(2.0 * np.pi * 10.0 * t), np.sin(2.0 * np.pi * 2.0 * t)]
    )[None, :, :].repeat(n_epochs, axis=0)

    batch = EpochTensorBatch(
        signals=signals,
        labels=(np.arange(n_epochs, dtype=np.int64) % 5),
        onsets_sec=np.arange(n_epochs, dtype=np.float64) * duration_sec,
        channel_names=("Fpz-Cz", "Pz-Oz"),
        sfreq=sfreq,
        epoch_duration_sec=duration_sec,
        subject_id="SC4001",
        recording_id="E0",
        ignore_index=-100,
    )
    vocabulary = LabelVocabulary(
        ignore_label=settings.encodings.ignore_label,
        ignore_index=settings.encodings.ignore_index,
    )
    return encoder(batch, vocabulary=vocabulary)


def test_existing_time_frequency_cache_is_still_a_cache_hit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "sc_to_st")
    settings = _settings()

    encoded = _real_time_frequency_dataset(settings)
    psg = Path("SC4001E0-PSG.edf")
    metadata = cache.build_cache_metadata(
        settings=settings,
        encoded=encoded,
        source_psg=psg,
        cohort="SC",
        representation="time_frequency",
    )
    cache.save_cache(encoded, metadata)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cache hit expected; builder must not run")

    monkeypatch.setattr(cache, "_preprocess_and_encode_one", fail_if_called)

    result, status = cache.load_or_build_encoded_dataset(
        psg,
        settings=settings,
        representation="time_frequency",
        encoder=None,
        vocabulary=None,
        verbose=False,
    )

    assert status.loaded_from_cache is True
    assert result.metadata.representation == "time_frequency"
    assert result.metadata.algorithm == "stft"
    np.testing.assert_allclose(result.features, encoded.features)
