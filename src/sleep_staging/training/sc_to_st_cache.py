"""Per-recording cache for the primary SC→ST PyTorch orchestration.

Mirrors the resumability guarantees ``classical_runner.py`` already has for
the BandPower + LogisticRegression baseline, but generalizes them to all
representations used by the SC→ST experiment (``raw``, ``bandpower``,
``time_frequency``) so every model path (Raw CNN, BandPower MLP, STFT CNN,
BandPower→LogisticRegression) can resume from cached per-recording tensors
instead of re-reading and re-preprocessing EDF files.

Cache layout
------------
``<CACHE_ROOT>/<representation>/<PSG-stem>.npz``   (features + labels + onsets)
``<CACHE_ROOT>/<representation>/<PSG-stem>.json``  (validity/fingerprint sidecar)
``<CACHE_ROOT>/<representation>/<PSG-stem>.features.npy``  (memory-mapped features)

``CACHE_ROOT`` defaults to ``D:/SleepStagingCache/sc_to_st`` and is overridable
via the ``SLEEP_CACHE_ROOT`` environment variable. The cache is deliberately
kept off the repository tree: at full scale it is several GB per
representation, which does not belong in version control.

The full PSG recording stem (e.g. ``SC4001E0-PSG`` / ``ST7011J0-PSG``) is the
cache key. This is intentionally *not* the legacy ``subject__recording``-style
key used elsewhere, and this cache tree is intentionally separate from the
``outputs/classical_baseline/cache`` SC-only legacy cache: SC→ST caches SC
*and* ST recordings, keyed uniquely by full PSG stem.

Every cache entry stores a metadata sidecar with enough information to reject
stale or incompatible entries: cache format version, package version, the
recording's cohort (SC/ST), representation, a preprocessing fingerprint, and
an encoder/representation fingerprint (which itself folds in the
preprocessing fingerprint, so any upstream preprocessing change invalidates
downstream representation caches too).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sleep_staging import __version__ as PACKAGE_VERSION
from sleep_staging.acquisition.loader import load_recording
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.preprocessing.pipeline import preprocess_recording
from sleep_staging.representations.factory import build_encoder
from sleep_staging.representations.types import (
    EncodedDataset,
    EncodedDatasetCollection,
    LabelVocabulary,
    RepresentationMetadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default root of the SC→ST cache tree, overridable via ``SLEEP_CACHE_ROOT``.
#: Deliberately outside the repo (multi-GB) and separate from
#: ``outputs/classical_baseline/cache`` (legacy SC-only classical cache).
DEFAULT_CACHE_ROOT = Path(r"D:\SleepStagingCache\sc_to_st")
CACHE_ROOT = Path(os.environ.get("SLEEP_CACHE_ROOT", DEFAULT_CACHE_ROOT))

CACHE_FORMAT_VERSION = 1

REPRESENTATIONS: tuple[str, ...] = ("raw", "bandpower", "time_frequency")

#: Cache subdirectory per representation:
#: <CACHE_ROOT>/{raw,bandpower,time_frequency}/
REPRESENTATION_CACHE_DIRS: dict[str, str] = {
    "raw": "raw",
    "bandpower": "bandpower",
    "time_frequency": "time_frequency",
}


@dataclasses.dataclass(frozen=True, slots=True)
class CacheStatus:
    """Outcome of a cache lookup/build for one recording."""

    stem: str
    cohort: str
    representation: str
    loaded_from_cache: bool
    data_path: Path
    meta_path: Path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _fingerprint(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def representation_cache_dir(representation: str) -> Path:
    if representation not in REPRESENTATION_CACHE_DIRS:
        raise ValueError(f"Unknown representation: {representation!r}")
    return CACHE_ROOT / REPRESENTATION_CACHE_DIRS[representation]


def cache_paths(representation: str, recording_stem: str) -> tuple[Path, Path]:
    """Return ``(data_path, meta_path)`` for a cache key (full PSG stem)."""
    cache_dir = representation_cache_dir(representation)
    return cache_dir / f"{recording_stem}.npz", cache_dir / f"{recording_stem}.json"


def cohort_for_recording(psg_path: Path) -> str:
    """SC or ST, derived from the Sleep-EDF filename convention."""
    return parse_psg_filename(psg_path).study


def _preprocessing_identity(settings) -> dict[str, object]:
    prep = settings.preprocessing
    return {
        "epoch_duration_sec": float(prep.epoch_duration_sec),
        "min_remainder_sec": float(prep.min_remainder_sec),
        "wake_crop": asdict(prep.wake_crop),
        "stage_map": asdict(prep.stage_map),
        "channels": asdict(prep.channels),
        "filter": asdict(prep.filter),
        "normalize": asdict(prep.normalize),
        "bad_channel": asdict(prep.bad_channel),
        "reference": asdict(prep.reference),
        "ica": asdict(prep.ica),
        "amplitude_reject": asdict(prep.amplitude_reject),
    }


def _encoding_identity(settings, representation: str) -> dict[str, object]:
    encodings = settings.encodings
    if representation == "raw":
        return {"raw": asdict(encodings.raw)}
    if representation == "bandpower":
        return {"bandpower": asdict(encodings.bandpower)}
    if representation == "time_frequency":
        return {"time_frequency": asdict(encodings.time_frequency)}
    raise ValueError(f"Unknown representation: {representation!r}")


def preprocessing_fingerprint(settings) -> str:
    """Fingerprint of preprocessing config only (shared across representations)."""
    return _fingerprint(
        {
            "package_version": PACKAGE_VERSION,
            "preprocessing": _preprocessing_identity(settings),
        }
    )


def encoder_fingerprint(settings, representation: str) -> str:
    """Fingerprint of preprocessing + representation-specific encoding config.

    Folding the preprocessing fingerprint in here means a change to either
    preprocessing *or* encoding settings invalidates the cache; downstream
    code only needs to compare this single value.
    """
    return _fingerprint(
        {
            "package_version": PACKAGE_VERSION,
            "representation": representation,
            "preprocessing": _preprocessing_identity(settings),
            "encoding": _encoding_identity(settings, representation),
        }
    )


def _config_identity(settings, representation: str) -> dict[str, object]:
    """Human-inspectable configuration identity, stored (not just hashed)."""
    return {
        "preprocessing": _preprocessing_identity(settings),
        "encoding": _encoding_identity(settings, representation),
    }


def build_cache_metadata(
    *,
    settings,
    encoded: EncodedDataset,
    source_psg: Path,
    cohort: str,
    representation: str,
) -> dict[str, object]:
    metadata = encoded.metadata
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "package_version": PACKAGE_VERSION,
        "representation": representation,
        "cohort": cohort,
        "source_recording_stem": source_psg.stem,
        "source_recording_name": source_psg.name,
        "source_psg": str(source_psg),
        "subject_id": encoded.subject_id,
        "recording_id": encoded.recording_id,
        "preprocessing_fingerprint": preprocessing_fingerprint(settings),
        "encoder_fingerprint": encoder_fingerprint(settings, representation),
        "config_identity": _config_identity(settings, representation),
        "channel_names": list(metadata.channel_names),
        "sfreq": float(metadata.sfreq),
        "epoch_duration_sec": float(metadata.epoch_duration_sec),
        "feature_shape": list(metadata.feature_shape),
        "algorithm": metadata.algorithm,
        "freqs_hz": None if metadata.freqs_hz is None else list(metadata.freqs_hz),
        "times_sec": None if metadata.times_sec is None else list(metadata.times_sec),
        "band_names": None if metadata.band_names is None else list(metadata.band_names),
        "extras": dict(metadata.extras),
        "ignore_index": int(encoded.ignore_index),
    }
    return payload


def save_cache(encoded: EncodedDataset, metadata: dict[str, object]) -> tuple[Path, Path]:
    representation = str(metadata["representation"])
    stem = str(metadata["source_recording_stem"])
    data_path, meta_path = cache_paths(representation, stem)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        features=np.asarray(encoded.features),
        labels=np.asarray(encoded.labels),
        onsets_sec=(
            np.asarray(encoded.onsets_sec)
            if encoded.onsets_sec is not None
            else np.array([], dtype=np.float64)
        ),
    )
    metadata = dict(metadata)
    metadata["cache_data_file"] = data_path.name
    metadata["cache_metadata_file"] = meta_path.name
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    return data_path, meta_path


def _metadata_from_payload(payload: dict[str, object]) -> RepresentationMetadata:
    extras = dict(payload.get("extras") or {})
    freqs_hz = payload.get("freqs_hz")
    times_sec = payload.get("times_sec")
    band_names = payload.get("band_names")
    return RepresentationMetadata(
        representation=payload["representation"],  # type: ignore[arg-type]
        channel_names=tuple(payload["channel_names"]),
        sfreq=float(payload["sfreq"]),
        epoch_duration_sec=float(payload["epoch_duration_sec"]),
        feature_shape=tuple(int(x) for x in payload["feature_shape"]),
        algorithm=payload.get("algorithm"),
        freqs_hz=None if freqs_hz is None else tuple(float(x) for x in freqs_hz),
        times_sec=None if times_sec is None else tuple(float(x) for x in times_sec),
        band_names=None if band_names is None else tuple(band_names),
        extras=extras,
    )


def _load_features_memmapped(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one cache entry's arrays, preferring a memory-mapped ``features`` read.

    ``save_cache`` writes ``features`` into a compressed ``.npz``. Reading it
    normally (``np.load(...)['features']``) forces a full decompression into
    RAM, and ``encode_recordings_with_cache`` accumulates one
    ``EncodedDataset`` per recording into an in-memory list *before* any
    training starts -- for the full 197-recording SC+ST ``raw`` dataset that
    is ~8 GB resident, regardless of how many representations are loaded at
    once (measured; not the same problem sequential per-representation
    processing already fixed).

    To avoid that, an uncompressed ``<stem>.features.npy`` sidecar is
    written next to the ``.npz`` on first load, then reopened immediately
    with ``mmap_mode='r'`` -- so even the very first pass over an
    unconverted cache stays low-memory, not just subsequent ones. Every load
    thereafter goes straight to ``mmap_mode='r'`` on the sidecar. The OS
    pages epoch data in from disk on demand instead of the process
    committing the whole array to RSS. Cached *values* are byte-identical
    either way -- this only changes how they reach memory, not what they
    contain, so it does not touch cache fingerprints, preprocessing, or any
    scientific content.
    """
    sidecar_path = data_path.with_name(data_path.stem + ".features.npy")
    if sidecar_path.exists():
        features = np.load(sidecar_path, mmap_mode="r")
        with np.load(data_path, allow_pickle=False) as npz:
            labels = np.asarray(npz["labels"], dtype=np.int64)
            onsets = np.asarray(npz["onsets_sec"], dtype=np.float64)
        return features, labels, onsets

    with np.load(data_path, allow_pickle=False) as npz:
        raw_features = np.asarray(npz["features"])
        labels = np.asarray(npz["labels"], dtype=np.int64)
        onsets = np.asarray(npz["onsets_sec"], dtype=np.float64)

    try:
        tmp_path = sidecar_path.with_name(sidecar_path.name + ".tmp")
        with open(tmp_path, "wb") as fh:
            np.save(fh, raw_features)
        tmp_path.replace(sidecar_path)
        features = np.load(sidecar_path, mmap_mode="r")
    except OSError:
        # Best-effort: fall back to the in-RAM array for this call only.
        features = raw_features
    return features, labels, onsets


def load_cache(
    data_path: Path,
    meta_path: Path,
    *,
    expected_fingerprint: str,
    representation: str,
    cohort: str,
    source_stem: str,
) -> EncodedDataset | None:
    """Load a cache entry, returning ``None`` for any missing/stale/mismatched entry."""
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Reject stale/incompatible cache entries.
    if meta.get("cache_format_version") != CACHE_FORMAT_VERSION:
        return None
    if meta.get("encoder_fingerprint") != expected_fingerprint:
        return None
    if meta.get("representation") != representation:
        return None
    if meta.get("cohort") != cohort:
        return None
    if meta.get("source_recording_stem") != source_stem:
        return None

    try:
        features, labels, onsets = _load_features_memmapped(data_path)
    except Exception:
        return None

    try:
        metadata = _metadata_from_payload(meta)
        onsets_arg = onsets if onsets.size else None
        return EncodedDataset(
            features=features,
            labels=labels,
            metadata=metadata,
            subject_id=str(meta["subject_id"]),
            recording_id=str(meta["recording_id"]),
            onsets_sec=onsets_arg,
            ignore_index=int(meta.get("ignore_index", -100)),
        )
    except Exception:
        return None


def _preprocess_and_encode_one(
    psg_path: Path,
    *,
    settings,
    encoder,
    vocabulary: LabelVocabulary,
) -> EncodedDataset:
    """Run the existing production preprocessing + encoding pipeline once.

    Intentionally the *only* place that reads/preprocesses/re-encodes an EDF
    for the SC→ST cache path, so preprocessing logic is never duplicated.
    """
    recording = load_recording(psg_path, settings=settings.acquisition, preload=settings.acquisition.preload)
    preprocessed = preprocess_recording(recording, settings.preprocessing, copy=False)
    return encoder(preprocessed, vocabulary=vocabulary)


def load_or_build_encoded_dataset(
    psg_path: Path,
    *,
    settings,
    representation: str,
    encoder,
    vocabulary: LabelVocabulary,
    verbose: bool = True,
) -> tuple[EncodedDataset, CacheStatus]:
    """Cache-first per-recording encode.

    1. Check for a valid cache entry.
    2. If valid, load it (no EDF read/preprocess/encode) and report a hit.
    3. Otherwise preprocess + encode via the production pipeline, persist the
       cache immediately, and report a write.
    """
    cohort = cohort_for_recording(psg_path)
    stem = psg_path.stem
    data_path, meta_path = cache_paths(representation, stem)
    fingerprint = encoder_fingerprint(settings, representation)

    cached = load_cache(
        data_path,
        meta_path,
        expected_fingerprint=fingerprint,
        representation=representation,
        cohort=cohort,
        source_stem=stem,
    )
    if cached is not None:
        if verbose:
            print(f"CACHE HIT: [{cohort}/{representation}] {stem}")
        return cached, CacheStatus(
            stem=stem,
            cohort=cohort,
            representation=representation,
            loaded_from_cache=True,
            data_path=data_path,
            meta_path=meta_path,
        )

    encoded = _preprocess_and_encode_one(psg_path, settings=settings, encoder=encoder, vocabulary=vocabulary)
    metadata = build_cache_metadata(
        settings=settings,
        encoded=encoded,
        source_psg=psg_path,
        cohort=cohort,
        representation=representation,
    )
    save_cache(encoded, metadata)
    if verbose:
        print(f"CACHE WRITTEN: [{cohort}/{representation}] {stem}")
    return encoded, CacheStatus(
        stem=stem,
        cohort=cohort,
        representation=representation,
        loaded_from_cache=False,
        data_path=data_path,
        meta_path=meta_path,
    )


def encode_recordings_with_cache(
    recording_paths,
    *,
    settings,
    representation: str,
    verbose: bool = True,
) -> EncodedDatasetCollection:
    """Encode many recordings for one representation, cache-first per recording.

    Works independently of which downstream model consumes the resulting
    collection: once a representation's cache exists on disk, any model that
    trains on that representation reuses it.
    """
    enc_settings = dataclasses.replace(settings.encodings, representation=representation)
    encoder = build_encoder(enc_settings)
    vocabulary = LabelVocabulary(ignore_label=settings.encodings.ignore_label, ignore_index=settings.encodings.ignore_index)

    datasets: list[EncodedDataset] = []
    for path in recording_paths:
        encoded, _status = load_or_build_encoded_dataset(
            path,
            settings=settings,
            representation=representation,
            encoder=encoder,
            vocabulary=vocabulary,
            verbose=verbose,
        )
        datasets.append(encoded)
    return EncodedDatasetCollection(items=tuple(datasets))
