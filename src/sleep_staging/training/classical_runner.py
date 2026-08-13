"""Resumable classical baseline runner for Phase 4a.

BandPower features are cached per recording under ``outputs/classical_baseline/cache``
so interrupted runs can resume without recomputing already-encoded recordings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import dataclasses
import numpy as np

from sleep_staging import __version__ as PACKAGE_VERSION
from sleep_staging.acquisition.loader import discover_recordings, load_recording
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.config import load_settings
from sleep_staging.preprocessing.pipeline import preprocess_recording
from sleep_staging.representations.factory import build_encoder
from sleep_staging.representations.types import EncodedDataset, EncodedDatasetCollection, LabelVocabulary, RepresentationMetadata
from sleep_staging.training.classical import ClassicalBaselineResult, train_bandpower_logistic_regression
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.experiment import build_shared_subject_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classical_baseline"
CACHE_ROOT = OUTPUT_ROOT / "cache"
RESULT_PATH = OUTPUT_ROOT / "bandpower_logistic_regression_phase4a.json"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "phase4_outputs" / "manifest.json"
CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CacheStatus:
    path: Path
    loaded_from_cache: bool
    cache_file: Path | None = None


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _cache_stem(recording_id: str, subject_id: str) -> str:
    return f"{subject_id}__{recording_id}"


def _cache_stem_for_source(source_stem: str) -> str:
    return source_stem


def _cache_paths(recording_stem: str) -> tuple[Path, Path]:
    stem = _cache_stem_for_source(recording_stem)
    return CACHE_ROOT / f"{stem}.npz", CACHE_ROOT / f"{stem}.json"


def _fingerprint_metadata(meta: dict[str, object]) -> str:
    blob = json.dumps(meta, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_cache_metadata(
    *,
    settings,
    encoded: EncodedDataset,
    source_psg: Path,
    source_hypnogram: Path,
) -> dict[str, object]:
    bp = settings.encodings.bandpower
    prep = settings.preprocessing
    meta = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "package_version": PACKAGE_VERSION,
        "representation": encoded.representation,
        "encoding_method": "bandpower",
        "source_recording_stem": source_psg.stem,
        "source_recording_name": source_psg.name,
        "subject_id": encoded.subject_id,
        "recording_id": encoded.recording_id,
        "source_psg": str(source_psg),
        "source_hypnogram": str(source_hypnogram),
        "channel_names": list(encoded.channel_names),
        "sfreq": float(encoded.sfreq),
        "epoch_duration_sec": float(encoded.metadata.epoch_duration_sec),
        "feature_shape": list(encoded.metadata.feature_shape),
        "band_names": list(encoded.metadata.band_names or ()),
        "algorithm": encoded.metadata.algorithm,
        "bands_hz": list(encoded.metadata.extras.get("bands_hz", ())),
        "feature_names": list(encoded.metadata.extras.get("feature_names", ())),
        "include_log_absolute": bool(encoded.metadata.extras.get("include_log_absolute", True)),
        "include_relative": bool(encoded.metadata.extras.get("include_relative", True)),
        "include_ratios": bool(encoded.metadata.extras.get("include_ratios", False)),
        "eps": float(encoded.metadata.extras.get("eps", bp.eps)),
        "expected_sfreq": float(encoded.metadata.extras.get("expected_sfreq", bp.expected_sfreq)),
        "welch": dict(encoded.metadata.extras.get("welch", {})),
        "preprocessing": {
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
        },
        "bandpower_settings": {
            "method": bp.method,
            "include_log_absolute": bp.include_log_absolute,
            "include_relative": bp.include_relative,
            "include_ratios": bp.include_ratios,
            "eps": bp.eps,
            "expected_sfreq": bp.expected_sfreq,
            "bands": bp.bands,
            "welch": asdict(bp.welch),
            "ratios": bp.ratios,
        },
        "encoder_config_fingerprint": _fingerprint_metadata(
            {
                "package_version": PACKAGE_VERSION,
                "representation": encoded.representation,
                "encoding_method": "bandpower",
                "preprocessing": {
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
                },
                "bandpower": {
                    "method": bp.method,
                    "include_log_absolute": bp.include_log_absolute,
                    "include_relative": bp.include_relative,
                    "include_ratios": bp.include_ratios,
                    "eps": bp.eps,
                    "expected_sfreq": bp.expected_sfreq,
                    "bands": bp.bands,
                    "welch": asdict(bp.welch),
                    "ratios": bp.ratios,
                },
            }
        ),
    }
    return meta


def _save_cache(encoded: EncodedDataset, metadata: dict[str, object]) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    source_stem = str(metadata.get("source_recording_stem") or encoded.recording_id)
    data_path, meta_path = _cache_paths(source_stem)
    np.savez_compressed(
        data_path,
        features=np.asarray(encoded.features),
        labels=np.asarray(encoded.labels),
        onsets_sec=np.asarray(encoded.onsets_sec) if encoded.onsets_sec is not None else np.array([], dtype=np.float64),
    )
    metadata = dict(metadata)
    metadata["cache_data_file"] = data_path.name
    metadata["cache_metadata_file"] = meta_path.name
    metadata["saved_at"] = "cache-only"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    return data_path


def _load_cache(data_path: Path, meta_path: Path, *, expected_fingerprint: str) -> EncodedDataset | None:
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("cache_format_version") != CACHE_FORMAT_VERSION:
        return None
    if meta.get("encoder_config_fingerprint") != expected_fingerprint:
        return None
    if meta.get("representation") != "bandpower":
        return None
    if meta.get("source_recording_stem") != data_path.stem:
        return None
    if meta.get("source_recording_name") != f"{data_path.stem}.edf":
        return None
    try:
        with np.load(data_path, allow_pickle=False) as npz:
            features = np.asarray(npz["features"])
            labels = np.asarray(npz["labels"], dtype=np.int64)
            onsets = np.asarray(npz["onsets_sec"], dtype=np.float64)
    except Exception:
        return None
    try:
        metadata = RepresentationMetadata(
            representation="bandpower",
            channel_names=tuple(meta["channel_names"]),
            sfreq=float(meta["sfreq"]),
            epoch_duration_sec=float(meta["epoch_duration_sec"]),
            feature_shape=tuple(int(x) for x in meta["feature_shape"]),
            algorithm=str(meta.get("algorithm") or "welch"),
            band_names=tuple(meta.get("band_names") or ()),
            extras={
                "layout": "NCF",
                "feature_names": tuple(meta.get("feature_names") or ()),
                "include_log_absolute": bool(meta.get("include_log_absolute", True)),
                "include_relative": bool(meta.get("include_relative", True)),
                "include_ratios": bool(meta.get("include_ratios", False)),
                "eps": float(meta.get("eps", 1e-10)),
                "expected_sfreq": float(meta.get("expected_sfreq", 100.0)),
                "bands_hz": tuple(tuple(x) for x in meta.get("bands_hz", ())),
                "welch": dict(meta.get("welch", {})),
            },
        )
        onsets_arg = onsets if onsets.size else None
        return EncodedDataset(
            features=features,
            labels=labels,
            metadata=metadata,
            subject_id=str(meta["subject_id"]),
            recording_id=str(meta["recording_id"]),
            onsets_sec=onsets_arg,
            ignore_index=-100,
        )
    except Exception:
        return None


def _build_bandpower_dataset_for_recording(
    psg_file: Path,
    *,
    settings,
    encoder,
    vocabulary: LabelVocabulary,
) -> EncodedDataset:
    recording = load_recording(psg_file, settings=settings.acquisition, preload=settings.acquisition.preload)
    preprocessed = preprocess_recording(recording, settings.preprocessing, copy=False)
    return encoder(preprocessed, vocabulary=vocabulary)


def load_or_build_bandpower_cache(
    psg_file: Path,
    *,
    settings,
    encoder,
    vocabulary: LabelVocabulary,
) -> tuple[EncodedDataset, CacheStatus]:
    ids = parse_psg_filename(psg_file)
    data_path, meta_path = _cache_paths(psg_file.stem)
    expected_fingerprint = _fingerprint_metadata(
        {
            "package_version": PACKAGE_VERSION,
            "representation": "bandpower",
            "encoding_method": "bandpower",
            "preprocessing": {
                "epoch_duration_sec": float(settings.preprocessing.epoch_duration_sec),
                "min_remainder_sec": float(settings.preprocessing.min_remainder_sec),
                "wake_crop": asdict(settings.preprocessing.wake_crop),
                "stage_map": asdict(settings.preprocessing.stage_map),
                "channels": asdict(settings.preprocessing.channels),
                "filter": asdict(settings.preprocessing.filter),
                "normalize": asdict(settings.preprocessing.normalize),
                "bad_channel": asdict(settings.preprocessing.bad_channel),
                "reference": asdict(settings.preprocessing.reference),
                "ica": asdict(settings.preprocessing.ica),
                "amplitude_reject": asdict(settings.preprocessing.amplitude_reject),
            },
            "bandpower": {
                "method": settings.encodings.bandpower.method,
                "include_log_absolute": settings.encodings.bandpower.include_log_absolute,
                "include_relative": settings.encodings.bandpower.include_relative,
                "include_ratios": settings.encodings.bandpower.include_ratios,
                "eps": settings.encodings.bandpower.eps,
                "expected_sfreq": settings.encodings.bandpower.expected_sfreq,
                "bands": settings.encodings.bandpower.bands,
                "welch": asdict(settings.encodings.bandpower.welch),
                "ratios": settings.encodings.bandpower.ratios,
            },
        }
    )
    cached = _load_cache(data_path, meta_path, expected_fingerprint=expected_fingerprint)
    if cached is not None:
        return cached, CacheStatus(path=psg_file, loaded_from_cache=True, cache_file=data_path)

    encoded = _build_bandpower_dataset_for_recording(psg_file, settings=settings, encoder=encoder, vocabulary=vocabulary)
    metadata = _build_cache_metadata(settings=settings, encoded=encoded, source_psg=psg_file, source_hypnogram=psg_file.with_name(psg_file.name.replace("-PSG.edf", "-Hypnogram.edf")))
    _save_cache(encoded, metadata)
    return encoded, CacheStatus(path=psg_file, loaded_from_cache=False, cache_file=data_path)


def _load_phase4_test_subjects() -> tuple[str, ...]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return tuple(data["test_subjects"])


def _subject_split_for_bandpower(collection: EncodedDatasetCollection):
    split = build_shared_subject_split(collection, ratios=(0.7, 0.15, 0.15), seed=42)
    expected_test = _load_phase4_test_subjects()
    if tuple(split.test) != expected_test:
        raise AssertionError(
            f"BandPower split mismatch. Expected test subjects {expected_test}, got {split.test}"
        )
    return split


def _save_final_result(
    *,
    split,
    result: ClassicalBaselineResult,
    settings,
    n_cached: int,
    n_new: int,
) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "BandPower + LogisticRegression",
        "algorithm": "LogisticRegression",
        "class_weight": "balanced",
        "seed": settings.experiment.train.seed,
        "split": {
            "cohort": settings.experiment.split.cohort,
            "seed": settings.experiment.split.seed,
            "train_ratio": settings.experiment.split.train_ratio,
            "val_ratio": settings.experiment.split.val_ratio,
            "test_ratio": settings.experiment.split.test_ratio,
            "train_subjects": list(split.train),
            "val_subjects": list(split.val),
            "test_subjects": list(split.test),
        },
        "cache": {
            "cache_root": str(CACHE_ROOT),
            "n_cached_recordings": n_cached,
            "n_new_recordings": n_new,
            "cache_format_version": CACHE_FORMAT_VERSION,
        },
        "validation_metrics": result.validation_metrics.as_dict(),
        "metrics": result.test_metrics.as_dict() if result.test_metrics is not None else None,
        "n_predictions": len(result.predictions),
        "predictions": [asdict(p) for p in result.predictions],
        "package_version": PACKAGE_VERSION,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return RESULT_PATH


def run_classical_baseline() -> Path:
    settings = load_settings(DEFAULT_CONFIG)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    psg_files = discover_recordings(settings.acquisition.data_root)
    sc_files = [p for p in psg_files if parse_psg_filename(p).study == "SC"]
    print(f"Found {len(sc_files)} SC recordings")

    enc_settings = dataclasses.replace(settings.encodings, representation="bandpower")
    encoder = build_encoder(enc_settings)
    vocabulary = LabelVocabulary(ignore_label=settings.encodings.ignore_label, ignore_index=settings.encodings.ignore_index)

    datasets: list[EncodedDataset] = []
    cached = 0
    new = 0
    for idx, psg_file in enumerate(sc_files, 1):
        data_path, meta_path = _cache_paths(psg_file.stem)
        if data_path.exists() and meta_path.exists():
            fingerprint = _fingerprint_metadata(
                {
                    "package_version": PACKAGE_VERSION,
                    "representation": "bandpower",
                    "encoding_method": "bandpower",
                    "preprocessing": {
                        "epoch_duration_sec": float(settings.preprocessing.epoch_duration_sec),
                        "min_remainder_sec": float(settings.preprocessing.min_remainder_sec),
                        "wake_crop": asdict(settings.preprocessing.wake_crop),
                        "stage_map": asdict(settings.preprocessing.stage_map),
                        "channels": asdict(settings.preprocessing.channels),
                        "filter": asdict(settings.preprocessing.filter),
                        "normalize": asdict(settings.preprocessing.normalize),
                        "bad_channel": asdict(settings.preprocessing.bad_channel),
                        "reference": asdict(settings.preprocessing.reference),
                        "ica": asdict(settings.preprocessing.ica),
                        "amplitude_reject": asdict(settings.preprocessing.amplitude_reject),
                    },
                    "bandpower": {
                        "method": settings.encodings.bandpower.method,
                        "include_log_absolute": settings.encodings.bandpower.include_log_absolute,
                        "include_relative": settings.encodings.bandpower.include_relative,
                        "include_ratios": settings.encodings.bandpower.include_ratios,
                        "eps": settings.encodings.bandpower.eps,
                        "expected_sfreq": settings.encodings.bandpower.expected_sfreq,
                        "bands": settings.encodings.bandpower.bands,
                        "welch": asdict(settings.encodings.bandpower.welch),
                        "ratios": settings.encodings.bandpower.ratios,
                    },
                }
            )
            encoded = _load_cache(data_path, meta_path, expected_fingerprint=fingerprint)
            if encoded is not None:
                cached += 1
                datasets.append(encoded)
                print(f"[{idx}/{len(sc_files)}] CACHE HIT: {psg_file.stem}")
                continue

        print(f"[{idx}/{len(sc_files)}] PROCESS: {psg_file.stem}")
        encoded, _status = load_or_build_bandpower_cache(
            psg_file,
            settings=settings,
            encoder=encoder,
            vocabulary=vocabulary,
        )
        datasets.append(encoded)
        new += 1
        print(f"[{idx}/{len(sc_files)}] CACHE WRITTEN: {psg_file.stem}")

    collection = EncodedDatasetCollection(tuple(datasets))
    split = _subject_split_for_bandpower(collection)

    train_ds = EpochDataset(collection, subject_ids=split.train, drop_ignore=False)
    val_ds = EpochDataset(collection, subject_ids=split.val, drop_ignore=False)
    test_ds = EpochDataset(collection, subject_ids=split.test, drop_ignore=False)

    result = train_bandpower_logistic_regression(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        ignore_index=settings.experiment.train.ignore_index,
        max_iter=settings.experiment.classical_baseline.max_iter,
        evaluate_test=True,
        random_state=settings.experiment.train.seed,
    )

    save_path = _save_final_result(split=split, result=result, settings=settings, n_cached=cached, n_new=new)
    print(f"WROTE {save_path}")
    return save_path


if __name__ == "__main__":
    run_classical_baseline()
