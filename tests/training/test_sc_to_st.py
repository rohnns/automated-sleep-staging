from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.phase4_helpers import make_collection
from sleep_staging.config import load_settings
from sleep_staging.acquisition.loader import discover_recordings
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.representations.types import (
    EncodedDataset,
    EncodedDatasetCollection,
    RepresentationMetadata,
)
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.sc_to_st import _subject_key, build_sc_split, discover_sc_st_recordings, run_primary_experiment
from sleep_staging.training import sc_to_st as orchestrator
from sleep_staging.training.split import assert_no_subject_leakage, subject_wise_split


def _dataset_root() -> Path:
    """Sleep-EDF root for tests: SLEEP_EDF_ROOT env var, else the config default."""
    return load_settings(Path('configs/default.yaml')).acquisition.data_root


#: Tests below need the real Sleep-EDF Expanded corpus on disk. They are skipped
#: (not failed) when it is absent, so the suite stays green on machines that only
#: have the source checkout -- while still validating real-data invariants here.
requires_dataset = pytest.mark.skipif(
    not _dataset_root().exists(),
    reason=f"Sleep-EDF dataset not found at {_dataset_root()} (set SLEEP_EDF_ROOT to override)",
)


def test_config_loads_and_dataset_root_is_configurable() -> None:
    settings = load_settings(Path('configs/default.yaml'))
    # Assert the root is *resolvable and configurable*, not that it equals one
    # machine's absolute path -- SLEEP_EDF_ROOT may legitimately override it.
    assert settings.acquisition.data_root == _dataset_root()
    assert settings.experiment.train.seed == 42


@requires_dataset
def test_dataset_discovery_separates_sc_and_st() -> None:
    sc, st = discover_sc_st_recordings(_dataset_root())
    assert len(sc) == 153
    assert len(st) == 44
    assert all(parse_psg_filename(p).study == 'SC' for p in sc)
    assert all(parse_psg_filename(p).study == 'ST' for p in st)


@requires_dataset
def test_sc_split_exact_counts_and_no_st_leakage() -> None:
    sc, _ = discover_sc_st_recordings(_dataset_root())
    train, val = build_sc_split(sc, seed=42, train_subjects=55, val_subjects=12)
    assert len(train) == 55
    assert len(val) == 12
    assert not set(train) & set(val)


def test_representation_shapes_and_model_construction() -> None:
    raw = make_collection('raw', subjects=('S1', 'S2'))
    bp = make_collection('bandpower', subjects=('S1', 'S2'))
    tf = make_collection('time_frequency', subjects=('S1', 'S2'))
    assert raw.items[0].features.shape[1:] == (1, 3000)
    assert bp.items[0].features.shape[1:] == (1, 10)
    assert tf.items[0].features.shape[1:] == (1, 75, 28)


def test_subject_key_matches_recording_metadata_subject_id_format() -> None:
    """Regression for the num_samples=0 DataLoader crash.

    ``_subject_key()`` builds the strings used to filter an already-encoded
    ``EncodedDatasetCollection`` (via ``EpochDataset(subject_ids=...)``). It
    must exactly equal the value ``extract_metadata()`` stores as
    ``RecordingMetadata.subject_id`` / ``EncodedDataset.subject_id`` -- the
    bare 2-digit Sleep-EDF subject code -- or every subject-filtered split
    (train and val) silently matches zero recordings.
    """
    for name in ("SC4001E0-PSG.edf", "SC4192E0-PSG.edf", "ST7011J0-PSG.edf", "ST7242J0-PSG.edf"):
        path = Path(name)
        ids = parse_psg_filename(path)
        assert _subject_key(path) == ids.subject_id


def test_sc_split_subject_keys_actually_match_encoded_dataset() -> None:
    """End-to-end regression: split subject keys must select real recordings.

    Builds a synthetic ``EncodedDatasetCollection`` whose ``subject_id``
    values are set exactly the way production code sets them
    (``parse_psg_filename(path).subject_id``, mirroring
    ``extract_metadata()``), then confirms ``EpochDataset`` filtering by the
    split's subject keys is non-empty. Before the fix, ``_subject_key`` used
    ``sleep_edf_subject_key`` (``study+series+subject``, e.g. ``"SC400"``)
    which never matches the bare ``"00"`` stored on ``EncodedDataset``, so
    this would have failed with ``len(train_ds) == 0`` regardless of split
    ratios.
    """
    sc_paths = tuple(
        Path(f"SC4{subj:02d}{night}E0-PSG.edf") for subj in range(6) for night in (1, 2)
    )
    metadata = RepresentationMetadata(
        representation="raw",
        channel_names=("Fpz-Cz",),
        sfreq=100.0,
        epoch_duration_sec=30.0,
        feature_shape=(1, 3000),
    )
    items = tuple(
        EncodedDataset(
            features=np.zeros((2, 1, 3000), dtype=np.float32),
            labels=np.asarray([0, 1], dtype=np.int64),
            metadata=metadata,
            subject_id=parse_psg_filename(path).subject_id,
            recording_id=parse_psg_filename(path).recording_id,
        )
        for path in sc_paths
    )
    collection = EncodedDatasetCollection(items=items)

    unique_subjects = sorted({_subject_key(path) for path in sc_paths})
    assert len(unique_subjects) == 6  # 2 nights per subject collapse to 1 key

    split = subject_wise_split(unique_subjects, ratios=(0.7, 0.15, 0.15), seed=42)
    assert_no_subject_leakage(split)
    assert len(split.train) > 0
    assert len(split.val) > 0

    train_ds = EpochDataset(collection, subject_ids=split.train, drop_ignore=False)
    val_ds = EpochDataset(collection, subject_ids=split.val, drop_ignore=False)

    assert len(train_ds) > 0, "train split matched zero recordings -- subject-id format mismatch"
    assert len(val_ds) > 0, "val split matched zero recordings -- subject-id format mismatch"
    # Every matched subject contributes 2 nights x 2 epochs each.
    assert len(train_ds) == len(split.train) * 2 * 2
    assert len(val_ds) == len(split.val) * 2 * 2


def test_sc_to_st_smoke_external_test(monkeypatch, tmp_path: Path) -> None:
    sc = tuple(Path(f'SC40{i}1E0-PSG.edf') for i in range(1, 4))
    st = tuple(Path(f'ST70{i}1J0-PSG.edf') for i in range(1, 3))
    sc_collection = {
        'bandpower': make_collection('bandpower', subjects=('S1', 'S2', 'S3')),
        'raw': make_collection('raw', subjects=('S1', 'S2', 'S3')),
        'time_frequency': make_collection('time_frequency', subjects=('S1', 'S2', 'S3')),
    }
    st_collection = make_collection('bandpower', subjects=('T1', 'T2'))

    def fake_build_primary_collections(**kwargs):
        settings = load_settings(Path('configs/default.yaml'))
        return settings, sc, st

    monkeypatch.setattr(orchestrator, 'build_primary_collections', fake_build_primary_collections)
    monkeypatch.setattr(
        orchestrator,
        '_subject_key',
        lambda path: 'S1' if 'SC401' in path.stem else ('S2' if 'SC402' in path.stem else ('S3' if 'SC403' in path.stem else ('T1' if 'ST701' in path.stem else 'T2'))),
    )

    def fake_encode(recording_paths, *, settings, representation):
        return st_collection if representation == 'bandpower' and list(recording_paths) == list(st) else sc_collection[representation]

    monkeypatch.setattr(orchestrator, '_preprocess_and_encode_recordings', fake_encode)
    monkeypatch.setattr(orchestrator, 'PRIMARY_MODEL_ROOT', tmp_path / 'models')
    monkeypatch.setattr(orchestrator, 'PRIMARY_ARTIFACT_ROOT', tmp_path / 'artifacts')

    # build_primary_collections is monkeypatched above, so this root is never
    # read from disk -- it only has to be a valid Path.
    report = run_primary_experiment(
        config_path=Path('configs/default.yaml'),
        dataset_root=_dataset_root(),
        model='classical',
        smoke=True,
        max_sc_recordings=3,
        max_st_recordings=2,
    )
    assert report['st_test_subjects']
    classical = report['models']['classical']
    assert Path(classical['metrics_path']).exists()
    assert Path(classical['predictions_path']).exists()
    assert Path(classical['metadata_path']).exists()
    assert 'sc_to_st' in classical['metrics_path']
