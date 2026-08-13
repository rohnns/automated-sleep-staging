from __future__ import annotations

from pathlib import Path

import pytest

from tests.phase4_helpers import make_collection
from sleep_staging.config import load_settings
from sleep_staging.acquisition.loader import discover_recordings
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.training.sc_to_st import build_sc_split, discover_sc_st_recordings, run_primary_experiment
from sleep_staging.training import sc_to_st as orchestrator


def test_config_loads_and_dataset_root_is_configurable() -> None:
    settings = load_settings(Path('configs/default.yaml'))
    assert str(settings.acquisition.data_root).replace('\\', '/').endswith('D:/SleepEDFX')
    assert settings.experiment.train.seed == 42


def test_dataset_discovery_separates_sc_and_st() -> None:
    sc, st = discover_sc_st_recordings(Path('D:/SleepEDFX'))
    assert len(sc) == 153
    assert len(st) == 44
    assert all(parse_psg_filename(p).study == 'SC' for p in sc)
    assert all(parse_psg_filename(p).study == 'ST' for p in st)


def test_sc_split_exact_counts_and_no_st_leakage() -> None:
    sc, _ = discover_sc_st_recordings(Path('D:/SleepEDFX'))
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
        return settings, sc, st, sc_collection

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

    report = run_primary_experiment(
        config_path=Path('configs/default.yaml'),
        dataset_root=Path('D:/SleepEDFX'),
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
