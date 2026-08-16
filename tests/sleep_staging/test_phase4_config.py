from __future__ import annotations

from pathlib import Path

import yaml

from sleep_staging.config import load_settings


def test_default_phase4_experiment_settings_load() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = load_settings(project_root / "configs" / "default.yaml")
    exp = settings.experiment
    assert exp.split.cohort == "SC"
    # Primary SC→ST experiment uses the 3-channel staging montage
    # (2 EEG derivations + 1 EOG), matching configs/default.yaml.
    assert exp.split.channels == ("Fpz-Cz", "Pz-Oz", "horizontal")
    assert exp.split.ratios == (0.70, 0.15, 0.15)
    assert exp.train.class_weighting == "balanced"
    assert exp.train.early_stopping_patience == 5
    assert exp.train.checkpoint_dir is not None
    assert exp.classical_baseline.enabled is True


def test_custom_phase4_experiment_settings_parse(tmp_path: Path) -> None:
    cfg = {
        "acquisition": {"data_root": "D:/SleepEDFX"},
        "experiment": {
            "split": {"channels": ["Fpz-Cz"], "seed": 9, "cohort": "SC"},
            "train": {"max_epochs": 2, "checkpoint_dir": "ckpts", "early_stopping_patience": None},
            "classical_baseline": {"max_iter": 123},
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    settings = load_settings(path, project_root=tmp_path)
    assert settings.experiment.split.seed == 9
    assert settings.experiment.train.max_epochs == 2
    assert settings.experiment.train.early_stopping_patience is None
    assert settings.experiment.train.checkpoint_dir == tmp_path / "ckpts"
    assert settings.experiment.classical_baseline.max_iter == 123
