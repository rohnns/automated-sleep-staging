from __future__ import annotations

from pathlib import Path

from sleep_staging.evaluation.output import build_epoch_predictions, compute_sleep_statistics, plot_hypnogram, save_predictions_csv


def test_sleep_statistics_basic() -> None:
    stats = compute_sleep_statistics([0, 0, 1, 2, 3, 4], epoch_duration_sec=30.0)
    assert stats.total_sleep_time_sec == 120.0
    assert stats.sleep_efficiency == 120.0 / 180.0
    assert stats.sleep_onset_latency_sec == 60.0
    assert stats.rem_latency_sec == 150.0
    assert stats.time_in_stage_sec["W"] == 60.0
    assert stats.time_in_stage_sec["REM"] == 30.0


def test_export_helpers_write_files(tmp_path: Path) -> None:
    recs = build_epoch_predictions(
        subject_id="S1",
        recording_id="R1",
        onsets_sec=[0.0, 30.0],
        duration_sec=30.0,
        y_true=[0, 1],
        y_pred=[0, 2],
    )
    csv_path = tmp_path / "predictions.csv"
    save_predictions_csv(recs, csv_path)
    assert csv_path.exists()
    assert "target_stage" in csv_path.read_text(encoding="utf-8")

    plot_path = tmp_path / "hypnogram.png"
    plot_hypnogram(
        onsets_sec=[0.0, 30.0],
        y_true=[0, 1],
        y_pred=[0, 2],
        out_path=plot_path,
        title="demo",
    )
    assert plot_path.exists()
