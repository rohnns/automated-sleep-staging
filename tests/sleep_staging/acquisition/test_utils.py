"""Unit tests for Sleep-EDF filename / path utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleep_staging.acquisition.exceptions import (
    MetadataExtractionError,
    MissingHypnogramFileError,
    MissingPSGFileError,
)
from sleep_staging.acquisition.utils import (
    discover_psg_files,
    ensure_file,
    infer_hypnogram_path,
    parse_psg_filename,
    resolve_hypnogram_path,
)


def test_parse_psg_filename_cassette() -> None:
    ids = parse_psg_filename("SC4001E0-PSG.edf")
    assert ids.study == "SC"
    assert ids.series == "4"
    assert ids.subject_id == "00"
    assert ids.recording_id == "1"
    assert ids.scorer_id == "E"
    assert ids.stem == "SC4001E0"


def test_parse_psg_filename_telemetry() -> None:
    ids = parse_psg_filename(Path("ST7011J0-PSG.edf"))
    assert ids.study == "ST"
    assert ids.series == "7"
    assert ids.subject_id == "01"
    assert ids.recording_id == "1"
    assert ids.scorer_id == "J"


def test_parse_psg_filename_invalid() -> None:
    with pytest.raises(MetadataExtractionError):
        parse_psg_filename("not-a-sleep-edf.edf")


def test_infer_hypnogram_path() -> None:
    psg = Path("/data/SC4001E0-PSG.edf")
    hyp = infer_hypnogram_path(psg)
    assert hyp.name == "SC4001E0-Hypnogram.edf"
    assert hyp.parent == psg.parent


def test_ensure_file_missing_psg(tmp_path: Path) -> None:
    with pytest.raises(MissingPSGFileError):
        ensure_file(tmp_path / "SC4001E0-PSG.edf", kind="PSG")


def test_resolve_hypnogram_path_missing(tmp_path: Path) -> None:
    psg = tmp_path / "SC4001E0-PSG.edf"
    psg.write_bytes(b"")
    with pytest.raises(MissingHypnogramFileError):
        resolve_hypnogram_path(psg)


def test_resolve_hypnogram_path_scorer_mismatch(tmp_path: Path) -> None:
    """Sleep-EDF Expanded often uses a different scorer code on the hypnogram."""
    psg = tmp_path / "SC4001E0-PSG.edf"
    hyp = tmp_path / "SC4001EC-Hypnogram.edf"
    psg.write_bytes(b"")
    hyp.write_bytes(b"")
    resolved = resolve_hypnogram_path(psg)
    assert resolved == hyp.resolve()


def test_discover_psg_files(tmp_path: Path) -> None:
    (tmp_path / "SC4001E0-PSG.edf").write_bytes(b"")
    (tmp_path / "SC4001E0-Hypnogram.edf").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ST7011J0-PSG.edf").write_bytes(b"")
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")

    found = discover_psg_files(tmp_path)
    names = {path.name for path in found}
    assert names == {"SC4001E0-PSG.edf", "ST7011J0-PSG.edf"}


def test_discover_missing_root(tmp_path: Path) -> None:
    with pytest.raises(MissingPSGFileError):
        discover_psg_files(tmp_path / "does-not-exist")
