"""Path and filename helpers for Sleep-EDF Expanded.

These helpers are deliberately Sleep-EDF-specific (filename conventions,
hypnogram pairing). They do not depend on MNE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sleep_staging.acquisition.exceptions import (
    MetadataExtractionError,
    MissingHypnogramFileError,
    MissingPSGFileError,
)
from sleep_staging.common.logging_utils import get_logger

logger = get_logger(__name__)

# Sleep-EDF Expanded PSG filenames, e.g. SC4001E0-PSG.edf / ST7011J0-PSG.edf
# Convention: {SC|ST}{series}{subject:02d}{night}{scorer}{suffix}-PSG.edf
_PSG_PATTERN = re.compile(
    r"^(?P<study>SC|ST)"
    r"(?P<series>\d)"
    r"(?P<subject>\d{2})"
    r"(?P<recording>\d)"
    r"(?P<scorer>[A-Za-z])"
    r"(?P<suffix>\d)"
    r"-PSG\.edf$",
    re.IGNORECASE,
)

_HYPNOGRAM_SUFFIX = "-Hypnogram.edf"


@dataclass(frozen=True, slots=True)
class SleepEDFFileIds:
    """Identifiers parsed from a Sleep-EDF Expanded PSG filename."""

    study: str
    series: str
    subject_id: str
    recording_id: str
    scorer_id: str
    stem: str


def ensure_file(path: Path, *, kind: str) -> Path:
    """Validate that ``path`` exists and is a file.

    Parameters
    ----------
    path:
        Candidate file path.
    kind:
        Human-readable label used in error messages (``PSG``, ``Hypnogram``).

    Returns
    -------
    Path
        Resolved absolute path.
    """
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        if kind.lower().startswith("psg"):
            raise MissingPSGFileError(f"{kind} file not found: {resolved}")
        if kind.lower().startswith("hypnogram"):
            raise MissingHypnogramFileError(f"{kind} file not found: {resolved}")
        raise FileNotFoundError(f"{kind} file not found: {resolved}")
    if not resolved.is_file():
        raise IsADirectoryError(f"{kind} path is not a file: {resolved}")
    return resolved


def parse_psg_filename(path: Path | str) -> SleepEDFFileIds:
    """Parse subject / recording identifiers from a PSG filename.

    Parameters
    ----------
    path:
        Path or filename of a ``*-PSG.edf`` file.

    Returns
    -------
    SleepEDFFileIds
        Parsed identifiers.

    Raises
    ------
    MetadataExtractionError
        If the filename does not match the Sleep-EDF Expanded convention.
    """
    name = Path(path).name
    match = _PSG_PATTERN.match(name)
    if match is None:
        raise MetadataExtractionError(
            f"Filename does not match Sleep-EDF Expanded PSG convention: {name}"
        )
    lower = name.lower()
    if not lower.endswith("-psg.edf"):
        raise MetadataExtractionError(f"Expected a *-PSG.edf filename, got: {name}")
    stem = name[: -len("-PSG.edf")]
    return SleepEDFFileIds(
        study=match.group("study").upper(),
        series=match.group("series"),
        subject_id=match.group("subject"),
        recording_id=match.group("recording"),
        scorer_id=match.group("scorer").upper(),
        stem=stem,
    )


def infer_hypnogram_path(psg_path: Path) -> Path:
    """Return the same-stem hypnogram candidate for a PSG file.

    Prefer :func:`resolve_hypnogram_path` for real datasets: Sleep-EDF Expanded
    often uses a different scorer code in the hypnogram filename
    (e.g. ``SC4001E0-PSG.edf`` pairs with ``SC4001EC-Hypnogram.edf``).
    """
    name = psg_path.name
    lower = name.lower()
    if not lower.endswith("-psg.edf"):
        raise MetadataExtractionError(
            f"Cannot infer hypnogram path from non-PSG filename: {name}"
        )
    stem = name[: -len("-PSG.edf")]
    return psg_path.with_name(f"{stem}{_HYPNOGRAM_SUFFIX}")


def _recording_prefix(psg_path: Path) -> str:
    """Return the stable recording key shared by PSG and hypnogram files.

    Example: ``SC4001E0-PSG.edf`` -> ``SC4001``.
    """
    ids = parse_psg_filename(psg_path)
    return f"{ids.study}{ids.series}{ids.subject_id}{ids.recording_id}"


def resolve_hypnogram_path(
    psg_path: Path,
    hypnogram_path: Path | None = None,
) -> Path:
    """Resolve and validate the hypnogram path for a PSG recording.

    Resolution order when ``hypnogram_path`` is omitted:

    1. Same-stem sibling (``SC4001E0-Hypnogram.edf``)
    2. Unique sibling matching the recording prefix
       (``SC4001*-Hypnogram.edf``), which covers Sleep-EDF Expanded scorer
       codes that differ between PSG and hypnogram files.
    """
    if hypnogram_path is not None:
        resolved = ensure_file(Path(hypnogram_path), kind="Hypnogram")
        logger.debug("Resolved hypnogram for %s -> %s", psg_path.name, resolved.name)
        return resolved

    exact = infer_hypnogram_path(psg_path)
    if exact.is_file():
        resolved = exact.resolve()
        logger.debug("Resolved hypnogram for %s -> %s", psg_path.name, resolved.name)
        return resolved

    prefix = _recording_prefix(psg_path)
    matches = sorted(
        path.resolve()
        for path in psg_path.parent.glob(f"{prefix}*{_HYPNOGRAM_SUFFIX}")
        if path.is_file()
    )
    if len(matches) == 1:
        logger.debug("Resolved hypnogram for %s -> %s", psg_path.name, matches[0].name)
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise MetadataExtractionError(
            f"Ambiguous hypnogram match for {psg_path.name} (prefix {prefix}): {names}"
        )
    raise MissingHypnogramFileError(
        f"Hypnogram file not found for {psg_path.name} "
        f"(tried {exact.name} and {prefix}*-Hypnogram.edf in {psg_path.parent})"
    )


def discover_psg_files(data_root: Path) -> list[Path]:
    """Discover Sleep-EDF ``*-PSG.edf`` files under ``data_root`` (recursive)."""
    root = data_root.expanduser().resolve()
    if not root.exists():
        raise MissingPSGFileError(f"Data root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {root}")

    files = sorted(
        {
            path.resolve()
            for path in root.rglob("*-PSG.edf")
            if path.is_file() and _PSG_PATTERN.match(path.name)
        }
    )
    logger.info("Discovered %d PSG file(s) under %s", len(files), root)
    return files
