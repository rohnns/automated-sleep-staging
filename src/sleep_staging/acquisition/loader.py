"""Sleep-EDF Expanded data acquisition via MNE.

MNE boundary
------------
This module is the designated place for MNE I/O
(``mne.io.read_raw_edf``, ``mne.read_annotations``, ``Raw.set_annotations``).
Downstream stages should consume :class:`SleepRecording` rather than repeating
EDF loading. Later preprocessing may call MNE APIs on ``recording.raw``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import mne

from sleep_staging.acquisition.dataclasses import SleepRecording
from sleep_staging.acquisition.exceptions import InvalidEDFFileError, MissingPSGFileError
from sleep_staging.acquisition.metadata import extract_metadata, validate_recording
from sleep_staging.acquisition.utils import (
    discover_psg_files,
    ensure_file,
    parse_psg_filename,
    resolve_hypnogram_path,
)
from sleep_staging.common.logging_utils import get_logger
from sleep_staging.config.settings import AcquisitionSettings

logger = get_logger(__name__)


class SleepEDFLoader:
    """Load Sleep-EDF Expanded PSG recordings and hypnogram annotations.

    The loader validates file presence, reads data with MNE, attaches
    hypnogram annotations to the ``Raw`` object (authoritative store),
    extracts structured metadata, and returns a :class:`SleepRecording`.
    """

    def __init__(
        self,
        settings: AcquisitionSettings | None = None,
        *,
        data_root: Path | str | None = None,
        preload: bool | None = None,
        stim_channel: str | None = None,
        infer_types: bool | None = None,
        mne_verbose: str | None = None,
    ) -> None:
        if settings is None:
            root = Path(data_root) if data_root is not None else Path.cwd()
            settings = AcquisitionSettings(data_root=root.expanduser().resolve())

        self._settings = AcquisitionSettings(
            data_root=(
                Path(data_root).expanduser().resolve()
                if data_root is not None
                else settings.data_root
            ),
            preload=settings.preload if preload is None else preload,
            stim_channel=settings.stim_channel if stim_channel is None else stim_channel,
            infer_types=settings.infer_types if infer_types is None else infer_types,
            mne_verbose=settings.mne_verbose if mne_verbose is None else mne_verbose,
        )
        logger.debug(
            "Initialized SleepEDFLoader(data_root=%s, preload=%s)",
            self._settings.data_root,
            self._settings.preload,
        )

    @property
    def settings(self) -> AcquisitionSettings:
        """Return the effective acquisition settings."""
        return self._settings

    def load_recording(
        self,
        psg_path: Path | str,
        hypnogram_path: Path | str | None = None,
        *,
        preload: bool | None = None,
    ) -> SleepRecording:
        """Load one PSG recording and its paired hypnogram."""
        resolved_psg = self._resolve_psg_path(psg_path)
        resolved_hyp = resolve_hypnogram_path(
            resolved_psg,
            Path(hypnogram_path) if hypnogram_path is not None else None,
        )
        use_preload = self._settings.preload if preload is None else preload

        logger.info("Loading PSG: %s", resolved_psg)
        raw = self._read_psg(resolved_psg, preload=use_preload)

        logger.info("Loading hypnogram: %s", resolved_hyp)
        annotations = self._read_hypnogram(resolved_hyp)

        # Authoritative annotation store: attached to Raw for all MNE workflows.
        raw.set_annotations(annotations, emit_warning=False)

        validate_recording(raw, psg_path=resolved_psg, hypnogram_path=resolved_hyp)
        file_ids = parse_psg_filename(resolved_psg)
        metadata = extract_metadata(
            raw,
            psg_path=resolved_psg,
            hypnogram_path=resolved_hyp,
            file_ids=file_ids,
        )
        recording = SleepRecording(raw=raw, metadata=metadata)
        logger.info("Loaded %s", recording.summary())
        return recording

    def iter_recordings(
        self,
        data_root: Path | str | None = None,
        *,
        preload: bool | None = None,
    ) -> Iterator[SleepRecording]:
        """Yield recordings discovered under a data root."""
        root = Path(data_root).expanduser().resolve() if data_root else self._settings.data_root
        for psg_path in discover_psg_files(root):
            yield self.load_recording(psg_path, preload=preload)

    def load_all(
        self,
        data_root: Path | str | None = None,
        *,
        preload: bool | None = None,
    ) -> list[SleepRecording]:
        """Load every discoverable recording under a data root."""
        return list(self.iter_recordings(data_root, preload=preload))

    def load_many(
        self,
        psg_paths: Sequence[Path | str],
        *,
        preload: bool | None = None,
    ) -> list[SleepRecording]:
        """Load an explicit sequence of PSG recordings."""
        return [self.load_recording(path, preload=preload) for path in psg_paths]

    def discover(self, data_root: Path | str | None = None) -> list[Path]:
        """Return sorted PSG paths discovered under ``data_root``."""
        root = Path(data_root).expanduser().resolve() if data_root else self._settings.data_root
        return discover_psg_files(root)

    def _resolve_psg_path(self, psg_path: Path | str) -> Path:
        path = Path(psg_path).expanduser()
        if not path.is_absolute():
            candidate = (self._settings.data_root / path).resolve()
            if candidate.exists():
                path = candidate
            else:
                cwd_candidate = path.resolve()
                path = cwd_candidate if cwd_candidate.exists() else candidate
        return ensure_file(path, kind="PSG")

    def _read_psg(self, psg_path: Path, *, preload: bool) -> mne.io.BaseRaw:
        try:
            return mne.io.read_raw_edf(
                str(psg_path),
                stim_channel=self._settings.stim_channel,
                infer_types=self._settings.infer_types,
                preload=preload,
                verbose=self._settings.mne_verbose,
            )
        except MissingPSGFileError:
            raise
        except Exception as exc:
            raise InvalidEDFFileError(
                f"Failed to read PSG EDF file '{psg_path}': {exc}"
            ) from exc

    def _read_hypnogram(self, hypnogram_path: Path) -> mne.Annotations:
        try:
            return mne.read_annotations(str(hypnogram_path))
        except Exception as exc:
            raise InvalidEDFFileError(
                f"Failed to read hypnogram EDF file '{hypnogram_path}': {exc}"
            ) from exc


def load_recording(
    psg_path: Path | str,
    hypnogram_path: Path | str | None = None,
    *,
    settings: AcquisitionSettings | None = None,
    preload: bool | None = None,
) -> SleepRecording:
    """Convenience function to load a single Sleep-EDF recording."""
    return SleepEDFLoader(settings=settings).load_recording(
        psg_path, hypnogram_path, preload=preload
    )


def discover_recordings(data_root: Path | str) -> list[Path]:
    """Discover PSG files under ``data_root`` without loading them."""
    return discover_psg_files(Path(data_root))


def load_recordings(
    sources: Path | str | Iterable[Path | str],
    *,
    settings: AcquisitionSettings | None = None,
    preload: bool | None = None,
) -> list[SleepRecording]:
    """Load recordings from a directory or an iterable of PSG paths."""
    loader = SleepEDFLoader(settings=settings)
    if isinstance(sources, (str, Path)):
        path = Path(sources)
        if path.is_dir() or not path.suffix:
            return loader.load_all(path, preload=preload)
        return [loader.load_recording(path, preload=preload)]
    return loader.load_many(list(sources), preload=preload)
