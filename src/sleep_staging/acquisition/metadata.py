"""Metadata extraction from loaded Sleep-EDF MNE objects.

MNE boundary
------------
Functions here read MNE ``Raw`` / annotation state to populate
:class:`~sleep_staging.acquisition.dataclasses.RecordingMetadata`.
Filename parsing itself remains MNE-agnostic (see ``utils``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np

from sleep_staging.acquisition.dataclasses import ChannelInfo, RecordingMetadata
from sleep_staging.acquisition.exceptions import MetadataExtractionError, RecordingValidationError
from sleep_staging.acquisition.utils import SleepEDFFileIds, parse_psg_filename
from sleep_staging.common.logging_utils import get_logger

logger = get_logger(__name__)

_BIPOLAR_REFERENCE_HINT = (
    "Bipolar derivations encoded in channel names "
    "(e.g. Fpz-Cz, Pz-Oz); no separate reference electrode recorded."
)


def _unit_for_channel(raw: mne.io.BaseRaw, ch_name: str) -> str | None:
    """Best-effort unit lookup from MNE channel metadata."""
    try:
        idx = raw.ch_names.index(ch_name)
    except ValueError:
        return None

    unit_dict = getattr(raw, "_orig_units", None)
    if isinstance(unit_dict, dict) and ch_name in unit_dict:
        value = unit_dict[ch_name]
        return str(value) if value is not None else None

    try:
        unit_code = raw.info["chs"][idx].get("unit")
        if unit_code is None:
            return None
        unit_map = {
            107: "V",  # FIFF_UNIT_V
            112: "T",  # FIFF_UNIT_T
            201: "°C",
            202: "%",
        }
        return unit_map.get(int(unit_code))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _infer_reference(channel_names: list[str], raw: mne.io.BaseRaw) -> str | None:
    """Infer reference / montage description from channel names and Raw state."""
    montage = raw.get_montage()
    if montage is not None and getattr(montage, "kind", None):
        return str(montage.kind)

    looks_bipolar = any(
        "-" in name
        and any(token in name.lower() for token in ("fpz", "pz", "cz", "oz", "eeg"))
        for name in channel_names
    )
    if looks_bipolar or any("Fpz-Cz" in name or "Pz-Oz" in name for name in channel_names):
        return _BIPOLAR_REFERENCE_HINT

    custom_ref = raw.info.get("custom_ref_applied")
    if custom_ref:
        return f"custom_ref_applied={custom_ref}"
    return None


def _montage_name(raw: mne.io.BaseRaw) -> str | None:
    montage = raw.get_montage()
    if montage is None:
        return None
    kind = getattr(montage, "kind", None)
    return str(kind) if kind is not None else None


def _meas_date(raw: mne.io.BaseRaw) -> datetime | None:
    value = raw.info.get("meas_date")
    if isinstance(value, datetime):
        return value
    return None


def build_channel_infos(raw: mne.io.BaseRaw) -> tuple[ChannelInfo, ...]:
    """Build per-channel metadata from an MNE Raw object."""
    types = raw.get_channel_types()
    sfreq = float(raw.info["sfreq"])
    return tuple(
        ChannelInfo(
            name=name,
            ch_type=ch_type,
            unit=_unit_for_channel(raw, name),
            sampling_frequency=sfreq,
        )
        for name, ch_type in zip(raw.ch_names, types, strict=True)
    )


def extract_metadata(
    raw: mne.io.BaseRaw,
    *,
    psg_path: Path,
    hypnogram_path: Path,
    file_ids: SleepEDFFileIds | None = None,
) -> RecordingMetadata:
    """Extract structured metadata from a loaded PSG with annotations attached.

    Annotations are read exclusively from ``raw.annotations`` (authoritative).
    """
    try:
        ids = file_ids or parse_psg_filename(psg_path)
    except MetadataExtractionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise MetadataExtractionError(
            f"Failed to parse identifiers from {psg_path.name}: {exc}"
        ) from exc

    try:
        channel_infos = build_channel_infos(raw)
        channel_names = tuple(raw.ch_names)
        channel_types = tuple(raw.get_channel_types())
        units = {ch.name: ch.unit for ch in channel_infos}
        duration = float(raw.n_times) / float(raw.info["sfreq"])
        n_annotations = len(raw.annotations)
        extra: dict[str, Any] = {
            "highpass": raw.info.get("highpass"),
            "lowpass": raw.info.get("lowpass"),
            "n_times": int(raw.n_times),
            "series": ids.series,
            "scorer_id": ids.scorer_id,
            "file_stem": ids.stem,
        }
        for key in ("subject_info", "device_info"):
            if key in raw.info and raw.info[key] is not None:
                extra[key] = raw.info[key]

        metadata = RecordingMetadata(
            subject_id=ids.subject_id,
            recording_id=ids.recording_id,
            study=ids.study,
            sampling_frequency=float(raw.info["sfreq"]),
            duration_sec=duration,
            n_channels=len(channel_names),
            channel_names=channel_names,
            channel_types=channel_types,
            channels=channel_infos,
            units=units,
            reference=_infer_reference(list(channel_names), raw),
            montage=_montage_name(raw),
            meas_date=_meas_date(raw),
            psg_path=psg_path,
            hypnogram_path=hypnogram_path,
            n_annotations=n_annotations,
            extra=extra,
        )
    except MetadataExtractionError:
        raise
    except Exception as exc:
        raise MetadataExtractionError(
            f"Failed to extract metadata for {psg_path.name}: {exc}"
        ) from exc

    logger.debug(
        "Extracted metadata: subject=%s recording=%s sfreq=%.2f duration=%.1fs channels=%d",
        metadata.subject_id,
        metadata.recording_id,
        metadata.sampling_frequency,
        metadata.duration_sec,
        metadata.n_channels,
    )
    return metadata


def validate_recording(raw: mne.io.BaseRaw, *, psg_path: Path, hypnogram_path: Path) -> None:
    """Validate that a Raw object has usable signals and attached annotations.

    Raises
    ------
    RecordingValidationError
        If the Raw object or its annotations are empty / inconsistent.
    """
    if raw is None:
        raise RecordingValidationError(f"Raw signal object is missing for {psg_path}")
    if len(raw.ch_names) == 0:
        raise RecordingValidationError(f"PSG contains no channels: {psg_path}")
    if raw.n_times <= 0:
        raise RecordingValidationError(f"PSG contains no samples: {psg_path}")
    sfreq = raw.info.get("sfreq")
    if sfreq is None or not np.isfinite(sfreq) or float(sfreq) <= 0:
        raise RecordingValidationError(f"Invalid sampling frequency in {psg_path}: {sfreq}")
    if raw.annotations is None:
        raise RecordingValidationError(
            f"Annotations are missing on Raw for hypnogram {hypnogram_path}"
        )
    if len(raw.annotations) == 0:
        raise RecordingValidationError(
            f"Hypnogram contains no annotations (or none fell within the "
            f"recording window): {hypnogram_path}"
        )

    logger.debug(
        "Validated recording for %s (%d ch, %d annot)",
        psg_path.name,
        len(raw.ch_names),
        len(raw.annotations),
    )
