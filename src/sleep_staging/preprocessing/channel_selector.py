"""Configuration-driven channel selection.

Dataset evidence: Sleep-EDF Expanded cassette recordings commonly expose
``Fpz-Cz``, ``Pz-Oz`` (EEG), ``horizontal`` (EOG), ``submental`` (EMG), plus
respiration / rectal temperature / stim that are usually dropped for staging.
Telemetry recordings omit some extras. Selection is therefore config-driven by
channel name (preferred) and/or MNE channel type.
"""

from __future__ import annotations

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import MissingChannelsError, TransformError
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)

# Defaults informed by analysis channel layouts for sleep staging.
DEFAULT_CHANNEL_NAMES: tuple[str, ...] = (
    "Fpz-Cz",
    "Pz-Oz",
    "horizontal",
    "submental",
)


class ChannelSelector(Transform):
    """Keep a subset of channels by name and/or type.

    Parameters
    ----------
    names:
        Exact channel names to keep (order preserved).
    types:
        MNE channel types to keep when ``names`` is empty/None.
    require_all_names:
        If true, missing requested names raise ``MissingChannelsError``.
        If false, keep the intersection.
    """

    name = "channel_selector"

    def __init__(
        self,
        *,
        names: list[str] | tuple[str, ...] | None = DEFAULT_CHANNEL_NAMES,
        types: list[str] | tuple[str, ...] | None = None,
        require_all_names: bool = True,
    ) -> None:
        self.names = tuple(names) if names is not None else None
        self.types = tuple(types) if types is not None else None
        self.require_all_names = require_all_names
        if self.names is None and self.types is None:
            raise ValueError("Provide names and/or types for channel selection")

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        picks = resolve_channel_picks(
            state.raw.ch_names,
            available_types=state.raw.get_channel_types(),
            names=self.names,
            types=self.types,
            require_all_names=self.require_all_names,
        )
        if not picks:
            raise TransformError("Channel selection produced an empty channel set")

        state.raw.pick(picks)
        state.extras["selected_channels"] = list(state.raw.ch_names)
        logger.info("Selected channels: %s", list(state.raw.ch_names))
        return state


def resolve_channel_picks(
    channel_names: list[str] | tuple[str, ...],
    *,
    available_types: list[str] | tuple[str, ...] | None = None,
    names: tuple[str, ...] | list[str] | None,
    types: tuple[str, ...] | list[str] | None,
    require_all_names: bool = True,
) -> list[str]:
    """Resolve the ordered list of channels to keep."""
    present = list(channel_names)

    if names is not None:
        missing = [name for name in names if name not in present]
        if missing and require_all_names:
            raise MissingChannelsError(
                f"Requested channels not found: {missing}; available={present}"
            )
        picks = [name for name in names if name in present]
        return picks

    assert types is not None
    if available_types is None:
        raise TransformError("available_types is required when selecting by type")
    if len(available_types) != len(present):
        raise TransformError("available_types must align with channel_names")

    wanted = {t.lower() for t in types}
    return [
        name
        for name, ch_type in zip(present, available_types, strict=True)
        if ch_type.lower() in wanted
    ]
