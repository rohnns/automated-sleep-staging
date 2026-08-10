"""Map Rechtschaffen & Kales hypnogram labels to AASM five-stage labels.

Dataset evidence: Sleep-EDF Expanded hypnograms use R&K labels including
separate stages 3 and 4. AASM merges 3+4 into N3. Movement time and '?' are
not part of the five target classes; they are marked with an ignore label
so epoch alignment is preserved for later encodings / loss masking.
"""

from __future__ import annotations

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.types import EpochLabels, PreprocessedRecording, Transform

logger = get_logger(__name__)

AASM_STAGES: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")

# String sentinel kept on epoch labels; encodings map this to ignore_index (-1).
IGNORE_LABEL = "IGNORE"

DEFAULT_RK_TO_AASM: dict[str, str] = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": IGNORE_LABEL,
    "Sleep stage ?": IGNORE_LABEL,
}


class StageMapper(Transform):
    """Map epoch labels from R&K (or already-AASM) names to AASM stages.

    Parameters
    ----------
    mapping:
        Source label → AASM label (or :data:`IGNORE_LABEL`).
    ignore_label:
        Sentinel written for Movement / unknown / explicitly ignored sources.
    unmapped_policy:
        How to handle source labels absent from ``mapping``:

        - ``"ignore"`` — keep the epoch, write ``ignore_label`` (default)
        - ``"drop"`` — remove the epoch (breaks contiguous index alignment;
          requires ``allow_length_change=True``)
        - ``"error"`` — raise :class:`TransformError`
    allow_length_change:
        Required opt-in when ``unmapped_policy="drop"``. Dropping epochs
        desynchronizes contiguous ``raw`` sample windows from label index
        ``k``; only safe if encodings slice exclusively via ``onsets_sec``.
    """

    name = "stage_mapper"

    def __init__(
        self,
        *,
        mapping: dict[str, str] | None = None,
        ignore_label: str = IGNORE_LABEL,
        unmapped_policy: str = "ignore",
        allow_length_change: bool = False,
        drop_unmapped: bool | None = None,
    ) -> None:
        if unmapped_policy not in {"ignore", "drop", "error"}:
            raise ValueError("unmapped_policy must be 'ignore', 'drop', or 'error'")
        # Backward-compatible alias used by earlier scaffolding / configs.
        if drop_unmapped is True:
            unmapped_policy = "drop"
        elif drop_unmapped is False and unmapped_policy == "ignore":
            unmapped_policy = "error"
        if unmapped_policy == "drop" and not allow_length_change:
            raise ValueError(
                "unmapped_policy='drop' removes epochs and desynchronizes "
                "contiguous signal/label indexing. Prefer 'ignore' (default), "
                "or pass allow_length_change=True if encodings index epochs "
                "only via EpochLabels.onsets_sec."
            )
        self.mapping = dict(mapping or DEFAULT_RK_TO_AASM)
        self.ignore_label = ignore_label
        self.unmapped_policy = unmapped_policy
        self.allow_length_change = allow_length_change

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        if state.epoch_labels is None:
            raise TransformError("StageMapper requires AnnotationUnroller to run first")

        mapped = map_epoch_labels(
            state.epoch_labels,
            mapping=self.mapping,
            ignore_label=self.ignore_label,
            unmapped_policy=self.unmapped_policy,
        )
        ignored = sum(1 for label in mapped.labels if label == self.ignore_label)
        dropped = state.epoch_labels.n_epochs - mapped.n_epochs
        state.epoch_labels = mapped
        state.extras["stage_mapping"] = {
            "ignored_epochs": ignored,
            "dropped_epochs": dropped,
            "ignore_label": self.ignore_label,
            "label_set": sorted(set(mapped.labels)),
        }
        logger.info(
            "Mapped stages to AASM (%d epoch(s); %d ignored as %s, %d dropped); labels=%s",
            mapped.n_epochs,
            ignored,
            self.ignore_label,
            dropped,
            sorted(set(mapped.labels)),
        )
        return state


def map_epoch_labels(
    epoch_labels: EpochLabels,
    *,
    mapping: dict[str, str] | None = None,
    ignore_label: str = IGNORE_LABEL,
    unmapped_policy: str = "ignore",
) -> EpochLabels:
    """Pure helper: map epoch labels; mark unknowns with ``ignore_label`` by default.

    The ``"drop"`` policy is available here for onset-indexed workflows. Prefer
    :class:`StageMapper` with ``unmapped_policy='ignore'`` for the default
    continuous pipeline.
    """
    if unmapped_policy not in {"ignore", "drop", "error"}:
        raise ValueError("unmapped_policy must be 'ignore', 'drop', or 'error'")

    table = mapping or DEFAULT_RK_TO_AASM
    onsets: list[float] = []
    labels: list[str] = []

    for onset, label in zip(epoch_labels.onsets_sec, epoch_labels.labels, strict=True):
        if label in AASM_STAGES:
            mapped: str | None = label
        elif label == ignore_label:
            mapped = ignore_label
        elif label in table:
            mapped = table[label]
        elif unmapped_policy == "ignore":
            mapped = ignore_label
        elif unmapped_policy == "drop":
            mapped = None
        else:
            raise TransformError(f"Unmapped stage label: {label!r}")

        # Optional drop policy removes ignore-mapped epochs (Movement / '?').
        if mapped is None or (unmapped_policy == "drop" and mapped == ignore_label):
            continue

        onsets.append(onset)
        labels.append(mapped)

    return EpochLabels(
        onsets_sec=tuple(onsets),
        duration_sec=epoch_labels.duration_sec,
        labels=tuple(labels),
    )
