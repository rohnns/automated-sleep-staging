"""Centralized configuration loading for the sleep-staging pipeline.

Configuration stays intentionally simple: a YAML file mapped to frozen
dataclasses. Do not introduce Hydra, Pydantic, or a plugin system here.

Preprocessing defaults are informed by ``analysis/dataset_statistics.py``:
30 s epochs, ~30 min wake buffers, R&K→AASM mapping, EEG/EOG/EMG channels,
0.5–30 Hz band-pass + optional line notch, per-recording z-score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from sleep_staging.config.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class AcquisitionSettings:
    """Settings that control how Sleep-EDF recordings are loaded."""

    data_root: Path
    preload: bool = False
    stim_channel: str = "Event marker"
    infer_types: bool = True
    mne_verbose: str = "ERROR"


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@dataclass(frozen=True, slots=True)
class WakeCropSettings:
    """Wake cropping around the scored sleep period.

    Prefer ``minutes`` in YAML (e.g. ``minutes: 30``). ``buffer_sec`` is the
    runtime value; if only minutes are provided, seconds are derived as
    ``minutes * 60``.
    """

    enabled: bool = True
    buffer_sec: float = 1800.0
    require_sleep: bool = True

    @property
    def minutes(self) -> float:
        return self.buffer_sec / 60.0


@dataclass(frozen=True, slots=True)
class StageMapSettings:
    """R&K → AASM mapping options.

    ``unmapped_policy`` defaults to ``ignore`` so Movement / ``?`` epochs keep
    their time slots and are marked for loss masking later.
    """

    unmapped_policy: str = "ignore"
    ignore_label: str = "IGNORE"


@dataclass(frozen=True, slots=True)
class ChannelSelectSettings:
    """Channel selection by name and/or type."""

    names: tuple[str, ...] | None = (
        "Fpz-Cz",
        "Pz-Oz",
        "horizontal",
        "submental",
    )
    types: tuple[str, ...] | None = None
    require_all_names: bool = True


@dataclass(frozen=True, slots=True)
class FilterSettings:
    """Band-pass / notch filter settings."""

    enabled: bool = True
    l_freq: float | None = 0.5
    h_freq: float | None = 30.0
    notch_freqs: tuple[float, ...] = (50.0,)


@dataclass(frozen=True, slots=True)
class NormalizeSettings:
    """Per-recording normalization settings."""

    enabled: bool = True
    method: str = "zscore"
    eps: float = 1e-8


@dataclass(frozen=True, slots=True)
class PreprocessingSettings:
    """Settings for the composable preprocessing pipeline."""

    epoch_duration_sec: float = 30.0
    min_remainder_sec: float = 30.0
    wake_crop: WakeCropSettings = field(default_factory=WakeCropSettings)
    stage_map: StageMapSettings = field(default_factory=StageMapSettings)
    channels: ChannelSelectSettings = field(default_factory=ChannelSelectSettings)
    filter: FilterSettings = field(default_factory=FilterSettings)
    normalize: NormalizeSettings = field(default_factory=NormalizeSettings)


@dataclass(frozen=True, slots=True)
class RawEncodingSettings:
    """Pass-through raw waveform encoding options."""

    dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class WelchPSDSettings:
    """Welch PSD parameters for band-power encoding.

    Defaults target 100 Hz Sleep-EDF epochs (``nperseg=400`` → 4 s segments,
    frequency resolution ``fs / nfft = 0.25`` Hz). These are experimental
    starting points, not claimed optima.
    """

    nperseg: int = 400
    noverlap: int = 200
    nfft: int = 400
    window: str = "hamming"
    average: str = "median"
    detrend: str = "constant"
    scaling: str = "density"


@dataclass(frozen=True, slots=True)
class BandPowerEncodingSettings:
    """Per-channel band-power encoding options.

    Default five bands (delta…beta) with log-absolute + relative powers →
    shape ``(N, C, 10)``. Gamma is omitted because preprocessing low-passes
    at 30 Hz. Spectral ratios are opt-in and disabled by default.
    """

    method: str = "welch"
    include_log_absolute: bool = True
    include_relative: bool = True
    include_ratios: bool = False
    eps: float = 1e-10
    expected_sfreq: float = 100.0
    bands: tuple[tuple[str, float, float], ...] = (
        ("delta", 0.5, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 12.0),
        ("sigma", 12.0, 16.0),
        ("beta", 16.0, 30.0),
    )
    welch: WelchPSDSettings = field(default_factory=WelchPSDSettings)
    ratios: tuple[tuple[str, str, str], ...] = ()
    """Optional (name, numerator_band, denominator_band) triples; unused unless
    ``include_ratios`` is true."""


@dataclass(frozen=True, slots=True)
class STFTEncodingSettings:
    """STFT geometry / options for the time-frequency representation.

    Defaults target 100 Hz Sleep-EDF 30 s epochs: 256-sample Hann window
    (Δf ≈ 0.391 Hz), 100-sample hop (1 s), log-power in 0.5–30 Hz.
    """

    n_fft: int = 256
    hop_length: int = 100
    win_length: int | None = 256
    window: str = "hann"
    fmin: float = 0.5
    fmax: float = 30.0
    power: int = 2
    log_scale: bool = True
    eps: float = 1e-10
    expected_sfreq: float = 100.0
    dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class CWTEncodingSettings:
    """CWT geometry / options (placeholder for a later increment)."""

    wavelet: str = "morlet"
    fmin: float = 0.5
    fmax: float = 30.0
    n_freqs: int = 30
    output: str = "magnitude"
    log_scale: bool = True


@dataclass(frozen=True, slots=True)
class TimeFrequencyEncodingSettings:
    """Time-frequency representation with swappable backend."""

    method: str = "stft"  # stft | cwt
    stft: STFTEncodingSettings = field(default_factory=STFTEncodingSettings)
    cwt: CWTEncodingSettings = field(default_factory=CWTEncodingSettings)


@dataclass(frozen=True, slots=True)
class EncodingSettings:
    """Settings for Phase 3 encodings / representations.

    ``representation`` selects which encoder the factory builds:
    ``raw`` | ``bandpower`` | ``time_frequency``.
    """

    representation: str = "raw"
    ignore_label: str = "IGNORE"
    ignore_index: int = -100
    raw: RawEncodingSettings = field(default_factory=RawEncodingSettings)
    bandpower: BandPowerEncodingSettings = field(default_factory=BandPowerEncodingSettings)
    time_frequency: TimeFrequencyEncodingSettings = field(
        default_factory=TimeFrequencyEncodingSettings
    )


@dataclass(frozen=True, slots=True)
class SplitSettings:
    """Subject-wise split ratios and seed (shared across representations)."""

    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    cohort: str = "SC"
    channels: tuple[str, ...] = ("Fpz-Cz",)

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (self.train_ratio, self.val_ratio, self.test_ratio)


@dataclass(frozen=True, slots=True)
class TrainSettings:
    """Fixed Phase 4a training recipe (identical for all representations)."""

    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    ignore_index: int = -100
    drop_ignore_from_loader: bool = True
    primary_metric: str = "macro_f1"
    class_weighting: str = "balanced"
    early_stopping_patience: int | None = 5
    checkpoint_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Phase 4a baseline model options."""

    in_channels: int = 1
    n_classes: int = 5
    n_band_features: int = 10


@dataclass(frozen=True, slots=True)
class ClassicalBaselineSettings:
    """Single classical baseline: BandPower → LogisticRegression."""

    enabled: bool = True
    max_iter: int = 1000


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """Phase 4a experiment controls."""

    split: SplitSettings = field(default_factory=SplitSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    classical_baseline: ClassicalBaselineSettings = field(
        default_factory=ClassicalBaselineSettings
    )


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Top-level pipeline settings container."""

    acquisition: AcquisitionSettings
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    encodings: EncodingSettings = field(default_factory=EncodingSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    project_root: Path = field(default_factory=Path.cwd)


def _as_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Config section '{key}' must be a mapping.")
    return value


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"Expected a list of strings, got {type(value)!r}")
    return tuple(str(item) for item in value)


def _float_tuple(value: Any, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"Expected a list of numbers, got {type(value)!r}")
    return tuple(float(item) for item in value)


def _parse_wake_crop(wake_raw: Mapping[str, Any]) -> WakeCropSettings:
    enabled = bool(wake_raw.get("enabled", True))
    require_sleep = bool(wake_raw.get("require_sleep", True))

    if "buffer_sec" in wake_raw and wake_raw["buffer_sec"] is not None:
        buffer_sec = float(wake_raw["buffer_sec"])
    elif "minutes" in wake_raw and wake_raw["minutes"] is not None:
        buffer_sec = float(wake_raw["minutes"]) * 60.0
    else:
        buffer_sec = 1800.0

    return WakeCropSettings(
        enabled=enabled,
        buffer_sec=buffer_sec,
        require_sleep=require_sleep,
    )


def _parse_channels(channels_raw: Mapping[str, Any] | Sequence[Any]) -> ChannelSelectSettings:
    """Accept either a bare channel list or a mapping with names/types."""
    defaults = ChannelSelectSettings()

    if isinstance(channels_raw, Sequence) and not isinstance(channels_raw, (str, bytes, Mapping)):
        return ChannelSelectSettings(
            names=_optional_str_tuple(channels_raw),
            types=None,
            require_all_names=True,
        )

    if not isinstance(channels_raw, Mapping):
        raise ConfigurationError("preprocessing.channels must be a list or mapping")

    names = (
        _optional_str_tuple(channels_raw["names"])
        if "names" in channels_raw
        else defaults.names
    )
    types = (
        _optional_str_tuple(channels_raw["types"])
        if "types" in channels_raw
        else defaults.types
    )
    return ChannelSelectSettings(
        names=names,
        types=types,
        require_all_names=bool(channels_raw.get("require_all_names", True)),
    )


def _parse_stage_map(stage_raw: Mapping[str, Any]) -> StageMapSettings:
    # Prefer explicit unmapped_policy; fall back to legacy drop_unmapped flag.
    if "unmapped_policy" in stage_raw:
        policy = str(stage_raw["unmapped_policy"])
    elif "drop_unmapped" in stage_raw:
        policy = "drop" if bool(stage_raw["drop_unmapped"]) else "error"
    else:
        policy = "ignore"

    if policy not in {"ignore", "drop", "error"}:
        raise ConfigurationError(
            "stage_map.unmapped_policy must be 'ignore', 'drop', or 'error'"
        )

    return StageMapSettings(
        unmapped_policy=policy,
        ignore_label=str(stage_raw.get("ignore_label", "IGNORE")),
    )


def _load_preprocessing(raw: Mapping[str, Any]) -> PreprocessingSettings:
    wake_raw = _require_mapping(raw, "wake_crop")
    stage_raw = _require_mapping(raw, "stage_map")
    filter_raw = _require_mapping(raw, "filter")
    norm_raw = _require_mapping(raw, "normalize")

    channels_value = raw.get("channels", {})
    if channels_value is None:
        channels_value = {}

    method = str(norm_raw.get("method", "zscore"))
    if method not in {"zscore", "robust", "center"}:
        raise ConfigurationError(
            "normalize.method must be 'zscore', 'robust', or 'center'"
        )

    return PreprocessingSettings(
        epoch_duration_sec=float(raw.get("epoch_duration_sec", 30.0)),
        min_remainder_sec=float(raw.get("min_remainder_sec", 30.0)),
        wake_crop=_parse_wake_crop(wake_raw),
        stage_map=_parse_stage_map(stage_raw),
        channels=_parse_channels(channels_value),
        filter=FilterSettings(
            enabled=bool(filter_raw.get("enabled", True)),
            l_freq=(
                None
                if filter_raw.get("l_freq", 0.5) is None
                else float(filter_raw.get("l_freq", 0.5))
            ),
            h_freq=(
                None
                if filter_raw.get("h_freq", 30.0) is None
                else float(filter_raw.get("h_freq", 30.0))
            ),
            notch_freqs=_float_tuple(
                filter_raw.get("notch_freqs", [50.0]),
                default=(50.0,),
            ),
        ),
        normalize=NormalizeSettings(
            enabled=bool(norm_raw.get("enabled", True)),
            method=method,
            eps=float(norm_raw.get("eps", 1e-8)),
        ),
    )


def _parse_bands(value: Any) -> tuple[tuple[str, float, float], ...]:
    defaults = BandPowerEncodingSettings().bands
    if value is None:
        return defaults
    if isinstance(value, Mapping):
        return tuple((str(name), float(bounds[0]), float(bounds[1])) for name, bounds in value.items())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError("bandpower.bands must be a mapping or list")
    parsed: list[tuple[str, float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            parsed.append(
                (str(item["name"]), float(item["fmin"]), float(item["fmax"]))
            )
        elif isinstance(item, Sequence) and len(item) == 3:
            parsed.append((str(item[0]), float(item[1]), float(item[2])))
        else:
            raise ConfigurationError(f"Invalid band entry: {item!r}")
    return tuple(parsed) if parsed else defaults


def _parse_ratios(value: Any) -> tuple[tuple[str, str, str], ...]:
    """Parse optional spectral ratio definitions; empty when omitted."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError("bandpower.ratios must be a list")
    parsed: list[tuple[str, str, str]] = []
    for item in value:
        if isinstance(item, Mapping):
            parsed.append(
                (
                    str(item["name"]),
                    str(item["numerator"]),
                    str(item["denominator"]),
                )
            )
        elif isinstance(item, Sequence) and len(item) == 3:
            parsed.append((str(item[0]), str(item[1]), str(item[2])))
        else:
            raise ConfigurationError(f"Invalid ratio entry: {item!r}")
    return tuple(parsed)


def _load_encodings(raw: Mapping[str, Any]) -> EncodingSettings:
    representation = str(raw.get("representation", "raw"))
    if representation not in {"raw", "bandpower", "time_frequency"}:
        raise ConfigurationError(
            "encodings.representation must be 'raw', 'bandpower', or 'time_frequency'"
        )

    raw_cfg = _require_mapping(raw, "raw")
    bp_cfg = _require_mapping(raw, "bandpower")
    tf_cfg = _require_mapping(raw, "time_frequency")
    stft_cfg = _require_mapping(tf_cfg, "stft")
    cwt_cfg = _require_mapping(tf_cfg, "cwt")

    tf_method = str(tf_cfg.get("method", "stft"))
    if tf_method not in {"stft", "cwt"}:
        raise ConfigurationError("time_frequency.method must be 'stft' or 'cwt'")

    bp_method = str(bp_cfg.get("method", "welch"))
    if bp_method not in {"welch"}:
        raise ConfigurationError(
            "bandpower.method must be 'welch' (other PSD estimators are not implemented yet)"
        )

    welch_cfg = _require_mapping(bp_cfg, "welch")
    welch = WelchPSDSettings(
        nperseg=int(welch_cfg.get("nperseg", 400)),
        noverlap=int(welch_cfg.get("noverlap", 200)),
        nfft=int(welch_cfg.get("nfft", 400)),
        window=str(welch_cfg.get("window", "hamming")),
        average=str(welch_cfg.get("average", "median")),
        detrend=str(welch_cfg.get("detrend", "constant")),
        scaling=str(welch_cfg.get("scaling", "density")),
    )


    # Backward-compatible: legacy ``relative: true`` maps to include_relative.
    include_relative = bool(
        bp_cfg.get(
            "include_relative",
            bp_cfg.get("relative", True),
        )
    )
    include_log_absolute = bool(bp_cfg.get("include_log_absolute", True))
    if not include_log_absolute and not include_relative:
        raise ConfigurationError(
            "bandpower requires at least one of include_log_absolute / include_relative"
        )

    win_length_raw = stft_cfg.get("win_length", 256)
    win_length = None if win_length_raw is None else int(win_length_raw)

    return EncodingSettings(
        representation=representation,
        ignore_label=str(raw.get("ignore_label", "IGNORE")),
        ignore_index=int(raw.get("ignore_index", -100)),
        raw=RawEncodingSettings(dtype=str(raw_cfg.get("dtype", "float32"))),
        bandpower=BandPowerEncodingSettings(
            method=bp_method,
            include_log_absolute=include_log_absolute,
            include_relative=include_relative,
            include_ratios=bool(bp_cfg.get("include_ratios", False)),
            eps=float(bp_cfg.get("eps", 1e-10)),
            expected_sfreq=float(bp_cfg.get("expected_sfreq", 100.0)),
            bands=_parse_bands(bp_cfg.get("bands")),
            welch=welch,
            ratios=_parse_ratios(bp_cfg.get("ratios")),
        ),
        time_frequency=TimeFrequencyEncodingSettings(
            method=tf_method,
            stft=STFTEncodingSettings(
                n_fft=int(stft_cfg.get("n_fft", 256)),
                hop_length=int(stft_cfg.get("hop_length", 100)),
                win_length=win_length,
                window=str(stft_cfg.get("window", "hann")),
                fmin=float(stft_cfg.get("fmin", 0.5)),
                fmax=float(stft_cfg.get("fmax", 30.0)),
                power=int(stft_cfg.get("power", 2)),
                log_scale=bool(stft_cfg.get("log_scale", True)),
                eps=float(stft_cfg.get("eps", 1e-10)),
                expected_sfreq=float(stft_cfg.get("expected_sfreq", 100.0)),
                dtype=str(stft_cfg.get("dtype", "float32")),
            ),
            cwt=CWTEncodingSettings(
                wavelet=str(cwt_cfg.get("wavelet", "morlet")),
                fmin=float(cwt_cfg.get("fmin", 0.5)),
                fmax=float(cwt_cfg.get("fmax", 30.0)),
                n_freqs=int(cwt_cfg.get("n_freqs", 30)),
                output=str(cwt_cfg.get("output", "magnitude")),
                log_scale=bool(cwt_cfg.get("log_scale", True)),
            ),
        ),
    )

def _load_experiment(raw: Mapping[str, Any], *, project_root: Path) -> ExperimentSettings:
    """Load Phase 4a controlled experiment settings."""
    split_raw = _require_mapping(raw, "split")
    train_raw = _require_mapping(raw, "train")
    model_raw = _require_mapping(raw, "model")
    classical_raw = _require_mapping(raw, "classical_baseline")

    checkpoint_raw = train_raw.get("checkpoint_dir")
    checkpoint_dir = None
    if checkpoint_raw not in {None, ""}:
        checkpoint_dir = _as_path(str(checkpoint_raw), base=project_root)

    patience_raw = train_raw.get("early_stopping_patience", 5)
    patience = None if patience_raw is None else int(patience_raw)

    channels = split_raw.get("channels", ["Fpz-Cz"])
    return ExperimentSettings(
        split=SplitSettings(
            seed=int(split_raw.get("seed", 42)),
            train_ratio=float(split_raw.get("train_ratio", 0.7)),
            val_ratio=float(split_raw.get("val_ratio", 0.15)),
            test_ratio=float(split_raw.get("test_ratio", 0.15)),
            cohort=str(split_raw.get("cohort", "SC")),
            channels=tuple(str(ch) for ch in channels),
        ),
        train=TrainSettings(
            seed=int(train_raw.get("seed", 42)),
            batch_size=int(train_raw.get("batch_size", 32)),
            max_epochs=int(train_raw.get("max_epochs", 20)),
            learning_rate=float(train_raw.get("learning_rate", 1e-3)),
            weight_decay=float(train_raw.get("weight_decay", 0.0)),
            num_workers=int(train_raw.get("num_workers", 0)),
            ignore_index=int(train_raw.get("ignore_index", -100)),
            drop_ignore_from_loader=bool(train_raw.get("drop_ignore_from_loader", True)),
            primary_metric=str(train_raw.get("primary_metric", "macro_f1")),
            class_weighting=str(train_raw.get("class_weighting", "balanced")),
            early_stopping_patience=patience,
            checkpoint_dir=checkpoint_dir,
        ),
        model=ModelSettings(
            in_channels=int(model_raw.get("in_channels", 1)),
            n_classes=int(model_raw.get("n_classes", 5)),
            n_band_features=int(model_raw.get("n_band_features", 10)),
        ),
        classical_baseline=ClassicalBaselineSettings(
            enabled=bool(classical_raw.get("enabled", True)),
            max_iter=int(classical_raw.get("max_iter", 1000)),
        ),
    )


def load_settings(config_path: Path | str, *, project_root: Path | None = None) -> PipelineSettings:
    """Load pipeline settings from a YAML configuration file."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")

    if project_root is None:
        project_root = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    project_root = project_root.resolve()

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")

    acq_raw = _require_mapping(raw, "acquisition")
    log_raw = _require_mapping(raw, "logging")
    prep_raw = _require_mapping(raw, "preprocessing")
    enc_raw = _require_mapping(raw, "encodings")
    exp_raw = _require_mapping(raw, "experiment")

    data_root_value = acq_raw.get("data_root", "D:/SleepEDFX")
    acquisition = AcquisitionSettings(
        data_root=_as_path(str(data_root_value), base=project_root),
        preload=bool(acq_raw.get("preload", False)),
        stim_channel=str(acq_raw.get("stim_channel", "Event marker")),
        infer_types=bool(acq_raw.get("infer_types", True)),
        mne_verbose=str(acq_raw.get("mne_verbose", "ERROR")),
    )
    logging_settings = LoggingSettings(
        level=str(log_raw.get("level", "INFO")),
        format=str(
            log_raw.get(
                "format",
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            )
        ),
    )
    return PipelineSettings(
        acquisition=acquisition,
        preprocessing=_load_preprocessing(prep_raw),
        encodings=_load_encodings(enc_raw),
        experiment=_load_experiment(exp_raw, project_root=project_root),
        logging=logging_settings,
        project_root=project_root,
    )
