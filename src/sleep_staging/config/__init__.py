"""Configuration loading and typed settings."""

from sleep_staging.config.exceptions import ConfigurationError
from sleep_staging.config.settings import (
    AcquisitionSettings,
    BandPowerEncodingSettings,
    CWTEncodingSettings,
    ChannelSelectSettings,
    EncodingSettings,
    FilterSettings,
    LoggingSettings,
    NormalizeSettings,
    PipelineSettings,
    PreprocessingSettings,
    RawEncodingSettings,
    STFTEncodingSettings,
    StageMapSettings,
    TimeFrequencyEncodingSettings,
    WakeCropSettings,
    WelchPSDSettings,
    load_settings,
)

__all__ = [
    "AcquisitionSettings",
    "BandPowerEncodingSettings",
    "CWTEncodingSettings",
    "ChannelSelectSettings",
    "ConfigurationError",
    "EncodingSettings",
    "FilterSettings",
    "LoggingSettings",
    "NormalizeSettings",
    "PipelineSettings",
    "PreprocessingSettings",
    "RawEncodingSettings",
    "STFTEncodingSettings",
    "StageMapSettings",
    "TimeFrequencyEncodingSettings",
    "WakeCropSettings",
    "WelchPSDSettings",
    "load_settings",
]
