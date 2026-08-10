"""Exceptions raised by the data acquisition layer."""

from __future__ import annotations


class AcquisitionError(Exception):
    """Base exception for all acquisition failures."""


class MissingPSGFileError(AcquisitionError):
    """Raised when a required PSG EDF file cannot be found."""


class MissingHypnogramFileError(AcquisitionError):
    """Raised when a required hypnogram annotation file cannot be found."""


class InvalidEDFFileError(AcquisitionError):
    """Raised when an EDF file exists but cannot be parsed by MNE."""


class MetadataExtractionError(AcquisitionError):
    """Raised when recording metadata cannot be extracted or parsed."""


class RecordingValidationError(AcquisitionError):
    """Raised when a loaded recording fails structural validation."""
