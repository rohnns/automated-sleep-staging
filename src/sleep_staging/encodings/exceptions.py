"""Encoding-related exceptions."""

from __future__ import annotations


class EncodingError(Exception):
    """Base exception for encoding failures."""


class EncoderNotImplementedError(EncodingError, NotImplementedError):
    """Raised when an encoder or backend algorithm is not yet implemented."""


class EpochExtractionError(EncodingError):
    """Raised when epochs cannot be sliced from a preprocessed recording."""


class ShapeValidationError(EncodingError):
    """Raised when an encoded tensor does not match the declared layout."""
