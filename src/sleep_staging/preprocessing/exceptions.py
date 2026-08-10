"""Preprocessing-related exceptions."""

from __future__ import annotations


class PreprocessingError(Exception):
    """Base exception for preprocessing failures."""


class TransformError(PreprocessingError):
    """Raised when a single transform cannot be applied."""


class MissingChannelsError(TransformError):
    """Raised when requested channels are not present on the recording."""


class NoSleepBoundaryError(TransformError):
    """Raised when sleep onset/offset cannot be determined and cropping requires them."""
