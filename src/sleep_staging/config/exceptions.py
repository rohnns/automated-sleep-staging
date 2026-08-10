"""Configuration-related exceptions."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Raised when pipeline configuration is missing or invalid."""
