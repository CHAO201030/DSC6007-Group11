"""Shared contracts and utilities for the project.

Keep this package small and stable: every other package depends on it, so
interface changes should be reviewed by the whole group.
"""

from .interfaces import Partner, Policy

__all__ = ["Policy", "Partner"]
