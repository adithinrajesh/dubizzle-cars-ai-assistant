"""Persistent application state, independent of provider transcripts."""

from .repository import SQLiteState, StateError

__all__ = ["SQLiteState", "StateError"]
