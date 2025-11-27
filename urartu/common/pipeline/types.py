"""
Type definitions for pipeline components.

This module contains dataclasses and type definitions used throughout the pipeline system.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ActionOutput:
    """Container for action outputs."""

    name: str
    action_name: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Container for cached action outputs."""

    action_output: ActionOutput
    cache_key: str
    timestamp: float
    config_hash: str
    action_version: Optional[str] = None

    def is_valid(self, max_age_seconds: Optional[int] = None) -> bool:
        """Check if cache entry is still valid."""
        if max_age_seconds is not None:
            age = time.time() - self.timestamp
            return age < max_age_seconds
        return True
