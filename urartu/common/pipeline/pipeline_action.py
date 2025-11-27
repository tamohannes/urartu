"""
Pipeline action representation.

This module contains the PipelineAction class which represents a single action
within a pipeline.
"""

from typing import Any, Callable, Dict, List, Optional


class PipelineAction:
    """Represents a single action in the pipeline."""

    def __init__(
        self,
        name: str,
        action_name: str,
        config_overrides: Optional[Dict] = None,
        outputs_to_track: Optional[List[str]] = None,
        condition: Optional[Callable] = None,
    ):
        """
        Initialize a pipeline action.

        Args:
            name: Unique name for this action
            action_name: Name of the action to run
            config_overrides: Configuration overrides for this action
            outputs_to_track: List of output keys to track from this action
            condition: Optional callable that determines if action should run
        """
        self.name = name
        self.action_name = action_name
        self.config_overrides = config_overrides or {}
        self.outputs_to_track = outputs_to_track or []
        self.condition = condition

    def should_run(self, context: Dict[str, Any]) -> bool:
        """Check if this action should run based on its condition."""
        if self.condition is None:
            return True
        return self.condition(context)
