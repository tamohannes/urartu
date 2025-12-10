"""
Pipeline Stage Data Structures

This module defines the data structures for representing pipeline stages.
A pipeline stage can be either a single action or a loopable block of actions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineStage:
    """
    Base class for pipeline stages.
    
    A stage represents a unit of work in a pipeline that can be executed.
    """
    stage_idx: int  # Index of this stage in the pipeline
    stage_type: str = ""  # "action" or "loopable" (set by subclasses in __post_init__)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stage_idx={self.stage_idx}, type={self.stage_type})"


@dataclass
class ActionStage(PipelineStage):
    """
    A stage representing a single action to be run once.
    
    Attributes:
        action_name: Name of the action
        action_config: Configuration overrides for the action
        outputs_to_track: List of output keys to track from this action
    """
    action_name: str = ""
    action_config: Dict[str, Any] = field(default_factory=dict)
    outputs_to_track: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.stage_type = "action"
    
    def __repr__(self) -> str:
        return f"ActionStage(idx={self.stage_idx}, action={self.action_name})"


@dataclass
class LoopableStage(PipelineStage):
    """
    A stage representing a block of actions to be run multiple times with different loop contexts.
    
    Attributes:
        loop_configs: Mapping from loop variable names to their injection paths
        loop_iterations: List of loop contexts (each context is a dict of loop variables)
        loopable_actions: List of action names in this loopable block
        action_configs: Dict mapping action names to their config overrides
    """
    loop_configs: Dict[str, str] = field(default_factory=dict)
    loop_iterations: List[Dict[str, Any]] = field(default_factory=list)
    loopable_actions: List[str] = field(default_factory=list)
    action_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.stage_type = "loopable"
    
    def __repr__(self) -> str:
        return f"LoopableStage(idx={self.stage_idx}, actions={self.loopable_actions}, iterations={len(self.loop_iterations)})"
    
    def get_num_iterations(self) -> int:
        """Return the number of iterations in this loopable block."""
        return len(self.loop_iterations)
    
    def get_num_actions(self) -> int:
        """Return the number of actions in this loopable block."""
        return len(self.loopable_actions)

