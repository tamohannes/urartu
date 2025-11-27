"""
Pipeline module for orchestrating sequences of actions.

This module provides a flexible Pipeline class that can run multiple actions in sequence,
manage data flow between steps, and handle configuration overrides.

The module is organized into submodules:
- types: Type definitions (ActionOutput, CacheEntry)
- resolvers: Data resolvers for configuration references
- action: PipelineAction class
- config: Configuration injection utilities
- cache: Cache management
- status: Status display utilities
"""

from .loopable_pipeline import LoopablePipeline

# Import Pipeline class from pipeline.py
from .pipeline import Pipeline
from .pipeline_action import PipelineAction
from .resolvers import ActionOutputResolver, DataResolver, LoopContextResolver

# Import main classes for backward compatibility
from .types import ActionOutput, CacheEntry

__all__ = [
    'Pipeline',
    'LoopablePipeline',
    'PipelineAction',
    'ActionOutput',
    'CacheEntry',
    'DataResolver',
    'ActionOutputResolver',
    'LoopContextResolver',
]
