"""
Unified cache key generation utility for Urartu.

This module provides centralized cache key generation logic that ensures
consistent naming across actions and pipelines, enabling cross-pipeline
cache sharing.
"""

import hashlib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger
from urartu.common.action import CACHE_IGNORE_KEYS

logger = get_logger(__name__)


def make_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.
    
    Handles:
    - DictConfig and OmegaConf objects
    - Path objects
    - Sets (converted to sorted lists)
    - Other non-serializable types
    """
    if isinstance(obj, (DictConfig, dict)):
        if isinstance(obj, DictConfig):
            obj = OmegaConf.to_container(obj, resolve=True)
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return sorted([make_serializable(item) for item in obj])
    elif isinstance(obj, Path):
        # Use string representation for paths
        return str(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except Exception:
            return repr(obj)


def filter_config_for_cache(config: Dict[str, Any], ignore_keys: Optional[set] = None) -> Dict[str, Any]:
    """
    Filter configuration to remove keys that don't affect outputs.
    
    Args:
        config: Configuration dictionary to filter
        ignore_keys: Set of keys to ignore (defaults to CACHE_IGNORE_KEYS)
        
    Returns:
        Filtered configuration dictionary
    """
    if ignore_keys is None:
        ignore_keys = CACHE_IGNORE_KEYS
    
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    
    filtered = {k: v for k, v in config.items() if k not in ignore_keys}
    return filtered


class CacheKeyGenerator:
    """
    Unified cache key generator for actions and pipelines.
    
    Provides consistent cache key generation that enables cross-pipeline
    cache sharing when actions have identical configurations.
    """
    
    @staticmethod
    def generate_action_cache_key(
        action_name: str,
        config: Dict[str, Any],
        dependency_hash: Optional[str] = None
    ) -> str:
        """
        Generate a cache key for an action.
        
        Format: {action_name}_{config_hash}_{dependency_hash}
        
        Args:
            action_name: Name of the action
            config: Action configuration (will be filtered)
            dependency_hash: Optional hash of dependencies (for dependent actions)
            
        Returns:
            Cache key string
        """
        # Filter config to remove non-output-affecting keys
        filtered_config = filter_config_for_cache(config)
        serializable_config = make_serializable(filtered_config)
        
        # Build key factors
        key_factors = {
            'action_name': action_name,
            'config': serializable_config
        }
        
        # Add dependency hash if provided
        if dependency_hash:
            key_factors['dependency_hash'] = dependency_hash
        
        # Convert to JSON for consistent ordering
        key_string = json.dumps(key_factors, sort_keys=True)
        
        # Generate hash
        config_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        # Build final cache key
        if dependency_hash:
            cache_key = f"{action_name}_{config_hash}_{dependency_hash[:8]}"
        else:
            cache_key = f"{action_name}_{config_hash}"
        
        logger.debug(f"Generated cache key for {action_name}: {cache_key}")
        logger.debug(f"  Config keys: {list(serializable_config.keys()) if isinstance(serializable_config, dict) else 'N/A'}")
        
        return cache_key
    
    @staticmethod
    def generate_pipeline_action_cache_key(
        action_name: str,
        config_overrides: Dict[str, Any],
        previous_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
        pipeline_config_hash: Optional[str] = None
    ) -> str:
        """
        Generate a cache key for an action within a pipeline.
        
        For actions with no dependencies, uses simplified key for cross-pipeline sharing.
        For dependent actions, includes pipeline context.
        
        Args:
            action_name: Name of the action
            config_overrides: Action-specific configuration overrides
            previous_outputs: Outputs from previous actions (if any)
            pipeline_config_hash: Hash of pipeline configuration (if any)
            
        Returns:
            Cache key string
        """
        # Filter config
        filtered_config = filter_config_for_cache(config_overrides)
        serializable_config = make_serializable(filtered_config)
        
        # Determine if this action has dependencies
        has_dependencies = previous_outputs is not None and len(previous_outputs) > 0
        
        if not has_dependencies:
            # First action or no dependencies - use simplified key for cross-pipeline sharing
            key_factors = {
                'action_name': action_name,
                'config': serializable_config
            }
        else:
            # Action depends on previous outputs - include pipeline context
            serializable_outputs = make_serializable(previous_outputs)
            key_factors = {
                'action_name': action_name,
                'config_overrides': serializable_config,
                'previous_outputs': serializable_outputs
            }
            
            # Include pipeline config hash if provided
            if pipeline_config_hash:
                key_factors['pipeline_config_hash'] = pipeline_config_hash
        
        # Convert to JSON for consistent ordering
        key_string = json.dumps(key_factors, sort_keys=True)
        
        # Generate hash
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        final_key = f"{action_name}_{cache_key}"
        
        logger.debug(f"Generated pipeline action cache key: {final_key}")
        logger.debug(f"  Has dependencies: {has_dependencies}")
        logger.debug(f"  Config keys: {list(serializable_config.keys()) if isinstance(serializable_config, dict) else 'N/A'}")
        
        return final_key
    
    @staticmethod
    def generate_dependency_hash(previous_outputs: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a hash of previous action outputs for dependency tracking.
        
        Args:
            previous_outputs: Dictionary mapping action names to their outputs
            
        Returns:
            Hash string
        """
        serializable_outputs = make_serializable(previous_outputs)
        key_string = json.dumps(serializable_outputs, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:12]

