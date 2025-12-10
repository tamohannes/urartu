"""
Unified cache key generation utility for Urartu.

This module provides centralized cache key generation logic that ensures
consistent naming across actions and pipelines, enabling cross-pipeline
cache sharing.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from urartu.common.action import CACHE_IGNORE_KEYS
from urartu.utils.logging import get_logger

logger = get_logger(__name__)


def make_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.

    Handles:
    - DictConfig and OmegaConf objects
    - Path objects (normalized to portable paths if within .runs)
    - Sets (converted to sorted lists)
    - Other non-serializable types

    CRITICAL: Paths within .runs are normalized to portable format to ensure
    consistent cache keys across different mount points or working directories.
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
        # Normalize paths to portable format if they're within .runs
        # This ensures consistent cache keys regardless of where .runs is mounted
        # Use Urartu's _make_path_portable function for consistent normalization
        from urartu.common.action import Action

        path_str = str(obj)
        portable_path = Action._make_path_portable(path_str)
        return portable_path
    elif isinstance(obj, str):
        # Check if string is a path that should be normalized
        # Use Urartu's _make_path_portable function for consistent normalization
        from urartu.common.action import Action

        if obj and (
            obj.startswith('.runs/')
            or obj.startswith('.runs\\')
            or '/.runs/' in obj
            or '\\.runs\\' in obj
            or obj.startswith('/')
            or (len(obj) > 2 and obj[1] == ':')
        ):
            portable_path = Action._make_path_portable(obj)
            return portable_path
        return obj
    elif isinstance(obj, (int, float, bool, type(None))):
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

    def filter_recursive(obj: Any) -> Any:
        """Recursively filter config, removing 'hash' fields from nested dicts."""
        if isinstance(obj, dict):
            filtered = {}
            for k, v in obj.items():
                # Skip top-level ignored keys
                if k in ignore_keys:
                    continue
                # Skip 'hash' fields in nested dicts (e.g., dataset.hash) - these are derived fields
                # that depend on the config itself, so including them would cause cache mismatches
                if k == 'hash':
                    continue
                # Recursively filter nested structures
                filtered[k] = filter_recursive(v)
            return filtered
        elif isinstance(obj, list):
            return [filter_recursive(item) for item in obj]
        else:
            return obj

    filtered = filter_recursive(config)
    return filtered


class CacheKeyGenerator:
    """
    Unified cache key generator for actions and pipelines.

    Provides consistent cache key generation that enables cross-pipeline
    cache sharing when actions have identical configurations.
    """

    @staticmethod
    def generate_action_cache_key(action_name: str, config: Dict[str, Any], dependency_hash: Optional[str] = None) -> str:
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

        # Check if pipeline-specific fields leaked through after filtering
        pipeline_specific_in_filtered = [
            k for k in serializable_config.keys() if k in ['experiment_name', 'pipeline_name', 'pipeline_id', 'pipeline_config_hash']
        ]
        if pipeline_specific_in_filtered:
            logger.error(f"❌ CRITICAL: Pipeline-specific fields found in filtered config for {action_name}: {pipeline_specific_in_filtered}")
            logger.error(f"   These should have been filtered out by CACHE_IGNORE_KEYS!")

        # Build key factors
        key_factors = {'action_name': action_name, 'config': serializable_config}

        # Add dependency hash if provided
        if dependency_hash:
            key_factors['dependency_hash'] = dependency_hash

        # Convert to JSON for consistent ordering
        key_string = json.dumps(key_factors, sort_keys=True)

        # Debug: Log the actual config values being hashed (first iteration only to avoid spam)
        if logger.isEnabledFor(logging.DEBUG):
            config_str = json.dumps(serializable_config, sort_keys=True)
            logger.debug(f"  Config JSON (first 500 chars): {config_str[:500]}...")
            logger.debug(f"  Full key_string hash input (first 500 chars): {key_string[:500]}...")

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
        pipeline_config_hash: Optional[str] = None,
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
            key_factors = {'action_name': action_name, 'config': serializable_config}
        else:
            # Action depends on previous outputs - include pipeline context
            serializable_outputs = make_serializable(previous_outputs)
            key_factors = {'action_name': action_name, 'config_overrides': serializable_config, 'previous_outputs': serializable_outputs}

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
