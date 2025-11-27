"""
Cache management for pipeline actions.

This module handles saving and loading cached action outputs.
"""

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from .types import ActionOutput, CacheEntry

logger = logging.getLogger(__name__)


class PipelineCache:
    """Manages caching for pipeline actions."""

    def __init__(self, cache_dir: Path, cache_enabled: bool = True, force_rerun: bool = False, cache_max_age: Optional[int] = None):
        """
        Initialize the pipeline cache.

        Args:
            cache_dir: Directory where cache entries are stored
            cache_enabled: Whether caching is enabled
            force_rerun: Whether to force rerun (skip cache)
            cache_max_age: Maximum age of cache entries in seconds
        """
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.force_rerun = force_rerun
        self.cache_max_age = cache_max_age

    def get_cache_path(self, cache_key: str) -> Path:
        """
        Get the directory path for a cache entry.

        Structure: cache/{action_name}/{cache_hash}/

        Args:
            cache_key: Cache key in format "{action_name}_{hash}"

        Returns:
            Path to the cache directory for this entry
        """
        # Extract action name and hash from cache key
        # Format: {action_name}_{hash}
        if '_' in cache_key:
            parts = cache_key.rsplit('_', 1)
            action_name = parts[0]
            cache_hash = parts[1] if len(parts) > 1 else cache_key
        else:
            # Fallback: use cache_key as both name and hash
            action_name = cache_key
            cache_hash = cache_key

        # Create hierarchical structure: cache/{action_name}/{cache_hash}/
        cache_entry_dir = self.cache_dir / action_name / cache_hash
        return cache_entry_dir

    def get_cache_file_path(self, cache_key: str) -> Path:
        """Get the pickle file path within a cache entry directory."""
        cache_dir = self.get_cache_path(cache_key)
        return cache_dir / "cache.pkl"

    def get_cache_metadata_path(self, cache_key: str) -> Path:
        """Get the metadata YAML file path within a cache entry directory."""
        cache_dir = self.get_cache_path(cache_key)
        return cache_dir / "metadata.yaml"

    def save_to_cache(
        self, cache_key: str, action_output: ActionOutput, config_hash: str, pipeline_action=None, resolve_value_func=None, get_config_hash_func=None
    ):
        """
        Save action output to cache.

        Args:
            cache_key: Unique cache key for this action output
            action_output: The action output to cache
            config_hash: Hash of the action configuration
            pipeline_action: Optional pipeline action (for metadata)
            resolve_value_func: Optional function to resolve config values
            get_config_hash_func: Optional function to get full config hash
        """
        if not self.cache_enabled:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Get cache directory for this entry
            cache_entry_dir = self.get_cache_path(cache_key)
            cache_file_path = self.get_cache_file_path(cache_key)
            metadata_path = self.get_cache_metadata_path(cache_key)

            # Create cache entry directory
            cache_entry_dir.mkdir(parents=True, exist_ok=True)

            cache_entry = CacheEntry(
                action_output=action_output,
                cache_key=cache_key,
                timestamp=time.time(),
                config_hash=config_hash,
                action_version=None,  # Could add git hash or file modification time
            )

            # Save cache entry to pickle file
            with open(cache_file_path, 'wb') as f:
                pickle.dump(cache_entry, f)

            # Also save a human-readable metadata file
            full_config = {}
            if pipeline_action and resolve_value_func:
                # Include the resolved config overrides
                resolved_config_overrides = resolve_value_func(pipeline_action.config_overrides)
                if hasattr(resolved_config_overrides, 'items'):
                    resolved_config_overrides = (
                        OmegaConf.to_container(resolved_config_overrides, resolve=True)
                        if OmegaConf.is_config(resolved_config_overrides)
                        else dict(resolved_config_overrides)
                    )
                    full_config = resolved_config_overrides

            metadata = {
                'cache_key': cache_key,
                'action_name': action_output.action_name,
                'timestamp': datetime.fromtimestamp(cache_entry.timestamp).isoformat(),
                'config_hash': config_hash,
                'outputs_keys': list(action_output.outputs.keys()),
                'full_config': full_config,  # Add full config content
            }

            if get_config_hash_func:
                metadata['full_config_hash'] = get_config_hash_func()

            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, indent=2, sort_keys=False)

            logger.info(f"Cached output for action '{action_output.name}' with key {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")

    def load_from_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """
        Load action output from cache if available and valid.

        Args:
            cache_key: The cache key to load

        Returns:
            CacheEntry if found and valid, None otherwise
        """
        if not self.cache_enabled:
            logger.info(f"Cache is disabled, skipping cache lookup for {cache_key}")
            return None

        if self.force_rerun:
            logger.info(f"Force rerun is enabled, skipping cache lookup for {cache_key}")
            return None

        cache_file_path = self.get_cache_file_path(cache_key)
        cache_entry_dir = self.get_cache_path(cache_key)
        logger.info(f"🔍 Checking cache directory: {cache_entry_dir}")
        logger.info(f"📂 Cache file exists: {cache_file_path.exists()}")

        if not cache_file_path.exists():
            logger.info(f"❌ Cache file not found: {cache_file_path}")
            return None

        try:
            logger.info(f"📖 Attempting to load cache from: {cache_file_path}")
            with open(cache_file_path, 'rb') as f:
                cache_entry = pickle.load(f)
            logger.info(f"✅ Successfully loaded cache entry for {cache_key}")

            # Check if cache is still valid
            if self.cache_max_age is not None:
                logger.info(f"⏰ Checking cache age (max_age: {self.cache_max_age}s)")

            if cache_entry.is_valid(self.cache_max_age):
                logger.info(f"✅ Found valid cache for key {cache_key}")
                return cache_entry
            else:
                logger.info(f"❌ Cache for key {cache_key} is expired")
                return None

        except Exception as e:
            logger.error(f"💥 Failed to load cache for {cache_key}: {e}")
            import traceback

            logger.error(f"💥 Cache loading error details:\n{traceback.format_exc()}")
            return None

    def clear_cache(self):
        """Clear all cached outputs."""
        if self.cache_dir.exists():
            import shutil

            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared pipeline cache at {self.cache_dir}")

    @staticmethod
    def generate_config_hash(cfg: DictConfig) -> str:
        """
        Generate a hash of the complete configuration to ensure cache invalidation
        when config files are modified.

        Args:
            cfg: The configuration to hash

        Returns:
            A hash string representing the complete configuration
        """
        from urartu.utils.cache import make_serializable

        # Include the complete configuration in the hash
        config_for_hash = make_serializable(cfg)

        # Convert to JSON for consistent ordering
        config_string = json.dumps(config_for_hash, sort_keys=True)

        # Generate hash
        config_hash = hashlib.sha256(config_string.encode()).hexdigest()[:12]

        return config_hash
