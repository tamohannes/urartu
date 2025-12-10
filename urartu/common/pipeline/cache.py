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
        self,
        cache_key: str,
        action_output: ActionOutput,
        config_hash: str,
        pipeline_action=None,
        resolve_value_func=None,
        get_config_hash_func=None,
        pipeline_common_configs=None,
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
            # CRITICAL: Use the same config that's used for cache key generation
            # This should be the filtered, serializable config (same as _get_serializable_config())
            # to ensure cache matching works correctly across pipelines
            # The config should include pipeline_common_configs merged in, then filtered
            full_config = {}
            try:
                if pipeline_action and resolve_value_func:
                    # Get resolved config overrides (action-specific config with dependencies injected)
                    resolved_config_overrides = resolve_value_func(pipeline_action.config_overrides)
                    if hasattr(resolved_config_overrides, 'items'):
                        resolved_config_overrides = (
                            OmegaConf.to_container(resolved_config_overrides, resolve=True)
                            if OmegaConf.is_config(resolved_config_overrides)
                            else dict(resolved_config_overrides)
                        )

                        # Merge with pipeline_common_configs if provided (same as _run_action does)
                        # This ensures full_config matches what's used for cache key generation
                        if pipeline_common_configs:
                            from omegaconf import OmegaConf

                            merged_config = OmegaConf.merge(
                                OmegaConf.create(pipeline_common_configs),  # Base (pipeline defaults)
                                OmegaConf.create(resolved_config_overrides),  # Override (action-specific)
                            )
                            merged_config_dict = OmegaConf.to_container(merged_config, resolve=True)
                        else:
                            merged_config_dict = resolved_config_overrides

                        # Filter the config the same way _get_serializable_config() does
                        from urartu.utils.cache import filter_config_for_cache, make_serializable

                        filtered_config = filter_config_for_cache(merged_config_dict)
                        full_config = make_serializable(filtered_config)
            except Exception as e:
                logger.debug(f"Could not get filtered config for metadata: {e}")
                # Fallback to resolved_config_overrides if filtering fails
                if pipeline_action and resolve_value_func:
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
    def generate_config_hash(cfg: DictConfig, action_name: Optional[str] = None, loop_configs: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a hash of the complete configuration to ensure cache invalidation
        when config files are modified.

        This method normalizes the config to ignore pipeline structure differences
        (e.g., single vs multiple loopable_actions blocks) so that the same action
        configs produce the same hash across different pipeline structures.

        If action_name and loop_configs are provided, only includes actions that match
        those criteria, enabling cross-pipeline cache sharing for the same action.

        Args:
            cfg: The configuration to hash
            action_name: Optional action name to filter actions (for cross-pipeline sharing)
            loop_configs: Optional loop_configs dict to filter actions (for cross-pipeline sharing)

        Returns:
            A hash string representing the complete configuration
        """
        from urartu.utils.cache import make_serializable

        # Normalize config to extract only action-relevant parts
        # This ensures same action configs produce same hash regardless of pipeline structure
        normalized_config = PipelineCache._normalize_config_for_hash(cfg, action_name=action_name, loop_configs=loop_configs)

        # Include the normalized configuration in the hash
        config_for_hash = make_serializable(normalized_config)

        # Convert to JSON for consistent ordering
        config_string = json.dumps(config_for_hash, sort_keys=True)

        # Generate hash
        config_hash = hashlib.sha256(config_string.encode()).hexdigest()[:12]

        return config_hash

    @staticmethod
    def _normalize_config_for_hash(
        cfg: DictConfig, action_name: Optional[str] = None, loop_configs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize config to extract only action-relevant parts, ignoring pipeline structure.

        This extracts:
        - Pipeline-level settings (experiment_name, seed, device, etc.)
        - Model config (shared model configuration)
        - Action configs (flattened from all loopable blocks)

        If action_name and loop_configs are provided, only includes actions that match,
        enabling cross-pipeline cache sharing for the same action.

        This ensures that the same action configs produce the same hash regardless of
        whether they're in a single loopable_actions block or multiple blocks.

        Args:
            cfg: The configuration to normalize
            action_name: Optional action name to filter actions (for cross-pipeline sharing)
            loop_configs: Optional loop_configs dict to filter actions (for cross-pipeline sharing)

        Returns:
            Normalized config dict with action-relevant parts only
        """
        normalized = {}

        # Extract pipeline-level settings (if present)
        if 'pipeline' in cfg:
            pipeline_cfg = OmegaConf.to_container(cfg.pipeline, resolve=False)
            if isinstance(pipeline_cfg, dict):
                # Extract only non-structural settings that affect action outputs
                # Exclude metadata fields like experiment_name, pipeline_name that don't affect outputs
                # This enables cross-pipeline cache sharing for the same action configs
                for key in ['seed', 'device', 'cache_enabled', 'force_rerun', 'cache_max_age_days', 'memory_management']:
                    if key in pipeline_cfg:
                        normalized[key] = pipeline_cfg[key]

        # Extract model config (shared across actions)
        if 'model' in cfg:
            normalized['model'] = OmegaConf.to_container(cfg.model, resolve=False)

        # Extract and normalize action configs
        # This flattens all actions from all loopable blocks into a single list
        normalized_actions = []

        if 'pipeline' in cfg and 'actions' in cfg.pipeline:
            actions_list = cfg.pipeline.actions
            for action_cfg in actions_list:
                if 'action_name' in action_cfg:
                    # Regular action
                    action_dict = OmegaConf.to_container(action_cfg, resolve=False)
                    # Remove structural keys that don't affect execution
                    if isinstance(action_dict, dict):
                        # Filter: if action_name is specified, only include matching actions
                        if action_name is not None:
                            if action_dict.get('action_name') != action_name:
                                continue
                        action_dict.pop('depends_on', None)  # Dependencies are resolved before execution
                        normalized_actions.append(action_dict)
                elif 'loopable_actions' in action_cfg:
                    # Loopable actions block - extract actions from within
                    loopable_cfg = action_cfg.loopable_actions
                    if 'actions' in loopable_cfg:
                        for loop_action in loopable_cfg.actions:
                            if 'action_name' in loop_action:
                                action_dict = OmegaConf.to_container(loop_action, resolve=False)
                                # Include loop_configs to preserve loop context injection
                                if isinstance(action_dict, dict):
                                    # Include loop_configs if present (affects how loop context is injected)
                                    current_loop_configs = None
                                    if 'loop_configs' in loopable_cfg:
                                        current_loop_configs = OmegaConf.to_container(loopable_cfg.loop_configs, resolve=False)
                                        action_dict['_loop_configs'] = current_loop_configs

                                    # Filter: if action_name and loop_configs are specified, only include matching actions
                                    if action_name is not None:
                                        if action_dict.get('action_name') != action_name:
                                            continue
                                        # If loop_configs is specified, only include actions with matching loop_configs
                                        if loop_configs is not None:
                                            if current_loop_configs != loop_configs:
                                                continue

                                    # Remove structural keys
                                    action_dict.pop('depends_on', None)
                                    normalized_actions.append(action_dict)

        # Sort actions by action_name for consistent ordering
        normalized_actions.sort(key=lambda x: x.get('action_name', ''))
        normalized['actions'] = normalized_actions

        return normalized
