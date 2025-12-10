import gc
import hashlib
import json
import pickle
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from aim import Run
from omegaconf import DictConfig, OmegaConf

from urartu.utils.hash import dict_to_8char_hash
from urartu.utils.logging import get_logger

from .device import Device

logger = get_logger(__name__)


# Keys in the action configuration that do NOT influence the produced outputs
# and therefore should be ignored when building cache keys. This enables cross-pipeline
# cache sharing when the same action has identical core configs but different pipeline contexts.
CACHE_IGNORE_KEYS = {
    # Cache-related settings
    "cache_enabled",
    "force_rerun",
    "cache_max_age_hours",
    "cache_max_age_days",
    "cache_max_age",
    # Memory management settings (don't affect outputs)
    "memory_management",
    "auto_cleanup",
    "force_cpu_offload",
    "aggressive_gc",
    # Pipeline-level settings that get merged but don't affect action outputs
    "experiment_name",
    "debug",
    # Dependency declarations (these are resolved before action runs)
    "depends_on",
    # Pipeline metadata
    "pipeline_id",
    "pipeline_name",
    "pipeline_config_hash",  # Pipeline-specific hash, shouldn't affect cross-pipeline cache sharing
    # Runtime/execution settings that don't affect outputs
    "action_name",  # This is metadata, not part of action logic
    "run_dir",  # Runtime directory, doesn't affect outputs
    "device",  # Device doesn't affect outputs, just where computation happens
    "iteration_id",  # Iteration identifier is only for organizing outputs, not for cache key
    # Tracking settings (don't affect computation)
    "outputs_to_track",  # Only affects what gets tracked, not what gets computed
    # Aim/logging settings
    "aim",
    "use_aim",
    "aim_run",
    "aim_repo",
    # Stage/block metadata (these are framework internals, not action inputs)
    "_stage_idx",
    "_block_idx",
    "_resume_at_stage",
    "_post_loopable_only",
    "_submit_array_only",
    "_is_iteration_task",
    "_iteration_idx",
    "_loop_iterations_dir",
    "_iteration_job_ids",
    "_loopable_stage",
}


class Action(ABC):
    """
    A class to manage and configure actions based on a configuration and an Aim run session.

    This class initializes an action with specific configurations for task execution,
    sets up device configuration for the action, and ties the action to an Aim run session
    for tracking and logging purposes.

    Attributes:
        cfg (DictConfig): The full configuration object, typically containing all settings.
        action_config (DictConfig): Configuration specific to the action.
        aim_run (Run): An Aim run instance for tracking and logging the execution of the action.
    """

    def __init__(self, cfg: DictConfig, aim_run: Run):
        """
        Initializes the Action object with the necessary configuration and Aim session.

        Args:
            cfg (DictConfig): The configuration object providing settings for various components of the action.
            aim_run (Run): The Aim run session to track and manage execution metrics and logs.

        Sets up the device configuration for the action by calling an external method set_device
        from the Device class with the device setting specified in the action configuration.
        """
        self.cfg = cfg
        # action_name is always the identifier
        # action_config contains the actual configuration for regular actions
        # pipeline_config contains the actual configuration for pipelines

        # Debug: Check what cfg.pipeline contains at the very start
        if hasattr(cfg, 'pipeline'):
            try:
                pipeline_keys_at_start = list(cfg.pipeline.keys()) if hasattr(cfg.pipeline, 'keys') else []
                pipeline_actions_at_start = len(cfg.pipeline.actions) if 'actions' in cfg.pipeline else 0
                logger.info(
                    f"Action.__init__(): At START, cfg.pipeline has {len(pipeline_keys_at_start)} keys: {pipeline_keys_at_start}, {pipeline_actions_at_start} actions"
                )
            except Exception as e:
                logger.warning(f"Action.__init__(): Could not check cfg.pipeline at start: {e}")

        # Priority: action > pipeline > cfg
        # When an action runs inside a pipeline, action is specifically created for it
        # Use direct access with safe checks to avoid triggering resolution of ??? values
        try:
            # Check what we have
            has_action = False
            has_pipeline = False
            try:
                has_action = hasattr(cfg, 'action') and cfg.action and cfg.action != '???'
                # If action is a dict/config (not a string), it's from defaults, not an entity name
                if has_action and not isinstance(cfg.action, (str, type(None))):
                    # It's a config dict, check if it's actually empty or just defaults
                    if isinstance(cfg.action, dict) and len(cfg.action) == 0:
                        has_action = False
            except Exception:
                pass
            try:
                has_pipeline = hasattr(cfg, 'pipeline') and cfg.pipeline and cfg.pipeline != '???'
            except Exception:
                pass

            logger.info(f"Action.__init__(): has_action={has_action}, has_pipeline={has_pipeline}")

            # Try direct access first (faster and preserves structure)
            if has_action and not has_pipeline:
                # This is a regular action or an action within a pipeline - use action
                # Make a copy to preserve structure (especially for pipelines with actions)
                self.action_config = OmegaConf.create(OmegaConf.to_container(cfg.action, resolve=False))
            elif has_pipeline:
                # This is a pipeline itself - use pipeline
                # IMPORTANT: Access cfg.pipeline immediately and convert to container BEFORE any other access
                # This prevents OmegaConf from resolving and potentially losing the actions list
                try:
                    # Get the pipeline config as a container (dict) without resolving
                    # Do this FIRST before any other access to cfg.pipeline
                    pipeline_container = OmegaConf.to_container(cfg.pipeline, resolve=False)
                    # Verify we got the actions
                    if isinstance(pipeline_container, dict):
                        logger.info(f"Action.__init__(): pipeline_container is dict with keys: {list(pipeline_container.keys())}")
                        if 'actions' in pipeline_container:
                            logger.info(f"Action.__init__(): Found {len(pipeline_container['actions'])} actions in pipeline container")
                        else:
                            logger.error(
                                f"Action.__init__(): pipeline_container is missing 'actions' key! Available keys: {list(pipeline_container.keys())}"
                            )
                    # Create a new DictConfig from the container to preserve all keys including actions
                    self.action_config = OmegaConf.create(pipeline_container)
                    # Debug: verify actions were preserved
                    final_keys = list(self.action_config.keys()) if hasattr(self.action_config, 'keys') else []
                    logger.info(f"Action.__init__(): After OmegaConf.create, action_config has keys: {final_keys}")
                    if 'actions' in self.action_config:
                        logger.info(f"Action.__init__(): Successfully loaded pipeline config with {len(self.action_config.actions)} actions")
                    else:
                        logger.error(f"Action.__init__(): Pipeline config loaded but 'actions' key is missing! Keys: {final_keys}")
                        # Try to get it from the original cfg.pipeline as fallback
                        if hasattr(cfg.pipeline, 'actions') and 'actions' in cfg.pipeline:
                            logger.error(
                                f"Action.__init__(): Original cfg.pipeline has {len(cfg.pipeline.actions)} actions, but copy lost them! Using direct assignment."
                            )
                            # Use direct assignment as fallback
                            self.action_config = cfg.pipeline
                except Exception as e:
                    logger.error(f"Action.__init__(): Error copying pipeline config: {e}, using direct assignment")
                    # Fallback to direct assignment
                    self.action_config = cfg.pipeline
            else:
                # Fallback: config is at the top level (flattened structure)
                self.action_config = cfg
        except Exception as e:
            # If direct access fails (e.g., due to ??? values), use OmegaConf.select as fallback
            logger.debug(f"Action.__init__(): Direct access failed ({e}), using OmegaConf.select fallback")
            try:
                action_val = OmegaConf.select(cfg, 'action', default=None)
                if action_val and action_val != '???':
                    self.action_config = action_val
                else:
                    pipeline_val = OmegaConf.select(cfg, 'pipeline', default=None)
                    if pipeline_val and pipeline_val != '???':
                        self.action_config = pipeline_val
                    else:
                        self.action_config = cfg
            except Exception:
                # Final fallback
                self.action_config = cfg

        self.aim_run = aim_run

        # Handle both regular dict and OmegaConf DictConfig
        has_device_config = isinstance(self.action_config, dict) or (hasattr(self.action_config, 'get') and hasattr(self.action_config, 'keys'))
        device = self.action_config.get('device', 'auto') if has_device_config else 'auto'
        Device.set_device(device)

        # Set up caching system
        if has_device_config:  # Use the same check as device config
            self.cache_enabled = self.action_config.get('cache_enabled', True)
            self.force_rerun = self.action_config.get('force_rerun', False)

            # Support both days and hours with proper conversion, prefer days
            cache_max_age_days = self.action_config.get('cache_max_age_days', None)
            cache_max_age_hours = self.action_config.get('cache_max_age_hours', None)

            if cache_max_age_days is not None:
                self.cache_max_age = cache_max_age_days * 24 * 3600  # Convert days to seconds
            elif cache_max_age_hours is not None:
                self.cache_max_age = cache_max_age_hours * 3600  # Convert hours to seconds
            else:
                self.cache_max_age = None
        else:
            # Fallback defaults when action_config is not a dict-like object
            self.cache_enabled = True
            self.force_rerun = False
            self.cache_max_age = None

        # Use a shared cache directory by extracting the base .runs folder from run_dir
        # run_dir is typically something like: /path/to/.runs/pipeline_name/timestamp/
        # We want to use: /path/to/.runs/action_cache/
        run_dir_path = Path(self.cfg.get('run_dir', '.'))

        # Find the .runs directory in the path hierarchy
        runs_dir = None
        for parent in [run_dir_path] + list(run_dir_path.parents):
            if parent.name == '.runs':
                runs_dir = parent
                break

        # Fallback: assume run_dir is inside .runs and go up until we find it
        if runs_dir is None:
            current_path = run_dir_path
            while current_path != current_path.parent:  # Stop at filesystem root
                if current_path.name == '.runs' or (current_path / '.runs').exists():
                    runs_dir = current_path if current_path.name == '.runs' else current_path / '.runs'
                    break
                current_path = current_path.parent

        # Final fallback: use current directory
        if runs_dir is None:
            runs_dir = Path('.') / '.runs'

        # Use universal cache directory shared across all actions and pipelines
        # This allows actions to reuse cache regardless of context (standalone vs pipeline)
        # Cache keys include config hashes, so different configs won't collide
        self.cache_dir = runs_dir / 'cache'

        logger.debug(f"📦 Using universal cache directory: {self.cache_dir}")

        # Store runs_dir for later path resolution
        self._runs_dir = runs_dir

        self._cached_outputs = None
        self._cache_key = None

    def get_cache_entry_dir(self, subdirectory: Optional[str] = None) -> Path:
        """
        Get the cache entry directory for this action using the proper hierarchical structure.

        This is the main API for actions to get their cache directory. It automatically uses
        the structure: cache/{action_name}/{cache_hash}/

        Args:
            subdirectory: Optional subdirectory name to append (e.g., "wikidata_entities", "samples")

        Returns:
            Path to the cache entry directory (or subdirectory if specified)

        Example:
            # Get the base cache entry directory
            cache_dir = self.get_cache_entry_dir()

            # Get a subdirectory for specific files
            entities_dir = self.get_cache_entry_dir("wikidata_entities")
            samples_dir = self.get_cache_entry_dir("samples")
        """
        if self._cache_key is None:
            self._cache_key = self._generate_cache_key()

        cache_entry_dir = self._get_cache_path(self._cache_key)

        if subdirectory:
            return cache_entry_dir / subdirectory
        return cache_entry_dir

    def get_run_dir(self, subdirectory: Optional[str] = None) -> Path:
        """
        Get the run directory for this action for saving plots, visualizations, and human-readable outputs.

        IMPORTANT: Plots and visualizations should ALWAYS be saved to run_dir, NOT to cache.
        They should be regenerated from cached data every time the action runs, ensuring they
        reflect the latest visualization code and are always up-to-date.

        This is the main API for actions to get their run directory. The run_dir is unique
        per run and contains human-readable outputs like plots, reports, and documentation.

        For loopable actions, the run_dir includes an iteration identifier subdirectory
        (e.g., iteration parameter value) to organize outputs by iteration.

        Args:
            subdirectory: Optional subdirectory name to append (e.g., "plots", "visualizations", "reports")

        Returns:
            Path to the run directory (or subdirectory if specified)

        Example:
            # Get the base run directory
            run_dir = self.get_run_dir()

            # Get a subdirectory for plots
            plots_dir = self.get_run_dir("plots")
            visualization_dir = self.get_run_dir("visualizations")

        Note:
            - Use get_cache_entry_dir() for machine-readable data that should be cached
            - Use get_run_dir() for human-readable outputs like plots that should be regenerated
            - For loopable actions, outputs are organized by iteration (e.g., .runs/pipeline/timestamp/action_name/iteration_id/plots/)
        """
        # Get run_dir - it should always be set by pipeline initialization
        run_dir_str = self.cfg.get('run_dir')
        if run_dir_str is None or run_dir_str == '':
            raise ValueError(
                f"run_dir is not set in config for action {self.__class__.__name__}. "
                f"This should be set during pipeline initialization. "
                f"Config keys: {list(self.cfg.keys())}"
            )
        run_dir = Path(run_dir_str)

        # Get action name from config (preferred) or fall back to class name
        action_name = getattr(self.cfg, 'action_name', None) or self.__class__.__name__

        # Check if this is a loopable action with iteration identifier
        iteration_id = None
        if hasattr(self, 'action_config'):
            # First check if iteration_id is set directly (by pipeline)
            if hasattr(self.action_config, 'iteration_id'):
                iteration_id = self.action_config.iteration_id
            # Check for common iteration identifier keys in model config
            # Subclasses can set iteration_id directly, or it can come from config
            elif hasattr(self.action_config, 'model') and hasattr(self.action_config.model, 'revision'):
                iteration_id = self.action_config.model.revision
            # Also check top-level common keys
            elif hasattr(self.action_config, 'revision'):
                iteration_id = self.action_config.revision
            # Note: Project-specific iteration identifier logic (e.g., model.name) should be handled
            # in pipeline subclasses by setting iteration_id in the action config
            # by setting iteration_id in the action config

        # If we have an iteration identifier, add it as a subdirectory
        if iteration_id:
            # Sanitize iteration_id for use in file paths
            sanitized_id = str(iteration_id).replace('/', '_').replace(':', '_').replace(' ', '_')
            # Add action name and iteration as subdirectories
            run_dir = run_dir / action_name / str(sanitized_id)
        else:
            # Regular action (not loopable) - use action name
            run_dir = run_dir / action_name

        run_dir.mkdir(parents=True, exist_ok=True)

        if subdirectory:
            subdir_path = run_dir / subdirectory
            subdir_path.mkdir(parents=True, exist_ok=True)
            return subdir_path
        return run_dir

    def _resolve_config_paths(self, runs_dir: Path):
        """
        Automatically resolve portable paths in action_config.

        Deterministic rule: If a string value is a relative path (contains '/' or '\\')
        and is not an absolute path, resolve it relative to .runs directory.

        Args:
            runs_dir: The .runs directory path
        """

        def is_portable_path(value):
            """Check if a string value is actually a portable path (not a model name, etc.)."""
            if not isinstance(value, str) or not value:
                return False

            # If it's already absolute, don't resolve it
            if Path(value).is_absolute():
                return False

            # Must contain path separators
            if '/' not in value and '\\' not in value:
                return False

            # Strip .runs/ prefix if present for checking
            check_value = value
            if check_value.startswith('.runs/') or check_value.startswith('.runs\\'):
                check_value = check_value[6:]

            # Check if this path exists under runs_dir (most reliable method)
            potential_path = runs_dir / check_value
            if potential_path.exists():
                return True

            # If it doesn't exist yet, check if it's under our known directories
            # (for paths that will be created later)
            first_part = check_value.split('/')[0].split('\\')[0]
            if first_part in ['cache', 'data', 'output', 'outputs', 'checkpoint', 'checkpoints']:
                return True

            # Otherwise, assume it's not a path (could be a model name like "organization/model")
            return False

        def resolve_value(value, key=''):
            """Recursively resolve paths in a value."""
            if isinstance(value, str):
                # Only process if it looks like an actual portable path
                if is_portable_path(value):
                    # Strip leading .runs/ if present (portable paths shouldn't include it)
                    cleaned_value = value
                    if cleaned_value.startswith('.runs/') or cleaned_value.startswith('.runs\\'):
                        cleaned_value = cleaned_value[6:]  # Remove '.runs/' or '.runs\'

                    absolute_path = runs_dir / cleaned_value
                    logger.debug(f"Resolved portable path '{key}': {value} -> {absolute_path}")
                    return str(absolute_path)
                return value
            elif isinstance(value, dict):
                # Recursively process dictionaries
                return {k: resolve_value(v, k) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                # Recursively process lists/tuples
                processed = [resolve_value(item, key) for item in value]
                return type(value)(processed)
            else:
                return value

        # Process the action_config
        if hasattr(self.action_config, 'items'):
            try:
                # For OmegaConf, we need to be careful about modifying
                from omegaconf import OmegaConf

                if OmegaConf.is_config(self.action_config):
                    # Convert to container, resolve paths, and recreate config
                    config_dict = OmegaConf.to_container(self.action_config, resolve=True)
                    resolved_dict = resolve_value(config_dict)
                    # Recreate the config from resolved dict to ensure all nested paths are updated
                    self.action_config = OmegaConf.create(resolved_dict)
                else:
                    # Regular dict - modify in place
                    for key, value in self.action_config.items():
                        self.action_config[key] = resolve_value(value)
            except Exception as e:
                logger.warning(f"Failed to resolve config paths: {e}")

    def get_outputs(self) -> Dict[str, Any]:
        """
        Return the outputs produced by this action.

        This method should be overridden by subclasses to return relevant outputs
        that can be used by subsequent actions in a pipeline.

        Returns:
            Dict[str, Any]: A dictionary of output keys and values.
        """
        return {}

    @staticmethod
    def _make_path_portable(path):
        """
        Convert an absolute path to a relative path based on the .runs directory.
        This makes paths portable across machines where .runs is mounted at different locations.

        Args:
            path: Absolute path (string or Path object)

        Returns:
            String path relative to .runs directory WITHOUT the .runs/ prefix
            (e.g., "cache/ranks_abc12345", NOT ".runs/cache/ranks_abc12345")
            Returns original path if conversion fails.
        """
        if path is None:
            return None

        try:
            path_obj = Path(path)

            # Find the .runs directory in the path
            runs_dir = None
            for parent in [path_obj] + list(path_obj.parents):
                if parent.name == '.runs':
                    runs_dir = parent
                    break

            if runs_dir is None:
                # Path doesn't contain .runs, return as-is
                logger.debug(f"Path {path} doesn't contain .runs directory, returning as-is")
                return str(path)

            # Make path relative to .runs (this automatically excludes .runs/ prefix)
            relative_path = path_obj.relative_to(runs_dir)
            portable_path = str(relative_path)

            # Sanity check: ensure we didn't accidentally include .runs/ prefix
            if portable_path.startswith('.runs/') or portable_path.startswith('.runs\\'):
                logger.warning(f"Portable path incorrectly starts with .runs/: {portable_path}")
                portable_path = portable_path[6:]  # Strip .runs/ prefix

            logger.debug(f"Converted {path} -> .runs/{portable_path} (runs_dir={runs_dir})")
            return portable_path

        except Exception as e:
            logger.warning(f"Failed to make path portable: {path}, error: {e}")
            return str(path)

    @staticmethod
    def _is_runs_path(path_str: str) -> bool:
        """
        Check if a string looks like an absolute path within the .runs directory.

        Args:
            path_str: String to check

        Returns:
            True if it looks like a .runs path, False otherwise
        """
        try:
            path_obj = Path(path_str)

            # Must be absolute to be converted
            if not path_obj.is_absolute():
                return False

            # Check if .runs is in the path
            for parent in [path_obj] + list(path_obj.parents):
                if parent.name == '.runs':
                    return True

            return False
        except Exception:
            # If it can't be converted to a Path, it's not a path
            return False

    # Universal directory attributes that should be made portable
    # These are the core directory attributes that exist across ALL actions
    PORTABLE_PATH_KEYS = {
        'run_dir',  # The run directory for this execution
        'output_dir',  # Output directory (alternative name)
        'output_path',  # Output path (common naming convention)
        'cache_dir',  # Cache directory
    }

    @staticmethod
    def _make_outputs_portable(outputs: Any) -> Any:
        """
        Convert standard path keys in outputs to portable format.

        This focuses on known directory/path keys (run_dir, output_dir, cache_dir, etc.)
        rather than recursively checking all strings, making it predictable and explicit.

        Args:
            outputs: The outputs dictionary or value to process

        Returns:
            The same structure with standard paths converted to portable format
        """
        if outputs is None:
            return None

        if isinstance(outputs, dict):
            result = {}
            for key, value in outputs.items():
                # Check if this is a known path key
                if key in Action.PORTABLE_PATH_KEYS:
                    # Convert paths to portable format
                    if isinstance(value, (str, Path)) and value:
                        result[key] = Action._make_path_portable(str(value))
                    else:
                        result[key] = value
                elif isinstance(value, dict):
                    # Recursively process nested dictionaries
                    result[key] = Action._make_outputs_portable(value)
                elif isinstance(value, (list, tuple)):
                    # Process lists/tuples in case they contain dicts with paths
                    processed = [Action._make_outputs_portable(item) if isinstance(item, dict) else item for item in value]
                    result[key] = type(value)(processed)
                else:
                    # Keep other values as-is
                    result[key] = value
            return result

        else:
            # For non-dict values, return as-is
            return outputs

    def _resolve_portable_path(self, portable_path):
        """
        Resolve a portable path (relative to .runs) back to an absolute path.
        This is the inverse of _make_path_portable().

        Args:
            portable_path: Path relative to .runs directory

        Returns:
            Absolute path by joining with the detected .runs directory
        """
        if portable_path is None:
            return None

        try:
            # If it's already an absolute path, return it
            path_obj = Path(portable_path)
            if path_obj.is_absolute():
                return str(portable_path)

            # Otherwise, join with our .runs directory
            # Use cache_dir which is always .runs/cache, so parent is .runs
            if hasattr(self, 'cache_dir'):
                runs_dir = self.cache_dir.parent
            else:
                # Fallback: try to find .runs from run_dir
                run_dir = Path(self.cfg.get('run_dir', '.'))
                runs_dir = None
                for parent in [run_dir] + list(run_dir.parents):
                    if parent.name == '.runs':
                        runs_dir = parent
                        break
                if runs_dir is None:
                    # Last resort: assume current directory has .runs
                    runs_dir = Path('.runs')

            absolute_path = runs_dir / portable_path

            logger.debug(f"Resolved .runs/{portable_path} -> {absolute_path}")
            return str(absolute_path)

        except Exception as e:
            logger.warning(f"Failed to resolve portable path: {portable_path}, error: {e}")
            return str(portable_path)

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on action configuration."""
        from urartu.utils.cache import CacheKeyGenerator

        # For cross-pipeline cache sharing, always use action_name from config
        # This ensures the same action (e.g., _2_sample_constructor) with same config
        # shares cache across standalone, pipeline A, pipeline B, etc.
        action_name = getattr(self.cfg, 'action_name', None) or self.__class__.__name__

        logger.debug(f"🔑 Action name determination:")
        logger.debug(f"   cfg.action_name: {getattr(self.cfg, 'action_name', None)}")
        logger.debug(f"   class name: {self.__class__.__name__}")
        logger.debug(f"   final action_name: {action_name}")

        # Get serializable config
        serializable_config = self._get_serializable_config()

        # Use unified cache key generator
        cache_key = CacheKeyGenerator.generate_action_cache_key(action_name=action_name, config=serializable_config)

        logger.debug(f"   Generated cache key: {cache_key}")

        # For debugging: if this is a different key from existing cache directories, show differences
        if self.cache_dir.exists():
            action_cache_dir = self.cache_dir / action_name
            if action_cache_dir.exists():
                existing_dirs = [d.name for d in action_cache_dir.iterdir() if d.is_dir()]
                cache_hash = cache_key.split('_')[-1] if '_' in cache_key else cache_key
                if existing_dirs and cache_hash not in existing_dirs:
                    logger.debug(f"🔍 Cache key mismatch detected! Generated: {cache_key}")
                    logger.debug(f"🔍 Existing cache directories: {existing_dirs}")

        return cache_key

    def _get_serializable_config(self) -> Dict[str, Any]:
        """Get a serializable version of the configuration for caching."""
        from omegaconf import OmegaConf

        from urartu.utils.cache import filter_config_for_cache

        try:
            if hasattr(self, 'action_config'):
                cfg_dict = OmegaConf.to_container(self.action_config, resolve=True)

                # Debug logging
                logger.debug(f"📋 Config serialization for {self.__class__.__name__}:")
                logger.debug(f"   Original config keys: {sorted(cfg_dict.keys())}")

                # Use unified filter function
                filtered_cfg = filter_config_for_cache(cfg_dict)

                removed_keys = set(cfg_dict.keys()) - set(filtered_cfg.keys())
                if removed_keys:
                    logger.debug(f"   Removed keys: {sorted(removed_keys)}")
                logger.debug(f"   Final config keys: {sorted(filtered_cfg.keys())}")

                # Check if pipeline-specific fields leaked through
                pipeline_specific_in_filtered = [
                    k for k in filtered_cfg.keys() if k in ['experiment_name', 'pipeline_name', 'pipeline_id', 'pipeline_config_hash']
                ]
                if pipeline_specific_in_filtered:
                    logger.warning(
                        f"   ⚠️  CRITICAL: Pipeline-specific fields found in FILTERED config (should be removed!): {pipeline_specific_in_filtered}"
                    )

                return filtered_cfg
            return {}
        except Exception:
            # Fallback to dict conversion
            if hasattr(self, 'action_config') and hasattr(self.action_config, 'items'):
                cfg_dict = dict(self.action_config)

                logger.debug(f"📋 Config serialization (fallback) for {self.__class__.__name__}:")
                logger.debug(f"   Original config keys: {sorted(cfg_dict.keys())}")

                # Use unified filter function
                filtered_cfg = filter_config_for_cache(cfg_dict)

                removed_keys = set(cfg_dict.keys()) - set(filtered_cfg.keys())
                if removed_keys:
                    logger.debug(f"   Removed keys: {sorted(removed_keys)}")
                logger.debug(f"   Final config keys: {sorted(filtered_cfg.keys())}")

                # Check if pipeline-specific fields leaked through
                pipeline_specific_in_filtered = [
                    k for k in filtered_cfg.keys() if k in ['experiment_name', 'pipeline_name', 'pipeline_id', 'pipeline_config_hash']
                ]
                if pipeline_specific_in_filtered:
                    logger.warning(
                        f"   ⚠️  CRITICAL: Pipeline-specific fields found in FILTERED config (should be removed!): {pipeline_specific_in_filtered}"
                    )

                return filtered_cfg
            return {}

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the directory path for a cache entry.

        Structure: cache/{action_name}/{cache_hash}/

        Args:
            cache_key: Cache key in format "{action_name}_{hash}"

        Returns:
            Path to the cache directory for this entry
        """
        # Ensure cache_dir is set
        if not hasattr(self, 'cache_dir') or self.cache_dir is None:
            # Fallback: use .runs/cache in current directory
            self.cache_dir = Path('.runs') / "cache"

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

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the pickle file path within a cache entry directory."""
        cache_dir = self._get_cache_path(cache_key)
        return cache_dir / "cache.pkl"

    def _get_cache_metadata_path(self, cache_key: str) -> Path:
        """Get the metadata YAML file path within a cache entry directory."""
        cache_dir = self._get_cache_path(cache_key)
        return cache_dir / "metadata.yaml"

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load outputs from cache if available and valid."""
        if not self.cache_enabled or self.force_rerun:
            logger.debug(
                f"💾 Cache disabled or force_rerun for {self.__class__.__name__} (enabled: {self.cache_enabled}, force_rerun: {self.force_rerun})"
            )
            return None

        # Use stored cache key if available, otherwise generate new one
        stored_key = getattr(self, '_cache_key', None)
        if stored_key:
            cache_key = stored_key
            logger.debug(f"💾 Using stored cache key: {cache_key}")
        else:
            cache_key = self._generate_cache_key()
            logger.debug(f"💾 Generated new cache key: {cache_key}")

        cache_file_path = self._get_cache_file_path(cache_key)
        cache_dir = self._get_cache_path(cache_key)
        logger.debug(f"💾 Looking for cache at: {cache_file_path}")

        # Debug: List existing cache directories
        if self.cache_dir.exists():
            action_name = getattr(self.cfg, 'action_name', None) or self.__class__.__name__
            action_cache_dir = self.cache_dir / action_name
            if action_cache_dir.exists():
                existing_dirs = [d for d in action_cache_dir.iterdir() if d.is_dir()]
                logger.debug(f"💾 Found {len(existing_dirs)} existing cache directories for {action_name}:")
                for d in existing_dirs:
                    logger.debug(f"   📁 {d.name}")

        if not cache_file_path.exists():
            logger.debug(f"💾 Cache file not found: {cache_file_path}")
            return None

        try:
            with open(cache_file_path, 'rb') as f:
                cache_data = pickle.load(f)

            logger.debug(f"💾 Loaded cache data from {cache_file_path}")
            logger.debug(f"   Cache timestamp: {cache_data.get('timestamp', 'N/A')}")
            logger.debug(f"   Cache action_name: {cache_data.get('action_name', 'N/A')}")
            logger.debug(f"   Cache config_hash: {cache_data.get('config_hash', 'N/A')}")
            logger.debug(f"   Cache has outputs: {'outputs' in cache_data}")

            # Check if cache is still valid
            if self.cache_max_age is not None:
                age = time.time() - cache_data['timestamp']
                logger.debug(f"   Cache age: {age:.1f}s, max_age: {self.cache_max_age}s")
                if age > self.cache_max_age:
                    logger.info(f"❌ Cache for {self.__class__.__name__} is expired (age: {age:.1f}s)")
                    return None
            else:
                logger.debug(f"   No cache age limit set")

            logger.debug(f"✅ Loading cached outputs for {self.__class__.__name__} from {cache_file_path}")
            if self.aim_run is not None:
                self.aim_run[f"action_{self.__class__.__name__}_cache_hit"] = True
            return cache_data['outputs']

        except Exception as e:
            logger.warning(f"Failed to load cache for {self.__class__.__name__}: {e}")
            return None

    def _save_to_cache(self, outputs: Dict[str, Any]):
        """Save outputs to cache."""
        if not self.cache_enabled:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Use stored cache key to ensure consistency with initial lookup
            stored_key = getattr(self, '_cache_key', None)
            if stored_key:
                cache_key = stored_key
                logger.debug(f"💾 Saving with stored cache key: {cache_key}")
            else:
                cache_key = self._generate_cache_key()
                logger.debug(f"💾 Saving with new cache key: {cache_key}")

            # Get cache directory for this entry
            cache_entry_dir = self._get_cache_path(cache_key)
            cache_file_path = self._get_cache_file_path(cache_key)
            metadata_path = self._get_cache_metadata_path(cache_key)

            # Create cache entry directory
            cache_entry_dir.mkdir(parents=True, exist_ok=True)

            # Use the same action name logic as cache key generation for consistency
            action_name = getattr(self.cfg, 'action_name', None) or self.__class__.__name__

            cache_data = {
                'outputs': outputs,
                'timestamp': time.time(),
                'action_name': action_name,
                'config_hash': hashlib.sha256(json.dumps(self._get_serializable_config(), sort_keys=True).encode()).hexdigest()[:8],
            }

            # Save cache data to pickle file
            with open(cache_file_path, 'wb') as f:
                pickle.dump(cache_data, f)

            # Also save human-readable metadata
            metadata = {
                'cache_key': cache_key,
                'action_name': action_name,  # Use the same action name as in cache_data
                'timestamp': datetime.fromtimestamp(cache_data['timestamp']).isoformat(),
                'config_hash': cache_data['config_hash'],
                'output_keys': list(outputs.keys()),
                'full_config': self._get_serializable_config(),  # Add full config content
            }
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, indent=2, sort_keys=False)

            logger.info(f"Cached outputs for {self.__class__.__name__} with key {cache_key}")
            if self.aim_run is not None:
                self.aim_run[f"action_{self.__class__.__name__}_cache_hit"] = False

        except Exception as e:
            logger.warning(f"Failed to save cache for {self.__class__.__name__}: {e}")

    def run_with_cache(self):
        """
        Run the action with caching support.

        This method should be called instead of run() directly to enable caching.
        It will check cache first, run the action if needed, and save to cache.
        """
        # Generate cache key once and store it to ensure consistency
        # Cache key is generated BEFORE resolving paths, so it uses portable paths
        self._cache_key = self._generate_cache_key()

        # Now resolve portable paths to absolute paths for actual use
        # This ensures paths work correctly on the current machine while keeping cache key consistent
        self._resolve_config_paths(self._runs_dir)

        # Try to load from cache first
        cached_outputs = self._load_from_cache()
        if cached_outputs is not None:
            self._cached_outputs = cached_outputs
            return

        # Cache miss - run the action
        logger.debug(f"Running {self.__class__.__name__} (cache miss)")

        # Call initialize() if it exists (for setup before running)
        if hasattr(self, 'initialize'):
            self.initialize()

        # Call the actual run method (must be implemented by subclasses)
        if hasattr(self, 'run'):
            self.run()
        elif hasattr(self, 'main'):
            self.main()
        else:
            raise NotImplementedError(f"Action {self.__class__.__name__} must implement run() or main() method")

        # Get outputs and save to cache
        outputs = self.get_outputs()
        if outputs:
            self._save_to_cache(outputs)
            # Don't set _cached_outputs here - it should only be set when loading FROM cache
            # This allows the pipeline to distinguish between "executed" vs "from cache"

    def clear_cache(self):
        """Clear the cache for this action."""
        if self.cache_dir.exists():
            cache_key = getattr(self, '_cache_key', None) or self._generate_cache_key()
            cache_entry_dir = self._get_cache_path(cache_key)

            if cache_entry_dir.exists():
                import shutil

                shutil.rmtree(cache_entry_dir)
                logger.info(f"Cleared cache directory for {self.__class__.__name__}: {cache_entry_dir}")

    def cleanup_memory(self):
        """
        Clean up memory resources used by this action.

        This method should be called after an action completes to free up GPU and RAM
        memory for subsequent actions in a pipeline. It performs generic cleanup and
        calls action-specific cleanup hooks.
        """
        logger.debug(f"🧹 Cleaning up memory for action: {self.__class__.__name__}")

        # Get initial memory stats
        initial_gpu_memory = self._get_gpu_memory_mb()
        initial_ram_gb = self._get_ram_usage_gb()

        # Call action-specific cleanup first
        try:
            self._action_specific_cleanup()
        except Exception as e:
            logger.warning(f"Action-specific cleanup failed: {e}")

        # Generic cleanup of common ML objects
        self._cleanup_common_attributes()

        # Clear GPU memory cache
        self._clear_gpu_cache()

        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collector freed {collected} objects")

        # Log memory savings
        final_gpu_memory = self._get_gpu_memory_mb()
        final_ram_gb = self._get_ram_usage_gb()

        gpu_freed = initial_gpu_memory - final_gpu_memory if initial_gpu_memory and final_gpu_memory else None
        ram_freed = initial_ram_gb - final_ram_gb if initial_ram_gb and final_ram_gb else None

        if gpu_freed and gpu_freed > 0:
            logger.debug(f"✅ Freed {gpu_freed:.1f} MB of GPU memory")
        if ram_freed and ram_freed > 0.1:  # Only log if significant
            logger.debug(f"✅ Freed {ram_freed:.2f} GB of RAM")

        logger.debug(f"🧹 Memory cleanup completed for {self.__class__.__name__}")

    def _action_specific_cleanup(self):
        """
        Hook for action subclasses to implement custom memory cleanup.

        Override this method in subclasses to clean up action-specific resources
        like models, datasets, or other large objects.
        """
        pass

    def _cleanup_common_attributes(self):
        """Clean up commonly used attributes in ML actions."""
        # List of common attribute names that might hold large objects
        common_large_attributes = [
            'model',
            '_model',
            'tokenizer',
            '_tokenizer',
            'dataset',
            '_dataset',
            'raw_dataset',
            'train_dataset',
            'test_dataset',
            'dataloader',
            'train_dataloader',
            'test_dataloader',
            'optimizer',
            '_optimizer',
            'scheduler',
            '_scheduler',
            'criterion',
            '_criterion',
            'loss_fn',
            'embeddings',
            '_embeddings',
            'features',
            '_features',
            'predictions',
            '_predictions',
            'logits',
            '_logits',
            'cache',
            '_cache',
            'memory_cache',
        ]

        cleaned_attributes = []
        for attr_name in common_large_attributes:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if attr_value is not None:
                    # Directly delete without moving to CPU (more efficient)
                    # Set to None to free reference
                    setattr(self, attr_name, None)
                    cleaned_attributes.append(attr_name)

        if cleaned_attributes:
            logger.debug(f"Cleaned up attributes: {', '.join(cleaned_attributes)}")

    def _clear_gpu_cache(self):
        """Clear GPU memory cache if available."""
        try:
            import torch

            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                final_memory = torch.cuda.memory_allocated()
                freed_mb = (initial_memory - final_memory) / (1024 * 1024)
                if freed_mb > 1:  # Only log if significant
                    logger.debug(f"Cleared GPU cache, freed {freed_mb:.1f} MB")
        except ImportError:
            pass  # PyTorch not available
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")

    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return None

    def _get_ram_usage_gb(self) -> Optional[float]:
        """Get current RAM usage in GB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024 * 1024)
        except ImportError:
            pass
        return None

    def run(self):
        """
        Optional method that can be implemented by action subclasses.
        This method should contain the main logic of the action.

        If not implemented, the action should have a main() method instead.
        """
        raise NotImplementedError(f"Action {self.__class__.__name__} must implement either run() or main() method")


class ActionDataset(Action):
    """
    A specialized Action class for dataset-related operations.

    This class extends the base Action class with functionality specific to dataset handling.
    It automatically generates a unique hash for the dataset based on its name and configuration,
    and sets this information in the Aim run for tracking purposes.

    Attributes:
        Inherits all attributes from the Action class.

    Note:
        The dataset hash is created using the dataset name combined with an 8-character hash
        derived from the dataset configuration, ensuring uniqueness for tracking.
    """

    def __init__(self, cfg: DictConfig, aim_run: Run):
        """
        Initializes the ActionDataset object with the necessary configuration and Aim session.

        Args:
            cfg (DictConfig): The configuration object providing settings for various components of the action.
            aim_run (Run): The Aim run session to track and manage execution metrics and logs.

        Before initializing the parent Action class, this constructor:
        1. Generates a unique hash for the dataset configuration
        2. Adds this hash to the dataset configuration
        3. Sets the complete configuration in the Aim run
        """
        # Get the dataset config from the appropriate location
        dataset_config = None

        # Try different locations for dataset config based on configuration structure
        if hasattr(cfg, 'action') and cfg.action and hasattr(cfg.action, 'dataset'):
            # Pipeline actions: cfg.action.dataset
            dataset_config = cfg.action.dataset
        elif hasattr(cfg, 'pipeline') and cfg.pipeline and hasattr(cfg.pipeline, 'dataset'):
            # Pipeline actions fallback: cfg.pipeline.dataset
            dataset_config = cfg.pipeline.dataset
        elif hasattr(cfg, 'dataset'):
            # Individual actions: cfg.dataset
            dataset_config = cfg.dataset

        if dataset_config and hasattr(dataset_config, 'name'):
            dataset_config["hash"] = f"{dataset_config.name}_{dict_to_8char_hash(dataset_config)}"

        if cfg.aim.use_aim:
            aim_run.set("cfg", cfg, strict=False)

        super().__init__(cfg, aim_run)
