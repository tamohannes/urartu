"""
Configuration injection and resolution for pipelines.

This module handles injecting action outputs and loop context into action configurations.
"""

import logging
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from .types import ActionOutput

logger = logging.getLogger(__name__)


class ConfigInjector:
    """Handles injection of dependencies and loop context into action configurations."""

    def __init__(self, action_outputs: Dict[str, Any], loopable_actions: list, loop_configs: Dict[str, Any]):
        """
        Initialize the config injector.

        Args:
            action_outputs: Dictionary of action outputs (may contain nested dicts for loopable actions)
            loopable_actions: List of loopable action names
            loop_configs: Dictionary mapping loop variable names to config paths
        """
        self.action_outputs = action_outputs
        self.loopable_actions = loopable_actions
        self.loop_configs = loop_configs

    def inject_action_outputs(
        self,
        action_config_dict: Dict[str, Any],
        current_action_name: str,
        loop_context: Optional[Dict[str, Any]] = None,
        get_iteration_outputs: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Inject outputs from previous actions into the current action's configuration.

        Uses the unified DependencyResolver for automatic dependency resolution.
        Handles iteration-aware outputs for loopable actions.

        Args:
            action_config_dict: The action's configuration dictionary
            current_action_name: Name of the current action being configured
            loop_context: Optional loop context to resolve iteration-aware dependencies
            get_iteration_outputs: Optional function to get all iteration outputs for an action

        Returns:
            Updated configuration dictionary with injected outputs
        """
        from urartu.common.dependency import DependencyResolver

        # Prepare action outputs dictionary for resolver
        # Handle both regular actions (ActionOutput) and loopable actions (dict of {iteration_id: ActionOutput})
        action_outputs_dict = {}
        iteration_id = None
        if loop_context:
            # Build iteration ID from loop context
            if len(loop_context) == 1:
                first_key, first_value = next(iter(loop_context.items()))
                sanitized = str(first_value).replace('/', '_').replace(':', '_').replace(' ', '_')
                iteration_id = f"{first_key}_{sanitized}"
            elif loop_context:
                parts_list = [f"{k}_{str(v).replace('/', '_').replace(':', '_').replace(' ', '_')}" for k, v in loop_context.items()]
                iteration_id = "_".join(parts_list)

        for name, entry in self.action_outputs.items():
            # Check if this is a loopable action (nested dict) or regular action (ActionOutput)
            if isinstance(entry, dict) and not isinstance(entry, ActionOutput):
                # Loopable action with multiple iterations
                if loop_context and iteration_id and iteration_id in entry:
                    # Use the current iteration's output
                    action_outputs_dict[name] = entry[iteration_id].outputs
                elif entry:
                    # Fallback: use first iteration if no loop context
                    first_iteration_output = next(iter(entry.values()))
                    action_outputs_dict[name] = first_iteration_output.outputs
            else:
                # Regular action - direct ActionOutput
                # Also handle backward compatibility: iteration-specific names
                if isinstance(entry, ActionOutput):
                    action_outputs_dict[name] = entry.outputs
                    # If this is an iteration-specific name, also map the base name
                    if loop_context and iteration_id and name.endswith(f"_{iteration_id}"):
                        base_name = name.rsplit(f"_{iteration_id}", 1)[0]
                        action_outputs_dict[base_name] = entry.outputs
                else:
                    # Shouldn't happen, but handle gracefully
                    logger.warning(f"Unexpected output type for '{name}': {type(entry)}")

        # For aggregator actions (actions that run after loopable actions),
        # provide access to all iteration outputs via a special key pattern
        # Check if this is likely an aggregator action (runs after loopable actions)
        if not loop_context and self.loopable_actions and get_iteration_outputs:
            # Check if current action depends on any loopable action
            depends_on = action_config_dict.get('depends_on', {})
            for dep_action_name in depends_on.keys():
                if dep_action_name in self.loopable_actions:
                    # This is an aggregator action - provide all iteration outputs
                    all_iteration_outputs = get_iteration_outputs(dep_action_name)
                    # Store as a special key that aggregator can access
                    all_iterations_data = {iter_id: output.outputs for iter_id, output in all_iteration_outputs.items()}
                    action_outputs_dict[f"{dep_action_name}_all_iterations"] = all_iterations_data
                    # Also inject directly into action_config_dict so aggregator can access it
                    action_config_dict[f"{dep_action_name}_all_iterations"] = all_iterations_data
                    logger.debug(f"📦 Provided all iteration outputs for aggregator action '{current_action_name}' from '{dep_action_name}'")

        # Create resolver and resolve dependencies
        resolver = DependencyResolver(action_outputs_dict)

        # Extract depends_on if present
        depends_on = action_config_dict.get('depends_on')

        # Resolve dependencies
        resolved_config = resolver.resolve_dependencies(
            action_config=action_config_dict, current_action_name=current_action_name, explicit_depends_on=depends_on
        )

        return resolved_config

    def inject_loop_context(self, action_config_dict: Dict[str, Any], loop_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject loop context parameters into the action's configuration.

        Uses loop_configs to determine where each variable should be injected.
        Supports multiple injection patterns:
        1. Single path: {"revision": "model.revision"} - injects at one location
        2. List of paths: {"revision": ["model.revision", "model.config.revision"]} - injects at multiple locations
        3. No mapping: Falls back to intelligent injection (top-level or common paths)

        Note: This function works with regular dicts locally for easier manipulation.
        The caller should convert the result back to DictConfig using OmegaConf.create().

        Args:
            action_config_dict: The action's configuration (dict or DictConfig, converted to dict internally)
            loop_context: Dictionary of loop iteration parameters (e.g., {'revision': 'stage1-step140000'})

        Returns:
            Updated configuration dictionary (regular dict) with loop context injected.
            Caller should convert to DictConfig: OmegaConf.create(result)
        """
        if not loop_context:
            return action_config_dict

        # Convert to dict for easier manipulation (local conversion)
        # Accepts both dict and DictConfig, but works with dict internally
        if isinstance(action_config_dict, DictConfig):
            # Convert DictConfig to dict, preserving all nested structures
            config = OmegaConf.to_container(action_config_dict, resolve=False)
            # Ensure it's a dict (not a list or other type)
            if not isinstance(config, dict):
                logger.warning(f"⚠️ Expected dict after conversion but got {type(config)}, using original")
                config = dict(action_config_dict) if isinstance(action_config_dict, dict) else action_config_dict

            # Recursively ensure all nested DictConfigs are converted to dicts
            def convert_nested_dictconfigs(d):
                if isinstance(d, dict):
                    for k, v in list(d.items()):
                        if isinstance(v, DictConfig):
                            d[k] = OmegaConf.to_container(v, resolve=False)
                            # Recursively convert nested structures
                            if isinstance(d[k], dict):
                                convert_nested_dictconfigs(d[k])
                        elif isinstance(v, dict):
                            convert_nested_dictconfigs(v)

            convert_nested_dictconfigs(config)
        else:
            # Make a deep copy to avoid modifying the original
            import copy

            if isinstance(action_config_dict, dict):
                config = copy.deepcopy(action_config_dict)
            else:
                config = action_config_dict

        # Log the config structure before injection for debugging
        if 'model' in config:
            logger.debug(
                f"🔄 Config has 'model' key before loop context injection: {list(config.get('model', {}).keys()) if isinstance(config.get('model'), dict) else type(config.get('model'))}"
            )
        else:
            logger.warning(
                f"⚠️ Config does NOT have 'model' key before loop context injection. Config keys: {list(config.keys()) if isinstance(config, dict) else 'N/A'}"
            )

        # Use loop_configs to determine injection paths
        for param_key, param_value in loop_context.items():
            # Skip iteration_id as it's metadata, not a config value
            if param_key == 'iteration_id':
                continue

            if param_key in self.loop_configs:
                # Get the injection path(s) from loop_configs
                injection_paths = self.loop_configs[param_key]

                # Support both single path (string) and multiple paths (list)
                if isinstance(injection_paths, str):
                    injection_paths = [injection_paths]
                elif not isinstance(injection_paths, list):
                    logger.warning(f"🔄 loop_configs['{param_key}'] must be a string or list, got {type(injection_paths)}. Skipping.")
                    continue

                # Inject at all specified paths
                for config_path in injection_paths:
                    # Debug: log model config before injection
                    model_before = None
                    if 'model' in config_path and 'model' in config:
                        model_before = config.get('model', {})
                        if isinstance(model_before, dict):
                            logger.debug(f"🔄 Before injection at {config_path}: model keys = {list(model_before.keys())}, model = {model_before}")
                        else:
                            logger.debug(f"🔄 Before injection at {config_path}: model type = {type(model_before)}, model = {model_before}")
                    else:
                        logger.warning(
                            f"⚠️ model not found in config before injection at {config_path}. Config keys: {list(config.keys()) if isinstance(config, dict) else 'N/A'}"
                        )

                    self._set_nested_config_value(config, config_path, param_value)

                    # Debug: log model config after injection
                    if 'model' in config_path and 'model' in config:
                        model_after = config.get('model', {})
                        if isinstance(model_after, dict):
                            logger.debug(f"🔄 After injection at {config_path}: model keys = {list(model_after.keys())}, model = {model_after}")
                        else:
                            logger.debug(f"🔄 After injection at {config_path}: model type = {type(model_after)}, model = {model_after}")

                        # Verify that type is still present
                        if isinstance(model_after, dict) and 'type' not in model_after:
                            logger.error(f"❌ ERROR: model.type was lost after injecting {param_value} at {config_path}!")
                            if model_before is not None:
                                logger.error(f"   model before: {model_before}")
                            else:
                                logger.error(f"   model before: (not captured - model was not in config)")
                            logger.error(f"   model after: {model_after}")
                            logger.error(f"   Full config keys: {list(config.keys()) if isinstance(config, dict) else 'N/A'}")
                    logger.debug(f"🔄 Injected {param_key} = {param_value} at {config_path}")
            else:
                # Fallback: intelligent injection
                # Try common paths first, then top-level
                injected = False

                # Common pattern fallback: if param_key matches a nested key in config, inject there
                # This is a convenience for common patterns (e.g., revision -> model.revision)
                # The primary mechanism should be through loop_configs
                if param_key == 'revision' and 'model' in config:
                    if isinstance(config['model'], dict):
                        config['model']['revision'] = param_value
                        logger.debug(f"🔄 Auto-injected {param_key} = {param_value} at model.{param_key} (fallback pattern)")
                        injected = True

                # If not injected yet, try top-level
                if not injected:
                    if param_key not in config:
                        config[param_key] = param_value
                        logger.debug(f"🔄 Auto-injected {param_key} = {param_value} at top level")
                    else:
                        # If key exists, check if it's a dict and merge
                        if isinstance(config[param_key], dict) and isinstance(param_value, dict):
                            config[param_key].update(param_value)
                            logger.debug(f"🔄 Merged {param_key} dict into existing config")
                        else:
                            config[param_key] = param_value
                            logger.debug(f"🔄 Overwrote {param_key} = {param_value} at top level")

        return config

    def _set_nested_config_value(self, config: Dict[str, Any], path: str, value: Any):
        """
        Set a value at a nested path in the configuration dictionary.

        Args:
            config: Configuration dictionary to modify
            path: Dot-separated path (e.g., 'dataset.data_files')
            value: Value to set
        """
        from urartu.common.dependency import DependencyResolver

        # Use DependencyResolver's method for consistency
        resolver = DependencyResolver({})
        resolver._set_nested_config_value(config, path, value)

    @staticmethod
    def get_common_pipeline_configs(pipeline_config: DictConfig, cfg: DictConfig) -> Dict[str, Any]:
        """
        Extract configuration values from pipeline config that should be
        propagated to all individual actions.

        Includes all pipeline configs except pipeline-specific ones.

        Args:
            pipeline_config: The pipeline configuration
            cfg: The full configuration (for accessing top-level values like debug)

        Returns:
            Dictionary of configs to be shared across actions
        """
        common_configs = {}

        # Define which pipeline-level configs should NOT be propagated to actions
        pipeline_specific_keys = {
            'actions',  # List of actions in the pipeline
            'cache_enabled',  # Pipeline-level caching control
            'force_rerun',  # Pipeline-level cache bypass
            'cache_max_age_hours',  # Pipeline-level cache expiry (hours)
            'cache_max_age_days',  # Pipeline-level cache expiry (days)
        }

        logger.debug(f"🔧 Pipeline config propagation debug:")
        logger.debug(f"   pipeline_config type: {type(pipeline_config)}")
        logger.debug(f"   pipeline_config keys: {list(pipeline_config.keys()) if hasattr(pipeline_config, 'keys') else 'N/A'}")
        if 'device' in pipeline_config:
            logger.debug(f"   pipeline_config.device: {pipeline_config['device']}")
        else:
            logger.debug(f"   device NOT found in pipeline_config")

        # Propagate all pipeline configs except pipeline-specific ones
        for key, value in pipeline_config.items():
            if key not in pipeline_specific_keys:
                common_configs[key] = value

        # Also check if debug is set at the top level (from CLI)
        if hasattr(cfg, 'debug') and 'debug' not in common_configs:
            common_configs['debug'] = cfg.debug

        logger.debug(f"   common_configs keys: {list(common_configs.keys())}")
        if 'device' in common_configs:
            logger.debug(f"   common_configs.device: {common_configs['device']}")
        else:
            logger.debug(f"   device NOT found in common_configs")

        logger.debug(f"Pipeline configs to propagate to actions: {list(common_configs.keys())}")
        return common_configs
