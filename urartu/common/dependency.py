"""
Automatic dependency resolution for Urartu pipelines.

This module provides automatic detection and injection of dependencies
between pipeline actions, reducing the need for manual dependency configuration.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger

logger = get_logger(__name__)


class DependencyResolver:
    """
    Automatic dependency resolver for pipeline actions.

    Detects required inputs from action configurations and automatically
    matches them with outputs from previous actions.
    """

    def __init__(self, action_outputs: Dict[str, Dict[str, Any]]):
        """
        Initialize the dependency resolver.

        Args:
            action_outputs: Dictionary mapping action names to their outputs
        """
        self.action_outputs = action_outputs

    def resolve_dependencies(
        self, action_config: Dict[str, Any], current_action_name: str, explicit_depends_on: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Resolve dependencies for an action configuration.

        Supports both explicit `depends_on` declarations and automatic
        detection based on action output schemas.

        Args:
            action_config: The action's configuration dictionary
            current_action_name: Name of the current action
            explicit_depends_on: Explicit dependency declarations (from depends_on)

        Returns:
            Updated configuration with dependencies injected
        """
        # Make a deep copy to avoid modifying the original
        config = copy.deepcopy(action_config)

        # Debug: Log original config keys to ensure nothing is lost
        original_keys = set(config.keys()) if isinstance(config, dict) else set()
        logger.debug(f"🔍 DependencyResolver.resolve_dependencies for '{current_action_name}': original keys = {original_keys}")

        # Remove depends_on from config (it's metadata, not action config)
        depends_on = config.pop('depends_on', None) or explicit_depends_on

        if not depends_on and not self.action_outputs:
            # No dependencies to resolve
            return config

        # Process explicit dependencies first
        if depends_on:
            config = self._inject_explicit_dependencies(config, depends_on, current_action_name)

        # Attempt automatic dependency detection
        config = self._inject_automatic_dependencies(config, current_action_name)

        # Debug: Verify all original keys are preserved (except depends_on which we removed)
        final_keys = set(config.keys()) if isinstance(config, dict) else set()
        expected_keys = original_keys - {'depends_on'}
        if not final_keys.issuperset(expected_keys):
            lost_keys = expected_keys - final_keys
            logger.error(f"❌ ERROR: Lost keys during dependency resolution for '{current_action_name}': {lost_keys}")
            logger.error(f"   Original keys: {original_keys}")
            logger.error(f"   Final keys: {final_keys}")
            logger.error(f"   Expected keys (original - depends_on): {expected_keys}")

        return config

    def _inject_explicit_dependencies(
        self, config: Dict[str, Any], depends_on: Dict[str, Dict[str, str]], current_action_name: str
    ) -> Dict[str, Any]:
        """
        Inject dependencies explicitly declared in depends_on.

        Args:
            config: Configuration dictionary to modify
            depends_on: Explicit dependency declarations
            current_action_name: Name of the current action

        Returns:
            Updated configuration
        """
        logger.debug(f"Processing explicit dependencies for {current_action_name}")

        for source_action_name, mappings in depends_on.items():
            if source_action_name not in self.action_outputs:
                logger.warning(
                    f"Source action '{source_action_name}' has not completed yet " f"or produced no outputs for action '{current_action_name}'"
                )
                continue

            source_outputs = self.action_outputs[source_action_name]
            if not source_outputs:
                logger.warning(f"Source action '{source_action_name}' produced no outputs")
                continue

            # Process each output->config mapping
            for output_key, config_path in mappings.items():
                if output_key in source_outputs:
                    output_value = source_outputs[output_key]
                    self._set_nested_config_value(config, config_path, output_value)
                    logger.debug(f"Injected {source_action_name}.{output_key} → {config_path} = {output_value}")
                else:
                    logger.warning(f"Output '{output_key}' not found in {source_action_name} outputs. " f"Available: {list(source_outputs.keys())}")

        return config

    def _inject_automatic_dependencies(self, config: Dict[str, Any], current_action_name: str) -> Dict[str, Any]:
        """
        Automatically detect and inject dependencies based on common patterns.

        This method looks for common patterns in action configurations that
        indicate dependencies on previous actions, such as:
        - Path fields that reference previous action outputs
        - Null/placeholder values that should be filled
        - Common naming patterns

        Args:
            config: Configuration dictionary to modify
            current_action_name: Name of the current action

        Returns:
            Updated configuration
        """
        # For now, automatic detection is limited to explicit depends_on
        # Future enhancement: analyze action schemas and output contracts
        # to automatically match inputs to outputs

        return config

    def _set_nested_config_value(self, config: Dict[str, Any], path: str, value: Any):
        """
        Set a value at a nested path in the configuration dictionary.

        Args:
            config: Configuration dictionary to modify
            path: Dot-separated path (e.g., 'dataset.data_files')
            value: Value to set
        """
        from omegaconf import DictConfig, OmegaConf

        keys = path.split('.')
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif isinstance(current[key], DictConfig):
                # Convert DictConfig to dict to preserve all fields
                # Use resolve=False to keep nested DictConfigs intact, then recursively convert them
                converted = OmegaConf.to_container(current[key], resolve=False)
                # Ensure it's a dict (not a list or other type)
                if not isinstance(converted, dict):
                    logger.warning(f"⚠️ Expected dict at '{key}' but got {type(converted)}, creating new dict")
                    current[key] = {}
                else:
                    # Store original keys to verify nothing is lost
                    original_keys = set(current[key].keys()) if isinstance(current[key], dict) else set()
                    current[key] = converted

                    # Recursively convert any nested DictConfigs to dicts to preserve all fields
                    def convert_nested(d):
                        if isinstance(d, dict):
                            for k, v in list(d.items()):
                                if isinstance(v, DictConfig):
                                    d[k] = OmegaConf.to_container(v, resolve=False)
                                    if isinstance(d[k], dict):
                                        convert_nested(d[k])
                                elif isinstance(v, dict):
                                    convert_nested(v)

                    convert_nested(current[key])
                    # Verify all original keys are preserved
                    new_keys = set(current[key].keys())
                    if original_keys and not new_keys.issuperset(original_keys):
                        lost_keys = original_keys - new_keys
                        logger.error(f"❌ ERROR: Lost keys during DictConfig conversion at '{key}': {lost_keys}")
                        logger.error(f"   Original keys: {original_keys}")
                        logger.error(f"   New keys: {new_keys}")
            elif not isinstance(current[key], dict):
                # If the path conflicts with a non-dict value, create a new dict
                # But preserve the old value with a warning
                logger.warning(f"⚠️ Overwriting non-dict value at '{key}' with dict to set nested path '{path}'")
                current[key] = {}
            else:
                # It's already a dict - ensure nested DictConfigs are converted
                if isinstance(current[key], dict):
                    # Store original keys to verify nothing is lost
                    original_keys = set(current[key].keys())
                    # Check if any nested values are DictConfigs and convert them
                    for nested_key, nested_value in list(current[key].items()):
                        if isinstance(nested_value, DictConfig):
                            current[key][nested_key] = OmegaConf.to_container(nested_value, resolve=False)
                            # Recursively convert nested dicts
                            if isinstance(current[key][nested_key], dict):

                                def convert_nested(d):
                                    if isinstance(d, dict):
                                        for k, v in list(d.items()):
                                            if isinstance(v, DictConfig):
                                                d[k] = OmegaConf.to_container(v, resolve=False)
                                                if isinstance(d[k], dict):
                                                    convert_nested(d[k])
                                            elif isinstance(v, dict):
                                                convert_nested(v)

                                convert_nested(current[key][nested_key])
                    # Verify all original keys are preserved
                    new_keys = set(current[key].keys())
                    if original_keys and not new_keys.issuperset(original_keys):
                        lost_keys = original_keys - new_keys
                        logger.error(f"❌ ERROR: Lost keys during nested DictConfig conversion at '{key}': {lost_keys}")
                        logger.error(f"   Original keys: {original_keys}")
                        logger.error(f"   New keys: {new_keys}")
            current = current[key]

        # Set the final value
        # This should NOT overwrite the parent dict - we're just setting one key
        current[keys[-1]] = value

    def get_required_inputs(self, action_config: Dict[str, Any]) -> Set[str]:
        """
        Detect required inputs from action configuration.

        Analyzes the configuration to identify fields that likely need
        to be filled from previous action outputs.

        Args:
            action_config: Action configuration dictionary

        Returns:
            Set of input field paths that need to be filled
        """
        required = set()

        # Look for common patterns indicating required inputs
        # - Null values
        # - Placeholder strings
        # - Path fields that are None or empty

        def scan_dict(d: Dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                current_path = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    scan_dict(value, current_path)
                elif value is None or value == "":
                    required.add(current_path)
                elif isinstance(value, str) and (value.startswith("{{") or value.startswith("TODO") or "placeholder" in value.lower()):
                    required.add(current_path)

        scan_dict(action_config)
        return required
