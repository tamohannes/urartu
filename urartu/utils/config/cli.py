"""
Simple CLI parser for Urartu.

Parses command-line arguments in the format:
  urartu <pipeline_name> [key=value ...] [--key value ...]
"""

import logging
import os
import pwd
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Get current user for config directory checking
try:
    current_user = pwd.getpwuid(os.getuid()).pw_name
except Exception:
    current_user = os.getenv("USER", "default")


def parse_override(key: str, value: str) -> Tuple[List[str], Any, bool]:
    """
    Parse a key=value override into a path and value.

    Supports nested keys like "pipeline.actions.0.model.name"

    Args:
        key: The key (may contain dots for nesting)
        value: The value (will be converted to appropriate type)

    Returns:
        Tuple of (path_list, value, is_config_group_selector) where:
        - path_list is the list of keys
        - value is the converted value
        - is_config_group_selector is True if unquoted and looks like a config group (simple identifier)
    """
    # Split by dots to handle nested keys
    path = key.split(".")

    # Check if value is quoted
    is_quoted = False
    original_value = value

    # Check for quoted strings (single or double quotes)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        is_quoted = True

    # Convert value to appropriate type
    converted_value = convert_value(value, is_quoted=is_quoted)

    # Determine if this is a config group selector
    # Config group selectors are: unquoted, simple string identifiers (no dots, no special chars except underscore)
    # Note: Boolean values (true/false), numbers, and None are NOT config group selectors
    # Examples:
    #   - machine=local → config group selector (loads machine/local.yaml)
    #   - machine="local" → string override (sets cfg.machine = "local")
    #   - debug=true → boolean override (sets cfg.debug = True, NOT a config group selector)
    #   - debug="true" → string override (sets cfg.debug = "true")
    is_config_group_selector = False
    if not is_quoted and isinstance(converted_value, str):
        # Simple identifier: alphanumeric + underscore/hyphen, no dots, no spaces
        # This excludes booleans, numbers, and None which are already converted to their types
        if converted_value.replace('_', '').replace('-', '').isalnum() and '.' not in converted_value:
            is_config_group_selector = True

    return path, converted_value, is_config_group_selector


def convert_value(value: str, is_quoted: bool = False) -> Any:
    """
    Convert a string value to appropriate Python type.

    Handles: bool, int, float, None, strings

    Args:
        value: The string value to convert
        is_quoted: Whether the value was quoted (single or double quotes)
                  If quoted, always return as string. If not quoted, may be config group selector.
    """
    # If quoted, always return as string (remove quotes)
    if is_quoted:
        # Remove surrounding quotes if present
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

    # For unquoted values, check if it's a special value or number
    # Handle None
    if value.lower() in ["null", "none", ""]:
        return None

    # Handle booleans (these are NOT config group selectors)
    # Examples: debug=true, debug=false, use_aim=true
    if value.lower() in ["true", "yes", "1"]:
        return True
    if value.lower() in ["false", "no", "0"]:
        return False

    # Handle integers
    try:
        return int(value)
    except ValueError:
        pass

    # Handle floats
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string (may be config group selector)
    return value


def set_nested_value(d: Dict[str, Any], path: List[str], value: Any) -> None:
    """
    Set a nested value in a dictionary using a path list.

    Args:
        d: Dictionary to modify
        path: List of keys (e.g., ["pipeline", "actions", "0", "model", "name"])
        value: Value to set
    """
    current = d

    # Navigate to the parent
    for key in path[:-1]:
        # Handle list indices
        if key.isdigit():
            idx = int(key)
            if not isinstance(current, list) or idx >= len(current):
                raise ValueError(f"Invalid list index in path: {'.'.join(path)}")
            current = current[idx]
        else:
            if key not in current:
                current[key] = {}
            current = current[key]

    # Set the final value
    final_key = path[-1]
    if final_key.isdigit():
        idx = int(final_key)
        if not isinstance(current, list) or idx >= len(current):
            raise ValueError(f"Invalid list index in path: {'.'.join(path)}")
        current[idx] = value
    else:
        current[final_key] = value


def is_config_group_key(key: str, cwd: Path) -> bool:
    """
    Check if a key corresponds to a config group by checking if config directories exist.

    A config group is identified by the existence of a directory with that name in:
    1. User-specific project configs: configs_{user}/{key}/
    2. General project configs: configs/{key}/
    3. Urartu package configs: urartu/urartu/config/{key}/

    Args:
        key: The key to check (e.g., "slurm", "aim", "machine")
        cwd: Current working directory (project root)

    Returns:
        True if the key corresponds to a config group, False otherwise
    """
    import urartu

    urartu_root = Path(urartu.__file__).parent.parent
    urartu_config_dir = urartu_root / "urartu" / "config"

    # Get current user
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        user = os.getenv("USER", "default")

    # Check user-specific project configs
    user_config_dir = cwd / f"configs_{user}" / key
    if user_config_dir.exists() and user_config_dir.is_dir():
        return True

    # Check general project configs
    project_config_dir = cwd / "configs" / key
    if project_config_dir.exists() and project_config_dir.is_dir():
        return True

    # Check Urartu package configs
    package_config_dir = urartu_config_dir / key
    if package_config_dir.exists() and package_config_dir.is_dir():
        return True

    return False


def parse_args() -> Tuple[str, Dict[str, Any], Dict[str, str]]:
    """
    Parse command-line arguments.

    Expected format:
      urartu <pipeline_name> [key=value ...] [--key value ...]

    Unquoted values for keys that match config group directories are treated as config group selectors.
    Quoted values or keys that don't match config groups are treated as direct overrides.

    Examples:
      - slurm=slurm → config group selector (loads slurm/slurm.yaml)
      - slurm="slurm" → string override (sets cfg.slurm = "slurm")
      - debug=true → boolean override (sets cfg.debug = True)
      - descr="new version" → string override (sets cfg.descr = "new version")

    Returns:
        Tuple of (pipeline_name, overrides_dict, config_group_selectors)
        where config_group_selectors maps config type to selector name
        (e.g., {"machine": "ukp", "slurm": "slurm", "aim": "no_aim"})

    Raises:
        ValueError: If no pipeline name is provided or arguments are malformed
    """
    import os
    import pwd
    from pathlib import Path

    # Get current user for config directory checking
    try:
        current_user = pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        current_user = os.getenv("USER", "default")

    args = sys.argv[1:]

    if not args:
        raise ValueError("No pipeline name provided. Usage: urartu <pipeline_name> [key=value ...]")

    # First argument is the pipeline name
    pipeline_name = args[0]

    # Get current working directory for config group detection
    cwd = Path.cwd()

    # Parse remaining arguments as overrides
    overrides = {}
    config_group_selectors = {}  # Track config group selectors separately
    i = 1

    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            # --key value format
            key = arg[2:]  # Remove --
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for argument: {key}")
            value = args[i + 1]
            i += 2

            # Parse the override
            path, converted_value, is_config_group = parse_override(key, value)

            # Check if this is a config group selector:
            # 1. Must be top-level (len(path) == 1)
            # 2. Must be unquoted string (is_config_group is True)
            # 3. The key must correspond to an existing config group directory
            if is_config_group and len(path) == 1 and is_config_group_key(path[0], cwd):
                config_group_selectors[path[0]] = converted_value
            else:
                set_nested_value(overrides, path, converted_value)
        elif "=" in arg:
            # key=value format
            key, value = arg.split("=", 1)
            path, converted_value, is_config_group = parse_override(key, value)

            # Check if this is a config group selector
            if is_config_group and len(path) == 1 and is_config_group_key(path[0], cwd):
                config_group_selectors[path[0]] = converted_value
            else:
                set_nested_value(overrides, path, converted_value)
            i += 1
        else:
            # Standalone value (treat as boolean flag)
            # This is less common but we'll support it for backward compatibility
            logger.warning(f"Standalone argument '{arg}' will be ignored. Use key=value format.")
            i += 1

    return pipeline_name, overrides, config_group_selectors


def print_usage():
    """Print usage information."""
    print(
        """Usage: urartu <pipeline_name> [overrides...]

Required:
  <pipeline_name>    Name of the pipeline to run (without .yaml extension)

Overrides (optional):
  key=value          Set a config value (supports nested: pipeline.actions.0.model.name=value)
  --key value        Alternative format for setting config values

Examples:
  urartu _attention_visualization_pipeline
  urartu _attention_visualization_pipeline aim.use_aim=true
  urartu _attention_visualization_pipeline pipeline.device=cuda --pipeline.seed 42
"""
    )


if __name__ == "__main__":
    try:
        pipeline_name, overrides = parse_args()
        print(f"Pipeline: {pipeline_name}")
        print(f"Overrides: {overrides}")
    except ValueError as e:
        print(f"Error: {e}")
        print_usage()
        sys.exit(1)
