"""
Custom config loader for Urartu that replaces Hydra.

Supports hierarchical config includes, config merging, and variable interpolation
using OmegaConf.
"""

import os
import pwd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger

logger = get_logger(__name__)

# Get current user for user-specific configs
try:
    current_user = pwd.getpwuid(os.getuid()).pw_name
except Exception:
    current_user = os.getenv("USER", "default")


def resolve_config_path(config_path: str, cwd: Path, config_type: str = "pipeline") -> Optional[Path]:
    """
    Resolve a config path to an actual file, checking user-specific paths first.

    Args:
        config_path: Relative path like "pipeline_name" or "/action/task/dataset/templates_v3"
        cwd: Current working directory (project root)
        config_type: Type of config ("pipeline", "action", or path component)

    Returns:
        Path to config file if found, None otherwise
    """
    # Handle absolute paths (starting with /)
    if config_path.startswith("/"):
        # Remove leading slash and split into components
        parts = config_path.lstrip("/").split("/")
        # Check user-specific first
        user_path = cwd / f"configs_{current_user}" / Path(*parts).with_suffix(".yaml")
        if user_path.exists():
            return user_path
        # Then general
        general_path = cwd / "configs" / Path(*parts).with_suffix(".yaml")
        if general_path.exists():
            return general_path
        return None

    # Handle relative paths (pipeline names, action names, etc.)
    # Check user-specific first
    user_path = cwd / f"configs_{current_user}" / config_type / f"{config_path}.yaml"
    if user_path.exists():
        return user_path

    # Then general
    general_path = cwd / "configs" / config_type / f"{config_path}.yaml"
    if general_path.exists():
        return general_path

    return None


def load_config_with_includes(config_path: Path, cwd: Path, visited: Optional[set] = None) -> DictConfig:
    """
    Load a config file and recursively resolve all includes from defaults.

    Args:
        config_path: Path to the config file
        cwd: Current working directory
        visited: Set of already visited config paths to detect cycles

    Returns:
        Merged DictConfig with all includes resolved
    """
    if visited is None:
        visited = set()

    # Normalize path to detect cycles
    normalized_path = config_path.resolve()
    if normalized_path in visited:
        raise ValueError(f"Circular dependency detected: {normalized_path} already visited")
    visited.add(normalized_path)

    # Load the base config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.debug(f"Loading config: {config_path}")
    base_cfg = OmegaConf.load(config_path)

    # Process defaults (includes)
    if "defaults" in base_cfg:
        defaults = base_cfg.defaults
        merged_configs = []

        for default_entry in defaults:
            if isinstance(default_entry, str):
                # Parse include syntax: "/path/to/config@alias" or just "/path/to/config"
                if "@" in default_entry:
                    include_path, alias = default_entry.rsplit("@", 1)
                    alias = alias.strip()
                else:
                    include_path = default_entry
                    alias = None

                include_path = include_path.strip()

                # Resolve the include path
                # Determine config type from path
                if include_path.startswith("/action/"):
                    config_type = "action"
                elif include_path.startswith("/pipeline/"):
                    config_type = "pipeline"
                else:
                    # Try to infer from path structure
                    parts = include_path.lstrip("/").split("/")
                    if len(parts) > 0:
                        config_type = parts[0] if parts[0] in ["action", "pipeline"] else "action"

                resolved_path = resolve_config_path(include_path, cwd, config_type)
                if resolved_path is None:
                    logger.warning(f"Could not resolve include path: {include_path}")
                    continue

                # Recursively load the included config
                included_cfg = load_config_with_includes(resolved_path, cwd, visited.copy())

                # If alias is specified, merge at that path, otherwise merge at root
                if alias:
                    # Create a nested structure with the alias as the key
                    alias_cfg = OmegaConf.create({alias: included_cfg})
                    merged_configs.append(alias_cfg)
                else:
                    # Merge at root level
                    merged_configs.append(included_cfg)

        # Remove defaults from base config before merging
        base_dict = OmegaConf.to_container(base_cfg, resolve=False)
        if "defaults" in base_dict:
            del base_dict["defaults"]
        base_cfg = OmegaConf.create(base_dict)

        # Merge all configs: includes first, then base (base overrides includes)
        all_configs = merged_configs + [base_cfg]
        merged = OmegaConf.merge(*all_configs)
    else:
        merged = base_cfg

    return merged


def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    """
    Apply command-line overrides to a config.

    Supports nested keys like "pipeline.actions.0.model.name=value"

    Args:
        cfg: Base config
        overrides: Dictionary of overrides (keys can be dot-separated paths)

    Returns:
        Config with overrides applied
    """
    if not overrides:
        return cfg

    # Convert to container for easier manipulation
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)

    for key, value in overrides.items():
        # Handle nested keys (e.g., "pipeline.actions.0.model.name")
        keys = key.split(".")
        current = cfg_dict

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            # Handle list indices
            if k.isdigit():
                k = int(k)
                if not isinstance(current, list) or k >= len(current):
                    raise ValueError(f"Invalid list index in override path: {key}")
                current = current[k]
            else:
                if k not in current:
                    current[k] = {}
                current = current[k]

        # Set the final value
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            if not isinstance(current, list) or final_key >= len(current):
                raise ValueError(f"Invalid list index in override path: {key}")
            current[final_key] = value
        else:
            current[final_key] = value

    return OmegaConf.create(cfg_dict)


def resolve_variables(cfg: DictConfig, pipeline_name: str) -> DictConfig:
    """
    Resolve variable interpolations in config (e.g., ${now:%Y-%m-%d}, ${oc.select:...}).

    Args:
        cfg: Config to resolve
        pipeline_name: Pipeline name for variable resolution

    Returns:
        Config with variables resolved
    """
    # Create a temporary config with pipeline_name for oc.select
    temp_cfg = OmegaConf.create({"pipeline_name": pipeline_name})

    # Register custom resolvers
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda pattern: datetime.now().strftime(pattern))

    if not OmegaConf.has_resolver("oc.select"):

        def oc_select(*args):
            """Select first non-empty value from args."""
            for arg in args:
                if arg and arg != "???":
                    return arg
            return None

        OmegaConf.register_new_resolver("oc.select", oc_select)

    # Resolve the config
    try:
        resolved = OmegaConf.to_container(cfg, resolve=True)
        return OmegaConf.create(resolved)
    except Exception as e:
        # Check if the error is about missing interpolation keys (like action_name)
        # This is OK for overrides that are already resolved (e.g., run_dir passed from local)
        error_str = str(e)
        if "Interpolation key" in error_str and "not found" in error_str:
            # Extract the key that's missing
            if "action_name" in error_str or "run_dir" in error_str:
                # This is likely a run_dir override that's already resolved
                # Just log a debug message and continue
                logger.debug(f"Variable resolution warning (likely already resolved override): {e}")
                # Try to resolve with missing keys as None
                try:
                    # Create a config with default values for common missing keys
                    temp_resolve_cfg = OmegaConf.create(
                        {
                            "pipeline_name": pipeline_name,
                            "action_name": None,  # Will be ignored by oc.select
                        }
                    )
                    # Merge with original config
                    merged = OmegaConf.merge(temp_resolve_cfg, cfg)
                    resolved = OmegaConf.to_container(merged, resolve=True)
                    return OmegaConf.create(resolved)
                except Exception:
                    # If that fails, just return the config as-is
                    logger.debug(f"Could not resolve with defaults, using config as-is")
                    return cfg
            else:
                logger.warning(f"Error resolving variables: {e}, continuing with unresolved config")
        else:
            logger.warning(f"Error resolving variables: {e}, continuing with unresolved config")
        return cfg


def load_config_group(
    cwd: Path, config_group: str, selector: Optional[str] = None, default_file: Optional[str] = None, raise_on_not_found: bool = False
) -> Optional[DictConfig]:
    """
    Load a config group (e.g., slurm, aim, machine) with support for selectors.

    Search order (matching Hydra plugin behavior):
    1. User-specific project configs: `configs_{user}/{config_group}/{selector}.yaml`
    2. General project configs: `configs/{config_group}/{selector}.yaml`
    3. Urartu package defaults: `urartu/urartu/config/{config_group}/{selector}.yaml`

    Args:
        cwd: Current working directory (project root)
        config_group: Name of the config group (e.g., "slurm", "aim", "machine")
        selector: Optional selector name (e.g., "slurm", "no_aim", "ukp")
                  If None, uses default_file
        default_file: Optional default file name (e.g., "default_slurm.yaml")
                      Used if selector is None
        raise_on_not_found: If True, raise FileNotFoundError if config is not found
                           (used when selector is explicitly provided by user)

    Returns:
        DictConfig with the loaded config, or None if not found (unless raise_on_not_found=True)

    Raises:
        FileNotFoundError: If raise_on_not_found=True and config file is not found
    """
    # Get Urartu package config directory
    import urartu

    urartu_root = Path(urartu.__file__).parent.parent
    urartu_config_dir = urartu_root / "urartu" / "config"

    # Determine which file to load
    if selector:
        config_file = f"{selector}.yaml"
    elif default_file:
        config_file = default_file
    else:
        # Try to find a default file
        config_file = f"default_{config_group}.yaml"

    # Collect all paths we checked for error message
    checked_paths = []

    # 1. Check user-specific project configs first
    user_config_path = cwd / f"configs_{current_user}" / config_group / config_file
    checked_paths.append(str(user_config_path))
    if user_config_path.exists():
        logger.info(f"Loading {config_group} from user-specific config: {user_config_path}")
        loaded = OmegaConf.load(user_config_path)
        # Handle both nested (slurm: {use_slurm: true}) and flat (use_slurm: true) configs
        if config_group in loaded:
            config_value = loaded[config_group]
        else:
            # Flat config - the entire file is the config value
            config_value = loaded
        logger.debug(f"Loaded {config_group} config value: {config_value}")
        return OmegaConf.create({config_group: config_value})

    # 2. Check general project configs
    project_config_path = cwd / "configs" / config_group / config_file
    checked_paths.append(str(project_config_path))
    if project_config_path.exists():
        logger.info(f"Loading {config_group} from project config: {project_config_path}")
        loaded = OmegaConf.load(project_config_path)
        # Handle both nested and flat configs
        if config_group in loaded:
            config_value = loaded[config_group]
        else:
            # Flat config - the entire file is the config value
            config_value = loaded
        logger.debug(f"Loaded {config_group} config value: {config_value}")
        return OmegaConf.create({config_group: config_value})

    # 3. Fall back to Urartu package defaults
    package_config_path = urartu_config_dir / config_group / config_file
    checked_paths.append(str(package_config_path))
    if package_config_path.exists():
        logger.debug(f"Loading {config_group} from package default: {package_config_path}")
        loaded = OmegaConf.load(package_config_path)
        # Handle both nested and flat configs
        if config_group in loaded:
            config_value = loaded[config_group]
        else:
            # Flat config - the entire file is the config value
            config_value = loaded
        return OmegaConf.create({config_group: config_value})

    # Config not found - raise error if requested
    if raise_on_not_found:
        error_msg = f"Config group '{config_group}' with selector '{selector}' not found.\n" f"Checked the following paths:\n"
        for path in checked_paths:
            error_msg += f"  - {path}\n"

        # Check if config group directory exists and list available selectors
        available_selectors = []
        for base_path in [cwd / f"configs_{current_user}", cwd / "configs", urartu_config_dir]:
            config_group_dir = base_path / config_group
            if config_group_dir.exists() and config_group_dir.is_dir():
                # List all .yaml files in the directory (excluding .yaml extension for selector names)
                for file in config_group_dir.glob("*.yaml"):
                    selector_name = file.stem
                    if selector_name not in available_selectors:
                        available_selectors.append(selector_name)

        if available_selectors:
            error_msg += (
                f"\nAvailable selectors for '{config_group}': {', '.join(available_selectors)}\n" f"Example: {config_group}={available_selectors[0]}"
            )
        else:
            error_msg += f"\nPlease ensure the config file exists in one of these locations, " f"or use a different selector."

        raise FileNotFoundError(error_msg)

    return None


def load_default_configs(cwd: Path, config_group_selectors: Optional[Dict[str, str]] = None) -> DictConfig:
    """
    Load default configs dynamically based on config group selectors.

    This function dynamically loads config groups (e.g., slurm, aim, machine) based on
    selectors provided in config_group_selectors. It discovers available config groups
    by checking what directories exist in the config paths.

    Search order (matching Hydra plugin behavior):
    1. User-specific project configs: `configs_{user}/{config_group}/`
    2. General project configs: `configs/{config_group}/`
    3. Urartu package defaults: `urartu/urartu/config/{config_group}/`

    Args:
        cwd: Current working directory (project root)
        config_group_selectors: Dictionary mapping config group names to selectors
                               (e.g., {"slurm": "slurm", "aim": "no_aim", "machine": "ukp"})

    Returns:
        DictConfig with default configs merged (user/project overrides take precedence)
    """
    if config_group_selectors is None:
        config_group_selectors = {}

    defaults = {}

    # Load each config group specified in selectors
    # If a selector is explicitly provided, we should raise an error if it's not found
    for config_group, selector in config_group_selectors.items():
        logger.info(f"Loading config group '{config_group}' with selector '{selector}'")
        loaded = load_config_group(cwd, config_group, selector=selector, raise_on_not_found=True)
        if loaded:
            logger.info(f"Successfully loaded config group '{config_group}': {list(loaded.keys())}")
            defaults.update(OmegaConf.to_container(loaded))
        else:
            logger.warning(f"Config group '{config_group}' with selector '{selector}' returned None")

    # Also load default configs for known groups if not already loaded
    # This ensures backward compatibility and loads defaults for groups not specified
    known_groups = ["aim", "slurm", "machine"]
    for config_group in known_groups:
        if config_group not in defaults:
            # Try to load default
            loaded = load_config_group(cwd, config_group, default_file=f"default_{config_group}.yaml")
            if loaded:
                defaults.update(OmegaConf.to_container(loaded))
            else:
                # Try machine=local as fallback
                if config_group == "machine":
                    loaded = load_config_group(cwd, config_group, selector="local")
                    if loaded:
                        defaults.update(OmegaConf.to_container(loaded))

    return OmegaConf.create(defaults)


def load_pipeline_config(
    pipeline_name: str,
    cwd: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    config_group_selectors: Optional[Dict[str, str]] = None,
) -> DictConfig:
    """
    Load a pipeline config with all includes resolved and overrides applied.

    Args:
        pipeline_name: Name of the pipeline (without .yaml extension)
        cwd: Current working directory (defaults to Path.cwd())
        overrides: Optional command-line overrides to apply (quoted values)
        config_group_selectors: Optional config group selectors (unquoted values like machine=local)

    Returns:
        Fully resolved and merged DictConfig
    """
    if cwd is None:
        cwd = Path.cwd()

    if config_group_selectors is None:
        config_group_selectors = {}

    # Load default configs dynamically based on config group selectors
    default_cfg = load_default_configs(cwd, config_group_selectors)

    # Resolve pipeline config path
    pipeline_path = resolve_config_path(pipeline_name, cwd, "pipeline")
    if pipeline_path is None:
        raise FileNotFoundError(
            f"Pipeline config not found: {pipeline_name}. "
            f"Checked: {cwd / f'configs_{current_user}/pipeline/{pipeline_name}.yaml'} "
            f"and {cwd / f'configs/pipeline/{pipeline_name}.yaml'}"
        )

    logger.info(f"Loading pipeline config: {pipeline_path}")

    # Load config with includes
    pipeline_cfg = load_config_with_includes(pipeline_path, cwd)

    # Ensure pipeline_name is set
    if "pipeline_name" not in pipeline_cfg or pipeline_cfg.pipeline_name is None:
        pipeline_cfg.pipeline_name = pipeline_name

    # Merge defaults with pipeline config (pipeline config overrides defaults)
    cfg = OmegaConf.merge(default_cfg, pipeline_cfg)

    # Apply remaining overrides
    if overrides:
        logger.info(f"Applying {len(overrides)} override(s)")
        cfg = apply_overrides(cfg, overrides)

    # Resolve variables
    cfg = resolve_variables(cfg, pipeline_name)

    return cfg
