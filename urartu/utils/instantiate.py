"""
Utility for instantiating objects from config, replacing hydra.utils.instantiate.

All configs are unified as DictConfig throughout Urartu. This function ensures
configs are always DictConfig and only converts to dict when necessary (e.g., for **kwargs).
"""

import importlib
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def instantiate(cfg, *args, **kwargs) -> Any:
    """
    Instantiate an object from a config dictionary.

    Compatible with hydra.utils.instantiate signature:
    - instantiate(cfg) where cfg has '_target_'
    - instantiate(cfg, **kwargs) where kwargs override cfg values
    - instantiate(cfg, extra_cfg) where extra_cfg is merged (Hydra-style)

    The config should have a '_target_' key specifying the class to instantiate,
    e.g., {'_target_': 'torch.nn.Linear', 'in_features': 10, 'out_features': 5}

    All configs are unified as DictConfig. If a regular dict is passed, it's converted to DictConfig.
    Only converts to dict when passing as **kwargs to constructors.

    Args:
        cfg: Configuration with '_target_' key (dict or DictConfig, will be converted to DictConfig)
        *args: Optional extra config (Hydra-style, merged with cfg)
        **kwargs: Additional keyword arguments to pass to the constructor (take precedence)

    Returns:
        Instantiated object
    """
    # Unify to DictConfig - convert dict to DictConfig if needed
    if not isinstance(cfg, DictConfig):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        else:
            raise ValueError(f"Config must be a dict or DictConfig, got {type(cfg)}")

    if '_target_' not in cfg:
        raise ValueError("Config must contain '_target_' key specifying the class to instantiate")

    # Handle Hydra-style extra config dict as second positional arg
    if args:
        extra_cfg = args[0]

        # Convert to DictConfig if needed
        if not isinstance(extra_cfg, DictConfig):
            if isinstance(extra_cfg, dict):
                extra_cfg = OmegaConf.create(extra_cfg)
            else:
                raise ValueError(f"Extra config must be a dict or DictConfig, got {type(extra_cfg)}")

        # Remove 'type' key from extra_cfg if present (it's metadata, not a constructor arg)
        # The 'type' key is used to specify _target_, but shouldn't be passed to the constructor
        extra_cfg_clean = OmegaConf.create({k: v for k, v in extra_cfg.items() if k != 'type'})

        # Merge extra config with cfg (cfg takes precedence for _target_)
        # Both are now DictConfig, so use OmegaConf.merge
        cfg = OmegaConf.merge(extra_cfg_clean, cfg)

    target = cfg['_target_']

    # Split target into module and class name
    if '.' not in target:
        raise ValueError(f"Target must be in format 'module.ClassName', got '{target}'")

    module_path, class_name = target.rsplit('.', 1)

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_path}' does not have attribute '{class_name}'")

    cls = getattr(module, class_name)

    # Create config without _target_ for constructor
    # cfg is now guaranteed to be DictConfig
    config = OmegaConf.create({k: v for k, v in cfg.items() if k != '_target_'})

    # Merge with kwargs (kwargs take precedence)
    if kwargs:
        # Convert kwargs to DictConfig and merge
        config = OmegaConf.merge(config, OmegaConf.create(kwargs))

    # Check if the class's __init__ accepts a single 'cfg' argument
    # This handles classes like ModelForCausalLM that expect cfg as a single DictConfig argument
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        # Remove 'self' from params
        if params and params[0] == 'self':
            params = params[1:]

        # If __init__ only takes one parameter (besides self), pass config as a single argument
        # This preserves DictConfig
        if len(params) == 1:
            return cls(config)
        # Otherwise, try to pass as keyword arguments
        # For keyword arguments, we need to convert DictConfig to dict
        else:
            config_dict = OmegaConf.to_container(config, resolve=True)
            return cls(**config_dict)
    except (ValueError, TypeError):
        # If we can't inspect the signature, try keyword arguments (default behavior)
        # Convert DictConfig to dict for **kwargs
        config_dict = OmegaConf.to_container(config, resolve=True)
        return cls(**config_dict)
