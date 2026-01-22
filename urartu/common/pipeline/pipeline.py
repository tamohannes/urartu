"""
Pipeline abstraction for orchestrating sequences of actions in urartu.

This module provides a flexible Pipeline class that can run multiple actions in sequence,
manage data flow between steps, and handle configuration overrides.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from aim import Run
from omegaconf import DictConfig, OmegaConf

from ..action import Action, ActionDataset
from ..device import Device
from .cache import PipelineCache
from .config import ConfigInjector
from .pipeline_action import PipelineAction
from .resolvers import ActionOutputResolver, DataResolver, LoopContextResolver
from .status import PipelineStatusDisplay

# Import from modular pipeline submodules
from .types import ActionOutput, CacheEntry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Re-export for backward compatibility
__all__ = ['Pipeline', 'PipelineAction', 'ActionOutput', 'CacheEntry', 'DataResolver', 'ActionOutputResolver', 'LoopContextResolver']


# Classes are now imported from submodules - no need to redefine them here


class Pipeline:
    """
    Pipeline class for orchestrating multiple actions in sequence.

    Features:
    - Sequential execution of actions
    - Data flow between actions via output tracking
    - Configuration override support
    - Conditional action execution
    - Extensible resolver system for dynamic values
    - Comprehensive error handling and logging
    """

    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        # Store config and aim_run
        self.cfg = cfg
        self.aim_run = aim_run

        # Extract pipeline config from cfg.pipeline
        # This logic is copied from Action.__init__ but simplified for pipelines
        try:
            has_pipeline = hasattr(cfg, 'pipeline') and cfg.pipeline and cfg.pipeline != '???'
            if has_pipeline:
                # Get the pipeline config as a container (dict) without resolving
                pipeline_container = OmegaConf.to_container(cfg.pipeline, resolve=False)
                # Create a new DictConfig from the container to preserve all keys including actions
                self.pipeline_config = OmegaConf.create(pipeline_container)
                logger.debug(f"Pipeline.__init__(): Created pipeline_config copy with {len(self.pipeline_config.keys())} keys")
                if 'actions' in self.pipeline_config:
                    logger.info(f"Pipeline.__init__(): pipeline_config has {len(self.pipeline_config.actions)} actions")
                else:
                    logger.warning(f"Pipeline.__init__(): pipeline_config missing 'actions'! Keys: {list(self.pipeline_config.keys())}")
            else:
                # Fallback: use cfg directly
                self.pipeline_config = cfg
                logger.warning(f"Pipeline.__init__(): No cfg.pipeline found, using cfg directly")
        except Exception as e:
            logger.warning(f"Pipeline.__init__(): Error copying pipeline config: {e}, using direct reference")
            self.pipeline_config = cfg.pipeline if hasattr(cfg, 'pipeline') else cfg

        # Set up device configuration
        has_device_config = isinstance(self.pipeline_config, dict) or (hasattr(self.pipeline_config, 'get') and hasattr(self.pipeline_config, 'keys'))
        device = self.pipeline_config.get('device', 'auto') if has_device_config else 'auto'
        Device.set_device(device)

        # Log AIM status for debugging
        if self.aim_run is not None:
            logger.info(f"📊 AIM tracking enabled: run hash = {self.aim_run.hash}")
        else:
            logger.info("📊 AIM tracking disabled: aim_run is None")

        # Initialize pipeline-specific attributes
        self.actions: List[PipelineAction] = []
        self.action_outputs: Dict[str, ActionOutput] = {}
        self.resolvers: List[DataResolver] = [ActionOutputResolver(), LoopContextResolver()]
        self._initialized = False

        # Cache configuration from pipeline
        self.cache_enabled = self.pipeline_config.get('cache_enabled', True)
        self.force_rerun = self.pipeline_config.get('force_rerun', False)

        # Support both days and hours with proper conversion, prefer days
        cache_max_age_days = self.pipeline_config.get('cache_max_age_days', None)
        cache_max_age_hours = self.pipeline_config.get('cache_max_age_hours', None)

        if cache_max_age_days is not None:
            self.cache_max_age = cache_max_age_days * 24 * 3600  # Convert days to seconds
        elif cache_max_age_hours is not None:
            self.cache_max_age = cache_max_age_hours * 3600  # Convert hours to seconds
        else:
            self.cache_max_age = None

        # Use a shared cache directory by extracting the base .runs folder from run_dir
        # run_dir is typically something like: /path/to/.runs/pipeline_name/timestamp/
        # We want to use: /path/to/.runs/pipeline_cache/
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
        cache_dir = runs_dir / 'cache'

        logger.info(f"📦 Using universal cache directory: {cache_dir}")

        # Convert cache_max_age to seconds if needed (already done above, but ensure it's correct)
        cache_max_age_seconds = self.cache_max_age
        if cache_max_age_seconds is not None and cache_max_age_seconds < 86400:  # If less than a day, might be in wrong units
            # Already converted above, so this is just a safety check
            pass

        # Initialize extracted helper classes
        self.cache_manager = PipelineCache(
            cache_dir=cache_dir, cache_enabled=self.cache_enabled, force_rerun=self.force_rerun, cache_max_age=cache_max_age_seconds
        )
        self.cache_dir = cache_dir  # Keep for backward compatibility

        # ConfigInjector will be initialized when needed
        self.config_injector = None

        # PipelineStatusDisplay will be initialized after run_dir is set
        self.status_display = None

    def _should_use_aim(self) -> bool:
        """Check if Aim tracking is enabled."""
        # The most reliable check: if aim_run is set and not None, AIM is enabled
        if self.aim_run is not None:
            return True

        # Fallback: check config (for cases where aim_run might not be set yet)
        # Check if aim config exists at top level (it's usually there, not in action config)
        try:
            # Try to access via parent Action's cfg which might have top-level access
            # Or check if we can access it through OmegaConf
            from omegaconf import OmegaConf

            aim_cfg = OmegaConf.select(self.cfg, 'aim', default=None)
            if aim_cfg is not None:
                if isinstance(aim_cfg, dict):
                    return aim_cfg.get('use_aim', False)
                elif hasattr(aim_cfg, 'get'):
                    return aim_cfg.get('use_aim', False)
                elif hasattr(aim_cfg, 'use_aim'):
                    return aim_cfg.use_aim
        except (AttributeError, KeyError, Exception):
            pass

        # Also check if it's nested in action config (legacy support)
        try:
            return self.cfg.get('aim', {}).get('use_aim', False)
        except (AttributeError, KeyError):
            try:
                return self.cfg.aim.use_aim
            except (AttributeError, KeyError):
                return False

    def _make_serializable(self, obj):
        """Convert OmegaConf objects to regular Python objects for JSON serialization."""
        from urartu.utils.cache import make_serializable

        return make_serializable(obj)

    def _extract_action_config(
        self, config_obj: Any, action_name: Optional[str] = None, block_keys: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generic method to extract and validate an action config from a config object.

        This method handles:
        - Converting DictConfig to plain dict
        - Recursively converting nested DictConfigs
        - Detecting if the config is actually a block structure (not an action config)
        - Extracting the action config from block structures if needed
        - Validating that we have a proper action config

        Args:
            config_obj: The config object (could be DictConfig, dict, or block structure)
            action_name: Optional action name to search for if config_obj is a block
            block_keys: List of keys that indicate a block structure (default: ['loopable_actions', 'actions'])

        Returns:
            Clean action config dict, or None if extraction fails
        """
        if block_keys is None:
            block_keys = ['loopable_actions', 'actions']

        # Convert to plain dict if it's a DictConfig
        if isinstance(config_obj, DictConfig):
            config_dict = OmegaConf.to_container(config_obj, resolve=False)
        elif isinstance(config_obj, dict):
            config_dict = dict(config_obj)
        else:
            logger.warning(f"_extract_action_config: config_obj is not a dict or DictConfig, got {type(config_obj)}")
            return None

        # Recursively convert nested DictConfigs to plain dicts
        def convert_nested_dictconfigs(d):
            if isinstance(d, dict):
                for k, v in list(d.items()):
                    if isinstance(v, DictConfig):
                        d[k] = OmegaConf.to_container(v, resolve=False)
                        if isinstance(d[k], dict):
                            convert_nested_dictconfigs(d[k])
                    elif isinstance(v, dict):
                        convert_nested_dictconfigs(v)

        convert_nested_dictconfigs(config_dict)

        # Check if this is a block structure (contains block keys)
        is_block = any(key in config_dict for key in block_keys)

        if is_block:
            # This is a block structure, try to extract the action config
            if action_name is None:
                logger.error(f"_extract_action_config: config_obj is a block structure but no action_name provided")
                return None

            # Try to find the action in the block's actions list
            for block_key in block_keys:
                if block_key in config_dict:
                    block = config_dict[block_key]
                    if isinstance(block, dict) and 'actions' in block:
                        actions_list = block['actions']
                    elif isinstance(block, list):
                        actions_list = block
                    else:
                        continue  # Skip if block is neither dict with 'actions' nor a list

                    if isinstance(actions_list, list):
                        for act_cfg in actions_list:
                            act_dict = (
                                OmegaConf.to_container(act_cfg, resolve=False)
                                if isinstance(act_cfg, DictConfig)
                                else dict(act_cfg) if isinstance(act_cfg, dict) else act_cfg
                            )
                            convert_nested_dictconfigs(act_dict)
                            if isinstance(act_dict, dict) and act_dict.get('action_name') == action_name:
                                logger.info(f"_extract_action_config: Extracted action config for '{action_name}' from block structure")
                                return act_dict
                else:
                    continue  # Skip to next block_key if this one is not in config_dict

            logger.error(f"_extract_action_config: Could not find action '{action_name}' in block structure")
            return None

        # This looks like an action config, validate it
        if not isinstance(config_dict, dict):
            logger.warning(f"_extract_action_config: config_dict is not a dict after conversion")
            return None

        # Check if it has action_name (or if we're searching for a specific one, verify it matches)
        if action_name is not None:
            if config_dict.get('action_name') != action_name:
                logger.warning(f"_extract_action_config: action_name mismatch: expected '{action_name}', got '{config_dict.get('action_name')}'")
                return None

        # Return cleaned config (remove metadata keys)
        cleaned_config = {k: v for k, v in config_dict.items() if k not in ['action_name', 'outputs_to_track'] + block_keys}

        return cleaned_config

    def _get_common_pipeline_configs(self) -> Dict[str, Any]:
        """Extract configuration values from pipeline config that should be propagated to all individual actions."""
        return ConfigInjector.get_common_pipeline_configs(self.pipeline_config, self.cfg)

    def _inject_action_outputs(
        self, action_config_dict: Dict[str, Any], current_action_name: str, loop_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Inject outputs from previous actions into the current action's configuration."""
        if self.config_injector is None:
            # Initialize config injector if not already done
            # For base Pipeline, we don't have loopable_actions or loop_configs
            self.config_injector = ConfigInjector(action_outputs=self.action_outputs, loopable_actions=[], loop_configs={})
        return self.config_injector.inject_action_outputs(
            action_config_dict=action_config_dict,
            current_action_name=current_action_name,
            loop_context=loop_context,
            get_iteration_outputs=None,  # Base Pipeline doesn't support iteration outputs
        )

    def _inject_loop_context(self, action_config_dict: Dict[str, Any], loop_context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject loop context parameters into the action's configuration."""
        # Base Pipeline doesn't support loop context injection
        # This should only be called by LoopablePipeline subclasses
        return action_config_dict

    def get_outputs(self) -> Dict[str, Any]:
        """
        Return the outputs produced by this pipeline.

        By default, returns the outputs from the last action in the pipeline.
        Can be overridden by subclasses to return custom pipeline outputs.

        Returns:
            Dict[str, Any]: A dictionary of output keys and values.
        """
        if self.action_outputs:
            # Return outputs from the last executed action
            last_action = list(self.action_outputs.values())[-1]
            return last_action.outputs
        return {}

    def run_with_cache(self):
        """
        Override the base Action's run_with_cache to handle pipeline-specific caching.
        Pipelines use their own main() method instead of the base caching logic.
        """
        # For pipelines, we use the pipeline's own caching system
        self.main()  # Calls initialize() and run()

    def add_action(self, action: Union[PipelineAction, Dict]) -> 'Pipeline':
        """
        Add an action to the pipeline.

        Args:
            action: Either a PipelineAction instance or a dict with action configuration

        Returns:
            Self for method chaining
        """
        if isinstance(action, dict):
            action = PipelineAction(**action)
        self.actions.append(action)
        return self

    def add_resolver(self, resolver: DataResolver) -> 'Pipeline':
        """
        Add a custom resolver for handling special values in configurations.

        Args:
            resolver: DataResolver instance

        Returns:
            Self for method chaining
        """
        self.resolvers.append(resolver)
        return self

    def _resolve_value(self, value: Any, context: Optional[Dict[str, Any]] = None, depth: int = 0) -> Any:
        """
        Resolve configuration values that may contain special references.

        Args:
            value: The value to resolve
            context: Context dictionary for resolvers
            depth: Recursion depth for debugging

        Returns:
            Resolved value
        """
        from omegaconf import DictConfig, ListConfig

        context = context or {"action_outputs": self.action_outputs}

        # Handle string values that might need resolution
        if isinstance(value, str):
            # DEBUG: Check if this is a template variable
            if value.startswith("{{actions."):
                logger.info(f"{'  ' * depth}🔍 Found template variable: {value}")
                for resolver in self.resolvers:
                    logger.info(f"{'  ' * depth}  Trying resolver: {resolver.__class__.__name__}")
                    if resolver.can_resolve(value):
                        logger.info(f"{'  ' * depth}  ✅ Resolver can handle it")
                        resolved = resolver.resolve(value, context)
                        logger.info(f"{'  ' * depth}  Resolved to: {resolved}")
                        return resolved
                    else:
                        logger.info(f"{'  ' * depth}  ❌ Resolver cannot handle it")
                logger.warning(f"{'  ' * depth}⚠️ No resolver could handle: {value}")
            else:
                for resolver in self.resolvers:
                    if resolver.can_resolve(value):
                        return resolver.resolve(value, context)

        # Recursively resolve nested structures (handle both dict and DictConfig)
        elif isinstance(value, (dict, DictConfig)):
            return {k: self._resolve_value(v, context, depth + 1) for k, v in value.items()}
        elif isinstance(value, (list, ListConfig)):
            return [self._resolve_value(v, context, depth + 1) for v in value]

        return value

    def _merge_configs(self, base_cfg: DictConfig, overrides: Dict) -> DictConfig:
        """Merge override configuration into base configuration."""
        # Convert to OmegaConf without resolving action references yet
        # Action references will be resolved later during actual execution
        if isinstance(overrides, dict):
            override_cfg = OmegaConf.create(overrides)
        else:
            override_cfg = overrides

        # Merge configurations
        return OmegaConf.merge(base_cfg, override_cfg)

    def _extract_outputs(self, pipeline_action: PipelineAction, action_instance: Any) -> Dict[str, Any]:
        """
        Extract outputs from an action instance.

        Args:
            pipeline_action: The pipeline action
            action_instance: The executed action instance

        Returns:
            Dictionary of extracted outputs
        """
        outputs = {}

        # Get outputs from the standard get_outputs method
        if hasattr(action_instance, 'get_outputs') and callable(action_instance.get_outputs):
            outputs = action_instance.get_outputs()
            if not isinstance(outputs, dict):
                logger.warning(f"get_outputs() from {pipeline_action.action_name} did not return a dict")
                outputs = {}
        else:
            logger.warning(f"Action {pipeline_action.action_name} does not implement get_outputs() method")

        return outputs

    def _print_pipeline_status(self, all_actions: List[PipelineAction], statuses: Dict[str, Dict], current_index: int):
        """Print a clean, colorful status view of pipeline progress."""
        # Initialize status display if not already done
        if self.status_display is None:
            run_dir = Path(self.cfg.get('run_dir', '.')) if self.cfg.get('run_dir') else None
            # Base Pipeline doesn't support loopable actions - use empty lists
            self.status_display = PipelineStatusDisplay(
                pipeline_config=self.pipeline_config,
                loopable_actions=[],  # Base Pipeline doesn't support loopable actions
                loop_iterations=[],  # Base Pipeline doesn't support loopable actions
                run_dir=run_dir,
            )
        self.status_display.print_status(all_actions, statuses, current_index)

    def _run_action_with_status(
        self, pipeline_action: PipelineAction, loop_context: Optional[Dict[str, Any]] = None, iteration_name: Optional[str] = None
    ) -> Tuple[ActionOutput, bool]:
        """
        Execute a single pipeline action and return its output with cache status.

        Args:
            pipeline_action: The action to run
            loop_context: Optional loop context parameters to inject
            iteration_name: Optional iteration-aware name for the action output

        Returns:
            Tuple of (ActionOutput, was_cached: bool)
        """
        # Track if cache was used
        was_cached = False

        # Log action start
        logger.debug(f"\n{'='*80}")
        log_name = iteration_name if iteration_name else pipeline_action.name
        logger.debug(f"Running pipeline action: {log_name} (action: {pipeline_action.action_name})")
        if loop_context:
            logger.debug(f"  Loop context: {loop_context}")
        logger.debug(f"{'='*80}")

        # Get the action output (will check cache internally)
        action_output = self._run_action(pipeline_action, loop_context=loop_context, iteration_name=iteration_name)

        # Check if it came from cache
        if hasattr(action_output, 'metadata') and action_output.metadata.get('from_cache', False):
            was_cached = True

        return action_output, was_cached

    def _run_action(
        self, pipeline_action: PipelineAction, loop_context: Optional[Dict[str, Any]] = None, iteration_name: Optional[str] = None
    ) -> ActionOutput:
        """Execute a single pipeline action."""
        action_name = iteration_name if iteration_name else pipeline_action.name
        logger.debug(f"\n{'='*80}")
        logger.debug(f"Running pipeline action: {action_name} (action: {pipeline_action.action_name})")
        if loop_context:
            logger.debug(f"  Loop context: {loop_context}")
        logger.debug(f"{'='*80}")

        # Check if action should run
        context = {"action_outputs": self.action_outputs}
        if loop_context:
            context["loop_context"] = loop_context
        if not pipeline_action.should_run(context):
            logger.info(f"Skipping action {action_name} due to condition")
            return ActionOutput(name=action_name, action_name=pipeline_action.action_name, metadata={"skipped": True})

        debug_mode = self.pipeline_config.get('debug', False)

        logger.debug(f"🔍 DEBUG: Available action_outputs before resolution: {list(self.action_outputs.keys())}")
        for name, output in self.action_outputs.items():
            # Handle both ActionOutput (regular actions) and dict (loopable actions)
            # Check ActionOutput first (it might be dict-like)
            if isinstance(output, ActionOutput):
                logger.debug(f"   - {name}: outputs={list(output.outputs.keys()) if output.outputs else 'None'}")
            elif isinstance(output, dict):
                # Loopable action - show iterations
                try:
                    iterations_info = ", ".join(
                        [
                            f"{iter_id}: {len(iter_output.outputs) if isinstance(iter_output, ActionOutput) and iter_output.outputs else 0} keys"
                            for iter_id, iter_output in output.items()
                            if isinstance(iter_output, ActionOutput)
                        ]
                    )
                    logger.debug(f"   - {name}: loopable action with iterations: {iterations_info}")
                except Exception as e:
                    logger.debug(f"   - {name}: dict but error processing: {e}, type={type(output)}")
            else:
                logger.debug(f"   - {name}: unknown type {type(output)}, value={str(output)[:100]}")

        # DEBUG: Write to file
        debug_file = Path(self.cfg.run_dir) / "pipeline_debug.txt"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_file, "a") as f:
            f.write(f"\n--- Running action: {pipeline_action.name} ---\n")
            f.write(f"Available action_outputs: {list(self.action_outputs.keys())}\n")
            for name, output in self.action_outputs.items():
                # Handle both ActionOutput (regular actions) and dict (loopable actions)
                # Check ActionOutput first (it might be dict-like)
                if isinstance(output, ActionOutput):
                    f.write(f"  {name}: {list(output.outputs.keys()) if output.outputs else 'None'}\n")
                elif isinstance(output, dict):
                    # Loopable action - show iterations
                    for iter_id, iter_output in output.items():
                        if isinstance(iter_output, ActionOutput):
                            f.write(f"  {name}[{iter_id}]: {list(iter_output.outputs.keys()) if iter_output.outputs else 'None'}\n")
                        else:
                            f.write(f"  {name}[{iter_id}]: unknown type {type(iter_output)}\n")
                else:
                    f.write(f"  {name}: unknown type {type(output)}, value={str(output)[:100]}\n")
            f.write(f"Config overrides before resolution: {str(pipeline_action.config_overrides)[:500]}\n")

        # Resolve config overrides to handle any references
        context = {"action_outputs": self.action_outputs}
        if loop_context:
            context["loop_context"] = loop_context

        # Always log for debugging this issue
        logger.info(f"🔍 Config overrides before resolution for '{action_name}':")
        logger.info(f"   Type: {type(pipeline_action.config_overrides)}")
        logger.info(f"   Keys: {list(pipeline_action.config_overrides.keys()) if isinstance(pipeline_action.config_overrides, dict) else 'N/A'}")
        # Check for model key if needed (base Pipeline doesn't track loopable actions)
        # Subclasses can override this behavior
        if False:  # Base Pipeline doesn't have loopable_action_configs
            if isinstance(pipeline_action.config_overrides, dict) and 'model' in pipeline_action.config_overrides:
                logger.info(
                    f"   model keys: {list(pipeline_action.config_overrides.get('model', {}).keys()) if isinstance(pipeline_action.config_overrides.get('model'), dict) else 'N/A'}"
                )
            else:
                logger.warning(f"   ⚠️ model key missing in config_overrides BEFORE resolution!")

        resolved_config_overrides = self._resolve_value(pipeline_action.config_overrides, context)

        # Always log after resolution
        logger.info(f"🔍 Config overrides after resolution for '{action_name}':")
        logger.info(f"   Type: {type(resolved_config_overrides)}")
        logger.info(f"   Keys: {list(resolved_config_overrides.keys()) if isinstance(resolved_config_overrides, dict) else 'N/A'}")
        # Check for model key if needed (base Pipeline doesn't track loopable actions)
        if False:  # Base Pipeline doesn't have loopable_action_configs
            if isinstance(resolved_config_overrides, dict) and 'model' in resolved_config_overrides:
                logger.info(
                    f"   model keys: {list(resolved_config_overrides.get('model', {}).keys()) if isinstance(resolved_config_overrides.get('model'), dict) else 'N/A'}"
                )
            else:
                logger.error(f"   ❌ ERROR: model key missing in config_overrides AFTER resolution!")
                logger.error(f"   Full resolved config: {resolved_config_overrides}")

        # DEBUG: Write resolution result to file
        with open(debug_file, "a") as f:
            f.write(f"Config overrides after resolution: {str(resolved_config_overrides)[:500]}\n")

        # Import the action module
        # Actions are always loaded from actions/ directory
        import sys

        current_dir = Path.cwd()

        # Detect project structure by walking up the directory tree to find the project root
        # The project root should contain 'self_aware/' as a subdirectory with 'actions/' inside it
        project_root = None
        actions_dir = None

        # Walk up the directory tree to find the project root
        search_dir = current_dir
        max_levels = 5  # Prevent infinite loops
        for _ in range(max_levels):
            # Check if this directory contains self_aware/actions/
            if (search_dir / "self_aware" / "actions").exists():
                project_root = search_dir
                actions_dir = search_dir / "self_aware" / "actions"
                break
            # Check if this directory itself is self_aware/ with actions/
            elif search_dir.name == "self_aware" and (search_dir / "actions").exists():
                project_root = search_dir.parent
                actions_dir = search_dir / "actions"
                break
            # Check if this directory has actions/ directly (flat structure)
            elif (search_dir / "actions").exists() and (search_dir / "self_aware").exists():
                project_root = search_dir
                actions_dir = search_dir / "actions"
                break
            # Go up one level
            if search_dir == search_dir.parent:
                break  # Reached filesystem root
            search_dir = search_dir.parent

        # Fallback: use current directory if we couldn't find the structure
        if project_root is None:
            project_root = current_dir
            actions_dir = current_dir / "actions"

        # Add project root to sys.path first (so imports like 'from self_aware.utils' work)
        # This allows actions to import from self_aware package
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Also add actions directory to sys.path for direct imports
        if str(actions_dir) not in sys.path:
            sys.path.append(str(actions_dir))

        logger.debug(f"📦 Current dir: {current_dir}")
        logger.debug(f"📦 Project root: {project_root}, Actions dir: {actions_dir}")
        logger.debug(f"📦 sys.path entries: {[p for p in sys.path if 'self_aware' in p or 'actions' in p]}")

        logger.info(f"📦 Attempting to import action: '{pipeline_action.action_name}'")
        try:
            # First try fully qualified package import (actions.<name>)
            action_module = importlib.import_module(f"actions.{pipeline_action.action_name}")
            logger.info(f"✅ Successfully imported action module: actions.{pipeline_action.action_name}")
        except ImportError:
            # Fall back to importing from local actions directory
            try:
                action_module = importlib.import_module(pipeline_action.action_name)
                logger.info(f"✅ Successfully imported action module: {pipeline_action.action_name}")
            except ImportError as e:
                logger.error(f"❌ Failed to import action '{pipeline_action.action_name}' from actions directory {actions_dir}: {e}")
                raise ImportError(f"Failed to import action '{pipeline_action.action_name}' from actions directory {actions_dir}: {e}")

        # Prepare configuration for this action
        action_cfg = OmegaConf.create(self.cfg)  # Deep copy base config
        action_cfg.action_name = pipeline_action.action_name  # Set the action name

        # Ensure run_dir is preserved (it should always be set by pipeline initialization)
        if 'run_dir' not in action_cfg or not action_cfg.run_dir:
            # Inherit from pipeline's run_dir
            if hasattr(self.cfg, 'run_dir') and self.cfg.run_dir:
                action_cfg.run_dir = self.cfg.run_dir
            else:
                raise ValueError(
                    f"run_dir is not set in pipeline config. "
                    f"This should be set during pipeline initialization. "
                    f"Pipeline cfg keys: {list(self.cfg.keys())}"
                )

        # Remove pipeline key so Action instances will use cfg.action instead
        if 'pipeline' in action_cfg:
            del action_cfg.pipeline

        # Get pipeline config hash to ensure cache invalidation when pipeline config changes
        pipeline_config_hash = PipelineCache.generate_config_hash(self.cfg)
        logger.debug(f"🔑 Adding pipeline config hash to action config: {pipeline_config_hash}")

        # Apply config overrides by creating proper action config structure
        if pipeline_action.config_overrides:
            # Use resolved config overrides (with pipeline references resolved)
            action_config_dict = resolved_config_overrides

            # Always log for debugging this issue
            logger.info(f"🔍 Before injecting action outputs for '{action_name}':")
            logger.info(f"   Config keys: {list(action_config_dict.keys()) if isinstance(action_config_dict, dict) else 'N/A'}")
            # Check for model key if needed (base Pipeline doesn't track loopable actions)
            if False:  # Base Pipeline doesn't have loopable_action_configs
                if isinstance(action_config_dict, dict) and 'model' in action_config_dict:
                    logger.info(
                        f"   model keys: {list(action_config_dict.get('model', {}).keys()) if isinstance(action_config_dict.get('model'), dict) else 'N/A'}"
                    )
                else:
                    logger.error(f"   ❌ ERROR: model key missing BEFORE injecting action outputs!")
                    logger.error(f"   Full config: {action_config_dict}")

            # Inject outputs from previous actions into config
            action_config_dict = self._inject_action_outputs(action_config_dict, pipeline_action.name, loop_context=loop_context)

            # Always log after injecting action outputs
            logger.info(f"🔍 After injecting action outputs for '{action_name}':")
            logger.info(f"   Config keys: {list(action_config_dict.keys()) if isinstance(action_config_dict, dict) else 'N/A'}")
            # Check for model key if needed (base Pipeline doesn't track loopable actions)
            if False:  # Base Pipeline doesn't have loopable_action_configs
                if isinstance(action_config_dict, dict) and 'model' in action_config_dict:
                    logger.info(
                        f"   model keys: {list(action_config_dict.get('model', {}).keys()) if isinstance(action_config_dict.get('model'), dict) else 'N/A'}"
                    )
                else:
                    logger.error(f"   ❌ ERROR: model key missing after injecting action outputs for '{action_name}'!")
                    logger.error(f"   Full config: {action_config_dict}")

            # Inject loop context if provided
            if loop_context:
                action_config_dict = self._inject_loop_context(action_config_dict, loop_context)
                logger.debug(f"🔄 Injected loop context into '{action_name}': {loop_context}")

            # Propagate common pipeline-level configs to individual actions
            pipeline_common_configs = self._get_common_pipeline_configs()

            logger.debug(f"🔧 Config merge debug for '{pipeline_action.name}':")
            logger.debug(f"   action_config_dict type: {type(action_config_dict)}")
            if 'device' in action_config_dict:
                logger.debug(f"   action_config_dict.device: {action_config_dict['device']}")
            else:
                logger.debug(f"   device NOT found in action_config_dict")

                # Log any dependencies that were processed
            if 'depends_on' in resolved_config_overrides:
                logger.debug(f"   🔗 Dependencies declared: {list(resolved_config_overrides['depends_on'].keys())}")
            else:
                logger.debug(f"   📝 No dependencies declared for this action")

            # Merge pipeline common configs with action-specific configs
            # Action-specific configs take precedence over pipeline configs
            # OmegaConf.merge: later arguments override earlier ones
            merged_config = OmegaConf.merge(
                OmegaConf.create(pipeline_common_configs),  # Base (pipeline defaults)
                OmegaConf.create(action_config_dict),  # Override (action-specific, with injections)
            )

            # Add pipeline config hash to ensure cache invalidation when pipeline config changes
            merged_config.pipeline_config_hash = pipeline_config_hash
            logger.debug(f"🔑 Added pipeline config hash {pipeline_config_hash} to action '{pipeline_action.name}'")

            logger.debug(f"   merged_config type: {type(merged_config)}")
            if 'device' in merged_config:
                logger.debug(f"   merged_config.device: {merged_config['device']}")
            else:
                logger.debug(f"   device NOT found in merged_config")

                # Log config override details
            overridden_keys = set(pipeline_common_configs.keys()) & set(action_config_dict.keys())
            if overridden_keys:
                logger.debug(f"Action '{pipeline_action.name}' overrides pipeline configs: {list(overridden_keys)}")

            # Loop context was already injected above, no need to inject again
            logger.debug(f"🔄 Loop context already injected into merged config for '{action_name}': {loop_context}")

            # Set iteration_id in action config for loopable actions (so get_run_dir() can use it)
            # Use a method that can be overridden by subclasses (like LoopablePipeline)
            if loop_context:
                iteration_id = self._get_iteration_id_from_context(loop_context)
                if iteration_id:
                    merged_config.iteration_id = iteration_id
                    logger.debug(f"🔑 Set iteration_id='{iteration_id}' in action config for '{action_name}'")

            action_cfg.action = merged_config  # Set the action configuration (Action instances look for cfg.action)
            logger.debug(f"Applied config overrides with pipeline common configs for {action_name}")
        else:
            # Even if no overrides, apply common pipeline configs
            pipeline_common_configs = self._get_common_pipeline_configs()
            base_config = OmegaConf.create(pipeline_common_configs)

            # Inject loop context if provided
            if loop_context:
                base_config_dict = OmegaConf.to_container(base_config, resolve=False)
                base_config_dict = self._inject_loop_context(base_config_dict, loop_context)
                base_config = OmegaConf.create(base_config_dict)
                logger.debug(f"🔄 Injected loop context into base config for '{action_name}': {loop_context}")

                # Set iteration_id in action config for loopable actions
                iteration_id = self._get_iteration_id_from_context(loop_context)
                if iteration_id:
                    base_config.iteration_id = iteration_id
                    logger.debug(f"🔑 Set iteration_id='{iteration_id}' in base config for '{action_name}'")

            # Add pipeline config hash even for actions without overrides
            base_config.pipeline_config_hash = pipeline_config_hash
            logger.debug(f"🔑 Added pipeline config hash {pipeline_config_hash} to action '{action_name}'")

            action_cfg.action = base_config  # Set the action configuration (Action.__init__ looks for cfg.action)

            logger.debug(f"🔧 No action overrides for '{pipeline_action.name}' - using pipeline configs:")
            if 'device' in pipeline_common_configs:
                logger.debug(f"   pipeline device will be used: {pipeline_common_configs['device']}")
            else:
                logger.debug(f"   No device in pipeline_common_configs - will default to 'auto'")

            logger.debug(f"Applied pipeline common configs for {pipeline_action.name}")

        # Create a sub-context in Aim for this action
        # Track action metadata (only if Aim is enabled)
        if self._should_use_aim():
            if self.aim_run is None:
                logger.warning(f"⚠️ _should_use_aim() returned True but self.aim_run is None for action {pipeline_action.name}")
            else:
                try:
                    self.aim_run[f"pipeline_action_{pipeline_action.name}_config"] = {
                        "action": pipeline_action.action_name,
                        "overrides": pipeline_action.config_overrides,
                    }
                    logger.debug(f"📊 Tracked action config in AIM: {pipeline_action.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to track action config in AIM: {e}")

        # Try to find action class - look for classes that inherit from Action
        action_class = None
        action_instance = None  # Initialize to None
        action_candidates = []
        was_cached = False  # Track whether action used cache

        for attr_name in dir(action_module):
            attr = getattr(action_module, attr_name)
            if isinstance(attr, type) and issubclass(attr, Action) and attr != Action and attr != ActionDataset:  # Exclude abstract base classes
                action_candidates.append(attr)

        # Prefer the most specific class (not imported from urartu.common)
        for candidate in action_candidates:
            if candidate.__module__ == action_module.__name__:
                action_class = candidate
                break

        # Fallback to any suitable candidate
        if not action_class and action_candidates:
            action_class = action_candidates[0]

        if action_class:
            # Use action class with run() method
            action_instance = action_class(action_cfg, self.aim_run)

            # Note: No need to override cache_dir - all actions and pipelines share universal cache
            # The cache_dir is already set correctly in Action instances

            # Run the action with full automation if it implements the new pattern
            # Otherwise fall back to run_with_cache() for backward compatibility
            if hasattr(action_instance, '_run_impl'):
                # New pattern: use full automation
                action_instance.run_with_full_automation()
            elif hasattr(action_instance, 'run_with_cache'):
                # Legacy pattern: use existing caching
                action_instance.run_with_cache()
            elif hasattr(action_instance, 'run'):
                action_instance.run()
            elif hasattr(action_instance, 'main'):
                action_instance.main()
            else:
                raise AttributeError(f"Action {pipeline_action.action_name} has no run() or main() method")

            # Check if plotting is enabled for this action
            plotting_enabled = True
            if hasattr(action_instance, 'action_config') and action_instance.action_config:
                plotting_config = action_instance.action_config.get('plotting', {})
                plotting_enabled = plotting_config.get('enabled', True)

            # Generate plots if enabled and action supports it
            if plotting_enabled:
                # Use new automation pattern if action implements _generate_plots()
                if hasattr(action_instance, '_generate_plots'):
                    logger.debug(f"📊 Calling create_plots_with_automation() for action {pipeline_action.name}")
                    try:
                        if hasattr(action_instance, 'create_plots_with_automation'):
                            action_instance.create_plots_with_automation()
                        else:
                            # Fallback: manually check and call _generate_plots
                            if hasattr(action_instance, 'should_generate_plots') and action_instance.should_generate_plots():
                                plot_data = action_instance._ensure_plot_data() if hasattr(action_instance, '_ensure_plot_data') else None
                                if plot_data:
                                    action_instance.apply_plot_style()
                                    action_instance._generate_plots(plot_data)
                                else:
                                    raise ValueError(f"No plot data available for {pipeline_action.name}")
                    except Exception as e:
                        logger.error(f"❌ Error in create_plots_with_automation() for {pipeline_action.name}: {e}", exc_info=True)
            else:
                logger.debug(f"⏭️  Skipping plots for {pipeline_action.name} (plotting.enabled=false)")

            # Extract outputs
            outputs = self._extract_outputs(pipeline_action, action_instance)
            logger.debug(f"🔍 DEBUG: Extracted outputs from {pipeline_action.name}: {list(outputs.keys()) if outputs else 'None'}")
            if outputs:
                for key, value in outputs.items():
                    logger.debug(f"   - {key}: {str(value)[:100]}")

            # Check if action loaded from cache
            if hasattr(action_instance, '_cached_outputs') and action_instance._cached_outputs:
                was_cached = True
                # Ensure we have outputs even from cached actions
                if not outputs:
                    outputs = action_instance._cached_outputs
                    logger.debug(f"📤 Using cached outputs for pipeline action {pipeline_action.name}: {list(outputs.keys()) if outputs else 'None'}")
        elif hasattr(action_module, 'main'):
            # Fallback to module-level main function
            action_module.main(cfg=action_cfg, aim_run=self.aim_run)
            outputs = {}  # Main-style actions don't return outputs directly
        else:
            raise AttributeError(f"Action {pipeline_action.action_name} has no Action class or main() function")

        # Automatically convert paths to portable format before storing outputs
        portable_outputs = Action._make_outputs_portable(outputs)

        # Use iteration_name if provided, otherwise use pipeline_action.name
        output_name = iteration_name if iteration_name else pipeline_action.name

        action_output = ActionOutput(
            name=output_name,
            action_name=pipeline_action.action_name,
            outputs=portable_outputs,
            metadata={"completed": True, "from_cache": was_cached},
        )

        # Track outputs in Aim (only if Aim is enabled)
        if self._should_use_aim() and outputs:
            if self.aim_run is None:
                logger.warning(f"⚠️ _should_use_aim() returned True but self.aim_run is None for action {pipeline_action.name}")
            else:
                try:
                    self.aim_run[f"pipeline_action_{pipeline_action.name}_outputs"] = outputs
                    logger.debug(f"📊 Tracked action outputs in AIM: {pipeline_action.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to track action outputs in AIM: {e}")

        # Clean up memory after action completes
        if action_instance is not None and hasattr(action_instance, 'cleanup_memory'):
            try:
                action_instance.cleanup_memory()
            except Exception as e:
                logger.warning(f"Memory cleanup failed for action '{pipeline_action.name}': {e}")

        return action_output

    def _get_iteration_id_from_context(self, loop_context: Dict[str, Any]) -> Optional[str]:
        """
        Generate iteration_id from loop context.

        Base Pipeline doesn't support loopable actions, so this returns None.
        Subclasses like LoopablePipeline should override this method.

        Args:
            loop_context: Dictionary of loop iteration parameters

        Returns:
            None (base Pipeline doesn't support iterations)
        """
        # Base Pipeline doesn't support loopable actions
        return None

    def initialize(self):
        """Initialize the pipeline and validate configuration."""
        if self._initialized:
            return

        # Load actions from configuration ONLY if not already added by subclass
        # This allows subclasses to programmatically define actions with custom logic
        logger.info(f"🔍 Pipeline.initialize(): self.actions = {len(self.actions)} items")
        logger.info(f"🔍 Pipeline.initialize(): pipeline_config type = {type(self.pipeline_config)}")
        logger.info(
            f"🔍 Pipeline.initialize(): pipeline_config keys = {list(self.pipeline_config.keys()) if hasattr(self.pipeline_config, 'keys') else 'N/A'}"
        )
        logger.info(f"🔍 Pipeline.initialize(): 'actions' in pipeline_config = {'actions' in self.pipeline_config}")
        if not self.actions and 'actions' in self.pipeline_config:
            logger.info("📥 Loading actions from YAML configuration")
            # Load actions from YAML configuration
            actions_list = self.pipeline_config.actions
            logger.info(f"📥 Found {len(actions_list)} action entries to load")
            for idx, action_cfg in enumerate(actions_list):
                # Check if this is a loopable_actions block
                # Base Pipeline doesn't handle loopable actions - skip them
                if 'loopable_actions' in action_cfg:
                    logger.warning(f"⚠️ Found loopable_actions block but Pipeline doesn't support it. Use LoopablePipeline instead.")
                    continue

                # Regular action
                action_name = action_cfg.get('action_name', 'unknown')
                logger.info(f"📥 Loading action {idx+1}/{len(actions_list)}: {action_name}")

                # Use generic method to extract action config (handles any block structures)
                config_overrides = self._extract_action_config(
                    config_obj=action_cfg, action_name=action_name, block_keys=['loopable_actions', 'actions']
                )

                # Fallback to simple extraction if generic method returns None
                if config_overrides is None:
                    logger.warning(f"⚠️ Generic extraction failed for '{action_name}', using simple extraction")
                    # Get all config except metadata keys
                    # IMPORTANT: Keep 'depends_on' - it will be processed during action execution by _inject_action_outputs
                    config_overrides = {
                        k: v for k, v in action_cfg.items() if k not in ['action_name', 'outputs_to_track', 'loopable_actions', 'actions']
                    }

                action = PipelineAction(
                    name=action_name,  # Use action_name as the name
                    action_name=action_name,  # Use action_name as the action_name
                    config_overrides=config_overrides,
                    outputs_to_track=action_cfg.get('outputs_to_track', []),
                )
                self.add_action(action)

        logger.debug(f"Pipeline initialized with {len(self.actions)} actions")

        # Validate all actions exist (actions are always in actions/ directory)
        for action in self.actions:
            action_path = Path("actions") / f"{action.action_name}.py"
            if not action_path.exists():
                raise FileNotFoundError(f"Action file not found: {action_path}. " f"Actions must be in the 'actions/' directory.")
            logger.debug(f"  ✓ Action '{action.name}': {action.action_name}")

        self._initialized = True

    def run(self):
        """Execute the pipeline by running all actions in sequence."""
        if not self._initialized:
            self.initialize()

        # Check debug mode
        debug_mode = self.pipeline_config.get('debug', False)

        # Filter out any loopable action placeholders (shouldn't exist in base Pipeline)
        regular_actions = [a for a in self.actions if a.name != "__loopable_actions__"]

        # In debug mode, show verbose header
        logger.debug(f"\nStarting pipeline execution with {len(regular_actions)} actions")

        successful_actions = 0
        # Track action statuses for clean display
        action_statuses = {}  # {action_name: {'cached': bool, 'completed': bool}}

        # Show initial progress box (all actions pending)
        if not debug_mode:
            self._print_pipeline_status(self.actions, action_statuses, 0)

        # DEBUG: Write to file for debugging
        debug_file = Path(self.cfg.run_dir) / "pipeline_debug.txt"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Starting pipeline with {len(regular_actions)} actions")
            f.write(f"\n{'='*80}\n")

        # Run all actions sequentially
        logger.info("=" * 80)
        logger.info(f"Running {len(regular_actions)} actions")
        logger.info("=" * 80)

        for i, action in enumerate(regular_actions):
            # Run the action (it will return metadata about cache usage)
            action_output, was_cached = self._run_action_with_status(action)

            # Store output for use by later actions
            self.action_outputs[action.name] = action_output
            action_statuses[action.name] = {'cached': was_cached, 'completed': True}

            # Debug mode: show detailed info
            logger.debug(f"🔍 DEBUG: Stored outputs for '{action.name}' in action_outputs")
            logger.debug(f"🔍 DEBUG: action_outputs keys are now: {list(self.action_outputs.keys())}")
            logger.debug(f"🔍 DEBUG: Outputs for '{action.name}': {list(action_output.outputs.keys()) if action_output.outputs else 'None'}")

            # Write debug info to file
            with open(debug_file, "a") as f:
                f.write(f"\nAction {i+1}: {action.name}\n")
                f.write(f"  Cached: {was_cached}\n")
                f.write(f"  Outputs: {list(action_output.outputs.keys()) if action_output.outputs else 'None'}\n")
                if action_output.outputs:
                    for key, value in action_output.outputs.items():
                        f.write(f"    {key}: {str(value)[:200]}\n")
                f.write(f"  action_outputs now has: {list(self.action_outputs.keys())}\n")

            if not action_output.metadata.get("skipped", False):
                successful_actions += 1

            # Clean mode: show progress status
            if not debug_mode:
                self._print_pipeline_status(self.actions, action_statuses, i + 1)
            else:
                logger.info(f"Completed action {i+1}/{len(regular_actions)}: {action.name}")

        # Final status - show final progress box with all completed
        if not debug_mode:
            # Show final progress box
            self._print_pipeline_status(self.actions, action_statuses, len(action_statuses))
            logger.info(f"✅ Pipeline completed successfully!")
        else:
            logger.info("\n" + "=" * 80)
            logger.info(f"Pipeline completed successfully! Executed {successful_actions}/{len(regular_actions)} actions.")
            logger.info("=" * 80)

        # Save final summary (only if Aim is enabled)
        if self._should_use_aim():
            self.aim_run["pipeline_summary"] = {
                "total_actions": len(action_statuses),
                "successful_actions": successful_actions,
                "action_names": list(action_statuses.keys()),
                "outputs": {name: output.outputs for name, output in self.action_outputs.items()},
            }

    def _generate_cache_key(
        self, pipeline_action: PipelineAction, resolved_config: Dict[str, Any], loop_configs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a unique cache key for an action based on its configuration.

        Uses the unified CacheKeyGenerator for consistent cache key generation.

        Args:
            pipeline_action: The pipeline action
            resolved_config: The resolved configuration (with references resolved)
            loop_configs: Optional loop_configs dict for cross-pipeline cache sharing

        Returns:
            A unique cache key string
        """
        from urartu.utils.cache import CacheKeyGenerator

        # Prepare previous outputs dictionary
        previous_outputs = None
        if self.action_outputs:
            previous_outputs = {name: output.outputs for name, output in self.action_outputs.items()}

        # CRITICAL: Do NOT include pipeline_config_hash in cache key for absolute cache sharing
        # Cache should depend only on action config and previous outputs, not pipeline structure
        # This enables cross-pipeline cache sharing when actions have identical configs and inputs
        pipeline_config_hash = None

        # Use unified cache key generator
        cache_key = CacheKeyGenerator.generate_pipeline_action_cache_key(
            action_name=pipeline_action.action_name,
            config_overrides=resolved_config,
            previous_outputs=previous_outputs,
            pipeline_config_hash=pipeline_config_hash,  # Always None for absolute cache sharing
        )

        return cache_key

    def _save_to_cache(self, cache_key: str, action_output: ActionOutput, config_hash: str):
        """Save action output to cache."""
        # Find the pipeline action for metadata
        pipeline_action = None
        for action in self.actions:
            if action.action_name == action_output.action_name:
                pipeline_action = action
                break

        # Get pipeline common configs to include in metadata (same as when action runs)
        pipeline_common_configs = self._get_common_pipeline_configs()

        self.cache_manager.save_to_cache(
            cache_key=cache_key,
            action_output=action_output,
            config_hash=config_hash,
            pipeline_action=pipeline_action,
            resolve_value_func=self._resolve_value,
            get_config_hash_func=self._get_config_hash,
            pipeline_common_configs=pipeline_common_configs,
        )

    def _load_from_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """Load action output from cache if available and valid."""
        return self.cache_manager.load_from_cache(cache_key)

    def clear_cache(self):
        """Clear all cached outputs."""
        self.cache_manager.clear_cache()

    def main(self):
        """Convenience method that calls initialize() and run()."""
        # Check if this is a submission-only job (for loopable pipelines)
        submit_array_only = self.cfg.get('_submit_array_only', False)
        if submit_array_only:
            # For loopable pipelines, this will be handled in run()
            # But we still need to initialize to get loop_iterations
            self.initialize()
            self.run()  # run() will check _submit_array_only and exit early
            return

        self.initialize()
        self.run()

    def get_outputs(self) -> Dict[str, Any]:
        """Return the combined outputs from all pipeline actions."""
        # Return the combined outputs from all executed actions
        return {
            "actions": {name: output.outputs for name, output in self.action_outputs.items()},
            "pipeline_metadata": {
                "total_actions": len(self.actions),
                "successful_actions": len([o for o in self.action_outputs.values() if not o.metadata.get("skipped", False)]),
                "pipeline_name": self.__class__.__name__,
            },
        }
