"""
Pipeline abstraction for orchestrating sequences of actions in urartu.

This module provides a flexible Pipeline class that can run multiple actions in sequence,
manage data flow between steps, and handle configuration overrides.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import importlib
import json
import hashlib
import pickle
import time
import yaml
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from aim import Run
from omegaconf import DictConfig, OmegaConf

from .action import Action, ActionDataset
from .device import Device


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ActionOutput:
    """Container for action outputs."""
    name: str
    action_name: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Container for cached action outputs."""
    action_output: ActionOutput
    cache_key: str
    timestamp: float
    config_hash: str
    action_version: Optional[str] = None
    
    def is_valid(self, max_age_seconds: Optional[int] = None) -> bool:
        """Check if cache entry is still valid."""
        if max_age_seconds is not None:
            age = time.time() - self.timestamp
            return age < max_age_seconds
        return True


class DataResolver(ABC):
    """Abstract base class for data resolvers that handle special value resolution in configs."""
    
    @abstractmethod
    def can_resolve(self, value: str) -> bool:
        """Check if this resolver can handle the given value."""
        pass
    
    @abstractmethod
    def resolve(self, value: str, context: Dict[str, Any]) -> Any:
        """Resolve the value using the provided context."""
        pass


class ActionOutputResolver(DataResolver):
    """Resolver for action output references in the format {{actions.action_name.output_key}}"""
    
    def can_resolve(self, value: str) -> bool:
        return isinstance(value, str) and value.startswith("{{actions.") and value.endswith("}}")
    
    def resolve(self, value: str, context: Dict[str, Any]) -> Any:
        # Parse reference like {{actions.construct_samples.data_files}}
        parts = value[10:-2].split(".")  # Remove {{actions. and }}
        if len(parts) < 2:
            raise ValueError(f"Invalid action reference: {value}")
        
        action_name = parts[0]
        output_key = ".".join(parts[1:])
        
        action_outputs = context.get("action_outputs", {})
        if action_name not in action_outputs:
            raise ValueError(f"Action '{action_name}' has not been executed yet or produced no outputs")
        
        # Navigate nested dictionaries
        output = action_outputs[action_name].outputs
        for key in output_key.split("."):
            if isinstance(output, dict) and key in output:
                output = output[key]
            else:
                raise ValueError(f"Cannot find '{output_key}' in action '{action_name}' outputs")
        
        return output


class PipelineAction:
    """Represents a single action in the pipeline."""
    
    def __init__(self, name: str, action_name: str, config_overrides: Optional[Dict] = None,
                 outputs_to_track: Optional[List[str]] = None, condition: Optional[Callable] = None):
        """
        Initialize a pipeline action.
        
        Args:
            name: Unique name for this action
            action_name: Name of the action to run
            config_overrides: Configuration overrides for this action
            outputs_to_track: List of output keys to track from this action
            condition: Optional callable that determines if action should run
        """
        self.name = name
        self.action_name = action_name
        self.config_overrides = config_overrides or {}
        self.outputs_to_track = outputs_to_track or []
        self.condition = condition
        
    def should_run(self, context: Dict[str, Any]) -> bool:
        """Check if this action should run based on its condition."""
        if self.condition is None:
            return True
        return self.condition(context)


class Pipeline(Action):
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
        super().__init__(cfg, aim_run)
        self.actions: List[PipelineAction] = []
        self.action_outputs: Dict[str, ActionOutput] = {}
        self.resolvers: List[DataResolver] = [ActionOutputResolver()]
        self._initialized = False
        
        # Set up pipeline config accessor
        # Use the action_config which was properly set by the parent Action class
        self.pipeline_config = self.action_config
        
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
        self.cache_dir = runs_dir / 'cache'
        
        logger.info(f"📦 Using universal cache directory: {self.cache_dir}")

        if self.cache_max_age is not None:
            self.cache_max_age = self.cache_max_age * 3600  # Convert to seconds
    
    def _make_serializable(self, obj):
        """Convert OmegaConf objects to regular Python objects for JSON serialization."""
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _get_common_pipeline_configs(self) -> Dict[str, Any]:
        """
        Extract configuration values from pipeline config that should be 
        propagated to all individual actions.
        
        Includes all pipeline configs except pipeline-specific ones.
        
        Returns:
            Dictionary of configs to be shared across actions
        """
        common_configs = {}
        
        # Define which pipeline-level configs should NOT be propagated to actions
        pipeline_specific_keys = {
            'actions',             # List of actions in the pipeline
            'cache_enabled',       # Pipeline-level caching control
            'force_rerun',         # Pipeline-level cache bypass
            'cache_max_age_hours', # Pipeline-level cache expiry (hours)
            'cache_max_age_days',  # Pipeline-level cache expiry (days)
        }
        
        logger.info(f"🔧 Pipeline config propagation debug:")
        logger.info(f"   pipeline_config type: {type(self.pipeline_config)}")
        logger.info(f"   pipeline_config keys: {list(self.pipeline_config.keys()) if hasattr(self.pipeline_config, 'keys') else 'N/A'}")
        if 'device' in self.pipeline_config:
            logger.info(f"   pipeline_config.device: {self.pipeline_config['device']}")
        else:
            logger.info(f"   device NOT found in pipeline_config")
        
        # Propagate all pipeline configs except pipeline-specific ones
        for key, value in self.pipeline_config.items():
            if key not in pipeline_specific_keys:
                common_configs[key] = value
                
        # Also check if debug is set at the top level (from CLI)
        if hasattr(self.cfg, 'debug') and 'debug' not in common_configs:
            common_configs['debug'] = self.cfg.debug
            
        logger.info(f"   common_configs keys: {list(common_configs.keys())}")
        if 'device' in common_configs:
            logger.info(f"   common_configs.device: {common_configs['device']}")
        else:
            logger.info(f"   device NOT found in common_configs")
            
        logger.debug(f"Pipeline configs to propagate to actions: {list(common_configs.keys())}")
        return common_configs
    
    def _inject_action_outputs(self, action_config_dict: Dict[str, Any], current_action_name: str) -> Dict[str, Any]:
        """
        Inject outputs from previous actions into the current action's configuration.
        
        This method handles the data flow between actions by:
        1. Checking if the current action declares dependencies (depends_on)
        2. For each dependency, extracting the specified output from the source action
        3. Dynamically setting the value at the specified config path
        
        Args:
            action_config_dict: The action's configuration dictionary
            current_action_name: Name of the current action being configured
            
        Returns:
            Updated configuration dictionary with injected outputs
        """
        # Make a deep copy to avoid modifying the original
        import copy
        config = copy.deepcopy(action_config_dict)
        
        # Check if this action declares dependencies
        if 'depends_on' not in config:
            logger.info(f"📝 Action '{current_action_name}' has no dependencies declared")
            return config
            
        logger.info(f"🔄 Processing dependencies for action '{current_action_name}'")
        dependencies = config['depends_on']
        
        # Process each dependency
        for source_action_name, mappings in dependencies.items():
            logger.info(f"   📤 Processing dependency on '{source_action_name}'")
            
            # Check if the source action has completed and produced outputs
            if source_action_name not in self.action_outputs:
                logger.error(f"   ❌ Source action '{source_action_name}' has not completed yet!")
                continue
                
            source_outputs = self.action_outputs[source_action_name].outputs
            if not source_outputs:
                logger.warning(f"   ⚠️ Source action '{source_action_name}' produced no outputs")
                continue
            
            # Process each output->config mapping
            for output_key, config_path in mappings.items():
                if output_key in source_outputs:
                    output_value = source_outputs[output_key]
                    
                    # Inject the value at the specified config path
                    self._set_nested_config_value(config, config_path, output_value)
                    logger.info(f"   ✅ Injected {source_action_name}.{output_key} → {config_path} = {output_value}")
                else:
                    logger.warning(f"   ❌ Output '{output_key}' not found in {source_action_name} outputs")
                    logger.warning(f"       Available outputs: {list(source_outputs.keys())}")
        
        # Remove depends_on from the final config (it's metadata, not action config)
        if 'depends_on' in config:
            del config['depends_on']
            
        return config
    
    def _set_nested_config_value(self, config: Dict[str, Any], path: str, value: Any):
        """
        Set a value at a nested path in the configuration dictionary.
        
        Args:
            config: Configuration dictionary to modify
            path: Dot-separated path (e.g., 'dataset.data_files')
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
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
        
    def _run_action(self, pipeline_action: PipelineAction) -> ActionOutput:
        """Execute a single pipeline action."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Running pipeline action: {pipeline_action.name} (action: {pipeline_action.action_name})")
        logger.info(f"{'='*80}")
        
        # Check if action should run
        context = {"action_outputs": self.action_outputs}
        if not pipeline_action.should_run(context):
            logger.info(f"Skipping action {pipeline_action.name} due to condition")
            return ActionOutput(
                name=pipeline_action.name,
                action_name=pipeline_action.action_name,
                metadata={"skipped": True}
            )
        
        # DEBUG: Log available action outputs before resolution
        logger.info(f"🔍 DEBUG: Available action_outputs before resolution: {list(self.action_outputs.keys())}")
        for name, output in self.action_outputs.items():
            logger.info(f"   - {name}: outputs={list(output.outputs.keys()) if output.outputs else 'None'}")
        
        # DEBUG: Write to file
        debug_file = Path(self.cfg.run_dir) / "pipeline_debug.txt"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_file, "a") as f:
            f.write(f"\n--- Running action: {pipeline_action.name} ---\n")
            f.write(f"Available action_outputs: {list(self.action_outputs.keys())}\n")
            for name, output in self.action_outputs.items():
                f.write(f"  {name}: {list(output.outputs.keys()) if output.outputs else 'None'}\n")
            f.write(f"Config overrides before resolution: {str(pipeline_action.config_overrides)[:500]}\n")
        
        # Resolve config overrides to handle any references
        context = {"action_outputs": self.action_outputs}
        logger.info(f"🔍 DEBUG: Config overrides before resolution: {pipeline_action.config_overrides}")
        resolved_config_overrides = self._resolve_value(pipeline_action.config_overrides, context)
        logger.info(f"🔍 DEBUG: Config overrides after resolution: {resolved_config_overrides}")
        
        # DEBUG: Write resolution result to file
        with open(debug_file, "a") as f:
            f.write(f"Config overrides after resolution: {str(resolved_config_overrides)[:500]}\n")
        
        # Import the action module
        try:
            # First try fully qualified package import (actions.<name>)
            action_module = importlib.import_module(f"actions.{pipeline_action.action_name}")
        except ImportError:
            # Fall back to importing from local actions directory added to sys.path
            import sys
            actions_dir = Path.cwd() / "actions"
            if str(actions_dir) not in sys.path:
                sys.path.append(str(actions_dir))
            try:
                action_module = importlib.import_module(pipeline_action.action_name)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import action '{pipeline_action.action_name}' from package or directory {actions_dir}: {e}"
                )
        
        # Prepare configuration for this action
        action_cfg = OmegaConf.create(self.cfg)  # Deep copy base config
        action_cfg.action_name = pipeline_action.action_name  # Set the action name
        
        # Get pipeline config hash to ensure cache invalidation when pipeline config changes
        pipeline_config_hash = self._get_config_hash()
        logger.info(f"🔑 Adding pipeline config hash to action config: {pipeline_config_hash}")
        
        # Apply config overrides by creating proper action config structure
        if pipeline_action.config_overrides:
            # Use resolved config overrides (with pipeline references resolved)
            action_config_dict = resolved_config_overrides
            
            # Inject outputs from previous actions into config
            action_config_dict = self._inject_action_outputs(action_config_dict, pipeline_action.name)
            
            # Propagate common pipeline-level configs to individual actions
            pipeline_common_configs = self._get_common_pipeline_configs()
            
            logger.info(f"🔧 Config merge debug for '{pipeline_action.name}':")
            logger.info(f"   action_config_dict type: {type(action_config_dict)}")
            if 'device' in action_config_dict:
                logger.info(f"   action_config_dict.device: {action_config_dict['device']}")
            else:
                logger.info(f"   device NOT found in action_config_dict")
            
            # Log any dependencies that were processed
            if 'depends_on' in resolved_config_overrides:
                logger.info(f"   🔗 Dependencies declared: {list(resolved_config_overrides['depends_on'].keys())}")
            else:
                logger.info(f"   📝 No dependencies declared for this action")
            
            # Merge pipeline common configs with action-specific configs
            # Action-specific configs take precedence over pipeline configs
            # OmegaConf.merge: later arguments override earlier ones
            merged_config = OmegaConf.merge(
                OmegaConf.create(pipeline_common_configs),  # Base (pipeline defaults)
                OmegaConf.create(action_config_dict)        # Override (action-specific, with injections)
            )
            
            # Add pipeline config hash to ensure cache invalidation when pipeline config changes
            merged_config.pipeline_config_hash = pipeline_config_hash
            logger.info(f"🔑 Added pipeline config hash {pipeline_config_hash} to action '{pipeline_action.name}'")
            
            logger.info(f"   merged_config type: {type(merged_config)}")
            if 'device' in merged_config:
                logger.info(f"   merged_config.device: {merged_config['device']}")
            else:
                logger.info(f"   device NOT found in merged_config")
            
            # Log config override details
            overridden_keys = set(pipeline_common_configs.keys()) & set(action_config_dict.keys())
            if overridden_keys:
                logger.info(f"Action '{pipeline_action.name}' overrides pipeline configs: {list(overridden_keys)}")
            
            action_cfg.action_config = merged_config  # Set the action configuration
            logger.info(f"Applied config overrides with pipeline common configs for {pipeline_action.name}")
        else:
            # Even if no overrides, apply common pipeline configs
            pipeline_common_configs = self._get_common_pipeline_configs()
            base_config = OmegaConf.create(pipeline_common_configs)
            
            # Add pipeline config hash even for actions without overrides
            base_config.pipeline_config_hash = pipeline_config_hash
            logger.info(f"🔑 Added pipeline config hash {pipeline_config_hash} to action '{pipeline_action.name}'")
            
            action_cfg.action_config = base_config  # Set the action configuration
            
            logger.info(f"🔧 No action overrides for '{pipeline_action.name}' - using pipeline configs:")
            if 'device' in pipeline_common_configs:
                logger.info(f"   pipeline device will be used: {pipeline_common_configs['device']}")
            else:
                logger.info(f"   No device in pipeline_common_configs - will default to 'auto'")
            
            logger.info(f"Applied pipeline common configs for {pipeline_action.name}")
        
        # Create a sub-context in Aim for this action
        # Track action metadata
        self.aim_run[f"pipeline_action_{pipeline_action.name}_config"] = {
            "action": pipeline_action.action_name,
            "overrides": pipeline_action.config_overrides
        }
        
        # Try to find action class - look for classes that inherit from Action
        action_class = None
        action_instance = None  # Initialize to None
        action_candidates = []
        
        for attr_name in dir(action_module):
            attr = getattr(action_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, Action) and 
                attr != Action and
                attr != ActionDataset):  # Exclude abstract base classes
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
            # The cache_dir is already set correctly in Action.__init__()
            
            # Run the action with its own caching support
            if hasattr(action_instance, 'run_with_cache'):
                action_instance.run_with_cache()
            elif hasattr(action_instance, 'run'):
                action_instance.run()
            elif hasattr(action_instance, 'main'):
                action_instance.main()
            else:
                raise AttributeError(f"Action {pipeline_action.action_name} has no run() or main() method")
            
            # Extract outputs
            outputs = self._extract_outputs(pipeline_action, action_instance)
            logger.info(f"🔍 DEBUG: Extracted outputs from {pipeline_action.name}: {list(outputs.keys()) if outputs else 'None'}")
            if outputs:
                for key, value in outputs.items():
                    logger.info(f"   - {key}: {str(value)[:100]}")
            
            # Ensure we have outputs even from cached actions
            if not outputs and hasattr(action_instance, '_cached_outputs') and action_instance._cached_outputs:
                outputs = action_instance._cached_outputs
                logger.info(f"📤 Using cached outputs for pipeline action {pipeline_action.name}: {list(outputs.keys()) if outputs else 'None'}")
        elif hasattr(action_module, 'main'):
            # Fallback to module-level main function
            action_module.main(cfg=action_cfg, aim_run=self.aim_run)
            outputs = {}  # Main-style actions don't return outputs directly
        else:
            raise AttributeError(f"Action {pipeline_action.action_name} has no Action class or main() function")
        
        action_output = ActionOutput(
            name=pipeline_action.name,
            action_name=pipeline_action.action_name,
            outputs=outputs,
            metadata={"completed": True}
        )
        
        # Track outputs in Aim
        if outputs:
            self.aim_run[f"pipeline_action_{pipeline_action.name}_outputs"] = outputs
        
        # Clean up memory after action completes
        if action_instance is not None and hasattr(action_instance, 'cleanup_memory'):
            try:
                action_instance.cleanup_memory()
            except Exception as e:
                logger.warning(f"Memory cleanup failed for action '{pipeline_action.name}': {e}")
            
        return action_output
        
    def initialize(self):
        """Initialize the pipeline and validate configuration."""
        if self._initialized:
            return
            

        # Load actions from configuration ONLY if not already added by subclass
        # This allows subclasses to programmatically define actions with custom logic
        logger.info(f"🔍 Pipeline.initialize(): self.actions = {len(self.actions)} items")
        logger.info(f"🔍 Pipeline.initialize(): 'actions' in pipeline_config = {'actions' in self.pipeline_config}")
        if not self.actions and 'actions' in self.pipeline_config:
            logger.info("📥 Loading actions from YAML configuration")
            # Load actions from YAML configuration
            for action_cfg in self.pipeline_config.actions:
                # Get all config except metadata keys
                # IMPORTANT: Keep 'depends_on' - it will be processed during action execution by _inject_action_outputs
                config_overrides = {k: v for k, v in action_cfg.items() 
                                  if k not in ['action_name', 'outputs_to_track']}
                
                action = PipelineAction(
                    name=action_cfg.action_name,  # Use action_name as the name
                    action_name=action_cfg.action_name,  # Use action_name as the action_name
                    config_overrides=config_overrides,
                    outputs_to_track=action_cfg.get('outputs_to_track', [])
                )
                self.add_action(action)
                    
        logger.info(f"Pipeline initialized with {len(self.actions)} actions")
        
        # Validate all actions exist
        for action in self.actions:
            action_path = Path("actions") / f"{action.action_name}.py"
            if not action_path.exists():
                raise FileNotFoundError(f"Action file not found: {action_path}")
            logger.info(f"  ✓ Action '{action.name}': {action.action_name}")
            
        self._initialized = True
        
    def run(self):
        """Execute the pipeline by running all actions in sequence."""
        if not self._initialized:
            self.initialize()
            
        logger.info(f"\nStarting pipeline execution with {len(self.actions)} actions")
        
        successful_actions = 0
        # DEBUG: Write to file for debugging
        debug_file = Path(self.cfg.run_dir) / "pipeline_debug.txt"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Starting pipeline with {len(self.actions)} actions\n")
            f.write(f"{'='*80}\n")
        
        for i, action in enumerate(self.actions):
            try:
                # Run the action
                action_output = self._run_action(action)
                
                # Store output for use by later actions
                self.action_outputs[action.name] = action_output
                logger.info(f"🔍 DEBUG: Stored outputs for '{action.name}' in action_outputs")
                logger.info(f"🔍 DEBUG: action_outputs keys are now: {list(self.action_outputs.keys())}")
                logger.info(f"🔍 DEBUG: Outputs for '{action.name}': {list(action_output.outputs.keys()) if action_output.outputs else 'None'}")
                
                # DEBUG: Write to file
                with open(debug_file, "a") as f:
                    f.write(f"\nAction {i+1}: {action.name}\n")
                    f.write(f"  Outputs: {list(action_output.outputs.keys()) if action_output.outputs else 'None'}\n")
                    if action_output.outputs:
                        for key, value in action_output.outputs.items():
                            f.write(f"    {key}: {str(value)[:200]}\n")
                    f.write(f"  action_outputs now has: {list(self.action_outputs.keys())}\n")
                
                if not action_output.metadata.get("skipped", False):
                    successful_actions += 1
                    
                logger.info(f"Completed action {i+1}/{len(self.actions)}: {action.name}")
                
            except Exception as e:
                logger.error(f"Failed at action '{action.name}': {str(e)}")
                self.aim_run[f"pipeline_action_{action.name}_error"] = str(e)
                self.aim_run["pipeline_failed_at_action"] = action.name
                raise
                
        logger.info("\n" + "="*80)
        logger.info(f"Pipeline completed successfully! Executed {successful_actions} actions.")
        logger.info("="*80)
        
        # Save final summary
        self.aim_run["pipeline_summary"] = {
            "total_actions": len(self.actions),
            "successful_actions": successful_actions,
            "action_names": [action.name for action in self.actions],
            "outputs": {name: output.outputs for name, output in self.action_outputs.items()}
        }
        
    def _get_config_hash(self) -> str:
        """
        Generate a hash of the complete configuration to ensure cache invalidation
        when config files are modified.
        
        Returns:
            A hash string representing the complete configuration
        """
        # Include the complete configuration in the hash
        config_for_hash = self._make_serializable(self.cfg)
        
        # Convert to JSON for consistent ordering
        config_string = json.dumps(config_for_hash, sort_keys=True)
        
        # Generate hash
        config_hash = hashlib.sha256(config_string.encode()).hexdigest()[:12]
        
        return config_hash

    def _generate_cache_key(self, pipeline_action: PipelineAction, resolved_config: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for an action based on its configuration.
        
        For cross-pipeline cache sharing, we focus on the action's core configuration
        and ignore pipeline context when the action doesn't depend on previous outputs.
        
        Args:
            pipeline_action: The pipeline action
            resolved_config: The resolved configuration (with references resolved)
            
        Returns:
            A unique cache key string
        """
        # Import the cache ignore keys from action module for consistency
        from .action import CACHE_IGNORE_KEYS
        
        # Filter config to enable cross-pipeline cache sharing
        # Remove pipeline-level keys that don't affect action outputs
        filtered_config = {k: v for k, v in resolved_config.items() if k not in CACHE_IGNORE_KEYS}
        
        # Get hash of complete config file to ensure cache invalidation when config changes
        config_hash = self._get_config_hash()
        logger.debug(f"Generated config hash for cache key: {config_hash}")
        
        # For cross-pipeline cache sharing: if this action has no previous outputs (first action),
        # generate a cache key based only on the action config, not pipeline context
        has_previous_outputs = bool(self.action_outputs)
        has_depends_on = 'depends_on' in resolved_config and resolved_config['depends_on']
        
        if not has_previous_outputs and not has_depends_on:
            # First action with no dependencies - use simplified cache key for cross-pipeline sharing
            # Only include action-specific config, not pipeline-wide config hash
            key_factors = {
                'action_name': pipeline_action.action_name,
                'config': self._make_serializable(filtered_config),
            }
        else:
            # Action depends on previous outputs - include pipeline context
            key_factors = {
                'action_name': pipeline_action.action_name,
                'config_overrides': self._make_serializable(filtered_config),
                'outputs_to_track': self._make_serializable(pipeline_action.outputs_to_track),
                'config_hash': config_hash,  # Include complete config hash
                # Include previous action outputs that might affect this action
                'previous_outputs': {
                    name: self._make_serializable(output.outputs) 
                    for name, output in self.action_outputs.items()
                }
            }
        
        # Convert to JSON for consistent ordering
        key_string = json.dumps(key_factors, sort_keys=True)
        
        # Generate hash
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{pipeline_action.action_name}_{cache_key}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _save_to_cache(self, cache_key: str, action_output: ActionOutput, config_hash: str):
        """Save action output to cache."""
        if not self.cache_enabled:
            return
            
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_entry = CacheEntry(
                action_output=action_output,
                cache_key=cache_key,
                timestamp=time.time(),
                config_hash=config_hash,
                action_version=None  # Could add git hash or file modification time
            )
            
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_entry, f)
                
            # Also save a human-readable metadata file
            metadata_path = cache_path.with_suffix('.yaml')
            
            # Get the full resolved config for this action
            pipeline_action = None
            for action in self.actions:
                if action.action_name == action_output.action_name:
                    pipeline_action = action
                    break
            
            # Build full config including pipeline context
            full_config = {}
            if pipeline_action:
                # Include the resolved config overrides
                resolved_config_overrides = self._resolve_value(pipeline_action.config_overrides)
                if hasattr(resolved_config_overrides, 'items'):
                    resolved_config_overrides = OmegaConf.to_container(resolved_config_overrides, resolve=True) if OmegaConf.is_config(resolved_config_overrides) else dict(resolved_config_overrides)
                full_config = resolved_config_overrides
            
            metadata = {
                'cache_key': cache_key,
                'action_name': action_output.action_name,
                'timestamp': datetime.fromtimestamp(cache_entry.timestamp).isoformat(),
                'config_hash': config_hash,
                'full_config_hash': self._get_config_hash(),  # Add complete config hash for debugging
                'outputs_keys': list(action_output.outputs.keys()),
                'full_config': full_config  # Add full config content
            }
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, indent=2, sort_keys=False)
                
            logger.info(f"Cached output for action '{action_output.name}' with key {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """Load action output from cache if available and valid."""
        if not self.cache_enabled:
            logger.info(f"Cache is disabled, skipping cache lookup for {cache_key}")
            return None
            
        if self.force_rerun:
            logger.info(f"Force rerun is enabled, skipping cache lookup for {cache_key}")
            return None
            
        cache_path = self._get_cache_path(cache_key)
        logger.info(f"🔍 Checking cache file: {cache_path}")
        logger.info(f"📂 Cache file exists: {cache_path.exists()}")
        
        if not cache_path.exists():
            logger.info(f"❌ Cache file not found: {cache_path}")
            return None
            
        try:
            logger.info(f"📖 Attempting to load cache from: {cache_path}")
            with open(cache_path, 'rb') as f:
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
    
    def main(self):
        """Convenience method that calls initialize() and run()."""
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
                "pipeline_name": self.__class__.__name__
            }
        }
