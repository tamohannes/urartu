"""
LoopablePipeline class for pipelines that support loopable actions.

This class extends Pipeline with functionality to run actions multiple times
with different parameter values (e.g., different hyperparameters, configurations, or variants).
"""

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger

from .loopable_cache import LoopableCacheMixin
from .loopable_jobs import LoopableJobsMixin
from .mixins.multi_block import MultiBlockLoopableMixin
from .pipeline import Pipeline
from .pipeline_action import PipelineAction
from .pipeline_stage import PipelineStage, ActionStage, LoopableStage
from .status import PipelineStatusDisplay
from .types import ActionOutput

logger = get_logger(__name__)


class LoopablePipeline(LoopableCacheMixin, LoopableJobsMixin, MultiBlockLoopableMixin, Pipeline):
    """
    Pipeline class that supports loopable actions.

    Loopable actions are actions that run multiple times with different
    parameter values. This is useful for experiments that need to iterate
    over different hyperparameters, configurations, or other variables.

    Features:
    - Run actions multiple times with different parameter values
    - Organize outputs by iteration
    - Support for aggregator actions that collect results from all iterations
    """

    def __init__(self, cfg: DictConfig, aim_run) -> None:
        """Initialize the loopable pipeline."""
        super().__init__(cfg, aim_run)

        # Loopable actions support
        self.loopable_actions: List[str] = []
        self.loop_iterations: List[Dict[str, Any]] = []
        self.loop_configs: Dict[str, str] = {}  # Maps loop variable names to config paths
        self.aggregation_action: Optional[str] = None
        self.loopable_action_configs: Dict[str, Dict[str, Any]] = {}
        self.current_loop_context: Optional[Dict[str, Any]] = None

    def _get_iteration_id(self, loop_context: Dict[str, Any]) -> Optional[str]:
        """
        Generate a meaningful iteration identifier from loop context.

        This method MUST be overridden by subclasses to provide meaningful names
        based on the actual parameter values being iterated over.

        The returned identifier will be used for organizing artifacts in directories,
        so it should be descriptive and based on the actual loop parameter values
        (e.g., revision name, model name, hyperparameter value).

        Args:
            loop_context: Dictionary of loop iteration parameters

        Returns:
            Meaningful iteration identifier string based on loop parameter values,
            or None if not applicable

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_iteration_id() to provide "
            "meaningful iteration identifiers based on loop parameter values"
        )

    def get_iteration_outputs(self, base_action_name: str) -> Dict[str, ActionOutput]:
        """
        Get all outputs from a loopable action across all iterations.

        This is useful for aggregator actions that need to collect results
        from all iterations of a loopable action.

        Args:
            base_action_name: The base action name (without iteration suffix)
                            (e.g., "_3_answer_rank_classifier_mock")

        Returns:
            Dictionary mapping iteration identifiers to ActionOutput objects
            (e.g., {"iteration_1": ActionOutput(...), "iteration_2": ActionOutput(...), ...})
        """
        iteration_outputs = {}

        # Check if this action is stored in the new nested structure
        if base_action_name in self.action_outputs:
            entry = self.action_outputs[base_action_name]
            if isinstance(entry, dict) and not isinstance(entry, ActionOutput):
                # New nested structure: action_outputs[base_name][iteration_id] = ActionOutput
                iteration_outputs = entry.copy()
            elif isinstance(entry, ActionOutput):
                # Regular action (not loopable) - return empty or single entry
                logger.warning(f"Action '{base_action_name}' is not a loopable action")
                return {}

        # Also check backward compatibility: iteration-specific names (e.g., action_name_iteration_id)
        for output_key, action_output in self.action_outputs.items():
            if output_key.startswith(f"{base_action_name}_") and isinstance(action_output, ActionOutput):
                # Extract iteration ID from the key (everything after base_action_name_)
                iteration_id = output_key[len(base_action_name) + 1 :]
                if iteration_id not in iteration_outputs:
                    iteration_outputs[iteration_id] = action_output

        logger.debug(f"📦 Found {len(iteration_outputs)} iteration outputs for '{base_action_name}': {list(iteration_outputs.keys())}")
        return iteration_outputs

    def _inject_action_outputs(
        self, action_config_dict: Dict[str, Any], current_action_name: str, loop_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Inject outputs from previous actions into the current action's configuration."""
        if self.config_injector is None:
            # Initialize config injector if not already done
            from .config import ConfigInjector

            self.config_injector = ConfigInjector(
                action_outputs=self.action_outputs, loopable_actions=self.loopable_actions, loop_configs=self.loop_configs
            )
        return self.config_injector.inject_action_outputs(
            action_config_dict=action_config_dict,
            current_action_name=current_action_name,
            loop_context=loop_context,
            get_iteration_outputs=self.get_iteration_outputs,
        )

    def _inject_loop_context(self, action_config_dict: Dict[str, Any], loop_context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject loop context parameters into the action's configuration."""
        if self.config_injector is None:
            # Initialize config injector if not already done
            from .config import ConfigInjector

            self.config_injector = ConfigInjector(
                action_outputs=self.action_outputs, loopable_actions=self.loopable_actions, loop_configs=self.loop_configs
            )
        return self.config_injector.inject_loop_context(action_config_dict, loop_context)

    def _get_iteration_id_from_context(self, loop_context: Dict[str, Any]) -> Optional[str]:
        """
        Override to use LoopablePipeline's _get_iteration_id method.

        This allows LoopablePipeline subclasses to customize iteration ID generation.
        """
        return self._get_iteration_id(loop_context)

    def _find_project_and_actions_dir(self) -> tuple[Optional[Path], Optional[Path]]:
        """
        Find project root and actions directory by walking up the directory tree.

        Returns:
            Tuple of (project_root, actions_dir) or (None, None) if not found
        """
        import sys

        current_dir = Path.cwd()
        project_root = None
        actions_dir = None

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

        return project_root, actions_dir

    def _setup_action_imports(self, project_root: Path, actions_dir: Path) -> None:
        """
        Add project root and actions directory to sys.path for action imports.

        Args:
            project_root: Project root directory
            actions_dir: Actions directory
        """
        import sys

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if str(actions_dir) not in sys.path:
            sys.path.append(str(actions_dir))

    def _import_action_class(self, action_module_name: str, use_strict_check: bool = False):
        """
        Import action module and find action class.

        Args:
            action_module_name: Name of the action module
            use_strict_check: If True, use stricter Action class checking (for _run_post_loopable_only)

        Returns:
            Action class or None if not found
        """
        import importlib

        try:
            action_module = importlib.import_module(f'actions.{action_module_name}')

            if use_strict_check:
                # Stricter check: must be subclass of Action but not Action or ActionDataset
                from urartu.common.action import Action, ActionDataset

                action_class = None
                action_candidates = []
                for attr_name in dir(action_module):
                    attr = getattr(action_module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Action) and attr != Action and attr != ActionDataset:
                        action_candidates.append(attr)

                # Prefer the most specific class
                for candidate in action_candidates:
                    if candidate.__module__ == action_module.__name__:
                        action_class = candidate
                        break

                if not action_class and action_candidates:
                    action_class = action_candidates[0]

                return action_class
            else:
                # Simple check: any class with a 'run' method
                action_class = None
                for attr_name in dir(action_module):
                    attr = getattr(action_module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, 'run'):
                        action_class = attr
                        break
                return action_class
        except ImportError as e:
            logger.error(f"Failed to import action module {action_module_name}: {e}")
            return None

    def _create_action_instance(self, action: PipelineAction, action_cfg: DictConfig):
        """
        Create an action instance using framework APIs.
        
        Args:
            action: The PipelineAction to create an instance for
            action_cfg: The action configuration (OmegaConf DictConfig)
            
        Returns:
            Action instance or None if creation failed
        """
        project_root, actions_dir = self._find_project_and_actions_dir()
        self._setup_action_imports(project_root, actions_dir)
        action_class = self._import_action_class(action.action_name, use_strict_check=True)
        if action_class:
            return action_class(action_cfg, self.aim_run)
        return None

    def _load_single_iteration_output(
        self,
        action_name: str,
        iteration_id: str,
        iteration_idx: int,
        block_idx: Optional[int],
        loop_context: Dict[str, Any],
        run_dir: Path,
    ) -> bool:
        """
        Load output for a single iteration from disk or cache.
        
        Args:
            action_name: Name of the action
            iteration_id: Unique identifier for this iteration
            iteration_idx: Index of the iteration
            block_idx: Index of the block (None for single-block pipelines)
            loop_context: Loop context for this iteration
            run_dir: Run directory path
            
        Returns:
            True if output was loaded successfully, False otherwise
        """
        # Try loading from disk first (for newly executed iterations)
        possible_dirs = [
            run_dir / "array_tasks" / f"task_{iteration_idx}" / action_name,
        ]
        if block_idx is not None:
            possible_dirs.append(
                run_dir / "array_tasks" / f"block_{block_idx}" / f"task_{iteration_idx}" / action_name
            )
        
        for action_output_dir in possible_dirs:
            output_file = action_output_dir / "output.pkl"
            if output_file.exists():
                try:
                    import pickle
                    from urartu.common.pipeline.types import ActionOutput
                    with open(output_file, 'rb') as f:
                        saved_output = pickle.load(f)
                        if not isinstance(saved_output, ActionOutput):
                            if isinstance(saved_output, dict):
                                action_output = ActionOutput(
                                    name=f"{action_name}_{iteration_id}",
                                    action_name=action_name,
                                    outputs=saved_output,
                                    metadata={"from_file": True},
                                )
                            else:
                                continue
                        else:
                            action_output = saved_output
                        
                        if action_name not in self.action_outputs:
                            self.action_outputs[action_name] = {}
                        self.action_outputs[action_name][iteration_id] = action_output
                        logger.info(f"Loaded output for {action_name}[{iteration_id}] from {output_file}")
                        return True
                except Exception as e:
                    logger.debug(f"Error loading from {output_file}: {e}")
                    continue
        
        # If not loaded from disk, try cache
        try:
            # Get block-specific config if available
            if hasattr(self, '_get_block_config_for_action') and block_idx is not None:
                block_config = self._get_block_config_for_action(action_name, block_idx)
            else:
                # Fall back to standard config
                action = None
                for a in self.actions:
                    if a.action_name == action_name:
                        action = a
                        break
                if action is None:
                    if hasattr(self, 'loopable_action_configs') and action_name in self.loopable_action_configs:
                        stored_config = self.loopable_action_configs[action_name]
                        if isinstance(stored_config, dict) and not any(k.startswith('block_') for k in stored_config.keys()):
                            block_config = copy.deepcopy(stored_config)
                        else:
                            block_config = None
                    else:
                        block_config = None
                else:
                    block_config = action.config_overrides
            
            if not block_config:
                logger.debug(f"Could not get config for action '{action_name}' for cache lookup")
                return False
            
            # Create action instance and load from cache
            from urartu.common.pipeline.pipeline_action import PipelineAction
            temp_action = PipelineAction(
                name=action_name,
                action_name=action_name,
                config_overrides=block_config,
                outputs_to_track=[],
            )
            
            # Resolve config and inject loop context
            context = {"action_outputs": self.action_outputs, "loop_context": loop_context}
            resolved_config = self._resolve_value(block_config, context)
            resolved_config = self._inject_action_outputs(resolved_config, action_name, loop_context=loop_context)
            
            pipeline_common_configs = self._get_common_pipeline_configs()
            merged_config = OmegaConf.merge(
                OmegaConf.create(pipeline_common_configs),
                OmegaConf.create(resolved_config),
            )
            resolved_config = OmegaConf.to_container(merged_config, resolve=True)
            
            if loop_context:
                resolved_config = self._inject_loop_context(resolved_config, loop_context)
            
            merged_config = OmegaConf.create(resolved_config)
            
            # Create action instance
            action_cfg = OmegaConf.create(self.cfg)
            action_cfg.action_name = action_name
            if 'run_dir' not in action_cfg or not action_cfg.run_dir:
                if hasattr(self.cfg, 'run_dir') and self.cfg.run_dir:
                    action_cfg.run_dir = self.cfg.run_dir
            if 'pipeline' in action_cfg:
                del action_cfg.pipeline
            action_cfg.action = merged_config
            
            action_instance = self._create_action_instance(temp_action, action_cfg)
            if action_instance:
                cached_outputs = action_instance._load_from_cache()
                if cached_outputs is not None:
                    from urartu.common.pipeline.types import ActionOutput
                    action_output = ActionOutput(
                        name=f"{action_name}_{iteration_id}",
                        action_name=action_name,
                        outputs=cached_outputs if isinstance(cached_outputs, dict) else {},
                        metadata={"from_cache": True},
                    )
                    if action_name not in self.action_outputs:
                        self.action_outputs[action_name] = {}
                    self.action_outputs[action_name][iteration_id] = action_output
                    logger.info(f"Loaded output for {action_name}[{iteration_id}] from cache")
                    return True
        except Exception as e:
            logger.debug(f"Error attempting cache load for {action_name}[{iteration_id}]: {e}")
        
        return False

    def _print_pipeline_status(self, all_actions: List[PipelineAction], statuses: Dict[str, Dict], current_index: int):
        """Print a clean, colorful status view of pipeline progress with loopable action support."""
        # Initialize status display if not already done
        if self.status_display is None:
            run_dir = Path(self.cfg.get('run_dir', '.')) if self.cfg.get('run_dir') else None
            # LoopablePipeline supports loopable actions - use actual values
            self.status_display = PipelineStatusDisplay(
                pipeline_config=self.pipeline_config, loopable_actions=self.loopable_actions, loop_iterations=self.loop_iterations, run_dir=run_dir
            )
        self.status_display.print_status(all_actions, statuses, current_index)

    def _should_use_parallel_processing(self) -> bool:
        """
        Auto-detect if parallel processing should be used for loopable actions.

        Returns True if:
        - SLURM is enabled (use_slurm: true)
        - Loopable actions exist
        - Loop iterations are defined
        - NOT running as an array task (array tasks should run sequentially)

        Returns:
            True if parallel processing should be used, False otherwise
        """
        # If this is an iteration task, don't use parallel processing
        if self.cfg.get('_is_iteration_task', False):
            logger.debug("Running as iteration task - parallel processing disabled")
            return False

        # Check if SLURM is enabled
        slurm_cfg = self.cfg.get('slurm', None)
        if slurm_cfg is None:
            return False
        if isinstance(slurm_cfg, DictConfig) or isinstance(slurm_cfg, dict):
            use_slurm = slurm_cfg.get('use_slurm', False)
        else:
            return False

        # Check if loopable actions and iterations exist
        has_loopable_actions = len(self.loopable_actions) > 0
        has_loop_iterations = len(self.loop_iterations) > 0

        should_parallel = use_slurm and has_loopable_actions and has_loop_iterations

        if should_parallel:
            logger.info("🚀 Parallel processing enabled: Individual jobs will be submitted for each loopable iteration")
        else:
            logger.debug(
                f"Parallel processing disabled: use_slurm={use_slurm}, has_loopable_actions={has_loopable_actions}, has_loop_iterations={has_loop_iterations}"
            )

        return should_parallel

    def _is_dynamic_loop_iterations(self, loop_iterations_cfg: DictConfig) -> bool:
        """
        Check if loop_iterations is a dynamic specification.

        This method can be overridden by subclasses to detect their own dynamic patterns.

        Args:
            loop_iterations_cfg: The loop_iterations configuration

        Returns:
            True if this is a dynamic specification, False otherwise
        """
        # Base implementation: no dynamic patterns by default
        return False

    def _load_dynamic_loop_iterations(self, loop_iterations_cfg: DictConfig) -> List[Dict[str, Any]]:
        """
        Load dynamic loop iterations.

        This method can be overridden by subclasses to implement their own dynamic loading logic.

        Args:
            loop_iterations_cfg: The loop_iterations configuration with dynamic specification

        Returns:
            List of loop iteration dictionaries
        """
        # Base implementation: return empty list
        logger.warning("Dynamic loop iterations detected but _load_dynamic_loop_iterations not implemented")
        return []

    def initialize(self):
        """Initialize the pipeline and validate configuration, including loopable actions."""
        if self._initialized:
            return

        # Load actions from configuration
        logger.info(f"🔍 LoopablePipeline.initialize(): self.actions = {len(self.actions)} items")
        if not self.actions and 'actions' in self.pipeline_config:
            logger.info("📥 Loading actions from YAML configuration")
            actions_list = self.pipeline_config.actions
            logger.info(f"📥 Found {len(actions_list)} action entries to load")

            for idx, action_cfg in enumerate(actions_list):
                # Check if this is a loopable_actions block
                if 'loopable_actions' in action_cfg:
                    logger.debug(f"📥 Loading loopable actions block {idx+1}/{len(actions_list)}")
                    loopable_block = action_cfg.loopable_actions

                    # Extract loop_configs
                    if 'loop_configs' in loopable_block:
                        self.loop_configs = OmegaConf.to_container(loopable_block.loop_configs, resolve=False)
                        logger.debug(f"🔄 Loop configs: {self.loop_configs}")

                    # Extract loop_iterations
                    if 'loop_iterations' in loopable_block:
                        loop_iterations_cfg = loopable_block.loop_iterations

                        # Check if this is a dynamic specification (can be overridden by subclasses)
                        if isinstance(loop_iterations_cfg, DictConfig) and self._is_dynamic_loop_iterations(loop_iterations_cfg):
                            # Let subclasses handle dynamic loop iterations
                            self.loop_iterations = self._load_dynamic_loop_iterations(loop_iterations_cfg)
                            if self.loop_iterations:
                                logger.debug(f"🔄 Generated {len(self.loop_iterations)} loop iterations dynamically")
                        else:
                            # Static loop_iterations
                            self.loop_iterations = OmegaConf.to_container(loop_iterations_cfg, resolve=False)

                        if not isinstance(self.loop_iterations, list):
                            self.loop_iterations = [self.loop_iterations]
                        logger.debug(f"🔄 Found {len(self.loop_iterations)} loop iterations")

                    # Extract loopable actions
                    if 'actions' in loopable_block:
                        loopable_actions_list = loopable_block.actions
                        logger.debug(f"🔄 Found {len(loopable_actions_list)} loopable actions")
                        for idx, loopable_action_cfg in enumerate(loopable_actions_list):
                            logger.debug(f"🔄 Processing loopable action config {idx+1}")

                            action_name = None
                            if isinstance(loopable_action_cfg, DictConfig):
                                action_name = loopable_action_cfg.get('action_name')
                            elif isinstance(loopable_action_cfg, dict):
                                action_name = loopable_action_cfg.get('action_name')

                            if action_name is None:
                                logger.error(f"❌ Could not extract action_name from config at index {idx}!")
                                continue

                            logger.debug(f"🔄 Extracted action_name: '{action_name}'")

                            # Use generic method to extract and validate action config
                            config_overrides = self._extract_action_config(
                                config_obj=loopable_action_cfg, action_name=action_name, block_keys=['loopable_actions', 'actions']
                            )

                            if config_overrides is None:
                                logger.error(f"❌ Failed to extract action config for '{action_name}'!")
                                continue

                            self.loopable_action_configs[action_name] = config_overrides
                            if action_name not in self.loopable_actions:
                                self.loopable_actions.append(action_name)
                            logger.debug(f"🔄 Loaded config for loopable action: {action_name}")

                    # Create a special PipelineAction to represent the loopable block
                    loopable_action = PipelineAction(
                        name="__loopable_actions__", action_name="__loopable_actions__", config_overrides={}, outputs_to_track=[]
                    )
                    self.add_action(loopable_action)
                else:
                    # Regular action - call parent to handle it
                    action_name = action_cfg.get('action_name', 'unknown')
                    logger.info(f"📥 Loading action {idx+1}/{len(actions_list)}: {action_name}")

                    config_overrides = self._extract_action_config(
                        config_obj=action_cfg, action_name=action_name, block_keys=['loopable_actions', 'actions']
                    )

                    if config_overrides is None:
                        config_overrides = {
                            k: v for k, v in action_cfg.items() if k not in ['action_name', 'outputs_to_track', 'loopable_actions', 'actions']
                        }

                    action = PipelineAction(
                        name=action_name,
                        action_name=action_name,
                        config_overrides=config_overrides,
                        outputs_to_track=action_cfg.get('outputs_to_track', []),
                    )
                    self.add_action(action)

        # Validate actions
        for action in self.actions:
            if action.action_name == "__loopable_actions__":
                continue
            action_path = Path("actions") / f"{action.action_name}.py"
            if not action_path.exists():
                raise FileNotFoundError(f"Action file not found: {action_path}")

        # Initialize ConfigInjector
        if self.config_injector is None:
            from .config import ConfigInjector

            self.config_injector = ConfigInjector(
                action_outputs=self.action_outputs, loopable_actions=self.loopable_actions, loop_configs=self.loop_configs
            )

        self._initialized = True

    def _run_loopable_actions(
        self, pre_loopable_actions: List[PipelineAction], action_statuses: Dict[str, Dict], debug_mode: bool, loopable_idx: Optional[int] = None
    ):
        """
        Run loopable actions for each iteration.

        This method can be overridden by subclasses to customize loop iteration handling.

        Args:
            pre_loopable_actions: List of actions that run before loopable actions
            action_statuses: Dictionary tracking action execution status
            debug_mode: Whether debug mode is enabled
            loopable_idx: Index of the loopable block in the actions list
        """
        # Check if parallel processing should be used
        if self._should_use_parallel_processing():
            logger.info("=" * 80)
            logger.info(
                f"Phase 2: Submitting parallel loopable actions ({len(self.loopable_actions)} actions × {len(self.loop_iterations)} iterations)"
            )
            logger.info("=" * 80)
            # Use the new individual job submission method (not arrays)
            self._submit_array_only_internal()
            return

        logger.info("=" * 80)
        logger.info(f"Phase 2: Running loopable actions ({len(self.loopable_actions)} actions × {len(self.loop_iterations)} iterations)")
        logger.info("=" * 80)

        for iteration_idx, loop_context in enumerate(self.loop_iterations):
            iteration_num = iteration_idx + 1
            logger.debug(f"\n🔄 Iteration {iteration_num}/{len(self.loop_iterations)} with parameters: {loop_context}")

            # Set current loop context for resolvers
            self.current_loop_context = loop_context

            # Generate iteration identifier using the overridable method
            iteration_id = self._get_iteration_id(loop_context)
            if not iteration_id:
                iteration_id = f"iteration_{iteration_num}"

            # Run each loopable action
            for action_name in self.loopable_actions:
                logger.debug(f"🔄 Running loopable action: '{action_name}'")

                # Find or create the action
                action = None
                for a in self.actions:
                    if a.name == action_name:
                        # Verify the action has correct config (not a block structure)
                        extracted_config = self._extract_action_config(
                            config_obj=a.config_overrides, action_name=action_name, block_keys=['loopable_actions', 'actions']
                        )
                        if extracted_config is None:
                            logger.error(f"❌ ERROR: Action '{action_name}' in self.actions has invalid config_overrides!")
                            logger.error(
                                f"   config_overrides keys: {list(a.config_overrides.keys()) if isinstance(a.config_overrides, dict) else 'N/A'}"
                            )
                            logger.error(f"   Removing from self.actions and recreating from loopable_action_configs...")
                            self.actions = [act for act in self.actions if act.name != action_name]
                            action = None
                            break
                        action = a
                        break

                if action is None:
                    # Create action from loopable_action_configs
                    if action_name in self.loopable_action_configs:
                        logger.debug(f"🔄 Creating PipelineAction for '{action_name}' from loopable_action_configs")
                        # The stored config is already cleaned (action_name removed), so use it directly
                        stored_config = self.loopable_action_configs[action_name]

                        config_overrides = copy.deepcopy(stored_config)

                        action = PipelineAction(name=action_name, action_name=action_name, config_overrides=config_overrides, outputs_to_track=[])
                    else:
                        raise ValueError(f"Loopable action '{action_name}' not found in actions or loopable_action_configs")

                # Create iteration-aware action name
                iteration_action_name = f"{action_name}_{iteration_id}"

                # Mark action as running
                action_statuses[iteration_action_name] = {'cached': False, 'completed': False, 'running': True}
                if action_name not in action_statuses:
                    action_statuses[action_name] = {'cached': False, 'completed': False, 'running': True}
                else:
                    action_statuses[action_name]['running'] = True

                # Update status display
                if not debug_mode:
                    current_progress = len(pre_loopable_actions)
                    self._print_pipeline_status(self.actions, action_statuses, current_progress)

                # Run the action with loop context
                action_output, was_cached = self._run_action_with_status(action, loop_context=loop_context, iteration_name=iteration_action_name)

                # Store output in nested dict structure
                if action_name not in self.action_outputs:
                    self.action_outputs[action_name] = {}
                self.action_outputs[action_name][iteration_id] = action_output
                self.action_outputs[iteration_action_name] = action_output
                action_statuses[iteration_action_name] = {'cached': was_cached, 'completed': True, 'running': False}

                # Update base action status
                has_running_iteration = any(
                    key.startswith(action_name + "_") and key != iteration_action_name and action_statuses.get(key, {}).get('running', False)
                    for key in action_statuses.keys()
                )

                if action_name not in action_statuses:
                    action_statuses[action_name] = {'cached': was_cached, 'completed': False, 'running': False}
                else:
                    if not was_cached:
                        action_statuses[action_name]['cached'] = False
                    action_statuses[action_name]['running'] = has_running_iteration

                    completed_iterations = sum(
                        1 for key in action_statuses.keys() if key.startswith(action_name + "_") and action_statuses[key].get('completed', False)
                    )
                    if completed_iterations >= len(self.loop_iterations):
                        action_statuses[action_name]['completed'] = True
                        action_statuses[action_name]['running'] = False

                logger.info(f"  ✅ Completed {action_name} (iteration {iteration_num})")

                # Update status display
                if not debug_mode:
                    current_progress = len(pre_loopable_actions)
                    self._print_pipeline_status(self.actions, action_statuses, current_progress)

            # Clear loop context after iteration
            self.current_loop_context = None

        # After all loopable iterations complete
        if not debug_mode and loopable_idx is not None:
            action_statuses["__loopable_actions__"] = {'cached': False, 'completed': True}
            current_progress = len(pre_loopable_actions) + 1
            self._print_pipeline_status(self.actions, action_statuses, current_progress)

    # _check_iteration_cache is now provided by LoopableCacheMixin

    def _create_iteration_configs(self) -> Path:
        """
        Create iteration-specific config files for parallel execution.

        Each iteration gets its own config file containing the loop context.

        Returns:
            Path to the loop_iterations directory
        """
        run_dir = Path(self.cfg.get('run_dir', '.'))
        loop_iterations_dir = run_dir / "loop_iterations"
        loop_iterations_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📝 Creating {len(self.loop_iterations)} iteration config files in {loop_iterations_dir}")

        for iteration_idx, loop_context in enumerate(self.loop_iterations):
            iteration_config_path = loop_iterations_dir / f"{iteration_idx}.yaml"
            # Save the loop context as a YAML file
            iteration_cfg = OmegaConf.create(loop_context)
            with open(iteration_config_path, 'w') as f:
                OmegaConf.save(iteration_cfg, f)
            logger.debug(f"  Created iteration config {iteration_idx}: {iteration_config_path}")

        return loop_iterations_dir

    def _submit_parallel_loopable_jobs(
        self, pre_loopable_actions: List[PipelineAction], action_statuses: Dict[str, Dict], debug_mode: bool, loopable_idx: Optional[int] = None
    ):
        """
        Submit parallel loopable actions as SLURM job array.

        NOTE: This method should NOT be called directly when using SLURM.
        Instead, use _submit_array_only() which handles caching checks and submission.
        This method is kept for backward compatibility and non-SLURM parallel execution.

        Creates iteration config files and submits:
        1. A job array with one task per iteration
        2. A dependency job for post-loopable actions

        Args:
            pre_loopable_actions: List of actions that run before loopable actions
            action_statuses: Dictionary tracking action execution status
            debug_mode: Whether debug mode is enabled
            loopable_idx: Index of the loopable block in the actions list
        """
        # Create iteration config files
        loop_iterations_dir = self._create_iteration_configs()

        # Import here to avoid circular dependencies
        from urartu.utils.execution.launcher import launch_on_slurm

        # Get pipeline name
        pipeline_name = self.cfg.get('pipeline_name')
        if not pipeline_name:
            raise ValueError("Config must specify 'pipeline_name' for parallel execution")

        # Prepare base config for array jobs
        base_array_cfg = OmegaConf.create(self.cfg)
        base_array_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
        
        # Create array_tasks directory structure for organizing outputs
        run_dir = Path(self.cfg.get('run_dir', '.'))
        array_tasks_dir = run_dir / "array_tasks"
        array_tasks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created array_tasks directory: {array_tasks_dir}")

        # Submit job array - SubmitIt will automatically create an array when we submit multiple jobs
        logger.info(f"🚀 Submitting SLURM job array with {len(self.loop_iterations)} tasks")
        logger.info(f"📋 Submission job will use minimal resources (just submits array, doesn't run work)")
        array_jobs = launch_on_slurm(
            module=str(Path.cwd()),
            action_name=pipeline_name,
            cfg=base_array_cfg,
            aim_run=self.aim_run,
            array_size=len(self.loop_iterations),
        )

        # Get array job ID (first job's ID, array tasks will have _<task_id> suffix)
        if isinstance(array_jobs, list) and len(array_jobs) > 0:
            array_job_id = array_jobs[0].job_id.split('_')[0]  # Extract base job ID
        else:
            array_job_id = array_jobs.job_id.split('_')[0] if hasattr(array_jobs, 'job_id') else str(array_jobs)
        
        logger.info(f"✅ Submitted job array {array_job_id} with {len(self.loop_iterations)} tasks")

        # Check if there are post-loopable actions
        loopable_idx = next((idx for idx, a in enumerate(self.actions) if a.name == "__loopable_actions__"), None)
        has_post_loopable = loopable_idx is not None and loopable_idx < len(self.actions) - 1

        if has_post_loopable:
            # Submit dependency job for post-loopable actions
            # Create dependency string: afterok:array_job_id (wait for all array tasks)
            dependency_str = f"afterok:{array_job_id}"
            logger.info(f"🚀 Submitting dependency job for post-loopable actions (depends on array job {array_job_id})")
            
            dep_cfg = OmegaConf.create(self.cfg)
            dep_cfg['_post_loopable_only'] = True
            dep_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
            dep_cfg['_iteration_job_ids'] = [array_job_id]  # Store array job ID for reference
            
            # CRITICAL: Remove _submit_array_only flag - this job should run post-loopable actions
            if '_submit_array_only' in dep_cfg:
                del dep_cfg['_submit_array_only']
            
            from urartu.utils.execution.launcher import launch_on_slurm

            dep_job = launch_on_slurm(
                module=str(Path.cwd()),
                action_name=pipeline_name,
                cfg=dep_cfg,
                aim_run=self.aim_run,
                array_size=None,
                dependency=dependency_str,
            )
            
            logger.info(f"✅ Submitted dependency job {dep_job.job_id} for post-loopable actions")
        else:
            logger.info("No post-loopable actions, skipping dependency job")

        # Mark loopable actions as submitted (not completed yet)
        action_statuses["__loopable_actions__"] = {
            'cached': False,
            'completed': False,
            'running': True,
            'submitted': True,
            'array_job_id': array_job_id,
        }
        if not debug_mode:
            self._print_pipeline_status(self.actions, action_statuses, len(pre_loopable_actions))

    # _submit_array_only_internal is now provided by LoopableJobsMixin

    def run(self):
        """Execute the pipeline by running all stages in sequence, with support for loopable actions."""
        # Check if this is a resume job (resumes from a specific stage)
        resume_at_stage = self.cfg.get('_resume_at_stage', None)
        if resume_at_stage is not None:
            logger.info(f"🔄 Detected _resume_at_stage={resume_at_stage} - resuming from stage {resume_at_stage}")
            self._run_from_stage(int(resume_at_stage))
            return
        
        # Check if this is a post-loopable-only job (DEPRECATED: kept for backward compatibility)
        post_loopable_only = self.cfg.get('_post_loopable_only', False)
        if post_loopable_only:
            logger.info("🔄 Detected _post_loopable_only flag (deprecated) - running only post-loopable actions")
            self._run_post_loopable_only()
            return
        
        # Check if this is a submission-only job (should only submit jobs and exit)
        submit_array_only = self.cfg.get('_submit_array_only', False)
        if submit_array_only:
            logger.info("🔄 Detected _submit_array_only flag - submitting iteration jobs and exiting")
            
            # Check if this is a multi-stage pipeline (multiple loopable blocks)
            # If so, use stage-based submission; otherwise use legacy method
            if not self._initialized:
                self.initialize()
            
            # Parse pipeline to detect stages
            from .pipeline_parser import PipelineParser
            parser = PipelineParser(self.pipeline_config, dynamic_loader=self)
            stages = parser.parse()
            
            # Count loopable stages
            from .pipeline_stage import LoopableStage
            loopable_stages = [s for s in stages if isinstance(s, LoopableStage)]
            
            if len(loopable_stages) > 1:
                # Multi-stage pipeline: use new stage-based submission
                logger.info(f"📊 Detected {len(loopable_stages)} loopable stages - using stage-based submission")
                self._submit_all_stages()
            else:
                # Single loopable block: use legacy method
                logger.info("📊 Single loopable stage - using legacy submission")
                self._submit_array_only_internal()
            return
        
        # If running as an iteration task, use _run_single_iteration
        is_iteration_task = self.cfg.get('_is_iteration_task', False)
        if is_iteration_task:
            iteration_idx = self.cfg.get('_iteration_idx')
            if iteration_idx is None:
                raise ValueError("_is_iteration_task is True but _iteration_idx is not set")
            
            iteration_idx = int(iteration_idx)
            # Load iteration context from disk
            run_dir = Path(self.cfg.get('run_dir', '.'))
            loop_iterations_dir = Path(self.cfg.get('_loop_iterations_dir', run_dir / 'loop_iterations'))
            
            # Check for stage-based directory structure first, fallback to legacy
            stage_idx = self.cfg.get('_stage_idx', None)
            if stage_idx is not None:
                iteration_config_path = loop_iterations_dir / f"stage_{stage_idx}" / f"{iteration_idx}.yaml"
            else:
                # Legacy: try block-based or flat structure
                block_idx = self.cfg.get('_block_idx', None)
                if block_idx is not None:
                    iteration_config_path = loop_iterations_dir / f"block_{block_idx}" / f"{iteration_idx}.yaml"
                else:
                    iteration_config_path = loop_iterations_dir / f"{iteration_idx}.yaml"
            
            if not iteration_config_path.exists():
                raise FileNotFoundError(f"Iteration config not found: {iteration_config_path}")
            
            iteration_cfg = OmegaConf.load(iteration_config_path)
            loop_context = OmegaConf.to_container(iteration_cfg, resolve=True)
            logger.info(f"📋 Running iteration {iteration_idx}")
            self._run_single_iteration(iteration_idx, loop_context)
            return
        
        # Default: run the entire pipeline from stage 0
        self._run_from_stage(0)
    
    def _submit_all_stages(self):
        """
        Submit all loopable stages for parallel execution.
        
        This method is called when _submit_array_only is set and the pipeline has multiple loopable stages.
        It submits all stages and sets up the dependency chain.
        """
        if not self._initialized:
            self.initialize()
        
        # Parse the pipeline into stages
        from .pipeline_parser import PipelineParser
        
        parser = PipelineParser(self.pipeline_config, dynamic_loader=self)
        stages = parser.parse()
        
        if not stages:
            logger.warning("No stages found in pipeline")
            return
        
        # Load pre-loopable actions from cache (needed for cache checking)
        from .pipeline_stage import ActionStage, LoopableStage
        
        # Find first loopable stage
        first_loopable_idx = next((i for i, s in enumerate(stages) if isinstance(s, LoopableStage)), None)
        if first_loopable_idx is not None and first_loopable_idx > 0:
            logger.info(f"Loading pre-loopable actions (stages 0-{first_loopable_idx-1}) from cache...")
            for stage_idx in range(first_loopable_idx):
                stage = stages[stage_idx]
                if isinstance(stage, ActionStage):
                    # Load action output from cache
                    try:
                        from .pipeline_action import PipelineAction
                        action = PipelineAction(
                            name=stage.action_name,
                            action_name=stage.action_name,
                            config_overrides=stage.action_config,
                            outputs_to_track=stage.outputs_to_track,
                        )
                        action_output, was_cached = self._run_action_with_status(action)
                        self.action_outputs[stage.action_name] = action_output
                        logger.info(f"Loaded pre-loopable action '{stage.action_name}' from cache")
                    except Exception as e:
                        logger.warning(f"Could not load pre-loopable action '{stage.action_name}': {e}")
        
        # Submit all loopable stages
        all_job_ids = []
        run_dir = Path(self.cfg.get("run_dir", "."))
        loop_iterations_dir = run_dir / "loop_iterations"
        
        for stage in stages:
            if isinstance(stage, LoopableStage):
                logger.info(f"Submitting loopable stage {stage.stage_idx}: {stage}")
                
                # Set up loop context for this stage
                self.loop_configs = stage.loop_configs
                self.loop_iterations = stage.loop_iterations
                self.loopable_actions = stage.loopable_actions
                
                # Store action configs
                if not hasattr(self, 'loopable_action_configs'):
                    self.loopable_action_configs = {}
                for action_name in stage.loopable_actions:
                    self.loopable_action_configs[action_name] = stage.action_configs.get(action_name, {})
                
                # Update config injector
                if self.config_injector is None:
                    from .config import ConfigInjector
                    self.config_injector = ConfigInjector(
                        action_outputs=self.action_outputs,
                        loopable_actions=stage.loopable_actions,
                        loop_configs=stage.loop_configs,
                    )
                else:
                    self.config_injector.loopable_actions = stage.loopable_actions
                    self.config_injector.loop_configs = stage.loop_configs
                
                # Store stage_idx in config for iteration tasks
                self.cfg["_stage_idx"] = stage.stage_idx
                
                # Check caching and submit
                uncached_iterations = []
                cached_iterations = []
                
                logger.info(f"Checking cache status for {stage.get_num_iterations()} iterations in stage {stage.stage_idx}...")
                for iteration_idx, loop_context in enumerate(stage.loop_iterations):
                    try:
                        is_cached = self._check_iteration_cache(iteration_idx, loop_context)
                        if is_cached:
                            cached_iterations.append(iteration_idx)
                        else:
                            uncached_iterations.append(iteration_idx)
                    except Exception as e:
                        logger.error(f"Error checking cache for iteration {iteration_idx}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        uncached_iterations.append(iteration_idx)
                
                logger.info(f"Cache summary for stage {stage.stage_idx}: {len(cached_iterations)} cached, {len(uncached_iterations)} uncached")
                
                # Create iteration configs (always create, even if cached)
                stage_dir = loop_iterations_dir / f"stage_{stage.stage_idx}"
                stage_dir.mkdir(parents=True, exist_ok=True)
                
                for iteration_idx, loop_context in enumerate(stage.loop_iterations):
                    iteration_config_path = stage_dir / f"{iteration_idx}.yaml"
                    iteration_cfg = OmegaConf.create(loop_context)
                    with open(iteration_config_path, "w") as f:
                        OmegaConf.save(iteration_cfg, f)
                
                # Submit jobs for uncached iterations
                if uncached_iterations:
                    stage_job_ids = self._submit_iteration_jobs(uncached_iterations, loop_iterations_dir)
                    all_job_ids.extend(stage_job_ids)
                    logger.info(f"Submitted {len(stage_job_ids)} jobs for stage {stage.stage_idx}")
                else:
                    logger.info(f"All iterations cached for stage {stage.stage_idx}, no jobs to submit")
        
        # Submit final dependency job to resume at the first post-loopable action
        # Find the index of the last loopable stage
        loopable_stages = [s for s in stages if isinstance(s, LoopableStage)]
        if not loopable_stages:
            logger.warning("No loopable stages found")
            return
        
        last_loopable_idx = max(s.stage_idx for s in loopable_stages)
        
        if last_loopable_idx < len(stages) - 1:
            # There are actions after the last loopable stage
            resume_at_stage = last_loopable_idx + 1
            logger.info(f"Submitting dependency job to resume at stage {resume_at_stage}")
            
            # Create dependency configuration
            dep_cfg = OmegaConf.create(self.cfg)
            dep_cfg['_resume_at_stage'] = resume_at_stage
            dep_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
            
            # Remove submission flags
            if '_submit_array_only' in dep_cfg:
                del dep_cfg['_submit_array_only']
            if '_stage_idx' in dep_cfg:
                del dep_cfg['_stage_idx']
            
            # Submit the dependency job
            if all_job_ids:
                # Filter out any empty or invalid job IDs
                valid_job_ids = [job_id for job_id in all_job_ids if job_id and str(job_id).strip()]
                if valid_job_ids:
                    dependency_str = "afterok:" + ":".join(valid_job_ids)
                else:
                    dependency_str = None
                    logger.info("No valid job IDs for dependency, submitting dependency job without dependencies")
            else:
                dependency_str = None
                logger.info("All iterations cached, submitting dependency job without dependencies")
            
            from urartu.utils.execution.launcher import launch_on_slurm
            pipeline_name = self.cfg.get('pipeline_name', 'pipeline')
            
            try:
                dep_job = launch_on_slurm(
                    module=str(Path.cwd()),
                    action_name=pipeline_name,
                    cfg=dep_cfg,
                    aim_run=self.aim_run,
                    array_size=None,
                    dependency=dependency_str,
                )
                
                logger.info(f"Submitted dependency job {dep_job.job_id} to resume at stage {resume_at_stage}")
            except Exception as e:
                logger.error(f"Failed to submit dependency job: {e}")
                logger.error(f"  Dependency string: {dependency_str}")
                logger.error(f"  All job IDs: {all_job_ids}")
                raise
        else:
            logger.info("No post-loopable actions, no dependency job needed")
    
    def _run_from_stage(self, start_stage: int = 0):
        """
        Run the pipeline from a specific stage.
        
        This is the new stage-based execution path that supports arbitrary loopable blocks.
        
        Args:
            start_stage: Stage index to start from (0 = beginning)
        """
        if not self._initialized:
            self.initialize()
        
        debug_mode = self.pipeline_config.get('debug', False)
        
        # Parse the pipeline into stages
        from .pipeline_parser import PipelineParser
        from .pipeline_executor import PipelineExecutor
        
        parser = PipelineParser(self.pipeline_config, dynamic_loader=self)
        stages = parser.parse()
        
        if not stages:
            logger.warning("No stages found in pipeline")
            return
        
        # Load outputs from previously completed loopable stages (if resuming)
        if start_stage > 0:
            logger.info(f"Loading outputs from completed stages (0-{start_stage-1})...")
            self._load_completed_stage_outputs(stages[:start_stage])
        
        # Create executor and execute from the specified stage
        executor = PipelineExecutor(self, stages)
        executor.execute(resume_at_stage=start_stage)
        
        logger.info("Pipeline execution completed")
    
    def _load_completed_stage_outputs(self, completed_stages: List[Any]):
        """
        Load outputs from stages that were completed in a previous run.
        
        Args:
            completed_stages: List of PipelineStage objects that were already completed
        """
        from .pipeline_stage import ActionStage, LoopableStage
        from .pipeline_executor import PipelineExecutor
        
        for stage in completed_stages:
            if isinstance(stage, ActionStage):
                # Load single action output from cache
                logger.info(f"Loading output for completed action stage: {stage.action_name}")
                # The action should be cached, so we can load it
                action = PipelineAction(
                    name=stage.action_name,
                    action_name=stage.action_name,
                    config_overrides=stage.action_config,
                    outputs_to_track=stage.outputs_to_track,
                )
                try:
                    action_output, was_cached = self._run_action_with_status(action)
                    self.action_outputs[action.name] = action_output
                except Exception as e:
                    logger.warning(f"Error loading action output for {stage.action_name}: {e}")
            elif isinstance(stage, LoopableStage):
                # Load all iteration outputs from this loopable stage
                logger.info(f"Loading outputs from completed loopable stage {stage.stage_idx}")
                executor = PipelineExecutor(self, [stage])
                executor.load_stage_outputs(stage)
    
    def _run_legacy(self):
        """
        Run the pipeline using the legacy execution path.
        
        This method preserves the original behavior for backward compatibility.
        It should only be used for pipelines that haven't been migrated to the stage-based model.
        """
        if not self._initialized:
            self.initialize()

        debug_mode = self.pipeline_config.get('debug', False)

        # Separate actions into: pre-loopable, loopable block, post-loopable
        loopable_idx = next((idx for idx, a in enumerate(self.actions) if a.name == "__loopable_actions__"), None)
        has_loopable_block = loopable_idx is not None

        if has_loopable_block:
            pre_loopable_actions = self.actions[:loopable_idx]
            post_loopable_actions = self.actions[loopable_idx + 1 :]
        else:
            pre_loopable_actions = [a for a in self.actions if a.name != "__loopable_actions__"]
            post_loopable_actions = []

        regular_actions = pre_loopable_actions
        successful_actions = 0
        action_statuses = {}

        if not debug_mode:
            self._print_pipeline_status(self.actions, action_statuses, 0)

        # Phase 1: Run regular actions once
        logger.info("=" * 80)
        logger.info("Phase 1: Running regular actions (once)")
        logger.info("=" * 80)

        for i, action in enumerate(regular_actions):
            action_output, was_cached = self._run_action_with_status(action)
            self.action_outputs[action.name] = action_output
            action_statuses[action.name] = {'cached': was_cached, 'completed': True}

            if not action_output.metadata.get("skipped", False):
                successful_actions += 1

            if not debug_mode:
                self._print_pipeline_status(self.actions, action_statuses, i + 1)

        # Phase 2: Run loopable actions for each iteration
        if has_loopable_block and self.loopable_actions and self.loop_iterations:
            # Check if parallel processing was used (jobs submitted)
            parallel_used = self._should_use_parallel_processing()
            self._run_loopable_actions(pre_loopable_actions, action_statuses, debug_mode, loopable_idx)
            
            # If parallel processing was used, jobs were submitted and dependency job will handle post-loopable actions
            if parallel_used:
                logger.info("⏸️  Parallel processing enabled - post-loopable actions will run in dependency job")
                return  # Exit early, dependency job will handle post-loopable actions

        # Phase 3: Run post-loopable actions (e.g., aggregator)
        if post_loopable_actions:
            logger.info("=" * 80)
            logger.info(f"Phase 3: Running post-loopable actions ({len(post_loopable_actions)} action(s))")
            logger.info("=" * 80)

            for i, action in enumerate(post_loopable_actions):
                action_output, was_cached = self._run_action_with_status(action)
                self.action_outputs[action.name] = action_output
                action_statuses[action.name] = {'cached': was_cached, 'completed': True}

                if not action_output.metadata.get("skipped", False):
                    successful_actions += 1

                if not debug_mode:
                    current_progress = len(pre_loopable_actions) + 1 + (i + 1)
                    self._print_pipeline_status(self.actions, action_statuses, current_progress)

        # Final status
        if not debug_mode:
            self._print_pipeline_status(self.actions, action_statuses, len(action_statuses))
            logger.info(f"✅ Pipeline completed successfully!")

    def _run_single_iteration(self, iteration_idx: int, loop_context: Dict[str, Any]):
        """
        Run a single iteration of loopable actions.

        This is called when running as a SLURM array task.

        Args:
            iteration_idx: Index of the iteration to run
            loop_context: The loop context for this iteration
        """
        # Initialize first (this will load all iterations from config)
        if not self._initialized:
            self.initialize()
        
        # Override loop_iterations AFTER initialization to only include this single iteration
        # This prevents the pipeline from trying to submit another array
        # Note: initialize() may have loaded all iterations, so we override it here
        num_iterations_before = len(self.loop_iterations)
        self.loop_iterations = [loop_context]
        logger.info(
            f"📋 Overrode loop_iterations after initialization to only include iteration {iteration_idx} (was {num_iterations_before} iterations, now 1)"
        )

        # Find pre-loopable actions (actions before the loopable block)
        loopable_idx = next((idx for idx, a in enumerate(self.actions) if a.name == "__loopable_actions__"), None)
        if loopable_idx is not None:
            pre_loopable_actions = self.actions[:loopable_idx]
        else:
            pre_loopable_actions = [a for a in self.actions if a.name != "__loopable_actions__"]

        # Run pre-loopable actions first (they may be dependencies for loopable actions)
        if pre_loopable_actions:
            logger.info("=" * 80)
            logger.info(f"Phase 1: Running pre-loopable actions ({len(pre_loopable_actions)} action(s))")
            logger.info("=" * 80)
            for action in pre_loopable_actions:
                logger.info(f"🔄 Running pre-loopable action: '{action.name}'")
                action_output, was_cached = self._run_action_with_status(action)
                self.action_outputs[action.name] = action_output
                logger.info(f"  ✅ Completed {action.name}")

        iteration_num = iteration_idx + 1
        logger.info("=" * 80)
        logger.info(f"Phase 2: Running iteration {iteration_num}/1 (array task {iteration_idx})")
        logger.info(f"Loop context: {loop_context}")
        logger.info("=" * 80)

        # Set current loop context for resolvers
        self.current_loop_context = loop_context

        # Generate iteration identifier
        iteration_id = self._get_iteration_id(loop_context)
        if not iteration_id:
            iteration_id = f"iteration_{iteration_num}"

        # Run each loopable action for this iteration
        for action_name in self.loopable_actions:
            logger.info(f"🔄 Running loopable action: '{action_name}' (iteration {iteration_num})")

            # Find or create the action
            action = None
            for a in self.actions:
                if a.name == action_name:
                    action = a
                    break

            if action is None:
                # Create action from loopable_action_configs
                if action_name in self.loopable_action_configs:
                    stored_config = self.loopable_action_configs[action_name]
                    config_overrides = copy.deepcopy(stored_config)
                    action = PipelineAction(name=action_name, action_name=action_name, config_overrides=config_overrides, outputs_to_track=[])
                else:
                    raise ValueError(f"Loopable action '{action_name}' not found")

            # Create iteration-aware action name
            iteration_action_name = f"{action_name}_{iteration_id}"

            # Run the action with loop context
            action_output, was_cached = self._run_action_with_status(action, loop_context=loop_context, iteration_name=iteration_action_name)

            # Store output in nested dict structure
            if action_name not in self.action_outputs:
                self.action_outputs[action_name] = {}
            self.action_outputs[action_name][iteration_id] = action_output
            self.action_outputs[iteration_action_name] = action_output

            # Save output to disk for dependency job to load
            # Save to run_dir / array_tasks / task_{iteration_idx} / action_name / output.pkl
            # Using simple structure since we're using individual jobs, not arrays
            run_dir = Path(self.cfg.get('run_dir', '.'))
            array_task_dir = run_dir / "array_tasks" / f"task_{iteration_idx}"
            action_output_dir = array_task_dir / action_name
            action_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = action_output_dir / "output.pkl"
            try:
                import pickle

                with open(output_file, 'wb') as f:
                    pickle.dump(action_output, f)
                logger.debug(f"💾 Saved output for {action_name}[{iteration_id}] to {output_file}")
            except Exception as e:
                logger.warning(f"Could not save output to {output_file}: {e}")

            logger.info(f"  ✅ Completed {action_name} (iteration {iteration_num})")

        # Clear loop context
        self.current_loop_context = None
        logger.info(f"✅ Completed iteration {iteration_num}")

    def _run_post_loopable_only(self):
        """
        Run only post-loopable actions.

        This is called by the dependency job after all array tasks complete.
        """
        if not self._initialized:
            self.initialize()

        logger.info("=" * 80)
        logger.info("Running post-loopable actions (dependency job)")
        logger.info("=" * 80)

        # Find loopable block index (supports both simple and combined pipelines)
        loopable_indices = [
            idx for idx, a in enumerate(self.actions)
            if a.name == "__loopable_actions__" or a.name.startswith("__loopable_actions_block_")
        ]
        
        if not loopable_indices:
            logger.warning("No loopable actions block found, nothing to do")
            return

        # Use the last loopable block index (for combined pipelines with multiple blocks)
        last_loopable_idx = max(loopable_indices)
        first_loopable_idx = min(loopable_indices)
        post_loopable_actions = self.actions[last_loopable_idx + 1 :]
        if not post_loopable_actions:
            logger.info("No post-loopable actions found")
            return

        logger.info(f"Found {len(post_loopable_actions)} post-loopable action(s)")

        # First, load pre-loopable action outputs (they were run in the initial submission job)
        # These are needed as dependencies for post-loopable actions
        pre_loopable_actions = self.actions[:first_loopable_idx]
        logger.info(f"Loading outputs for {len(pre_loopable_actions)} pre-loopable action(s)...")
        for action in pre_loopable_actions:
            # Try to load from cache or run directory
            try:
                # Get action config (same way _run_action does it)
                # Resolve config overrides
                context = {"action_outputs": self.action_outputs}
                resolved_config = self._resolve_value(action.config_overrides, context)
                
                # Inject any dependencies (though pre-loopable actions typically don't depend on each other)
                resolved_config = self._inject_action_outputs(resolved_config, action.name)
                
                # Merge with base config (same structure as _run_action)
                from omegaconf import OmegaConf

                action_cfg = OmegaConf.create(self.cfg)
                action_cfg.action_name = action.action_name
                
                # Ensure run_dir is preserved
                if 'run_dir' not in action_cfg or not action_cfg.run_dir:
                    if hasattr(self.cfg, 'run_dir') and self.cfg.run_dir:
                        action_cfg.run_dir = self.cfg.run_dir
                
                # Remove pipeline key so Action instances will use cfg.action instead
                if 'pipeline' in action_cfg:
                    del action_cfg.pipeline
                
                # Merge resolved config and set action config structure (same as _run_action)
                # Apply pipeline common configs
                pipeline_common_configs = self._get_common_pipeline_configs()
                merged_config = OmegaConf.merge(
                    OmegaConf.create(pipeline_common_configs),
                    OmegaConf.create(resolved_config),
                )
                action_cfg.action = merged_config
                
                # Create action instance to load from cache (same logic as _run_action)
                # Import action module
                import sys
                import importlib
                from pathlib import Path
                from urartu.common.action import Action, ActionDataset
                
                action_module_name = action.action_name
                current_dir = Path.cwd()
                
                # Find project root and actions directory (same logic as _run_action)
                project_root = None
                actions_dir = None
                search_dir = current_dir
                max_levels = 5
                for _ in range(max_levels):
                    if (search_dir / "self_aware" / "actions").exists():
                        project_root = search_dir
                        actions_dir = search_dir / "self_aware" / "actions"
                        break
                    elif search_dir.name == "self_aware" and (search_dir / "actions").exists():
                        project_root = search_dir.parent
                        actions_dir = search_dir / "actions"
                        break
                    elif (search_dir / "actions").exists() and (search_dir / "self_aware").exists():
                        project_root = search_dir
                        actions_dir = search_dir / "actions"
                        break
                    if search_dir == search_dir.parent:
                        break
                    search_dir = search_dir.parent
                
                if project_root is None:
                    project_root = current_dir
                    actions_dir = current_dir / "actions"
                
                if actions_dir.exists():
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))
                    if str(actions_dir) not in sys.path:
                        sys.path.append(str(actions_dir))
                    
                    try:
                        action_module = importlib.import_module(f'actions.{action_module_name}')
                        # Find action class
                        action_class = None
                        action_candidates = []
                        for attr_name in dir(action_module):
                            attr = getattr(action_module, attr_name)
                            if isinstance(attr, type) and issubclass(attr, Action) and attr != Action and attr != ActionDataset:
                                action_candidates.append(attr)
                        
                        # Prefer the most specific class
                        for candidate in action_candidates:
                            if candidate.__module__ == action_module.__name__:
                                action_class = candidate
                                break
                        
                        if not action_class and action_candidates:
                            action_class = action_candidates[0]
                        
                        if action_class:
                            # Create action instance
                            action_instance = action_class(action_cfg, self.aim_run)
                        else:
                            logger.warning(f"Could not find action class in {action_module_name}")
                            action_instance = None
                    except ImportError as e:
                        logger.error(f"Failed to import action module {action_module_name}: {e}")
                        action_instance = None
                else:
                    logger.warning(f"Actions directory not found: {actions_dir}")
                    action_instance = None
                
                if action_instance:
                    # Try to load from cache
                    cached_outputs = action_instance._load_from_cache()
                    if cached_outputs is not None:
                        # _load_from_cache returns a dict of outputs, wrap it in ActionOutput
                        from urartu.common.pipeline.types import ActionOutput

                        action_output = ActionOutput(
                            name=action.name,
                            action_name=action.action_name,
                            outputs=cached_outputs if isinstance(cached_outputs, dict) else {},
                            metadata={"from_cache": True},
                        )
                        self.action_outputs[action.name] = action_output
                        logger.info(f"📦 Loaded output for pre-loopable action '{action.name}' from cache")
                    else:
                        # Try to load from run directory
                        run_dir = Path(self.cfg.get('run_dir', '.'))
                        action_output_dir = run_dir / action.name
                        output_file = action_output_dir / "output.pkl"
                        if output_file.exists():
                            import pickle
                            from urartu.common.pipeline.types import ActionOutput

                            with open(output_file, 'rb') as f:
                                saved_output = pickle.load(f)
                                # Ensure it's an ActionOutput object
                                if not isinstance(saved_output, ActionOutput):
                                    # If it's a dict, wrap it
                                    if isinstance(saved_output, dict):
                                        action_output = ActionOutput(
                                            name=action.name, action_name=action.action_name, outputs=saved_output, metadata={"from_file": True}
                                        )
                                    else:
                                        logger.warning(f"⚠️  Unexpected type for saved output: {type(saved_output)}")
                                        continue
                                else:
                                    action_output = saved_output
                                self.action_outputs[action.name] = action_output
                                logger.info(f"📦 Loaded output for pre-loopable action '{action.name}' from {output_file}")
                        else:
                            logger.warning(f"⚠️  Could not load output for pre-loopable action '{action.name}' (not in cache or run directory)")
            except Exception as e:
                logger.warning(f"⚠️  Error loading output for pre-loopable action '{action.name}': {e}")

        # Check if this is a multi-block pipeline
        if hasattr(self, 'loopable_blocks') and self.loopable_blocks:
            # Multi-block pipeline: use mixin method if available
            if hasattr(self, '_load_iteration_outputs_from_blocks'):
                logger.info("Detected multi-block pipeline, using _load_iteration_outputs_from_blocks()")
                self._load_iteration_outputs_from_blocks()
            else:
                logger.warning("Multi-block pipeline detected but _load_iteration_outputs_from_blocks() not available")
                # Fall through to single-block logic
        else:
            # Single-block pipeline: use existing logic
            # Load all iteration outputs from disk
            # Array tasks save their outputs to run_dir, we need to load them
            run_dir = Path(self.cfg.get('run_dir', '.'))
            loop_iterations_dir = Path(self.cfg.get('_loop_iterations_dir', run_dir / 'loop_iterations'))

            # Initialize action_outputs structure for all loopable actions
            for action_name in self.loopable_actions:
                if action_name not in self.action_outputs:
                    self.action_outputs[action_name] = {}

            logger.info(f"📦 Loading outputs for {len(self.loop_iterations)} iterations from task directories and cache...")
            logger.info(f"   Loop iterations directory: {loop_iterations_dir}")
            logger.info(f"   Loopable actions: {self.loopable_actions}")

            # Load iteration contexts and try to load outputs from disk
            # Outputs are typically saved by actions in their output directories
            # The get_iteration_outputs() method will handle loading them when needed
            # For now, we just ensure the structure is initialized
            
            # Check if we have block-specific directories (for combined pipelines)
            block_dirs = []
            if loop_iterations_dir.exists():
                block_dirs = [d for d in loop_iterations_dir.iterdir() if d.is_dir() and d.name.startswith('block_')]
            
            if block_dirs:
                # Combined pipeline: load from all blocks
                for block_dir in sorted(block_dirs):
                    block_iteration_configs = sorted(block_dir.glob("*.yaml"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
                    for iteration_config_path in block_iteration_configs:
                        if iteration_config_path.exists():
                            iteration_cfg = OmegaConf.load(iteration_config_path)
                            loop_context = OmegaConf.to_container(iteration_cfg, resolve=True)
                            iteration_id = self._get_iteration_id(loop_context)
                            if not iteration_id:
                                # Extract iteration index from filename
                                iteration_idx = int(iteration_config_path.stem) if iteration_config_path.stem.isdigit() else 0
                                iteration_id = f"iteration_{iteration_idx + 1}"

                            # Try to load outputs from disk if they exist
                            # Outputs are saved by iteration jobs to: run_dir / array_tasks / task_{iteration_idx} / action_name / output.pkl
                            # Using simple structure since we're using individual jobs, not arrays
                            for action_name in self.loopable_actions:
                                # Extract iteration index from filename for task directory lookup
                                iteration_idx = int(iteration_config_path.stem) if iteration_config_path.stem.isdigit() else 0
                                # Use simple structure: array_tasks/task_{iteration_idx}/
                                array_task_dir = run_dir / "array_tasks" / f"task_{iteration_idx}"
                                action_output_dir = array_task_dir / action_name
                                output_file = action_output_dir / "output.pkl"
                                
                                loaded = False
                                if output_file.exists():
                                    # Try to load from task directory (for newly executed iterations)
                                    try:
                                        import pickle

                                        with open(output_file, 'rb') as f:
                                            # Load the saved ActionOutput
                                            saved_output = pickle.load(f)
                                            if action_name not in self.action_outputs:
                                                self.action_outputs[action_name] = {}
                                            self.action_outputs[action_name][iteration_id] = saved_output
                                            # Also store with iteration-specific name for backward compatibility
                                            iteration_action_name = f"{action_name}_{iteration_id}"
                                            self.action_outputs[iteration_action_name] = saved_output
                                            logger.info(f"📦 Loaded output for {action_name}[{iteration_id}] from task directory: {output_file}")
                                            loaded = True
                                    except Exception as e:
                                        logger.warning(f"Could not load output from {output_file}: {e}")
                                
                                # If not found in task directory, try to load from cache (for cached iterations)
                                if not loaded:
                                    logger.debug(f"   Task directory output not found for {action_name}[{iteration_id}], trying cache...")
                                    try:
                                        # Find the action in the pipeline to get its full config (same logic as _check_iteration_cache)
                                        action = None
                                        for a in self.actions:
                                            if a.action_name == action_name:
                                                action = a
                                                break
                                        
                                        if action is None:
                                            # Create action from loopable_action_configs (same as _check_iteration_cache)
                                            if action_name in self.loopable_action_configs:
                                                stored_config = self.loopable_action_configs[action_name]
                                                # Handle combined pipeline structure: {block_0: {...}, block_1: {...}}
                                                if isinstance(stored_config, dict):
                                                    block_keys = [k for k in stored_config.keys() if k.startswith('block_')]
                                                    if block_keys:
                                                        # Extract block index from block_dir name (e.g., "block_0" -> 0)
                                                        block_idx = int(block_dir.name.split('_')[1]) if '_' in block_dir.name else 0
                                                        block_key = f"block_{block_idx}"
                                                        if block_key in stored_config:
                                                            config_overrides = copy.deepcopy(stored_config[block_key])
                                                        else:
                                                            # Fallback: use first available block
                                                            block_key = sorted(block_keys)[0]
                                                            config_overrides = copy.deepcopy(stored_config[block_key])
                                                            logger.warning(f"⚠️ Block key '{block_key}' not found, using {sorted(block_keys)[0]} for action '{action_name}'")
                                                    else:
                                                        # Standard pipeline structure - use stored_config directly
                                                        config_overrides = copy.deepcopy(stored_config)
                                                else:
                                                    config_overrides = copy.deepcopy(stored_config)
                                                action = PipelineAction(
                                                    name=action_name, action_name=action_name, config_overrides=config_overrides, outputs_to_track=[]
                                                )
                                            else:
                                                logger.warning(f"⚠️  Action '{action_name}' not found in pipeline actions or loopable_action_configs")
                                                continue
                                        
                                        # Get action config (same logic as _check_iteration_cache and _run_action)
                                        # Start with action's config_overrides
                                        action_config_dict = copy.deepcopy(action.config_overrides) if action.config_overrides else {}
                                        
                                        # Resolve any dynamic values in the config
                                        context = {"action_outputs": self.action_outputs, "loop_context": loop_context}
                                        resolved_config = self._resolve_value(action_config_dict, context)
                                        
                                        # Inject action outputs
                                        resolved_config = self._inject_action_outputs(resolved_config, action_name, loop_context=loop_context)
                                        
                                        # Merge with base config BEFORE injecting loop context
                                        # This ensures model.type and other base configs are present before we inject model.revision
                                        from omegaconf import OmegaConf

                                        pipeline_common_configs = self._get_common_pipeline_configs()
                                        merged_config = OmegaConf.merge(
                                            OmegaConf.create(pipeline_common_configs),  # Base (pipeline defaults)
                                            OmegaConf.create(resolved_config),  # Override (action-specific, with injections)
                                        )
                                        
                                        # Convert to dict for _inject_loop_context (it expects a dict and modifies it)
                                        resolved_config = OmegaConf.to_container(merged_config, resolve=True)
                                        
                                        # Inject loop context AFTER merging (ensures model.type is present)
                                        if loop_context:
                                            resolved_config = self._inject_loop_context(resolved_config, loop_context)
                                        
                                        # Convert back to OmegaConf for final use
                                        merged_config = OmegaConf.create(resolved_config)
                                        
                                        # Add pipeline config hash
                                        try:
                                            from urartu.common.pipeline.cache import PipelineCache

                                            pipeline_config_hash = PipelineCache.generate_config_hash(self.cfg)
                                            merged_config.pipeline_config_hash = pipeline_config_hash
                                        except Exception:
                                            pass
                                        
                                        # Set iteration_id
                                        if loop_context:
                                            iteration_id_from_context = self._get_iteration_id_from_context(loop_context)
                                            if iteration_id_from_context:
                                                merged_config.iteration_id = iteration_id_from_context
                                        
                                        # Create action_cfg structure
                                        action_cfg = OmegaConf.create(self.cfg)
                                        action_cfg.action_name = action_name
                                        if 'run_dir' not in action_cfg or not action_cfg.run_dir:
                                            if hasattr(self.cfg, 'run_dir') and self.cfg.run_dir:
                                                action_cfg.run_dir = self.cfg.run_dir
                                        if 'pipeline' in action_cfg:
                                            del action_cfg.pipeline
                                        action_cfg.action = merged_config
                                        
                                        # Import and create action instance
                                        import sys
                                        import importlib
                                        from pathlib import Path
                                        from urartu.common.action import Action, ActionDataset
                                        
                                        action_module_name = action_name
                                        current_dir = Path.cwd()
                                        
                                        # Find project root and actions directory
                                        project_root = None
                                        actions_dir = None
                                        search_dir = current_dir
                                        max_levels = 5
                                        for _ in range(max_levels):
                                            if (search_dir / "self_aware" / "actions").exists():
                                                project_root = search_dir
                                                actions_dir = search_dir / "self_aware" / "actions"
                                                break
                                            elif search_dir.name == "self_aware" and (search_dir / "actions").exists():
                                                project_root = search_dir.parent
                                                actions_dir = search_dir / "actions"
                                                break
                                            elif (search_dir / "actions").exists() and (search_dir / "self_aware").exists():
                                                project_root = search_dir
                                                actions_dir = search_dir / "actions"
                                                break
                                            if search_dir == search_dir.parent:
                                                break
                                            search_dir = search_dir.parent
                                        
                                        if project_root is None:
                                            project_root = current_dir
                                            actions_dir = current_dir / "actions"
                                        
                                        if actions_dir.exists():
                                            if str(project_root) not in sys.path:
                                                sys.path.insert(0, str(project_root))
                                            if str(actions_dir) not in sys.path:
                                                sys.path.append(str(actions_dir))
                                            
                                            try:
                                                action_module = importlib.import_module(f'actions.{action_module_name}')
                                                action_class = None
                                                for attr_name in dir(action_module):
                                                    attr = getattr(action_module, attr_name)
                                                    if isinstance(attr, type) and issubclass(attr, Action) and attr != Action and attr != ActionDataset:
                                                        action_class = attr
                                                        break
                                                
                                                if action_class:
                                                    action_instance = action_class(action_cfg, self.aim_run)
                                                    # Try to load from cache
                                                    cached_outputs = action_instance._load_from_cache()
                                                    if cached_outputs is not None:
                                                        # Wrap in ActionOutput
                                                        from urartu.common.pipeline.types import ActionOutput

                                                        action_output = ActionOutput(
                                                            name=action_name,
                                                            action_name=action_name,
                                                            outputs=cached_outputs if isinstance(cached_outputs, dict) else {},
                                                            metadata={"from_cache": True, "iteration_id": iteration_id},
                                                        )
                                                        if action_name not in self.action_outputs:
                                                            self.action_outputs[action_name] = {}
                                                        self.action_outputs[action_name][iteration_id] = action_output
                                                        # Also store with iteration-specific name for backward compatibility
                                                        iteration_action_name = f"{action_name}_{iteration_id}"
                                                        self.action_outputs[iteration_action_name] = action_output
                                                        logger.info(f"📦 Loaded output for {action_name}[{iteration_id}] from cache")
                                                        loaded = True
                                            except Exception as e:
                                                logger.warning(f"⚠️  Could not load {action_name}[{iteration_id}] from cache: {e}")
                                        
                                        if not loaded:
                                            logger.warning(f"⚠️  Output not found for {action_name}[{iteration_id}] (not in task directory or cache)")
                                    except Exception as e:
                                        logger.warning(f"⚠️  Error loading {action_name}[{iteration_id}] from cache: {e}")
                                        import traceback
                                        logger.debug(f"   Traceback: {traceback.format_exc()}")
            else:
                # Simple pipeline: load from flat structure
                for iteration_idx in range(len(self.loop_iterations)):
                    iteration_config_path = loop_iterations_dir / f"{iteration_idx}.yaml"
                if iteration_config_path.exists():
                    iteration_cfg = OmegaConf.load(iteration_config_path)
                    loop_context = OmegaConf.to_container(iteration_cfg, resolve=True)
                    iteration_id = self._get_iteration_id(loop_context)
                    if not iteration_id:
                        iteration_id = f"iteration_{iteration_idx + 1}"

                    # Try to load outputs from disk if they exist
                    for action_name in self.loopable_actions:
                        array_task_dir = run_dir / "array_tasks" / f"task_{iteration_idx}"
                        action_output_dir = array_task_dir / action_name
                        output_file = action_output_dir / "output.pkl"
                        
                        loaded = False
                        if output_file.exists():
                            try:
                                import pickle
                                with open(output_file, 'rb') as f:
                                    saved_output = pickle.load(f)
                                    if action_name not in self.action_outputs:
                                        self.action_outputs[action_name] = {}
                                    self.action_outputs[action_name][iteration_id] = saved_output
                                    iteration_action_name = f"{action_name}_{iteration_id}"
                                    self.action_outputs[iteration_action_name] = saved_output
                                    logger.info(f"📦 Loaded output for {action_name}[{iteration_id}] from task directory: {output_file}")
                                    loaded = True
                            except Exception as e:
                                logger.warning(f"Could not load output from {output_file}: {e}")
                        
                        # If not found, try cache (same logic as block-based loading above, but simplified)
                        if not loaded:
                            logger.debug(f"   Task directory output not found for {action_name}[{iteration_id}], trying cache...")
                            # Cache loading logic would go here (similar to block-based version)

        # Log summary of loaded outputs
        # Count total expected iterations (from all blocks if block-based structure exists)
        total_expected_iterations = len(self.loop_iterations)  # Default to current block's iterations
        if block_dirs:
            # Combined pipeline: count iterations from all blocks
            total_expected_iterations = sum(len(list(block_dir.glob("*.yaml"))) for block_dir in block_dirs)
        
        for action_name in self.loopable_actions:
            if action_name in self.action_outputs and isinstance(self.action_outputs[action_name], dict):
                num_loaded = len(self.action_outputs[action_name])
                logger.info(f"📊 Loaded {num_loaded}/{total_expected_iterations} outputs for '{action_name}'")
                if num_loaded == 0:
                    logger.warning(f"⚠️  No outputs loaded for '{action_name}' - this may cause issues for post-loopable actions")
            else:
                logger.warning(f"⚠️  '{action_name}' not found in action_outputs or has wrong structure")

        # Run post-loopable actions
        for action in post_loopable_actions:
            logger.info(f"🔄 Running post-loopable action: '{action.name}'")
            action_output, was_cached = self._run_action_with_status(action)
            self.action_outputs[action.name] = action_output
            logger.info(f"  ✅ Completed {action.name}")

        logger.info("✅ All post-loopable actions completed")
