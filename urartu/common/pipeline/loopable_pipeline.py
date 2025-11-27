"""
LoopablePipeline class for pipelines that support loopable actions.

This class extends Pipeline with functionality to run actions multiple times
with different parameter values (e.g., different hyperparameters, configurations, or variants).
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from .pipeline import Pipeline
from .pipeline_action import PipelineAction
from .status import PipelineStatusDisplay
from .types import ActionOutput

logger = logging.getLogger(__name__)


class LoopablePipeline(Pipeline):
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

    def run(self):
        """Execute the pipeline by running all actions in sequence, with support for loopable actions."""
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
            self._run_loopable_actions(pre_loopable_actions, action_statuses, debug_mode, loopable_idx)

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
