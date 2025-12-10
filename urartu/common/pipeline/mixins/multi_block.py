"""
Multi-block loopable pipeline mixin.

This mixin provides functionality for pipelines that support multiple sequential
loopable blocks, where each block can iterate over different variables.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger

from ..pipeline_action import PipelineAction

logger = get_logger(__name__)


class MultiBlockLoopableMixin:
    """
    Mixin class providing multi-block loopable pipeline support.
    
    This mixin enables pipelines to have multiple sequential loopable blocks,
    where each block can iterate over different variables (e.g., revisions vs model names).
    """
    
    def _process_loopable_block(self, loopable_block: DictConfig, block_idx: int) -> Dict[str, Any]:
        """
        Process a single loopable actions block and return block info.
        
        Args:
            loopable_block: The loopable_actions block configuration
            block_idx: Index of this block
            
        Returns:
            Dictionary with block info: loop_configs, loop_iterations, loopable_actions, block_idx
        """
        # Extract loop_configs
        loop_configs = {}
        if "loop_configs" in loopable_block:
            loop_configs = OmegaConf.to_container(loopable_block.loop_configs, resolve=False)
            logger.debug(f"Loop configs for block {block_idx + 1}: {loop_configs}")
        
        # Extract loop_iterations
        loop_iterations = []
        if "loop_iterations" in loopable_block:
            loop_iterations_cfg = loopable_block.loop_iterations
            
            # Check if this is a dynamic specification
            if isinstance(loop_iterations_cfg, DictConfig) and self._is_dynamic_loop_iterations(loop_iterations_cfg):
                loop_iterations = self._load_dynamic_loop_iterations(loop_iterations_cfg)
                if loop_iterations:
                    logger.info(f"Generated {len(loop_iterations)} loop iterations dynamically for block {block_idx + 1}")
            else:
                # Static loop_iterations
                loop_iterations = OmegaConf.to_container(loop_iterations_cfg, resolve=False)
                if not isinstance(loop_iterations, list):
                    loop_iterations = [loop_iterations]
                logger.info(f"Found {len(loop_iterations)} static loop iterations for block {block_idx + 1}")
        
        # Extract loopable actions
        loopable_actions = []
        if "actions" in loopable_block:
            loopable_actions_list = loopable_block.actions
            logger.debug(f"Found {len(loopable_actions_list)} loopable actions in block {block_idx + 1}")
            for action_idx, loopable_action_cfg in enumerate(loopable_actions_list):
                action_name = None
                if isinstance(loopable_action_cfg, DictConfig):
                    action_name = loopable_action_cfg.get("action_name")
                elif isinstance(loopable_action_cfg, dict):
                    action_name = loopable_action_cfg.get("action_name")
                
                if action_name is None:
                    logger.error(f"Could not extract action_name from config at index {action_idx}!")
                    continue
                
                # Extract action config
                config_overrides = self._extract_action_config(
                    config_obj=loopable_action_cfg,
                    action_name=action_name,
                    block_keys=["loopable_actions", "actions"],
                )
                
                if config_overrides is None:
                    logger.error(f"Failed to extract action config for '{action_name}'!")
                    continue
                
                # Store config for this block (block-specific storage)
                if not hasattr(self, 'loopable_action_configs'):
                    self.loopable_action_configs = {}
                if action_name not in self.loopable_action_configs:
                    self.loopable_action_configs[action_name] = {}
                block_key = f"block_{block_idx}"
                if block_key not in self.loopable_action_configs[action_name]:
                    self.loopable_action_configs[action_name][block_key] = config_overrides
                
                loopable_actions.append(action_name)
                if not hasattr(self, 'loopable_actions') or action_name not in self.loopable_actions:
                    if not hasattr(self, 'loopable_actions'):
                        self.loopable_actions = []
                    self.loopable_actions.append(action_name)
                logger.debug(f"Loaded config for loopable action: {action_name} in block {block_idx + 1}")
        
        return {
            "loop_configs": loop_configs,
            "loop_iterations": loop_iterations,
            "loopable_actions": loopable_actions,
            "block_idx": block_idx,
        }
    
    def _get_block_config_for_action(self, action_name: str, block_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get block-specific config for an action in a multi-block pipeline.
        
        Args:
            action_name: Name of the action
            block_idx: Index of the loopable block
            
        Returns:
            Config overrides for this action in this block, or None if not found
        """
        if not hasattr(self, 'loopable_action_configs') or action_name not in self.loopable_action_configs:
            return None
        
        stored_config = self.loopable_action_configs[action_name]
        # Handle multi-block structure: {block_0: {...}, block_1: {...}}
        if isinstance(stored_config, dict):
            block_key = f"block_{block_idx}"
            if block_key in stored_config:
                return copy.deepcopy(stored_config[block_key])
            else:
                # Fallback: use first available block
                block_keys = [k for k in stored_config.keys() if k.startswith("block_")]
                if block_keys:
                    block_key = sorted(block_keys)[0]
                    logger.warning(f"Block key 'block_{block_idx}' not found for '{action_name}', using {block_key}")
                    return copy.deepcopy(stored_config[block_key])
                else:
                    # Not a block-structured config, return as-is
                    return copy.deepcopy(stored_config)
        else:
            # Not a block-structured config, return as-is
            return copy.deepcopy(stored_config)
    
    def _load_iteration_outputs_from_blocks(self) -> None:
        """
        Load iteration outputs from all blocks (both from disk and cache).
        
        This method loads outputs for all loopable actions across all blocks,
        storing them in self.action_outputs[action_name][iteration_id] structure.
        """
        if not hasattr(self, 'loopable_blocks') or not self.loopable_blocks:
            logger.warning("No loopable blocks found, nothing to load")
            return
        
        run_dir = Path(self.cfg.get('run_dir', '.'))
        loop_iterations_dir = Path(self.cfg.get('_loop_iterations_dir', run_dir / 'loop_iterations'))
        
        # Get all loopable actions from all blocks
        all_loopable_actions = set()
        for block in self.loopable_blocks:
            all_loopable_actions.update(block['loopable_actions'])
        
        # Initialize action_outputs structure for all loopable actions
        for action_name in all_loopable_actions:
            if action_name not in self.action_outputs:
                self.action_outputs[action_name] = {}
        
        logger.info(f"Loading outputs from all {len(self.loopable_blocks)} blocks...")
        logger.info(f"   Loop iterations directory: {loop_iterations_dir}")
        logger.info(f"   Loopable actions: {list(all_loopable_actions)}")
        
        # Load iterations from all blocks
        if loop_iterations_dir.exists():
            block_dirs = sorted([d for d in loop_iterations_dir.iterdir() if d.is_dir() and d.name.startswith('block_')])
            
            for block_dir in block_dirs:
                try:
                    block_idx = int(block_dir.name.split('_')[1]) if block_dir.name.split('_')[1].isdigit() else 0
                    if block_idx >= len(self.loopable_blocks):
                        continue
                    
                    block = self.loopable_blocks[block_idx]
                    logger.info(f"Processing block {block_idx} from disk ({len(list(block_dir.glob('*.yaml')))} YAML files)...")
                    
                    # Set up this block's context for cache checking
                    self.loop_configs = block["loop_configs"]
                    self.loopable_actions = block["loopable_actions"]
                    if hasattr(self, 'current_block_idx'):
                        self.current_block_idx = block_idx
                    
                    # Update config injector
                    if self.config_injector is not None:
                        from urartu.common.pipeline.config import ConfigInjector
                        self.config_injector = ConfigInjector(
                            action_outputs=self.action_outputs,
                            loopable_actions=self.loopable_actions,
                            loop_configs=self.loop_configs,
                        )
                    
                    # Load iteration configs from YAML files
                    iteration_configs = sorted(block_dir.glob("*.yaml"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
                    
                    for iteration_config_path in iteration_configs:
                        if not iteration_config_path.exists():
                            continue
                        
                        iteration_cfg = OmegaConf.load(iteration_config_path)
                        loop_context = OmegaConf.to_container(iteration_cfg, resolve=True)
                        iteration_idx = int(iteration_config_path.stem) if iteration_config_path.stem.isdigit() else 0
                        
                        iteration_id = self._get_iteration_id(loop_context)
                        if not iteration_id:
                            iteration_id = f"block_{block_idx}_iteration_{iteration_idx + 1}"
                        
                        # Load outputs for each loopable action in this iteration
                        for action_name in block['loopable_actions']:
                            # Check if already loaded
                            if action_name in self.action_outputs and isinstance(self.action_outputs[action_name], dict) and iteration_id in self.action_outputs[action_name]:
                                logger.debug(f"{action_name}[{iteration_id}] already loaded, skipping")
                                continue
                            
                            # Try loading from disk first
                            loaded = self._load_single_iteration_output(
                                action_name=action_name,
                                iteration_id=iteration_id,
                                iteration_idx=iteration_idx,
                                block_idx=block_idx,
                                loop_context=loop_context,
                                run_dir=run_dir,
                            )
                            
                            if not loaded:
                                logger.warning(f"Could not load output for {action_name}[{iteration_id}] from disk or cache")
                except Exception as e:
                    logger.warning(f"Error processing block {block_dir.name}: {e}")
                    continue
    
    def _run_block_iterations(self, block_idx: int, debug_mode: bool = False) -> List[str]:
        """
        Run iterations for a specific block.
        
        Args:
            block_idx: Index of the block to run
            debug_mode: Whether to run in debug mode
            
        Returns:
            List of job IDs if parallel processing was used, empty list otherwise
        """
        if block_idx >= len(self.loopable_blocks):
            logger.error(f"Block index {block_idx} out of range (have {len(self.loopable_blocks)} blocks)")
            return []
        
        block = self.loopable_blocks[block_idx]
        logger.info(f"Running loopable block {block_idx + 1}/{len(self.loopable_blocks)}")
        logger.info(f"   {len(block['loopable_actions'])} actions × {len(block['loop_iterations'])} iterations")
        
        # Set up this block's context
        self.loop_configs = block["loop_configs"]
        self.loop_iterations = block["loop_iterations"]
        self.loopable_actions = block["loopable_actions"]
        if hasattr(self, 'current_block_idx'):
            self.current_block_idx = block_idx
        
        # Update config injector
        if self.config_injector is not None:
            from urartu.common.pipeline.config import ConfigInjector
            self.config_injector = ConfigInjector(
                action_outputs=self.action_outputs,
                loopable_actions=self.loopable_actions,
                loop_configs=self.loop_configs,
            )
        
        # Check if parallel processing should be used
        if self._should_use_parallel_processing():
            logger.info(f"Using parallel processing for block {block_idx + 1}")
            # Store block_idx in config for iteration tasks
            self.cfg["_block_idx"] = block_idx
            # Submit this block's iterations
            block_job_ids = self._submit_block_iterations_only()
            return block_job_ids
        else:
            # Sequential execution
            for iteration_idx, loop_context in enumerate(self.loop_iterations):
                iteration_num = iteration_idx + 1
                logger.info(f"Block {block_idx + 1}, Iteration {iteration_num}/{len(self.loop_iterations)} with parameters: {loop_context}")
                
                self.current_loop_context = loop_context
                iteration_id = self._get_iteration_id(loop_context)
                if not iteration_id:
                    iteration_id = f"block_{block_idx}_iteration_{iteration_num}"
                
                # Run each loopable action for this iteration
                for action_name in self.loopable_actions:
                    logger.info(f"Running loopable action: '{action_name}' (block {block_idx + 1}, iteration {iteration_num})")
                    
                    # Find or create the action
                    action = None
                    for a in self.actions:
                        if a.name == action_name:
                            action = a
                            break
                    
                    if action is None:
                        # Create action from loopable_action_configs
                        config_overrides = self._get_block_config_for_action(action_name, block_idx)
                        if config_overrides is not None:
                            action = PipelineAction(
                                name=action_name,
                                action_name=action_name,
                                config_overrides=config_overrides,
                                outputs_to_track=[],
                            )
                    
                    if action is None:
                        logger.error(f"Action '{action_name}' not found for block {block_idx + 1}")
                        continue
                    
                    # Run action with loop context
                    action_output, was_cached = self._run_action_with_status(
                        action,
                        loop_context=loop_context,
                        iteration_name=iteration_id,
                    )
                    
                    # Store output in nested structure
                    if action_name not in self.action_outputs:
                        self.action_outputs[action_name] = {}
                    self.action_outputs[action_name][iteration_id] = action_output
                    
                    # Also store with iteration-specific key for backward compatibility
                    iteration_action_name = f"{action_name}_{iteration_id}"
                    self.action_outputs[iteration_action_name] = action_output
                    
                    # Save output to disk for dependency job to load (same as base class)
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
            
            return []

