"""
Pipeline Parser

This module provides functionality to parse pipeline configurations into a list of stages.
A stage can be either a single action or a loopable block.
"""

import logging
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from .pipeline_stage import ActionStage, LoopableStage, PipelineStage

logger = logging.getLogger(__name__)


class PipelineParser:
    """
    Parses pipeline configurations into a list of stages.
    
    The parser reads the pipeline.actions list and creates a stage for each entry:
    - Regular actions become ActionStage objects
    - loopable_actions blocks become LoopableStage objects
    """
    
    def __init__(self, pipeline_config: DictConfig, dynamic_loader: Optional[Any] = None):
        """
        Initialize the parser.
        
        Args:
            pipeline_config: The pipeline configuration (cfg.pipeline)
            dynamic_loader: Optional object with _is_dynamic_loop_iterations() and _load_dynamic_loop_iterations()
        """
        self.pipeline_config = pipeline_config
        self.dynamic_loader = dynamic_loader
    
    def parse(self) -> List[PipelineStage]:
        """
        Parse the pipeline configuration into a list of stages.
        
        Returns:
            List of PipelineStage objects (ActionStage or LoopableStage)
        """
        if "actions" not in self.pipeline_config:
            logger.warning("No actions found in pipeline configuration")
            return []
        
        actions_list = self.pipeline_config.actions
        stages = []
        
        for idx, action_cfg in enumerate(actions_list):
            if "loopable_actions" in action_cfg:
                # This is a loopable block
                loopable_stage = self._parse_loopable_block(action_cfg.loopable_actions, len(stages))
                stages.append(loopable_stage)
            else:
                # This is a regular action
                action_stage = self._parse_action(action_cfg, len(stages))
                stages.append(action_stage)
        
        logger.info(f"Parsed pipeline into {len(stages)} stages:")
        for stage in stages:
            logger.info(f"  {stage}")
        
        return stages
    
    def _parse_action(self, action_cfg: DictConfig, stage_idx: int) -> ActionStage:
        """
        Parse a regular action into an ActionStage.
        
        Args:
            action_cfg: Action configuration
            stage_idx: Index of this stage
            
        Returns:
            ActionStage object
        """
        action_name = action_cfg.get("action_name", "unknown")
        
        # Extract config overrides (everything except action_name and outputs_to_track)
        config_overrides = {
            k: v for k, v in action_cfg.items()
            if k not in ["action_name", "outputs_to_track", "loopable_actions", "actions"]
        }
        
        outputs_to_track = action_cfg.get("outputs_to_track", [])
        
        return ActionStage(
            stage_idx=stage_idx,
            action_name=action_name,
            action_config=config_overrides,
            outputs_to_track=outputs_to_track,
        )
    
    def _parse_loopable_block(self, loopable_block: DictConfig, stage_idx: int) -> LoopableStage:
        """
        Parse a loopable_actions block into a LoopableStage.
        
        Args:
            loopable_block: The loopable_actions configuration
            stage_idx: Index of this stage
            
        Returns:
            LoopableStage object
        """
        # Extract loop_configs
        loop_configs = {}
        if "loop_configs" in loopable_block:
            loop_configs = OmegaConf.to_container(loopable_block.loop_configs, resolve=False)
        
        # Extract loop_iterations
        loop_iterations = []
        if "loop_iterations" in loopable_block:
            loop_iterations_cfg = loopable_block.loop_iterations
            
            # Check if this is a dynamic specification (requires dynamic_loader)
            if self.dynamic_loader and self._is_dynamic_loop_iterations(loop_iterations_cfg):
                loop_iterations = self._load_dynamic_loop_iterations(loop_iterations_cfg)
                if loop_iterations:
                    logger.info(f"Generated {len(loop_iterations)} loop iterations dynamically for stage {stage_idx}")
            else:
                # Static loop_iterations
                loop_iterations = OmegaConf.to_container(loop_iterations_cfg, resolve=False)
                if not isinstance(loop_iterations, list):
                    loop_iterations = [loop_iterations]
                logger.info(f"Found {len(loop_iterations)} static loop iterations for stage {stage_idx}")
        
        # Extract loopable actions and their configs
        loopable_actions = []
        action_configs = {}
        
        if "actions" in loopable_block:
            for loopable_action_cfg in loopable_block.actions:
                action_name = loopable_action_cfg.get("action_name")
                if not action_name:
                    logger.error("Could not extract action_name from loopable action config")
                    continue
                
                # Extract config overrides
                config_overrides = {
                    k: v for k, v in loopable_action_cfg.items()
                    if k not in ["action_name", "outputs_to_track", "loopable_actions", "actions"]
                }
                
                loopable_actions.append(action_name)
                action_configs[action_name] = config_overrides
        
        return LoopableStage(
            stage_idx=stage_idx,
            loop_configs=loop_configs,
            loop_iterations=loop_iterations,
            loopable_actions=loopable_actions,
            action_configs=action_configs,
        )
    
    def _is_dynamic_loop_iterations(self, loop_iterations_cfg: DictConfig) -> bool:
        """Check if loop_iterations uses dynamic specification."""
        if self.dynamic_loader and hasattr(self.dynamic_loader, '_is_dynamic_loop_iterations'):
            return self.dynamic_loader._is_dynamic_loop_iterations(loop_iterations_cfg)
        # Default: check for common dynamic patterns
        return isinstance(loop_iterations_cfg, DictConfig) and "dynamic_revisions" in loop_iterations_cfg
    
    def _load_dynamic_loop_iterations(self, loop_iterations_cfg: DictConfig) -> List[Dict[str, Any]]:
        """Load dynamic loop iterations using the dynamic_loader."""
        if self.dynamic_loader and hasattr(self.dynamic_loader, '_load_dynamic_loop_iterations'):
            return self.dynamic_loader._load_dynamic_loop_iterations(loop_iterations_cfg)
        logger.warning("Dynamic loop iterations requested but no dynamic_loader provided")
        return []

