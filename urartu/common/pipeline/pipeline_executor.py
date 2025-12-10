"""
Pipeline Executor

This module provides the PipelineExecutor class that executes pipeline stages.
It handles both sequential and parallel execution, supporting arbitrary numbers of 
loopable blocks at any position in the pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from .pipeline_stage import ActionStage, LoopableStage, PipelineStage
from .pipeline_action import PipelineAction
from .types import ActionOutput

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Executes a pipeline as a sequence of stages.
    
    This executor supports:
    - Sequential execution of actions
    - Parallel execution of loopable blocks via job arrays
    - Dependency management between stages
    - Resume functionality (continuing from a specific stage)
    """
    
    def __init__(self, pipeline: Any, stages: List[PipelineStage]):
        """
        Initialize the executor.
        
        Args:
            pipeline: The pipeline instance (provides methods like _run_action_with_status, etc.)
            stages: List of PipelineStage objects to execute
        """
        self.pipeline = pipeline
        self.stages = stages
        self.cfg = pipeline.cfg
    
    def execute(self, resume_at_stage: Optional[int] = None) -> None:
        """
        Execute the pipeline stages.
        
        Args:
            resume_at_stage: If provided, resume execution from this stage index
        """
        start_stage = resume_at_stage if resume_at_stage is not None else 0
        
        if start_stage > 0:
            logger.info(f"Resuming pipeline execution from stage {start_stage}")
        
        # Execute stages sequentially
        for stage_idx in range(start_stage, len(self.stages)):
            stage = self.stages[stage_idx]
            logger.info("=" * 80)
            logger.info(f"Executing Stage {stage_idx + 1}/{len(self.stages)}: {stage}")
            logger.info("=" * 80)
            
            if isinstance(stage, ActionStage):
                self._execute_action_stage(stage)
            elif isinstance(stage, LoopableStage):
                # Check if we should use parallel processing
                if self._should_use_parallel_processing():
                    # Submit the loopable block and set up dependency job to resume
                    self._submit_loopable_stage(stage, resume_at_stage=stage_idx + 1)
                    # Exit here - the dependency job will resume from the next stage
                    logger.info("Submitted loopable stage and dependency job. Pipeline will resume in dependency job.")
                    return
                else:
                    # Sequential execution
                    self._execute_loopable_stage(stage)
            else:
                logger.error(f"Unknown stage type: {type(stage)}")
        
        logger.info("=" * 80)
        logger.info("Pipeline execution completed successfully!")
        logger.info("=" * 80)
    
    def _execute_action_stage(self, stage: ActionStage) -> None:
        """
        Execute a single action stage.
        
        Args:
            stage: The ActionStage to execute
        """
        # Create a PipelineAction
        action = PipelineAction(
            name=stage.action_name,
            action_name=stage.action_name,
            config_overrides=stage.action_config,
            outputs_to_track=stage.outputs_to_track,
        )
        
        # Run the action
        action_output, was_cached = self.pipeline._run_action_with_status(action)
        self.pipeline.action_outputs[action.name] = action_output
        
        if was_cached:
            logger.info(f"Action '{stage.action_name}' loaded from cache")
        else:
            logger.info(f"Action '{stage.action_name}' completed successfully")
    
    def _execute_loopable_stage(self, stage: LoopableStage) -> None:
        """
        Execute a loopable stage sequentially (no parallel processing).
        
        Args:
            stage: The LoopableStage to execute
        """
        logger.info(f"Executing loopable stage with {stage.get_num_actions()} actions × {stage.get_num_iterations()} iterations")
        
        # Set up loop context for this stage
        self.pipeline.loop_configs = stage.loop_configs
        self.pipeline.loop_iterations = stage.loop_iterations
        self.pipeline.loopable_actions = stage.loopable_actions
        
        # Update config injector
        if self.pipeline.config_injector is not None:
            from .config import ConfigInjector
            self.pipeline.config_injector = ConfigInjector(
                action_outputs=self.pipeline.action_outputs,
                loopable_actions=stage.loopable_actions,
                loop_configs=stage.loop_configs,
            )
        
        # Execute each iteration
        for iteration_idx, loop_context in enumerate(stage.loop_iterations):
            iteration_num = iteration_idx + 1
            logger.info(f"Iteration {iteration_num}/{stage.get_num_iterations()} with context: {loop_context}")
            
            self.pipeline.current_loop_context = loop_context
            iteration_id = self.pipeline._get_iteration_id(loop_context)
            if not iteration_id:
                iteration_id = f"stage_{stage.stage_idx}_iteration_{iteration_num}"
            
            # Execute each action in this iteration
            for action_name in stage.loopable_actions:
                logger.info(f"Running loopable action: '{action_name}' (iteration {iteration_num})")
                
                # Get action config
                action_config = stage.action_configs.get(action_name, {})
                action = PipelineAction(
                    name=action_name,
                    action_name=action_name,
                    config_overrides=action_config,
                    outputs_to_track=[],
                )
                
                # Run the action with loop context
                action_output, was_cached = self.pipeline._run_action_with_status(
                    action,
                    loop_context=loop_context,
                    iteration_name=iteration_id,
                )
                
                # Store output in nested structure
                if action_name not in self.pipeline.action_outputs:
                    self.pipeline.action_outputs[action_name] = {}
                self.pipeline.action_outputs[action_name][iteration_id] = action_output
                
                # Also store with iteration-specific key for backward compatibility
                iteration_action_name = f"{action_name}_{iteration_id}"
                self.pipeline.action_outputs[iteration_action_name] = action_output
                
                # Save output to disk for potential dependency jobs
                self._save_iteration_output(action_output, action_name, iteration_id, iteration_idx, stage.stage_idx)
                
                logger.info(f"Completed {action_name} (iteration {iteration_num})")
        
        logger.info(f"Completed loopable stage {stage.stage_idx}")
    
    def _submit_loopable_stage(self, stage: LoopableStage, resume_at_stage: int) -> None:
        """
        Submit a loopable stage for parallel execution via job array.
        
        Args:
            stage: The LoopableStage to submit
            resume_at_stage: The stage index to resume at after this loopable stage completes
        """
        logger.info(f"Submitting loopable stage {stage.stage_idx} for parallel execution")
        logger.info(f"  {stage.get_num_actions()} actions × {stage.get_num_iterations()} iterations")
        
        # Set up loop context for this stage
        self.pipeline.loop_configs = stage.loop_configs
        self.pipeline.loop_iterations = stage.loop_iterations
        self.pipeline.loopable_actions = stage.loopable_actions
        
        # Store stage info in config for iteration tasks
        self.cfg["_stage_idx"] = stage.stage_idx
        self.cfg["_loopable_stage"] = True
        
        # Update config injector
        if self.pipeline.config_injector is not None:
            from .config import ConfigInjector
            self.pipeline.config_injector = ConfigInjector(
                action_outputs=self.pipeline.action_outputs,
                loopable_actions=stage.loopable_actions,
                loop_configs=stage.loop_configs,
            )
        
        # Check caching for iterations
        uncached_iterations = []
        cached_iterations = []
        
        logger.info(f"Checking cache status for {stage.get_num_iterations()} iterations...")
        for iteration_idx, loop_context in enumerate(stage.loop_iterations):
            try:
                # Store action configs in loopable_action_configs for cache checking
                if not hasattr(self.pipeline, 'loopable_action_configs'):
                    self.pipeline.loopable_action_configs = {}
                for action_name in stage.loopable_actions:
                    if action_name not in self.pipeline.loopable_action_configs:
                        self.pipeline.loopable_action_configs[action_name] = stage.action_configs.get(action_name, {})
                
                is_cached = self.pipeline._check_iteration_cache(iteration_idx, loop_context)
                if is_cached:
                    cached_iterations.append(iteration_idx)
                    logger.info(f"Iteration {iteration_idx} is cached, skipping submission")
                else:
                    uncached_iterations.append(iteration_idx)
                    logger.info(f"Iteration {iteration_idx} is not cached, will submit")
            except Exception as e:
                logger.error(f"Error checking cache for iteration {iteration_idx}: {e}")
                uncached_iterations.append(iteration_idx)
        
        logger.info(f"Cache check summary: {len(cached_iterations)} cached, {len(uncached_iterations)} uncached")
        
        # Create iteration config files
        loop_iterations_dir = self._create_iteration_configs(stage)
        
        # Submit jobs for uncached iterations
        iteration_job_ids = []
        if uncached_iterations:
            iteration_job_ids = self._submit_iteration_jobs(stage, uncached_iterations, loop_iterations_dir)
            logger.info(f"Submitted {len(iteration_job_ids)} iteration jobs")
        else:
            logger.info("All iterations are cached, no jobs to submit")
        
        # Submit dependency job to resume at next stage
        if resume_at_stage < len(self.stages):
            self._submit_resume_job(iteration_job_ids, resume_at_stage, loop_iterations_dir)
        else:
            logger.info("No more stages to execute after this loopable stage")
    
    def _save_iteration_output(
        self, 
        action_output: ActionOutput, 
        action_name: str, 
        iteration_id: str, 
        iteration_idx: int,
        stage_idx: int
    ) -> None:
        """
        Save iteration output to disk for dependency jobs to load.
        
        Args:
            action_output: The ActionOutput to save
            action_name: Name of the action
            iteration_id: Unique iteration identifier
            iteration_idx: Index of the iteration
            stage_idx: Index of the stage
        """
        run_dir = Path(self.cfg.get('run_dir', '.'))
        array_task_dir = run_dir / "array_tasks" / f"stage_{stage_idx}" / f"task_{iteration_idx}"
        action_output_dir = array_task_dir / action_name
        action_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = action_output_dir / "output.pkl"
        
        try:
            import pickle
            with open(output_file, 'wb') as f:
                pickle.dump(action_output, f)
            logger.debug(f"Saved output for {action_name}[{iteration_id}] to {output_file}")
        except Exception as e:
            logger.warning(f"Could not save output to {output_file}: {e}")
    
    def _create_iteration_configs(self, stage: LoopableStage) -> Path:
        """
        Create iteration config files for a loopable stage.
        
        Args:
            stage: The LoopableStage to create configs for
            
        Returns:
            Path to the loop_iterations directory
        """
        run_dir = Path(self.cfg.get("run_dir", "."))
        loop_iterations_dir = run_dir / "loop_iterations"
        stage_dir = loop_iterations_dir / f"stage_{stage.stage_idx}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating {stage.get_num_iterations()} iteration config files in {stage_dir}")
        
        for iteration_idx, loop_context in enumerate(stage.loop_iterations):
            iteration_config_path = stage_dir / f"{iteration_idx}.yaml"
            iteration_cfg = OmegaConf.create(loop_context)
            with open(iteration_config_path, "w") as f:
                OmegaConf.save(iteration_cfg, f)
            logger.debug(f"Created iteration config {iteration_idx} for stage {stage.stage_idx}: {iteration_config_path}")
        
        return loop_iterations_dir
    
    def _submit_iteration_jobs(
        self, 
        stage: LoopableStage, 
        uncached_iterations: List[int],
        loop_iterations_dir: Path
    ) -> List[str]:
        """
        Submit iteration jobs for a loopable stage.
        
        Args:
            stage: The LoopableStage to submit jobs for
            uncached_iterations: List of iteration indices to submit
            loop_iterations_dir: Path to loop iterations directory
            
        Returns:
            List of job IDs
        """
        # Delegate to the pipeline's job submission logic
        # The pipeline should have _submit_iteration_jobs method from LoopableJobsMixin
        if hasattr(self.pipeline, '_submit_iteration_jobs'):
            return self.pipeline._submit_iteration_jobs(uncached_iterations, loop_iterations_dir)
        else:
            logger.error("Pipeline does not have _submit_iteration_jobs method")
            return []
    
    def _submit_resume_job(
        self, 
        dependency_job_ids: List[str], 
        resume_at_stage: int,
        loop_iterations_dir: Path
    ) -> None:
        """
        Submit a dependency job that resumes the pipeline at a specific stage.
        
        Args:
            dependency_job_ids: List of job IDs to wait for
            resume_at_stage: Stage index to resume at
            loop_iterations_dir: Path to loop iterations directory
        """
        logger.info("=" * 80)
        logger.info(f"Submitting dependency job to resume at stage {resume_at_stage}")
        logger.info(f"  Depends on {len(dependency_job_ids)} iteration jobs")
        logger.info("=" * 80)
        
        # Create dependency configuration
        dep_cfg = OmegaConf.create(self.cfg)
        dep_cfg['_resume_at_stage'] = resume_at_stage
        dep_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
        dep_cfg['_iteration_job_ids'] = dependency_job_ids
        
        # Remove flags that shouldn't be in the dependency job
        if '_submit_array_only' in dep_cfg:
            del dep_cfg['_submit_array_only']
        if '_is_iteration_task' in dep_cfg:
            del dep_cfg['_is_iteration_task']
        if '_stage_idx' in dep_cfg:
            del dep_cfg['_stage_idx']
        if '_loopable_stage' in dep_cfg:
            del dep_cfg['_loopable_stage']
        
        # Submit the dependency job
        if dependency_job_ids:
            # Create dependency string (wait for all iteration jobs)
            if len(dependency_job_ids) == 1:
                dependency_str = f"afterok:{dependency_job_ids[0]}"
            else:
                dependency_str = "afterok:" + ":".join(dependency_job_ids)
        else:
            # No jobs to wait for (all cached), but still submit dependency job to continue
            dependency_str = None
        
        from urartu.utils.execution.launcher import launch_on_slurm
        
        # Get pipeline name from config
        pipeline_name = self.cfg.get('pipeline_name', 'pipeline')
        
        dep_job = launch_on_slurm(
            module=str(Path.cwd()),
            action_name=pipeline_name,
            cfg=dep_cfg,
            aim_run=self.pipeline.aim_run,
            array_size=None,
            dependency=dependency_str,
        )
        
        logger.info(f"Submitted dependency job {dep_job.job_id} to resume at stage {resume_at_stage}")
    
    def _execute_loopable_stage(self, stage: LoopableStage) -> None:
        """
        Execute a loopable stage sequentially (no parallel processing).
        
        Args:
            stage: The LoopableStage to execute
        """
        logger.info(f"Executing loopable stage with {stage.get_num_actions()} actions × {stage.get_num_iterations()} iterations")
        
        # Set up loop context
        self.pipeline.loop_configs = stage.loop_configs
        self.pipeline.loop_iterations = stage.loop_iterations
        self.pipeline.loopable_actions = stage.loopable_actions
        
        # Store action configs in loopable_action_configs for the pipeline to use
        if not hasattr(self.pipeline, 'loopable_action_configs'):
            self.pipeline.loopable_action_configs = {}
        for action_name in stage.loopable_actions:
            self.pipeline.loopable_action_configs[action_name] = stage.action_configs.get(action_name, {})
        
        # Update config injector
        if self.pipeline.config_injector is not None:
            from .config import ConfigInjector
            self.pipeline.config_injector = ConfigInjector(
                action_outputs=self.pipeline.action_outputs,
                loopable_actions=stage.loopable_actions,
                loop_configs=stage.loop_configs,
            )
        
        # Execute each iteration
        for iteration_idx, loop_context in enumerate(stage.loop_iterations):
            iteration_num = iteration_idx + 1
            logger.info(f"Iteration {iteration_num}/{stage.get_num_iterations()} with context: {loop_context}")
            
            self.pipeline.current_loop_context = loop_context
            iteration_id = self.pipeline._get_iteration_id(loop_context)
            if not iteration_id:
                iteration_id = f"stage_{stage.stage_idx}_iteration_{iteration_num}"
            
            # Execute each action in this iteration
            for action_name in stage.loopable_actions:
                logger.info(f"Running loopable action: '{action_name}' (iteration {iteration_num})")
                
                # Get action config
                action_config = stage.action_configs.get(action_name, {})
                action = PipelineAction(
                    name=action_name,
                    action_name=action_name,
                    config_overrides=action_config,
                    outputs_to_track=[],
                )
                
                # Run the action with loop context
                action_output, was_cached = self.pipeline._run_action_with_status(
                    action,
                    loop_context=loop_context,
                    iteration_name=iteration_id,
                )
                
                # Store output in nested structure
                if action_name not in self.pipeline.action_outputs:
                    self.pipeline.action_outputs[action_name] = {}
                self.pipeline.action_outputs[action_name][iteration_id] = action_output
                
                # Also store with iteration-specific key for backward compatibility
                iteration_action_name = f"{action_name}_{iteration_id}"
                self.pipeline.action_outputs[iteration_action_name] = action_output
                
                # Save output to disk for potential dependency jobs
                self._save_iteration_output(action_output, action_name, iteration_id, iteration_idx, stage.stage_idx)
                
                logger.info(f"Completed {action_name} (iteration {iteration_num})")
        
        logger.info(f"Completed loopable stage {stage.stage_idx}")
    
    def _should_use_parallel_processing(self) -> bool:
        """Check if parallel processing should be used."""
        if hasattr(self.pipeline, '_should_use_parallel_processing'):
            return self.pipeline._should_use_parallel_processing()
        return False
    
    def load_stage_outputs(self, stage: LoopableStage) -> None:
        """
        Load outputs from all iterations of a completed loopable stage.
        
        This is called when resuming the pipeline after a loopable stage has completed.
        
        Args:
            stage: The LoopableStage to load outputs for
        """
        logger.info(f"Loading outputs from completed loopable stage {stage.stage_idx}")
        
        run_dir = Path(self.cfg.get('run_dir', '.'))
        loop_iterations_dir = Path(self.cfg.get('_loop_iterations_dir', run_dir / 'loop_iterations'))
        stage_dir = loop_iterations_dir / f"stage_{stage.stage_idx}"
        
        if not stage_dir.exists():
            logger.warning(f"Stage directory not found: {stage_dir}")
            return
        
        # Initialize action_outputs structure for all actions in this stage
        for action_name in stage.loopable_actions:
            if action_name not in self.pipeline.action_outputs:
                self.pipeline.action_outputs[action_name] = {}
        
        # Load iteration configs
        iteration_configs = sorted(stage_dir.glob("*.yaml"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        
        for iteration_config_path in iteration_configs:
            if not iteration_config_path.exists():
                continue
            
            iteration_cfg = OmegaConf.load(iteration_config_path)
            loop_context = OmegaConf.to_container(iteration_cfg, resolve=True)
            iteration_idx = int(iteration_config_path.stem) if iteration_config_path.stem.isdigit() else 0
            
            iteration_id = self.pipeline._get_iteration_id(loop_context)
            if not iteration_id:
                iteration_id = f"stage_{stage.stage_idx}_iteration_{iteration_idx + 1}"
            
            # Load outputs for each action in this iteration
            for action_name in stage.loopable_actions:
                # Try loading from disk
                loaded = self._load_iteration_output_from_disk(
                    action_name, iteration_id, iteration_idx, stage.stage_idx, run_dir
                )
                
                # If not found on disk, try cache
                if not loaded:
                    loaded = self._load_iteration_output_from_cache(
                        action_name, iteration_id, iteration_idx, loop_context, stage
                    )
                
                if not loaded:
                    logger.warning(f"Could not load output for {action_name}[{iteration_id}]")
    
    def _load_iteration_output_from_disk(
        self,
        action_name: str,
        iteration_id: str,
        iteration_idx: int,
        stage_idx: int,
        run_dir: Path
    ) -> bool:
        """Load iteration output from disk."""
        possible_dirs = [
            run_dir / "array_tasks" / f"stage_{stage_idx}" / f"task_{iteration_idx}" / action_name,
            run_dir / "array_tasks" / f"task_{iteration_idx}" / action_name,  # Backward compatibility
        ]
        
        for action_output_dir in possible_dirs:
            output_file = action_output_dir / "output.pkl"
            if output_file.exists():
                try:
                    import pickle
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
                        
                        if action_name not in self.pipeline.action_outputs:
                            self.pipeline.action_outputs[action_name] = {}
                        self.pipeline.action_outputs[action_name][iteration_id] = action_output
                        logger.info(f"Loaded output for {action_name}[{iteration_id}] from {output_file}")
                        return True
                except Exception as e:
                    logger.debug(f"Error loading from {output_file}: {e}")
                    continue
        
        return False
    
    def _load_iteration_output_from_cache(
        self,
        action_name: str,
        iteration_id: str,
        iteration_idx: int,
        loop_context: Dict[str, Any],
        stage: LoopableStage
    ) -> bool:
        """Load iteration output from cache."""
        try:
            # Get action config
            action_config = stage.action_configs.get(action_name, {})
            
            # Use pipeline's _load_single_iteration_output if available
            if hasattr(self.pipeline, '_load_single_iteration_output'):
                run_dir = Path(self.cfg.get('run_dir', '.'))
                return self.pipeline._load_single_iteration_output(
                    action_name=action_name,
                    iteration_id=iteration_id,
                    iteration_idx=iteration_idx,
                    block_idx=None,  # Using stage_idx instead
                    loop_context=loop_context,
                    run_dir=run_dir,
                )
            else:
                logger.debug(f"_load_single_iteration_output not available, skipping cache load for {action_name}[{iteration_id}]")
                return False
        except Exception as e:
            logger.debug(f"Error loading from cache for {action_name}[{iteration_id}]: {e}")
            return False

