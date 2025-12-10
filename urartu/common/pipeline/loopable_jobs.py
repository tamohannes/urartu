"""
Job submission functionality for LoopablePipeline.

This module handles SLURM job submission, polling, and dependency management for loopable pipelines.
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import getpass
from omegaconf import DictConfig, OmegaConf

from urartu.utils.logging import get_logger

logger = get_logger(__name__)


class LoopableJobsMixin:
    """
    Mixin class providing job submission functionality for LoopablePipeline.

    This mixin handles:
    - Polling-based job submission
    - SLURM queue management
    - Dependency job submission
    - Job tracking and error handling
    """

    def _submit_array_only_internal(self):
        """
        Internal method to submit array jobs and exit (called from run() when _submit_array_only is set).

        This method:
        1. Initializes the pipeline to get loop_iterations
        2. Loads pre-loopable actions from cache
        3. Checks caching for each iteration
        4. Only submits array tasks for uncached iterations (or if force_rerun is set)
        5. Submits dependency job for post-loopable actions
        6. Exits immediately without running the pipeline
        """
        logger.info("🚀 Submission-only job: Initializing pipeline to get loop iterations...")

        # Initialize pipeline to get loop_iterations
        if not self._initialized:
            self.initialize()

        # Check if pipeline has loopable actions
        if not hasattr(self, 'loopable_actions') or len(self.loopable_actions) == 0:
            raise ValueError("Pipeline has no loopable actions, but _submit_array_only was called")

        if not hasattr(self, 'loop_iterations') or len(self.loop_iterations) == 0:
            raise ValueError("Pipeline has no loop iterations, but _submit_array_only was called")

        logger.info(f"📋 Found {len(self.loop_iterations)} loop iterations")

        # Load pre-loopable actions from cache (provided by LoopableCacheMixin)
        # This ensures that _inject_action_outputs has the dependencies it needs for all iterations
        if hasattr(self, '_load_pre_loopable_actions'):
            self._load_pre_loopable_actions()
        else:
            logger.warning("⚠️  _load_pre_loopable_actions not available - cache checks may fail")

        # Check caching for each iteration and only submit uncached ones
        uncached_iterations = []
        cached_iterations = []

        logger.info(f"🔍 Checking cache status for {len(self.loop_iterations)} iterations...")
        for iteration_idx, loop_context in enumerate(self.loop_iterations):
            # Check if this iteration is cached
            try:
                is_cached = self._check_iteration_cache(iteration_idx, loop_context)
                if is_cached:
                    cached_iterations.append(iteration_idx)
                    logger.info(f"✅ Iteration {iteration_idx} is cached, skipping submission")
                else:
                    uncached_iterations.append(iteration_idx)
                    logger.info(f"❌ Iteration {iteration_idx} is not cached, will submit")
            except Exception as e:
                # If cache check fails, assume not cached and log the error
                logger.error(f"⚠️  Error checking cache for iteration {iteration_idx}: {e}")
                logger.error(f"   Assuming iteration {iteration_idx} is not cached and will submit job")
                uncached_iterations.append(iteration_idx)
                import traceback

                logger.debug(traceback.format_exc())

        # Log summary
        logger.info(
            f"📊 Cache check summary: {len(cached_iterations)} cached, {len(uncached_iterations)} uncached out of {len(self.loop_iterations)} total iterations"
        )
        if len(cached_iterations) > 0:
            logger.info(f"   Cached iterations: {cached_iterations}")
        if len(uncached_iterations) > 0:
            logger.info(f"   Uncached iterations (will submit jobs): {uncached_iterations}")

        if len(uncached_iterations) == 0:
            logger.info("✅ All iterations are cached! No array tasks to submit.")
            # Still need to submit dependency job if there are post-loopable actions
            self._submit_dependency_job_if_needed([])
            return

        # Create iteration config files for all iterations
        loop_iterations_dir = self._create_iteration_configs()

        # Submit individual jobs for each uncached iteration
        iteration_job_ids = self._submit_iteration_jobs(uncached_iterations, loop_iterations_dir)

        # Submit dependency job if needed
        self._submit_dependency_job_if_needed(iteration_job_ids, loop_iterations_dir)

        logger.info("✅ Job submission complete. Exiting submission-only job.")

    def _submit_iteration_jobs(self, uncached_iterations: List[int], loop_iterations_dir: Path) -> List[str]:
        """
        Submit individual jobs for uncached iterations using polling-based approach.

        Args:
            uncached_iterations: List of iteration indices to submit
            loop_iterations_dir: Directory containing iteration configs

        Returns:
            List of job IDs for submitted iterations
        """
        from urartu.utils.execution.launcher import launch_on_slurm

        run_dir = Path(self.cfg.get('run_dir', '.'))
        pipeline_name = self.cfg.get('pipeline_name')
        if not pipeline_name:
            raise ValueError("Config must specify 'pipeline_name' for parallel execution")

        logger.info(f"🚀 Submitting {len(uncached_iterations)} individual jobs (one per uncached iteration)")

        # Polling-based job submission: maintain a steady number of running jobs
        iteration_job_ids = []
        target_running_jobs = 40  # Target number of jobs to keep running
        poll_interval = 30.0  # Check queue status every 30 seconds
        delay_between_jobs = 0.3  # Small delay between individual job submissions

        total_uncached = len(uncached_iterations)
        logger.info(f"📋 Submitting {total_uncached} uncached iterations using polling-based approach")
        logger.info(f"   Target: {target_running_jobs} running jobs, polling every {poll_interval}s")

        # Track which iterations have been submitted
        submitted_iterations = set()
        iteration_to_job_id = {}

        # Initial submission: submit up to target_running_jobs initially
        logger.info(f"🚀 Initial submission: checking queue status and submitting up to {target_running_jobs} jobs...")
        current_running = self._get_current_running_jobs()
        logger.info(f"📊 Current queue status: {current_running} jobs already running/pending")

        # Calculate how many slots are available
        available_slots = max(0, target_running_jobs - current_running)
        initial_submissions = min(available_slots, total_uncached)

        if initial_submissions > 0:
            logger.info(f"📤 Submitting {initial_submissions} initial jobs...")
            for i in range(initial_submissions):
                iteration_idx = uncached_iterations[i]
                try:
                    job_id = self._submit_single_iteration_job(iteration_idx, loop_iterations_dir, pipeline_name, run_dir)
                    iteration_job_ids.append(job_id)
                    submitted_iterations.add(iteration_idx)
                    iteration_to_job_id[iteration_idx] = job_id
                    logger.info(f"  ✅ Submitted job {job_id} for iteration {iteration_idx} ({i+1}/{initial_submissions})")
                    if i < initial_submissions - 1:  # Don't delay after last job
                        time.sleep(delay_between_jobs)
                except Exception as e:
                    error_msg = str(e)
                    if "QOSMaxSubmitJobPerUserLimit" in error_msg or "submit limit" in error_msg.lower():
                        logger.warning(
                            f"⚠️  Hit SLURM submission limit during initial submission (submitted {len(iteration_job_ids)}/{initial_submissions})"
                        )
                        logger.warning(f"   Will continue with polling approach to submit remaining jobs")
                        break
                    else:
                        logger.error(f"❌ Failed to submit job for iteration {iteration_idx}: {e}")
                        raise

        # Polling loop: check every 30 seconds and submit new jobs as slots become available
        next_iteration_idx = len(submitted_iterations)
        while next_iteration_idx < total_uncached:
            logger.info(
                f"⏳ Polling queue status... ({len(submitted_iterations)}/{total_uncached} submitted, {total_uncached - next_iteration_idx} remaining)"
            )
            time.sleep(poll_interval)

            # Check current running jobs
            current_running = self._get_current_running_jobs()
            logger.info(f"📊 Current queue status: {current_running} jobs running/pending (target: {target_running_jobs})")

            # Calculate how many slots are available
            available_slots = max(0, target_running_jobs - current_running)

            if available_slots > 0:
                # Submit new jobs to fill available slots
                submissions_this_round = min(available_slots, total_uncached - next_iteration_idx)
                logger.info(f"📤 Submitting {submissions_this_round} new jobs (filling {available_slots} available slots)...")

                for i in range(submissions_this_round):
                    if next_iteration_idx >= total_uncached:
                        break

                    iteration_idx = uncached_iterations[next_iteration_idx]
                    next_iteration_idx += 1

                    try:
                        job_id = self._submit_single_iteration_job(iteration_idx, loop_iterations_dir, pipeline_name, run_dir)
                        iteration_job_ids.append(job_id)
                        submitted_iterations.add(iteration_idx)
                        iteration_to_job_id[iteration_idx] = job_id
                        logger.info(f"  ✅ Submitted job {job_id} for iteration {iteration_idx} ({len(submitted_iterations)}/{total_uncached} total)")
                        if i < submissions_this_round - 1:  # Don't delay after last job
                            time.sleep(delay_between_jobs)
                    except Exception as e:
                        error_msg = str(e)
                        if "QOSMaxSubmitJobPerUserLimit" in error_msg or "submit limit" in error_msg.lower():
                            logger.warning(f"⚠️  Hit SLURM submission limit (submitted {len(submitted_iterations)}/{total_uncached} so far)")
                            logger.warning(f"   Will continue polling and try again in {poll_interval}s")
                            # Don't break - continue polling, maybe slots will open up
                        else:
                            logger.error(f"❌ Failed to submit job for iteration {iteration_idx}: {e}")
                            raise
            else:
                logger.info(f"⏸️  No available slots (all {target_running_jobs} slots filled). Waiting for jobs to complete...")

        # Check if all iterations were submitted
        if len(submitted_iterations) < total_uncached:
            remaining_iterations = [idx for idx in uncached_iterations if idx not in submitted_iterations]
            if remaining_iterations:
                remaining_file = run_dir / "remaining_iterations.txt"
                with open(remaining_file, 'w') as f:
                    for rem_idx in remaining_iterations:
                        f.write(f"{rem_idx}\n")
                logger.warning(f"⚠️  {len(remaining_iterations)} iterations were NOT submitted (hit submission limits)")
                logger.warning(f"   📝 Saved remaining iterations to {remaining_file}")
                logger.warning(f"   📝 You may need to resubmit these manually after jobs complete")

        logger.info(f"✅ Submission complete: {len(iteration_job_ids)} jobs submitted for {len(submitted_iterations)}/{total_uncached} iterations")
        return iteration_job_ids

    def _submit_single_iteration_job(self, iteration_idx: int, loop_iterations_dir: Path, pipeline_name: str, run_dir: Path) -> str:
        """
        Submit a single iteration job and return its job ID.

        Args:
            iteration_idx: Index of the iteration to submit
            loop_iterations_dir: Directory containing iteration configs
            pipeline_name: Name of the pipeline
            run_dir: Run directory for the pipeline

        Returns:
            Job ID string
        """
        from urartu.utils.execution.launcher import launch_on_slurm

        # Create config for this specific iteration
        iteration_cfg = OmegaConf.create(self.cfg)
        iteration_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
        iteration_cfg['_is_iteration_task'] = True  # Mark as iteration task
        iteration_cfg['_iteration_idx'] = iteration_idx  # Store iteration index

        # Support for block_idx (used by combined pipelines with multiple loopable blocks)
        # Check both _block_idx and _current_block_idx for compatibility
        block_idx = self.cfg.get('_block_idx') or self.cfg.get('_current_block_idx')
        if block_idx is not None:
            iteration_cfg['_block_idx'] = block_idx

        # CRITICAL: Remove _submit_array_only flag - this job should run the iteration
        if '_submit_array_only' in iteration_cfg:
            del iteration_cfg['_submit_array_only']

        # Set log paths and executor folder to task-specific directories
        task_dir = run_dir / "array_tasks" / f"task_{iteration_idx}"
        task_dir_str = str(task_dir)
        iteration_cfg['slurm'] = OmegaConf.create(iteration_cfg.get('slurm', {}))
        iteration_cfg['slurm']['additional_parameters'] = OmegaConf.create(iteration_cfg['slurm'].get('additional_parameters', {}))
        iteration_cfg['slurm']['additional_parameters']['output'] = f"{task_dir_str}/%j_log.out"
        iteration_cfg['slurm']['additional_parameters']['error'] = f"{task_dir_str}/%j_log.err"
        iteration_cfg['slurm']['log_folder'] = task_dir_str

        # Submit individual job (no array)
        job = launch_on_slurm(
            module=str(Path.cwd()),
            action_name=pipeline_name,
            cfg=iteration_cfg,
            aim_run=self.aim_run,
            array_size=None,  # No array - individual job
            dependency=None,  # No dependency - all run in parallel
        )

        # Extract job ID
        job_id = job.job_id if hasattr(job, 'job_id') else str(job)
        return job_id

    def _get_current_running_jobs(self) -> int:
        """Get the current number of running/pending jobs for the current user."""
        try:
            username = getpass.getuser()
            squeue_result = subprocess.run(['squeue', '-u', username, '-h', '-o', '%T'], capture_output=True, text=True, timeout=5)
            if squeue_result.returncode == 0:
                queue_statuses = [line.strip() for line in squeue_result.stdout.strip().split('\n') if line.strip()]
                running_count = len([s for s in queue_statuses if s in ['PENDING', 'CONFIGURING', 'RUNNING']])
                return running_count
        except Exception as e:
            logger.debug(f"Could not check queue status: {e}")
        return 0

    def _submit_dependency_job_if_needed(self, iteration_job_ids: List[str], loop_iterations_dir: Optional[Path] = None):
        """
        Submit dependency job for post-loopable actions if needed.

        Args:
            iteration_job_ids: List of job IDs for iteration jobs
            loop_iterations_dir: Optional directory containing iteration configs (created if needed)
        """
        # Check if there are post-loopable actions
        # Find all loopable action placeholders (supports both simple and combined pipelines)
        loopable_indices = [
            idx for idx, a in enumerate(self.actions)
            if a.name == "__loopable_actions__" or a.name.startswith("__loopable_actions_block_")
        ]
        
        # If we have loopable actions, check if there are actions after the last loopable block
        if loopable_indices:
            last_loopable_idx = max(loopable_indices)
            has_post_loopable = last_loopable_idx < len(self.actions) - 1
        else:
            has_post_loopable = False

        if not has_post_loopable:
            logger.info("No post-loopable actions, skipping dependency job")
            return

        # Create iteration configs if not provided (needed for dependency job)
        if loop_iterations_dir is None:
            loop_iterations_dir = self._create_iteration_configs()
            logger.info(f"📁 Created iteration configs in {loop_iterations_dir} for dependency job")

        # Create dependency string: afterok:job1,afterok:job2,afterok:job3,...
        if iteration_job_ids:
            dependency_str = ",".join([f"afterok:{job_id}" for job_id in iteration_job_ids])
            logger.info(f"🚀 Submitting dependency job for post-loopable actions (depends on {len(iteration_job_ids)} iteration jobs)")
        else:
            dependency_str = None  # No dependencies - all cached
            logger.info("📋 Submitting dependency job for post-loopable actions (all iterations cached)")

        dep_cfg = OmegaConf.create(self.cfg)
        dep_cfg['_post_loopable_only'] = True
        dep_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
        dep_cfg['_iteration_job_ids'] = iteration_job_ids  # Store job IDs for reference

        # CRITICAL: Remove _submit_array_only flag - this job should run post-loopable actions
        if '_submit_array_only' in dep_cfg:
            del dep_cfg['_submit_array_only']

        # Set dependency job resources
        if 'slurm' in dep_cfg:
            slurm_cfg = dep_cfg['slurm']
            dependency_mem = slurm_cfg.get('dependency_mem', None)
            if dependency_mem is None:
                dependency_mem = slurm_cfg.get('mem', 80)
            dep_cfg['slurm'] = OmegaConf.create(
                {
                    **slurm_cfg,
                    'mem': dependency_mem,
                    'gpus_per_node': slurm_cfg.get('dependency_gpus_per_node', slurm_cfg.get('gpus_per_node', 1)),
                    'cpus_per_task': slurm_cfg.get('dependency_cpus_per_task', slurm_cfg.get('cpus_per_task', 4)),
                    'nodes': slurm_cfg.get('dependency_nodes', slurm_cfg.get('nodes', 1)),
                    'nodelist': slurm_cfg.get('dependency_nodelist', slurm_cfg.get('nodelist', None)),
                }
            )
            logger.info(
                f"📋 Dependency job resources: {dependency_mem}GB mem, {dep_cfg['slurm']['gpus_per_node']} GPUs, {dep_cfg['slurm']['cpus_per_task']} CPUs"
            )

        from urartu.utils.execution.launcher import launch_on_slurm

        dep_job = launch_on_slurm(
            module=str(Path.cwd()),
            action_name=self.cfg.get('pipeline_name'),
            cfg=dep_cfg,
            aim_run=self.aim_run,
            array_size=None,
            dependency=dependency_str,
        )

        logger.info(f"✅ Submitted dependency job {dep_job.job_id} for post-loopable actions")
