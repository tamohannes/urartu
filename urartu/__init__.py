import logging
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from aim import Repo, Run
from omegaconf import DictConfig, OmegaConf

from urartu.utils.config.cli import parse_args, print_usage
from urartu.utils.config.config_loader import load_pipeline_config
from urartu.utils.execution.launcher import launch, launch_on_slurm, launch_remote
from urartu.utils.logging import configure_logging, get_logger


class Command(ABC):
    """Base class for all commands."""

    @abstractmethod
    def execute(self) -> None:
        """Execute the command."""
        pass

    @staticmethod
    @abstractmethod
    def get_command_name() -> str:
        """Get the command name used in CLI."""
        pass


class CleanCommand(Command):
    """Command to clean up runs that are not present in the Aim repository."""

    def __init__(self, aim_repo_path: str, runs_dir: str):
        self.aim_repo_path = aim_repo_path
        self.runs_dir = runs_dir

    @staticmethod
    def get_command_name() -> str:
        return "clean"

    def execute(self) -> None:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

        # Get run hashes from Aim repo
        repo = Repo(self.aim_repo_path)
        run_hashes = repo.list_all_runs()
        logging.info(f"Total number of runs found: {len(run_hashes)}")

        run_dir = Path(self.runs_dir)
        if not run_dir.exists():
            logging.warning(f"Run directory {run_dir} does not exist")
            return

        self._process_action_directories(run_dir, run_hashes)

    def _process_action_directories(self, run_dir: Path, run_hashes: List[str]) -> None:
        """Process each action directory."""
        for action_dir in run_dir.iterdir():
            if not action_dir.is_dir():
                continue

            logging.info(f"Processing action directory: {action_dir.name}")
            self._handle_debug_directory(action_dir)
            self._clean_run_directories(action_dir, run_hashes)

    def _handle_debug_directory(self, action_dir: Path) -> None:
        """Delete debug directory if it exists."""
        debug_dir = action_dir / "debug"
        if debug_dir.exists():
            try:
                shutil.rmtree(debug_dir)
                logging.info(f"Deleted debug directory: {debug_dir}")
            except Exception as e:
                logging.error(f"Failed to delete debug directory: {str(e)}")

    def _clean_run_directories(self, action_dir: Path, run_hashes: List[str]) -> None:
        """Clean run directories that don't have valid hashes."""
        dirs_to_delete = []
        date_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"

        for dir_path in action_dir.glob("**/"):
            if "_multirun" in str(dir_path):
                self._handle_multirun_directory(dir_path, date_pattern, run_hashes, dirs_to_delete)
            elif re.match(date_pattern, dir_path.name):
                self._handle_regular_directory(dir_path, run_hashes, dirs_to_delete)

        self._delete_marked_directories(action_dir.name, dirs_to_delete)

    def _handle_multirun_directory(
        self,
        dir_path: Path,
        date_pattern: str,
        run_hashes: List[str],
        dirs_to_delete: List[Path],
    ) -> None:
        """Handle multirun directory checking."""
        if re.match(date_pattern + "_multirun$", dir_path.name):
            all_num_dirs = []
            invalid_num_dirs = []

            for num_dir in dir_path.iterdir():
                if num_dir.is_dir() and num_dir.name.isdigit():
                    all_num_dirs.append(num_dir)
                    if not self._has_valid_hash(num_dir, run_hashes):
                        invalid_num_dirs.append(num_dir)

            if all_num_dirs and len(all_num_dirs) == len(invalid_num_dirs):
                dirs_to_delete.append(dir_path)
                logging.info(f"Multirun directory {dir_path} has no valid runs, marking for deletion")
            elif invalid_num_dirs:
                dirs_to_delete.extend(invalid_num_dirs)
                logging.info(f"Found {len(invalid_num_dirs)} invalid runs in multirun directory {dir_path}")

    def _handle_regular_directory(self, dir_path: Path, run_hashes: List[str], dirs_to_delete: List[Path]) -> None:
        """Handle regular directory checking."""
        if not self._has_valid_hash(dir_path, run_hashes):
            dirs_to_delete.append(dir_path)
            logging.info(f"Directory {dir_path} has no matching run hash yaml file")

    def _has_valid_hash(self, dir_path: Path, run_hashes: List[str]) -> bool:
        """Check if directory contains a yaml file with a valid hash."""
        yaml_files = list(dir_path.glob("*.yaml"))
        return any(yaml_file.stem in run_hashes for yaml_file in yaml_files)

    def _delete_marked_directories(self, action_name: str, dirs_to_delete: List[Path]) -> None:
        """Delete all marked directories."""
        if dirs_to_delete:
            logging.info(f"Found {len(dirs_to_delete)} directories to delete in action {action_name}")
            for dir_path in dirs_to_delete:
                try:
                    shutil.rmtree(dir_path)
                    logging.info(f"Deleted directory: {dir_path}")
                except Exception as e:
                    logging.error(f"Failed to delete directory {dir_path}: {str(e)}")
        else:
            logging.info(f"No directories to delete in action {action_name}")


class CommandRegistry:
    """Registry for all available commands."""

    _commands = {CleanCommand.get_command_name(): CleanCommand}

    @classmethod
    def get_command(cls, command_name: str, **kwargs) -> Optional[Command]:
        """Get a command instance by name."""
        command_class = cls._commands.get(command_name)
        if command_class:
            return command_class(**kwargs)
        return None

    @classmethod
    def register_command(cls, command_class: type) -> None:
        """Register a new command."""
        cls._commands[command_class.get_command_name()] = command_class


def parse_command_args(args: List[str]) -> dict:
    """Parse command arguments in the format key=value.

    Args:
        args: List of command line arguments

    Returns:
        Dictionary of parsed arguments
    """
    parsed_args = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        parsed_args[key] = value
    return parsed_args


def main():
    """Main entry point for the package."""
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
        print_usage()
        return

    # Parse CLI arguments
    pipeline_name, overrides, config_group_selectors = parse_args()

    # Load pipeline config
    cwd = Path.cwd()
    cfg = load_pipeline_config(pipeline_name, cwd, overrides, config_group_selectors)
    logging.debug(f"Config group selectors used: {config_group_selectors}")

    # Execute pipeline
    _execute_pipeline(cfg, cwd)


def _submit_array_only(cfg: DictConfig, cwd: Path, pipeline_name: str, aim_run):
    """
    Submit array jobs for loopable actions and exit immediately (no pipeline execution).

    This function:
    1. Initializes the pipeline to get loop_iterations
    2. Checks caching for each iteration
    3. Only submits array tasks for uncached iterations (or if force_rerun is set)
    4. Submits dependency job for post-loopable actions
    5. Exits immediately without running the pipeline

    Args:
        cfg: Configuration dictionary
        cwd: Current working directory
        pipeline_name: Name of the pipeline
        aim_run: Aim run object
    """
    logging.info("🚀 Submission-only job: Initializing pipeline to get loop iterations...")

    # Load and instantiate pipeline
    sys.path.append(str(cwd / "pipelines"))
    pipeline_module = __import__(pipeline_name, fromlist=[pipeline_name])
    pipeline_class = getattr(pipeline_module, pipeline_name.replace('_', ' ').title().replace(' ', ''), None)
    if not pipeline_class:
        # Try to find any class that looks like a pipeline
        for attr_name in dir(pipeline_module):
            attr = getattr(pipeline_module, attr_name)
            if isinstance(attr, type) and hasattr(attr, 'main'):
                pipeline_class = attr
                break

    if not pipeline_class:
        raise ValueError(f"Could not find pipeline class in {pipeline_name}")

    pipeline = pipeline_class(cfg, aim_run)

    # Initialize pipeline to get loop_iterations
    if not pipeline._initialized:
        pipeline.initialize()

    # Check if pipeline has loopable actions
    if not hasattr(pipeline, 'loopable_actions') or len(pipeline.loopable_actions) == 0:
        raise ValueError("Pipeline has no loopable actions, but _submit_array_only was called")

    if not hasattr(pipeline, 'loop_iterations') or len(pipeline.loop_iterations) == 0:
        raise ValueError("Pipeline has no loop iterations, but _submit_array_only was called")

    logging.info(f"📋 Found {len(pipeline.loop_iterations)} loop iterations")

    # Check caching for each iteration and only submit uncached ones
    uncached_iterations = []
    cached_iterations = []

    for iteration_idx, loop_context in enumerate(pipeline.loop_iterations):
        # Check if this iteration is cached
        is_cached = pipeline._check_iteration_cache(iteration_idx, loop_context)
        if is_cached:
            cached_iterations.append(iteration_idx)
            logging.info(f"✅ Iteration {iteration_idx} is cached, skipping submission")
        else:
            uncached_iterations.append(iteration_idx)
            logging.info(f"❌ Iteration {iteration_idx} is not cached, will submit")

    if len(cached_iterations) > 0:
        logging.info(f"📊 {len(cached_iterations)}/{len(pipeline.loop_iterations)} iterations are cached")

    if len(uncached_iterations) == 0:
        logging.info("✅ All iterations are cached! No jobs to submit.")
        # Still need to submit dependency job if there are post-loopable actions
        loopable_idx = next((idx for idx, a in enumerate(pipeline.actions) if a.name == "__loopable_actions__"), None)
        has_post_loopable = loopable_idx is not None and loopable_idx < len(pipeline.actions) - 1
        if has_post_loopable:
            logging.info("📋 Submitting dependency job for post-loopable actions (all iterations cached)")
            # All iterations are cached, so no jobs to wait for
            loop_iterations_dir = Path(cfg.get('run_dir', '.')) / "loop_iterations"
            dep_cfg = OmegaConf.create(cfg)
            dep_cfg['_post_loopable_only'] = True
            dep_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
            dep_cfg['_iteration_job_ids'] = []  # No jobs - all cached

            from urartu.utils.execution.launcher import launch_on_slurm

            dep_job = launch_on_slurm(
                module=str(cwd),
                action_name=pipeline_name,
                cfg=dep_cfg,
                aim_run=aim_run,
                array_size=None,
                dependency=None,  # No dependencies - all cached
            )
            logging.info(f"✅ Submitted dependency job {dep_job.job_id} for post-loopable actions (all iterations cached)")
        return

    # Create iteration config files for all iterations
    loop_iterations_dir = pipeline._create_iteration_configs()

    # Submit individual jobs for each uncached iteration (NO ARRAYS - simpler and more reliable)
    from urartu.utils.execution.launcher import launch_on_slurm

    logging.info(f"🚀 Submitting {len(uncached_iterations)} individual jobs (one per uncached iteration)")

    # Submit one job per uncached iteration
    iteration_job_ids = []
    for task_idx, iteration_idx in enumerate(uncached_iterations):
        # Create config for this specific iteration
        iteration_cfg = OmegaConf.create(cfg)
        iteration_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
        iteration_cfg['_is_iteration_task'] = True  # Mark as iteration task
        iteration_cfg['_iteration_idx'] = iteration_idx  # Store iteration index

        # CRITICAL: Remove _submit_array_only flag - this job should run the iteration
        if '_submit_array_only' in iteration_cfg:
            del iteration_cfg['_submit_array_only']

        # Submit individual job (no array)
        logging.info(f"  Submitting job for iteration {iteration_idx} (task {task_idx + 1}/{len(uncached_iterations)})")
        job = launch_on_slurm(
            module=str(cwd),
            action_name=pipeline_name,
            cfg=iteration_cfg,
            aim_run=aim_run,
            array_size=None,  # No array - individual job
            dependency=None,  # No dependency - all run in parallel
        )

        # Extract job ID
        job_id = job.job_id if hasattr(job, 'job_id') else str(job)
        iteration_job_ids.append(job_id)
        logging.info(f"    ✅ Submitted job {job_id} for iteration {iteration_idx}")

    logging.info(f"✅ Submitted {len(iteration_job_ids)} individual jobs for uncached iterations")

    # Check if there are post-loopable actions
    loopable_idx = next((idx for idx, a in enumerate(pipeline.actions) if a.name == "__loopable_actions__"), None)
    has_post_loopable = loopable_idx is not None and loopable_idx < len(pipeline.actions) - 1

    if has_post_loopable:
        # Submit dependency job that waits for all iteration jobs
        # If all iterations are cached, submit immediately (no dependencies)
        if len(iteration_job_ids) == 0:
            dependency_str = None
            logging.info("📋 Submitting dependency job for post-loopable actions (all iterations cached, no dependencies)")
        else:
            # Create dependency string: afterok:job1,afterok:job2,afterok:job3,...
            # Use commas to chain multiple afterok clauses (AND condition - wait for ALL jobs)
            dependency_str = ",".join([f"afterok:{job_id}" for job_id in iteration_job_ids])
            logging.info(f"📋 Submitting dependency job for post-loopable actions (depends on {len(iteration_job_ids)} iteration jobs)")

        dep_cfg = OmegaConf.create(cfg)
        dep_cfg['_post_loopable_only'] = True
        dep_cfg['_loop_iterations_dir'] = str(loop_iterations_dir)
        dep_cfg['_iteration_job_ids'] = iteration_job_ids

        # Remove _submit_array_only flag
        if '_submit_array_only' in dep_cfg:
            del dep_cfg['_submit_array_only']

        # Set dependency job resources (needs more memory than submission job for aggregation)
        # Use dependency_mem if specified, otherwise use main job memory (not submission memory)
        if 'slurm' in dep_cfg:
            slurm_cfg = dep_cfg['slurm']
            dependency_mem = slurm_cfg.get('dependency_mem', None)
            if dependency_mem is None:
                # Fall back to main job memory (not submission memory)
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
            logging.info(
                f"📋 Dependency job resources: {dependency_mem}GB mem, {dep_cfg['slurm']['gpus_per_node']} GPUs, {dep_cfg['slurm']['cpus_per_task']} CPUs"
            )

        from urartu.utils.execution.launcher import launch_on_slurm

        dep_job = launch_on_slurm(
            module=str(cwd),
            action_name=pipeline_name,
            cfg=dep_cfg,
            aim_run=aim_run,
            array_size=None,
            dependency=dependency_str,
        )

        logging.info(f"✅ Submitted dependency job {dep_job.job_id} for post-loopable actions")
    else:
        logging.info("No post-loopable actions, skipping dependency job")

    logging.info("✅ Job submission complete. Exiting submission-only job.")


def _run_array_task(cfg: DictConfig, cwd: Path, pipeline_name: str, aim_run):
    """
    Run a single iteration as an individual job.

    Args:
        cfg: Configuration dictionary
        cwd: Current working directory
        pipeline_name: Name of the pipeline
        aim_run: Aim run object
    """
    # Get iteration index from config (set when job was submitted)
    iteration_idx = cfg.get('_iteration_idx')
    if iteration_idx is None:
        raise ValueError("Iteration index not found in config (_iteration_idx)")

    iteration_idx = int(iteration_idx)
    logging.info(f"📋 Running iteration {iteration_idx} (individual job)")

    # Get loop iterations directory
    run_dir = cfg.get('run_dir', '.')
    loop_iterations_dir = Path(cfg.get('_loop_iterations_dir', str(Path(run_dir) / 'loop_iterations')))
    iteration_config_path = loop_iterations_dir / f"{iteration_idx}.yaml"

    if not iteration_config_path.exists():
        raise FileNotFoundError(f"Iteration config not found: {iteration_config_path}")

    # Load iteration context
    iteration_cfg = OmegaConf.load(iteration_config_path)
    loop_context = OmegaConf.to_container(iteration_cfg, resolve=True)
    logging.info(f"📋 Loaded iteration context: {loop_context}")

    # Mark this as an iteration task in the config to prevent nested parallel processing
    cfg['_is_iteration_task'] = True
    cfg['_iteration_idx'] = iteration_idx

    # Override loop_iterations to only include this single iteration
    # This prevents the pipeline from trying to submit another array
    if hasattr(cfg, 'pipeline') and cfg.pipeline:
        pipeline_cfg = cfg.pipeline
        if isinstance(pipeline_cfg, DictConfig) or isinstance(pipeline_cfg, dict):
            # Find the loopable_actions block and replace loop_iterations with just this one
            actions = pipeline_cfg.get('actions', [])
            for action in actions:
                if isinstance(action, DictConfig) or isinstance(action, dict):
                    if 'loopable_actions' in action:
                        loopable_block = action['loopable_actions']
                        # Replace loop_iterations with just this single iteration
                        loopable_block['loop_iterations'] = [loop_context]
                        logging.info(f"📋 Overrode loop_iterations to only include iteration {iteration_idx}")

    # Load and instantiate pipeline
    sys.path.append(str(cwd / "pipelines"))
    pipeline_module = __import__(pipeline_name, fromlist=[pipeline_name])
    pipeline_class = getattr(pipeline_module, pipeline_name.replace('_', ' ').title().replace(' ', ''), None)
    if not pipeline_class:
        # Try to find any class that looks like a pipeline
        for attr_name in dir(pipeline_module):
            attr = getattr(pipeline_module, attr_name)
            if isinstance(attr, type) and hasattr(attr, 'main'):
                pipeline_class = attr
                break

    if not pipeline_class:
        raise ValueError(f"Could not find pipeline class in {pipeline_name}")

    pipeline = pipeline_class(cfg, aim_run)
    if not hasattr(pipeline, '_run_single_iteration'):
        raise AttributeError(f"Pipeline {pipeline_class.__name__} does not support single iteration execution")

    # Run single iteration
    pipeline._run_single_iteration(iteration_idx, loop_context)


def _run_post_loopable_only(cfg: DictConfig, cwd: Path, pipeline_name: str, aim_run):
    """
    Run only post-loopable actions (for dependency job).

    Args:
        cfg: Configuration dictionary
        cwd: Current working directory
        pipeline_name: Name of the pipeline
        aim_run: Aim run object
    """
    logging.info("📋 Running post-loopable actions only")

    # Load and instantiate pipeline
    sys.path.append(str(cwd / "pipelines"))
    pipeline_module = __import__(pipeline_name, fromlist=[pipeline_name])
    pipeline_class = getattr(pipeline_module, pipeline_name.replace('_', ' ').title().replace(' ', ''), None)
    if not pipeline_class:
        # Try to find any class that looks like a pipeline
        for attr_name in dir(pipeline_module):
            attr = getattr(pipeline_module, attr_name)
            if isinstance(attr, type) and hasattr(attr, 'main'):
                pipeline_class = attr
                break

    if not pipeline_class:
        raise ValueError(f"Could not find pipeline class in {pipeline_name}")

    pipeline = pipeline_class(cfg, aim_run)
    if not hasattr(pipeline, '_run_post_loopable_only'):
        raise AttributeError(f"Pipeline {pipeline_class.__name__} does not support post-loopable-only execution")

    # Run post-loopable actions
    pipeline._run_post_loopable_only()


def _execute_pipeline(cfg: DictConfig, cwd: Path) -> None:
    """Execute a pipeline with the given configuration."""
    # Verify current directory is a Python module
    is_module = cwd.joinpath("__init__.py").exists()
    if not is_module:
        logging.error(
            f"The directory '{cwd}' is not recognized as a Python module because it lacks an '__init__.py' file."
            " To resolve this issue, ensure that an '__init__.py' file exists in the directory if it's intended to be a Python module."
        )
        raise FileNotFoundError("Missing '__init__.py' file.")

    # Verify actions directory exists
    are_actions = cwd.joinpath("actions").is_dir()
    if not are_actions:
        logging.error(
            f"The required 'actions' subdirectory does not exist in the directory '{cwd}'."
            " Please ensure that an 'actions' directory is created within the current directory to proceed."
        )
        raise FileNotFoundError("Missing 'actions' directory.")

    # Get pipeline name from config
    pipeline_name = cfg.get('pipeline_name')
    if not pipeline_name:
        raise ValueError("Config must specify 'pipeline_name' to identify the pipeline to run")

    # Verify pipeline file exists
    pipeline_file_path = cwd.joinpath("pipelines", f"{pipeline_name}.py")
    if not pipeline_file_path.exists():
        raise FileNotFoundError(
            f"Pipeline file not found: {pipeline_file_path}. " "Please ensure that the pipeline file exists in the 'pipelines' directory."
        )

    logging.info(f"The pipeline file '{pipeline_file_path}' is located and is ready to be used!")

    # Set run_dir - only create new one if not already set (e.g., from CLI override)
    if not cfg.get('run_dir') or cfg.get('run_dir') == '':
        # No run_dir specified, create a new timestamp-based one
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(f".runs/{pipeline_name}/{timestamp}")
        if cfg.get('debug', False):
            parts = list(run_dir.parts)
            parts.insert(-1, "debug")
            run_dir = Path(*parts)
        os.makedirs(run_dir, exist_ok=True)
        cfg.run_dir = str(run_dir)
    else:
        # run_dir already set (e.g., from CLI override or remote execution)
        # Just ensure the directory exists
        run_dir = Path(cfg.run_dir)
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"Using existing run_dir from config: {run_dir}")

    # Set up logging BEFORE checking for remote execution
    # This ensures sync and launch logs are captured locally
    # Use OmegaConf.select for DictConfig to properly access nested configs
    aim_cfg = OmegaConf.select(cfg, 'aim', default=None)
    if aim_cfg is None:
        aim_cfg = {}
        use_aim = False
    elif isinstance(aim_cfg, DictConfig) or isinstance(aim_cfg, dict):
        use_aim = aim_cfg.get('use_aim', False)
    else:
        use_aim = False

    # Set up AIM if needed (for local logging)
    aim_run = None
    if use_aim:
        # Get experiment name from pipeline config
        if hasattr(cfg, 'pipeline') and cfg.pipeline and 'experiment_name' in cfg.pipeline:
            experiment_name = cfg.pipeline.experiment_name
        else:
            experiment_name = pipeline_name

        aim_run = Run(
            repo=aim_cfg.get('repo', '.aim'),
            experiment=experiment_name,
            log_system_params=aim_cfg.get('log_system_params', True),
        )
        aim_run.set("cfg", cfg, strict=False)
        if cfg.get('debug', False):
            aim_run.add_tag("debug")
        # Update cfg.aim.hash safely
        if isinstance(aim_cfg, DictConfig) or isinstance(aim_cfg, dict):
            if isinstance(aim_cfg, DictConfig):
                # For DictConfig, use OmegaConf.update
                OmegaConf.set(cfg, 'aim.hash', aim_run.hash)
            else:
                aim_cfg['hash'] = aim_run.hash
                cfg.aim = aim_cfg

    # Set up logging to file
    if use_aim and aim_run:
        log_file = run_dir.joinpath(f"{aim_run.hash}.log")
    else:
        log_file = run_dir.joinpath("output.log")

    # Create a custom file handler that flushes immediately
    class ImmediateFlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()  # Force immediate write to disk

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Check if log file already exists (for appending)
    log_exists = log_file.exists()

    # Use append mode to preserve previous runs' logs
    file_handler = ImmediateFlushFileHandler(log_file, mode='a', encoding='utf-8')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger.handlers = [file_handler, stream_handler]

    # Add separator if appending to existing log
    if log_exists:
        separator = "\n" + "=" * 80 + "\n"
        separator += f"New run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        separator += "=" * 80 + "\n"
        # Write separator directly to file (before first log message)
        file_handler.stream.write(separator)
        file_handler.flush()

    logging.info(f"📁 Set run_dir to: {cfg.run_dir}")

    # Handle description field
    description = None
    if hasattr(cfg, 'descr') and cfg.descr:
        description = cfg.descr
    elif hasattr(cfg, 'description') and cfg.description:
        description = cfg.description
    elif (isinstance(aim_cfg, DictConfig) or isinstance(aim_cfg, dict)) and hasattr(aim_cfg, 'description') and aim_cfg.description:
        description = aim_cfg.description

    # Log description if provided
    if description:
        logging.info(f"📝 Description: {description}")
        if aim_run:
            aim_run.description = description

    # Check for remote execution
    # Use OmegaConf.select for DictConfig to properly access nested configs
    machine_cfg = OmegaConf.select(cfg, 'machine', default=None)

    if machine_cfg is None:
        machine_type = 'local'
        logging.debug("No machine config found, defaulting to 'local'")
    elif isinstance(machine_cfg, DictConfig) or isinstance(machine_cfg, dict):
        # Handle both DictConfig and dict
        machine_type = machine_cfg.get('type', 'local')
        logging.debug(f"Machine config found: type={machine_type}, full config: {machine_cfg}")
    else:
        # Handle case where machine is a string (e.g., "local")
        machine_type = str(machine_cfg) if machine_cfg else 'local'
        logging.debug(f"Machine config is a string: {machine_type}")

    if machine_type == "remote":
        logging.info(f"🚀 Remote execution detected (machine type: {machine_type}), launching on remote machine...")
        # Save config to file for reference
        cfg_file = run_dir.joinpath("cfg.yaml")
        with open(cfg_file, "w") as f:
            OmegaConf.save(cfg, f)
        logging.info(f"💾 Saved config to: {cfg_file}")
        launch_remote(cfg=cfg)
        return
    else:
        logging.debug(f"Local execution (machine type: {machine_type})")

    class TeeHandler:
        def __init__(self, filename, stream):
            self.terminal = stream
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def fileno(self):
            return self.terminal.fileno()

        def isatty(self):
            return self.terminal.isatty()

        def close(self):
            self.log.close()

    sys.stdout = TeeHandler(log_file, sys.stdout)
    sys.stderr = TeeHandler(log_file, sys.stderr)

    cfg.run_dir = str(run_dir)
    # Re-check aim_cfg (already set above, but ensure consistency)
    aim_cfg = OmegaConf.select(cfg, 'aim', default=None)
    if aim_cfg is None:
        aim_cfg = {}
        use_aim = False
    elif isinstance(aim_cfg, DictConfig) or isinstance(aim_cfg, dict):
        use_aim = aim_cfg.get('use_aim', False)
    else:
        use_aim = False
    if use_aim and aim_run:
        with open(run_dir.joinpath(f"{aim_run.hash}.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f)
    else:
        with open(run_dir.joinpath("cfg.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f)

    try:
        # Use OmegaConf.select for DictConfig to properly access nested configs
        slurm_cfg = OmegaConf.select(cfg, 'slurm', default=None)

        if slurm_cfg is None:
            use_slurm = False
            logging.debug("No SLURM config found")
        elif isinstance(slurm_cfg, DictConfig) or isinstance(slurm_cfg, dict):
            # Handle both DictConfig and dict
            use_slurm = slurm_cfg.get('use_slurm', False)
            logging.info(
                f"SLURM config check: slurm_cfg type={type(slurm_cfg)}, keys={list(slurm_cfg.keys()) if hasattr(slurm_cfg, 'keys') else 'N/A'}, use_slurm={use_slurm}"
            )
            logging.debug(f"SLURM config full: {slurm_cfg}")
        else:
            # Handle case where slurm is a string (shouldn't happen, but be safe)
            use_slurm = False
            logging.warning(f"SLURM config is not a dict/DictConfig: {type(slurm_cfg)}")

        if use_slurm:
            logging.info(f"🚀 SLURM execution detected (use_slurm: {use_slurm}), submitting to SLURM...")
            try:
                import submitit  # NOQA
            except ImportError:
                raise ImportError("Please 'pip install submitit' to schedule jobs on SLURM")

            # Check if pipeline has loopable actions (will use parallel processing)
            # If so, create a submission-only job that just submits the array and exits
            has_loopable_actions = False
            if hasattr(cfg, 'pipeline') and cfg.pipeline:
                pipeline_cfg = cfg.pipeline
                if isinstance(pipeline_cfg, DictConfig) or isinstance(pipeline_cfg, dict):
                    actions = pipeline_cfg.get('actions', [])
                    if actions:
                        # Check if any action has loopable_actions
                        for action in actions:
                            if isinstance(action, DictConfig) or isinstance(action, dict):
                                if 'loopable_actions' in action:
                                    has_loopable_actions = True
                                    logging.info("🔍 Detected loopable actions in config - will create submission-only job")
                                    break

            # If loopable actions detected, create a submission-only job
            if has_loopable_actions:
                slurm_cfg = cfg.get('slurm', {})
                if isinstance(slurm_cfg, DictConfig) or isinstance(slurm_cfg, dict):
                    # Create a submission-only config that just submits the array and exits
                    submission_cfg = OmegaConf.create(cfg)
                    submission_cfg['slurm'] = OmegaConf.create(
                        {
                            **slurm_cfg,
                            'mem': slurm_cfg.get('submission_mem', 8),
                            'gpus_per_node': slurm_cfg.get('submission_gpus_per_node', 0),
                            'cpus_per_task': slurm_cfg.get('submission_cpus_per_task', 1),
                            'nodes': slurm_cfg.get('submission_nodes', 1),
                            'nodelist': slurm_cfg.get('submission_nodelist', None),
                        }
                    )
                    # Add flag to indicate this job should only submit array and exit
                    submission_cfg['_submit_array_only'] = True
                    cfg = submission_cfg
                    logging.info(
                        f"📋 Using submission resources: {slurm_cfg.get('submission_mem', 8)}GB mem, {slurm_cfg.get('submission_gpus_per_node', 0)} GPUs, {slurm_cfg.get('submission_cpus_per_task', 1)} CPUs"
                    )
                    logging.info(f"📋 Initial job will only submit array and exit (no pipeline execution)")

            try:
                launch_on_slurm(
                    module=cwd,
                    action_name=pipeline_name,
                    cfg=cfg,
                    aim_run=aim_run,
                )
            except submitit.core.utils.FailedJobError as e:
                logging.error(f"Slurm job failed: {e}")
                raise
            except submitit.core.utils.FailedSubmissionError as e:
                logging.error(f"Failed to submit job to SLURM: {e}")
                raise
            except RuntimeError as e:
                if "Could not detect 'srun'" in str(e):
                    logging.error("Not running on a SLURM cluster or 'srun' command not available")
                else:
                    logging.error(f"Runtime error during SLURM job execution: {e}")
                raise
        else:
            # Check if this is a submission-only job, array task, or post-loopable-only job
            # IMPORTANT: Check post_loopable_only FIRST (most specific) to prevent dependency jobs
            # from incorrectly matching submit_array_only if both flags are somehow set
            submit_array_only = cfg.get('_submit_array_only', False)
            is_iteration_task = cfg.get('_is_iteration_task', False)
            post_loopable_only = cfg.get('_post_loopable_only', False)

            if post_loopable_only:
                # This is the dependency job - run only post-loopable actions
                logging.info("🔄 Detected post-loopable-only job - running post-loopable actions")
                _run_post_loopable_only(cfg, cwd, pipeline_name, aim_run)
            elif is_iteration_task:
                # This is running as an iteration job - run single iteration
                logging.info("🔄 Detected iteration task - running single iteration")
                _run_array_task(cfg, cwd, pipeline_name, aim_run)
            elif submit_array_only:
                # This is the initial submission job - only submit iteration jobs and exit
                logging.info("🔄 Detected submission-only job - submitting iteration jobs and exiting")
                _submit_array_only(cfg, cwd, pipeline_name, aim_run)
            else:
                try:
                    # Debug: verify pipeline config before launching
                    if hasattr(cfg, 'pipeline'):
                        pipeline_keys = list(cfg.pipeline.keys()) if hasattr(cfg.pipeline, 'keys') else []
                        actions_count = len(cfg.pipeline.actions) if 'actions' in cfg.pipeline else 0
                        logging.info(f"🚀 Launching pipeline with cfg.pipeline keys: {pipeline_keys[:10]}, actions: {actions_count}")
                    launch(
                        module=cwd,
                        action_name=pipeline_name,
                        cfg=cfg,
                        aim_run=aim_run,
                    )
                except ImportError as e:
                    logging.error(f"Failed to import required module for local execution: {e}")
                    raise
                except Exception as e:
                    logging.error(f"Error during local job execution: {e}")
                    raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
    finally:
        if aim_run:
            aim_run.close()


if __name__ == "__main__":
    main()
