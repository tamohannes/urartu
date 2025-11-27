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
