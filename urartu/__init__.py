import logging
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from aim import Repo, Run
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from omegaconf import OmegaConf

from urartu.utils.hydra_plugin import UrartuPlugin
from urartu.utils.launcher import launch, launch_on_slurm

Plugins.instance().register(UrartuPlugin)

import hydra
from omegaconf import DictConfig


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
                self._handle_multirun_directory(
                    dir_path, date_pattern, run_hashes, dirs_to_delete
                )
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
                logging.info(
                    f"Multirun directory {dir_path} has no valid runs, marking for deletion"
                )
            elif invalid_num_dirs:
                dirs_to_delete.extend(invalid_num_dirs)
                logging.info(
                    f"Found {len(invalid_num_dirs)} invalid runs in multirun directory {dir_path}"
                )

    def _handle_regular_directory(
        self, dir_path: Path, run_hashes: List[str], dirs_to_delete: List[Path]
    ) -> None:
        """Handle regular directory checking."""
        if not self._has_valid_hash(dir_path, run_hashes):
            dirs_to_delete.append(dir_path)
            logging.info(f"Directory {dir_path} has no matching run hash yaml file")

    def _has_valid_hash(self, dir_path: Path, run_hashes: List[str]) -> bool:
        """Check if directory contains a yaml file with a valid hash."""
        yaml_files = list(dir_path.glob("*.yaml"))
        return any(yaml_file.stem in run_hashes for yaml_file in yaml_files)

    def _delete_marked_directories(
        self, action_name: str, dirs_to_delete: List[Path]
    ) -> None:
        """Delete all marked directories."""
        if dirs_to_delete:
            logging.info(
                f"Found {len(dirs_to_delete)} directories to delete in action {action_name}"
            )
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
        print("""Usage: urartu action_config=ACTION_NAME [other_params]

Required arguments:
  action_config=ACTION_NAME    Name of the action to run (must exist in actions/ directory)

Optional arguments:
  debug=true                  Run in debug mode
  slurm.use_slurm=true       Run on SLURM cluster
  aim.use_aim=true           Use Aim for experiment tracking

Example:
  urartu action_config=generate aim=aim slurm=slurm
""")
        return

    # If we get here, proceed with normal Hydra execution
    _hydra_main()


@hydra.main(version_base=None, config_path="config", config_name="main")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra main function for running experiments."""
    hydra_cfg = HydraConfig.get()
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))

    cwd = Path.cwd()

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

    # Verify specific action file exists
    action_file_path = cwd.joinpath("actions", f"{cfg.action_name}.py")
    if action_file_path.exists():
        logging.info(
            f"The action file '{action_file_path}' is located and is ready to be used!"
        )
    else:
        logging.error(
            f"The action file '{action_file_path}' does not exist."
            " Please ensure that the file is correctly named and located in the 'actions' directory."
        )
        raise FileNotFoundError("Missing action file.")

    is_multirun = hydra_cfg.mode.name == "MULTIRUN"

    if is_multirun:
        run_dir = Path(hydra_cfg.runtime.output_dir, hydra_cfg.job.id).parent
    else:
        run_dir = Path(cfg.run_dir)
        if cfg.debug:
            parts = list(Path(run_dir).parts)
            parts.insert(-1, "debug")
            run_dir = Path(*parts)
        os.makedirs(run_dir, exist_ok=True)

    aim_run = None
    if cfg.aim.use_aim:
        aim_run = Run(
            repo=cfg.aim.repo,
            experiment=cfg.action_config.experiment_name,
            log_system_params=cfg.aim.log_system_params,
        )
        aim_run.set("cfg", cfg, strict=False)
        if cfg.debug:
            aim_run.add_tag("debug")
        cfg.aim.hash = aim_run.hash

    if cfg.aim.use_aim:
        log_file = run_dir.joinpath(f"{aim_run.hash}.log")
    else:
        log_file = run_dir.joinpath("output.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger.handlers = [file_handler, stream_handler]

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
    if cfg.aim.use_aim:
        with open(run_dir.joinpath(f"{aim_run.hash}.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f)
    else:
        with open(run_dir.joinpath("cfg.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f)

    try:
        if cfg.slurm.use_slurm:
            try:
                import submitit  # NOQA
            except ImportError:
                raise ImportError(
                    "Please 'pip install submitit' to schedule jobs on SLURM"
                )

            try:
                launch_on_slurm(
                    module=cwd,
                    action_name=cfg.action_name,
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
                    logging.error(
                        "Not running on a SLURM cluster or 'srun' command not available"
                    )
                else:
                    logging.error(f"Runtime error during SLURM job execution: {e}")
                raise
        else:
            try:
                launch(
                    module=cwd,
                    action_name=cfg.action_name,
                    cfg=cfg,
                    aim_run=aim_run,
                )
            except ImportError as e:
                logging.error(
                    f"Failed to import required module for local execution: {e}"
                )
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
