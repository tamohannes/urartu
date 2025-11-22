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
from urartu.utils.launcher import launch, launch_on_slurm, launch_remote
from urartu.utils.logging import get_logger, configure_logging

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
        print("""Usage: urartu action=ACTION_NAME [other_params]
       urartu pipeline=PIPELINE_NAME [other_params]

Required arguments:
  action=ACTION_NAME          Name of the action to run (must exist in actions/ directory)
  pipeline=PIPELINE_NAME      Name of the pipeline to run (must exist in pipelines/ directory)

Optional arguments:
  debug=true                  Run in debug mode
  slurm.use_slurm=true       Run on SLURM cluster
  aim.use_aim=true           Use Aim for experiment tracking

Examples:
  urartu action=generate aim=aim slurm=slurm
  urartu pipeline=my_pipeline aim=aim slurm=slurm
""")
        return

    # If we get here, proceed with normal Hydra execution
    _hydra_main()


@hydra.main(version_base=None, config_path="config", config_name="main")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra main function for running experiments."""
    hydra_cfg = HydraConfig.get()
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

    # Note: pipeline=name and action=name are config groups, not entity name shorthands.
    # Users should use pipeline_name=name and action_name=name to specify entity names.
    # However, if pipeline or action are strings (not dicts), they might be from command line overrides.
    # In that case, we treat them as entity names for backward compatibility.
    # Use OmegaConf.select() to safely check values without triggering resolution
    try:
        pipeline_val = OmegaConf.select(cfg, 'pipeline', default=None)
        if isinstance(pipeline_val, str) and pipeline_val not in ['???', 'default_pipeline', None]:
            # Check if this is a config file that exists
            pipeline_config_path = cwd.joinpath("configs", "pipeline", f"{pipeline_val}.yaml")
            if not pipeline_config_path.exists():
                # Not a config file, treat as entity name (backward compatibility)
                cfg.pipeline_name = pipeline_val
                # Reset to default pipeline config
                try:
                    default_pipeline_path = Path(__file__).parent.parent / "config" / "pipeline" / "default_pipeline.yaml"
                    if default_pipeline_path.exists():
                        default_cfg = OmegaConf.load(default_pipeline_path)
                        cfg.pipeline = default_cfg.get('pipeline', {})
                except Exception:
                    try:
                        cfg.pipeline = {}
                    except Exception:
                        pass
    except Exception:
        pass
    
    try:
        action_val = OmegaConf.select(cfg, 'action', default=None)
        if isinstance(action_val, str) and action_val not in ['???', 'default_action', None]:
            # Check if this is a config file that exists
            action_config_path = cwd.joinpath("configs", "action", f"{action_val}.yaml")
            if not action_config_path.exists():
                # Not a config file, treat as entity name (backward compatibility)
                cfg.action_name = action_val
                # Reset to default action config
                try:
                    default_action_path = Path(__file__).parent.parent / "config" / "action" / "default_action.yaml"
                    if default_action_path.exists():
                        default_cfg = OmegaConf.load(default_action_path)
                        cfg.action = default_cfg.get('action', {})
                except Exception:
                    try:
                        cfg.action = {}
                    except Exception:
                        pass
    except Exception:
        pass
    
    # Verify specific action or pipeline file exists
    # Determine entity name - can be action_name or pipeline_name
    entity_name = None
    entity_type = None
    
    # Check for pipeline_name first (pipelines use pipeline_name)
    # Use OmegaConf.select() to safely check values without triggering resolution
    entity_name = None
    entity_type = None
    
    try:
        pipeline_name_val = OmegaConf.select(cfg, 'pipeline_name', default=None)
        if pipeline_name_val is not None and pipeline_name_val != '???':
            entity_name = pipeline_name_val
            entity_type = 'pipeline'
    except Exception:
        pass
    
    # Fall back to action_name (actions use action_name, or legacy support)
    if entity_name is None:
        try:
            action_name_val = OmegaConf.select(cfg, 'action_name', default=None)
            if action_name_val is not None and action_name_val != '???':
                entity_name = action_name_val
                entity_type = 'action'  # Will be determined by file existence
        except Exception:
            pass
    
    if entity_name is None:
        raise ValueError(
            "Config must specify either 'action_name' (for actions) or 'pipeline_name' (for pipelines) "
            "to identify the entity to run"
        )
    
    # Check if it's a pipeline (in pipelines/ directory) or action (in actions/ directory)
    pipeline_file_path = cwd.joinpath("pipelines", f"{entity_name}.py")
    action_file_path = cwd.joinpath("actions", f"{entity_name}.py")
    
    is_pipeline = pipeline_file_path.exists()
    is_action = action_file_path.exists()
    
    # Update entity_type based on actual file existence
    if is_pipeline:
        entity_type = 'pipeline'
        # Set pipeline_name if not already set
        try:
            pipeline_name_val = OmegaConf.select(cfg, 'pipeline_name', default=None)
            if pipeline_name_val is None or pipeline_name_val == '???':
                cfg.pipeline_name = entity_name
        except Exception:
            cfg.pipeline_name = entity_name
    elif is_action:
        entity_type = 'action'
        # Set action_name if not already set
        try:
            action_name_val = OmegaConf.select(cfg, 'action_name', default=None)
            if action_name_val is None or action_name_val == '???':
                cfg.action_name = entity_name
        except Exception:
            cfg.action_name = entity_name
    
    if is_pipeline:
        logging.info(
            f"The pipeline file '{pipeline_file_path}' is located and is ready to be used!"
        )
        # Check if pipeline config exists and load it from pipeline directory
        pipeline_name = getattr(cfg, 'pipeline_name', entity_name)
        pipeline_config_path = cwd.joinpath("configs", "pipeline", f"{pipeline_name}.yaml")
        if pipeline_config_path.exists():
            logging.info(f"📋 Loading pipeline config from: {pipeline_config_path}")
            # Load pipeline config (Hydra will handle defaults via search path)
            pipeline_cfg = OmegaConf.load(pipeline_config_path)
            
            # Merge pipeline section
            # Note: OmegaConf.merge(base, override) - override takes precedence
            # We want the loaded config to override defaults, so put defaults first
            if 'pipeline' in pipeline_cfg:
                # Get current pipeline config (might be from defaults)
                current_pipeline = cfg.get('pipeline', {})
                if not current_pipeline:
                    current_pipeline = {}
                # Merge: defaults first, then loaded config (loaded config overrides defaults)
                merged_pipeline = OmegaConf.merge(
                    OmegaConf.create(current_pipeline) if not isinstance(current_pipeline, DictConfig) else current_pipeline,
                    pipeline_cfg.pipeline
                )
                cfg.pipeline = merged_pipeline
                logging.info(f"📋 Merged pipeline config with {len(merged_pipeline.get('actions', []))} actions")
            
            # Merge other top-level keys (like pipeline_name, debug, etc.)
            for key in pipeline_cfg:
                if key not in ['pipeline', 'defaults']:
                    # Special handling for pipeline_name
                    if key == 'pipeline_name':
                        cfg.pipeline_name = pipeline_cfg[key]
                    # Only override if the value in main config is None or missing
                    elif not hasattr(cfg, key) or cfg[key] is None:
                        cfg[key] = pipeline_cfg[key]
                    # Or if the pipeline config explicitly sets it (non-None value)
                    elif pipeline_cfg[key] is not None:
                        cfg[key] = pipeline_cfg[key]
        else:
            logging.warning(
                f"⚠️  Pipeline config not found at {pipeline_config_path}. "
                f"Using default pipeline configuration."
            )
    elif is_action:
        logging.info(
            f"The action file '{action_file_path}' is located and is ready to be used!"
        )
        # Check if action config exists and load it from action directory
        action_name = getattr(cfg, 'action_name', entity_name)
        action_config_path = cwd.joinpath("configs", "action", f"{action_name}.yaml")
        if action_config_path.exists():
            logging.info(f"📋 Loading action config from: {action_config_path}")
            # Load action config (Hydra will handle defaults via search path)
            action_cfg = OmegaConf.load(action_config_path)
            
            # Merge action section
            if 'action' in action_cfg:
                cfg.action = OmegaConf.merge(cfg.action, action_cfg.action)
            
            # Merge other top-level keys (like action_name, debug, etc.)
            for key in action_cfg:
                if key not in ['action', 'defaults']:
                    # Special handling for action_name
                    if key == 'action_name':
                        cfg.action_name = action_cfg[key]
                    # Only override if the value in main config is None or missing
                    elif not hasattr(cfg, key) or cfg[key] is None:
                        cfg[key] = action_cfg[key]
                    # Or if the action config explicitly sets it (non-None value)
                    elif action_cfg[key] is not None:
                        cfg[key] = action_cfg[key]
        else:
            logging.warning(
                f"⚠️  Action config not found at {action_config_path}. "
                f"Using default action configuration."
            )
    else:
        logging.error(
            f"Neither action nor pipeline file found for '{entity_name}'."
            f" Checked: {action_file_path} and {pipeline_file_path}."
            " Please ensure that the file exists in either the 'actions' or 'pipelines' directory."
        )
        raise FileNotFoundError("Missing action or pipeline file.")
    
    # Set the appropriate name field for consistency
    # Note: run_dir uses oc.select to choose pipeline_name or action_name automatically
    # We need to set the unused name to None BEFORE resolving to avoid interpolation errors
    # oc.select will fail if one of the values is still ??? (mandatory missing)
    # Convert to container, modify, and recreate to avoid triggering resolution of ??? values
    if is_pipeline:
        cfg.pipeline_name = entity_name
        # Set action_name to None for pipelines to avoid interpolation errors in run_dir
        # Convert to container to safely modify without triggering resolution
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            cfg_dict['action_name'] = None
            cfg = OmegaConf.create(cfg_dict)
        except Exception:
            # Fallback: try direct assignment (might fail if action_name is ???)
            try:
                cfg.action_name = None
            except Exception:
                pass
    elif is_action:
        cfg.action_name = entity_name
        # Set pipeline_name to None for actions to avoid interpolation errors in run_dir
        # Convert to container to safely modify without triggering resolution
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            cfg_dict['pipeline_name'] = None
            cfg = OmegaConf.create(cfg_dict)
        except Exception:
            # Fallback: try direct assignment (might fail if pipeline_name is ???)
            try:
                cfg.pipeline_name = None
            except Exception:
                pass
    
    # Now resolve the config after loading the appropriate config file
    # At this point, both pipeline_name and action_name are set (one to entity_name, one to None)
    # so oc.select in run_dir will work correctly
    # Preserve the pipeline config structure during resolution (especially the actions list)
    pipeline_backup = None
    if is_pipeline and hasattr(cfg, 'pipeline'):
        pipeline_backup = OmegaConf.to_container(cfg.pipeline, resolve=False)
        logging.info(f"📋 Backing up pipeline config with {len(pipeline_backup.get('actions', []))} actions before resolution")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    # Always restore pipeline config after resolution to ensure actions are preserved
    if is_pipeline and pipeline_backup is not None:
        current_pipeline = cfg.get('pipeline', {})
        if not current_pipeline or 'actions' not in current_pipeline:
            cfg.pipeline = OmegaConf.merge(current_pipeline or {}, pipeline_backup)
            logging.info(f"📋 Restored pipeline config with {len(cfg.pipeline.get('actions', []))} actions after resolution")
        else:
            logging.info(f"📋 Pipeline config preserved with {len(cfg.pipeline.get('actions', []))} actions after resolution")

    if cfg.machine.type == "remote":
        launch_remote(cfg=cfg)
        return

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

    # Handle description field (general field, not AIM-specific)
    # Prefer general description field, fall back to description and cfg.aim.description for backward compatibility
    description = None
    if hasattr(cfg, 'descr') and cfg.descr:
        description = cfg.descr
    elif hasattr(cfg, 'description') and cfg.description:
        description = cfg.description
    elif hasattr(cfg, 'aim') and hasattr(cfg.aim, 'description') and cfg.aim.description:
        description = cfg.aim.description

    aim_run = None
    if cfg.aim.use_aim:
        # Get experiment name - use pipeline_name for pipelines, action_name for actions
        entity_name_for_aim = getattr(cfg, 'pipeline_name', None) or getattr(cfg, 'action_name', None)
        if hasattr(cfg, 'pipeline') and cfg.pipeline and 'experiment_name' in cfg.pipeline:
            experiment_name = cfg.pipeline.experiment_name
        elif hasattr(cfg, 'action') and cfg.action and 'experiment_name' in cfg.action:
            experiment_name = cfg.action.experiment_name
        else:
            experiment_name = entity_name_for_aim
        
        aim_run = Run(
            repo=cfg.aim.repo,
            experiment=experiment_name,
            log_system_params=cfg.aim.log_system_params,
        )
        aim_run.set("cfg", cfg, strict=False)
        if cfg.debug:
            aim_run.add_tag("debug")
        # Set AIM description if description is provided
        if description:
            aim_run.description = description
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
    
    # Log description if provided (appears in both console and log file)
    if description:
        logging.info(f"📝 Description: {description}")

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
                # Use pipeline_name for pipelines, action_name for actions
                entity_name = getattr(cfg, 'pipeline_name', None) or getattr(cfg, 'action_name', None)
                launch_on_slurm(
                    module=cwd,
                    action_name=entity_name,
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
                # Use pipeline_name for pipelines, action_name for actions
                entity_name = getattr(cfg, 'pipeline_name', None) or getattr(cfg, 'action_name', None)
                # Debug: verify pipeline config before launching
                if is_pipeline and hasattr(cfg, 'pipeline'):
                    pipeline_keys = list(cfg.pipeline.keys()) if hasattr(cfg.pipeline, 'keys') else []
                    actions_count = len(cfg.pipeline.actions) if 'actions' in cfg.pipeline else 0
                    logging.info(f"🚀 Launching pipeline with cfg.pipeline keys: {pipeline_keys[:10]}, actions: {actions_count}")
                launch(
                    module=cwd,
                    action_name=entity_name,
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
