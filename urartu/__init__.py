import logging
import os
import secrets
import sys
from pathlib import Path

from aim import Run
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from omegaconf import OmegaConf

from urartu.utils.hydra_plugin import UrartuPlugin
from urartu.utils.launcher import launch, launch_on_slurm

Plugins.instance().register(UrartuPlugin)


import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    """
    Main function that sets up and executes a job based on provided configuration. It prepares
    the environment, handles logging, and directs job execution either locally or on a Slurm cluster
    depending on the configuration. Utilizes the Hydra framework for dynamic configuration management.

    Args:
        cfg (DictConfig): A Hydra-generated configuration object that includes settings for job execution,
                          directory paths, and optional SLURM and AIM integration.

    Raises:
        FileNotFoundError: If necessary directories or files are missing in the expected paths.
        ImportError: If required libraries (like submitit for SLURM) are not available.
    """
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
        logging.info(f"The action file '{action_file_path}' is located and is ready to be used!")
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
            self.log = open(filename, 'a')

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

    with open(run_dir.joinpath("notes.md"), "w") as f:
        pass

    cfg.run_dir = str(run_dir)
    run_hash = secrets.token_hex(8)
    cfg.action_config.run_hash = run_hash
    with open(run_dir.joinpath("cfg.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    aim_run = None
    if cfg.aim.use_aim:
        aim_run = Run(
            repo=cfg.aim.repo,
            experiment=cfg.action_config.experiment_name,
            log_system_params=cfg.aim.log_system_params,
        )
        aim_run.set("cfg", cfg, strict=False)

    try:
        if cfg.slurm.use_slurm:
            try:
                import submitit  # NOQA
            except ImportError:
                raise ImportError("Please 'pip install submitit' to schedule jobs on SLURM")

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
                    logging.error("Not running on a SLURM cluster or 'srun' command not available")
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
