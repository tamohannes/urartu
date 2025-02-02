import os
import sys
import logging
import secrets

from aim import Run
from hydra import compose, initialize
from hydra.core.plugins import Plugins
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime

from urartu.utils.launcher import launch, launch_on_slurm
from urartu.utils.hydra_plugin import UrartuPlugin

Plugins.instance().register(UrartuPlugin)


import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))

    cwd = Path.cwd()

    is_module = cwd.joinpath('__init__.py').exists()
    if not is_module:
        logging.error(
            f"The directory '{cwd}' is not recognized as a Python module because it lacks an '__init__.py' file."
            " To resolve this issue, ensure that an '__init__.py' file exists in the directory if it's intended to be a Python module."
        )
        raise FileNotFoundError("Missing '__init__.py' file.")

    are_actions = cwd.joinpath('actions').is_dir()
    if not are_actions:
        logging.error(
            f"The required 'actions' subdirectory does not exist in the directory '{cwd}'."
            " Please ensure that an 'actions' directory is created within the current directory to proceed."
        )
        raise FileNotFoundError("Missing 'actions' directory.")

    action_file_path = cwd.joinpath('actions', f"{cfg.action_name}.py")
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Redirect stdout and stderr to log file
    sys.stdout = open(log_file, 'a')
    sys.stderr = sys.stdout

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

    if cfg.slurm.use_slurm:
        try:
            import submitit  # NOQA
        except ImportError:
            raise ImportError("Please 'pip install submitit' to schedule jobs on SLURM")

        launch_on_slurm(
            module=cwd,
            action_name=cfg.action_name,
            cfg=cfg,
            aim_run=aim_run,
        )
    else:
        launch(
            module=cwd,
            action_name=cfg.action_name,
            cfg=cfg,
            aim_run=aim_run,
        )

    if cfg.aim.use_aim and aim_run.active:
        aim_run.close()
