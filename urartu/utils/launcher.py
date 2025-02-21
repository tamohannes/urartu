import logging
from pathlib import Path
from typing import Dict

from aim import Run
from iopath.common.file_io import g_pathmgr

from .job import ResumableJob, ResumableSlurmJob


def create_submitit_executor(cfg: Dict):
    """
    Creates and configures a SubmitIt executor based on the provided configuration.
    Ensures the log directory exists and is accessible.

    Args:
        cfg (Dict): A dictionary containing configuration settings for the executor,
                    including directory paths and Slurm specific options.

    Returns:
        submitit.AutoExecutor: A configured executor ready to handle job submissions.

    Raises:
        AssertionError: If the log directory does not exist or if required Slurm configuration
                        parameters are missing.
    """
    import submitit

    log_folder = Path(cfg["run_dir"])
    try:
        if not g_pathmgr.exists(log_folder):
            g_pathmgr.mkdirs(log_folder)
    except BaseException:
        logging.error(f"Error creating directory: {log_folder}")

    assert g_pathmgr.exists(
        log_folder
    ), f"Specified cfg['slurm']['log_folder']={log_folder} doesn't exist"
    assert cfg["slurm"]["partition"], "slurm.PARTITION must be set when using slurm"

    executor = submitit.AutoExecutor(folder=log_folder)

    # Update parameters to align with _make_sbatch_string
    executor.update_parameters(
        name=cfg["slurm"]["name"],
        slurm_comment=cfg["slurm"]["comment"],
        slurm_account=cfg["slurm"]["account"],
        slurm_partition=cfg["slurm"]["partition"],
        timeout_min=cfg["slurm"]["timeout_min"],
        slurm_constraint=cfg["slurm"]["constraint"],
        slurm_mem=f"{cfg['slurm']['mem']}G",
        slurm_nodelist=cfg["slurm"]["nodelist"],
        nodes=cfg["slurm"]["nodes"],
        tasks_per_node=cfg["slurm"]["tasks_per_node"],
        gpus_per_node=cfg["slurm"]["gpus_per_node"],
        cpus_per_task=cfg["slurm"]["cpus_per_task"],
        slurm_additional_parameters=cfg["slurm"]["additional_parameters"],
    )
    return executor


def launch_on_slurm(module: str, action_name: str, cfg: Dict, aim_run: Run):
    """
    Submits a job to a Slurm cluster using the provided module, action, configuration, and Aim run.
    Utilizes a SubmitIt executor for job management.

    Args:
        module (str): The module where the job's action is defined.
        action_name (str): The function or method to execute within the module.
        cfg (Dict): Configuration dictionary for the Slurm environment and the job specifics.
        aim_run (Run): An Aim toolkit Run object to track the job.

    Returns:
        submitit.Job: The submitted job object containing job management details and status.
    """
    executor = create_submitit_executor(cfg)
    trainer = ResumableSlurmJob(
        module=module, action_name=action_name, cfg=cfg, aim_run=aim_run
    )

    job = executor.submit(trainer)
    logging.info(f"Submitted job {job.job_id}")

    return job


def launch(module: str, action_name: str, cfg: Dict, aim_run: Run):
    """
    Executes a job directly, without using Slurm, using the specified module, action, configuration,
    and Aim run.

    Args:
        module (str): The module where the job's action is defined.
        action_name (str): The function or method to execute within the module.
        cfg (Dict): Configuration dictionary for the job specifics.
        aim_run (Run): An Aim toolkit Run object to track the job.
    """
    trainer = ResumableJob(
        module=module, action_name=action_name, cfg=cfg, aim_run=aim_run
    )
    trainer()
