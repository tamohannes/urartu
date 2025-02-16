from pathlib import Path
from typing import Dict

from aim import Run
from iopath.common.file_io import g_pathmgr


from .job import ResumableJob, ResumableSlurmJob
import logging
from iopath.common.file_io import g_pathmgr


def create_submitit_executor(cfg: Dict):
    import submitit

    log_folder = Path(cfg.run_dir)
    try:
        if not g_pathmgr.exists(log_folder):
            g_pathmgr.mkdirs(log_folder)
    except BaseException:
        logging.error(f"Error creating directory: {log_folder}")

    assert g_pathmgr.exists(log_folder), f"Specified cfg.slurm.log_folder={log_folder} doesn't exist"
    assert cfg.slurm.partition, "slurm.PARTITION must be set when using slurm"

    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Update parameters to align with _make_sbatch_string
    executor.update_parameters(
        name=cfg.slurm.name,
        comment=cfg.slurm.comment,  # Align with _make_sbatch_string
        account=cfg.slurm.account,
        partition=cfg.slurm.partition,
        timeout_min=cfg.slurm.timeout_min,
        constraint=cfg.slurm.constraint,
        # cpus_per_task=cfg.slurm.num_cpu_per_proc * cfg.slurm.num_proc_per_node,
        mem=f"{cfg.slurm.mem}G",  # Align with _make_sbatch_string
        nodelist=cfg.slurm.nodelist,
        nodes=cfg.slurm.nodes,
        tasks_per_node=cfg.slurm.tasks_per_node,
        gpus_per_node=cfg.slurm.gpus_per_node,
        cpus_per_task=cfg.slurm.cpus_per_task,
        additional_parameters=cfg.slurm.additional_parameters,
        # Add any other relevant parameters from _make_sbatch_string
    )
    return executor


def launch_on_slurm(module: str, action_name: str, cfg: Dict, aim_run: Run):
    executor = create_submitit_executor(cfg)
    trainer = ResumableSlurmJob(module=module, action_name=action_name, cfg=cfg, aim_run=aim_run)

    job = executor.submit(trainer)
    logging.info(f"Submitted job {job.job_id}")

    return job


def launch(module: str, action_name: str, cfg: Dict, aim_run: Run):
    trainer = ResumableJob(module=module, action_name=action_name, cfg=cfg, aim_run=aim_run)
    trainer()
