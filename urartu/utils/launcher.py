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
    timeout_min = cfg.slurm.time_hours * 60 + cfg.slurm.time_minutes
    executor.update_parameters(
        name=cfg.slurm.name,
        slurm_comment=cfg.slurm.comment,
        slurm_partition=cfg.slurm.partition,
        slurm_account=cfg.slurm.account,
        slurm_constraint=cfg.slurm.constraint,
        timeout_min=timeout_min,
        nodes=cfg.slurm.num_nodes,
        cpus_per_task=cfg.slurm.num_cpu_per_proc * cfg.slurm.num_proc_per_node,
        tasks_per_node=cfg.slurm.num_proc_per_node,
        gpus_per_node=cfg.slurm.num_gpu_per_node,
        slurm_mem=f"{cfg.slurm.mem_gb}G",
        mem_gb=cfg.slurm.mem_gb,
        slurm_additional_parameters=cfg.slurm.additional_parameters,
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
