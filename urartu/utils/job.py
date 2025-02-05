import sys
from importlib import import_module
from typing import Dict

from aim import Run


class ResumableSlurmJob:
    def __init__(self, module: str, action_name: str, cfg: Dict, aim_run: Run):
        self.module = module
        self.action_name = action_name
        self.cfg = cfg

        self.aim_run = None
        if self.cfg.aim.use_aim:
            self.aim_run_hash = aim_run.hash

    def get_aim_run(self):
        if self.cfg.aim.use_aim and self.aim_run is None:
            self.aim_run = Run(self.aim_run_hash, repo=self.cfg.aim.repo)
        return self.aim_run

    def __call__(self):
        import submitit

        environment = submitit.JobEnvironment()
        master_ip = environment.hostnames[0]
        master_port = self.cfg.slurm.port_id
        self.cfg.slurm.init_method = "tcp"
        self.cfg.slurm.run_id = f"{master_ip}:{master_port}"

        if self.cfg.aim.use_aim:
            self.get_aim_run()
            self.aim_run.set("job", {"job_id": int(environment.job_id), "hostname": environment.hostname})

        sys.path.append(f"{self.module}/actions")
        action = import_module(self.action_name)
        action.main(cfg=self.cfg, aim_run=self.aim_run)

    def checkpoint(self):
        import submitit

        runner = ResumableSlurmJob(
            module=self.module,
            action_name=self.action_name,
            cfg=self.cfg,
            aim_run=self.aim_run,
        )
        return submitit.helpers.DelayedSubmission(runner)

    def on_job_fail(self):
        self.get_aim_run()
        self.aim_run.close()


class ResumableJob:
    def __init__(self, module: str, action_name: str, cfg: Dict, aim_run: Run):
        self.module = module
        self.action_name = action_name
        self.cfg = cfg
        self.aim_run = aim_run

    def __call__(self):
        sys.path.append(f"{self.module}/actions")
        action = import_module(self.action_name)
        action.main(cfg=self.cfg, aim_run=self.aim_run)
