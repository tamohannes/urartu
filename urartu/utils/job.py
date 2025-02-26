import sys
from importlib import import_module
from typing import Dict

from aim import Run


class ResumableSlurmJob:
    """
    A class designed to handle the resumption and management of jobs submitted to a Slurm
    cluster, with integration for tracking and managing experiments using the Aim toolkit.

    Attributes:
        module (str): The module path where the job action is defined.
        action_name (str): The specific action to be executed within the module.
        cfg (Dict): Configuration dictionary containing settings for the job and Aim integration.
        aim_run (Run): An Aim toolkit Run object for tracking the job execution.

    Methods:
        get_aim_run: Retrieves or initializes the Aim run associated with this job.
        __call__: Executes the job's main action, managing the experiment's lifecycle and Slurm environment.
        checkpoint: Prepares a job checkpoint that can be resumed if the job is preempted.
        on_job_fail: Handles clean-up and closure of the Aim run in the event of job failure.
    """

    def __init__(self, module: str, action_name: str, cfg: Dict, aim_run: Run):
        """
        Initializes the ResumableSlurmJob with necessary parameters for job execution and
        experiment tracking.

        Args:
            module (str): Path to the module where job actions are located.
            action_name (str): Name of the function or action to run within the module.
            cfg (Dict): Configuration settings including Slurm and Aim details.
            aim_run (Run): An Aim toolkit Run object for tracking experiment data.
        """
        self.module = module
        self.action_name = action_name
        self.cfg = cfg
        self.aim_run = None
        if self.cfg.aim.use_aim:
            self.aim_run_hash = aim_run.hash

    def get_aim_run(self):
        """
        Retrieves or initializes the Aim run object based on configuration settings.

        Returns:
            Run: The Aim run object associated with the current job.
        """
        if self.cfg.aim.use_aim and self.aim_run is None:
            self.aim_run = Run(self.aim_run_hash, repo=self.cfg.aim.repo)
        return self.aim_run

    def __call__(self):
        """
        Executes the job action specified in the configuration. Handles the setup of the
        Slurm environment and tracks the job execution within an Aim run if configured.
        """
        import submitit

        environment = submitit.JobEnvironment()
        master_ip = environment.hostnames[0]
        master_port = self.cfg.slurm.port_id
        self.cfg.slurm.init_method = "tcp"
        self.cfg.slurm.run_id = f"{master_ip}:{master_port}"

        if self.cfg.aim.use_aim:
            self.get_aim_run()
            self.aim_run.set(
                "job",
                {"job_id": int(environment.job_id), "hostname": environment.hostname},
            )

        sys.path.append(f"{self.module}/actions")
        action = import_module(self.action_name)
        action.main(cfg=self.cfg, aim_run=self.aim_run)

    def checkpoint(self):
        """
        Prepares a checkpoint of the current job state that can be resumed later.

        Returns:
            DelayedSubmission: A submission object that can be used to resume the job.
        """
        import submitit

        runner = ResumableSlurmJob(
            module=self.module,
            action_name=self.action_name,
            cfg=self.cfg,
            aim_run=self.aim_run,
        )
        return submitit.helpers.DelayedSubmission(runner)

    def on_job_fail(self):
        """
        Handles the necessary cleanup and closure of the Aim run in case the job fails.
        """
        self.get_aim_run()
        self.aim_run.close()


class ResumableJob:
    """
    A simpler version of the ResumableSlurmJob for running and managing resumable jobs
    without the integration of Slurm-specific settings.

    Inherits similar attributes and methods from ResumableSlurmJob but tailored for non-Slurm environments.
    """

    def __init__(self, module: str, action_name: str, cfg: Dict, aim_run: Run):
        """
        Initializes the ResumableJob with necessary parameters for job execution and
        experiment tracking without Slurm integration.

        Args:
            module (str): Path to the module where job actions are located.
            action_name (str): Name of the function or action to run within the module.
            cfg (Dict): Configuration settings primarily for the job execution.
            aim_run (Run): An Aim toolkit Run object for tracking experiment data.
        """
        self.module = module
        self.action_name = action_name
        self.cfg = cfg
        self.aim_run = aim_run

    def __call__(self):
        """
        Executes the job action specified in the configuration. Similar to the Slurm version,
        but without managing Slurm-specific environment settings.
        """
        sys.path.append(f"{self.module}/actions")
        action = import_module(self.action_name)
        action.main(cfg=self.cfg, aim_run=self.aim_run)
