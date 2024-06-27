from aim import Run
from omegaconf import DictConfig

from .device import set_device


class Action:
    def __init__(self, cfg: DictConfig, aim_run: Run):
        self.cfg = cfg
        self.action_cfg = cfg.action_config
        self.task_cfg = self.action_cfg.get("task")
        self.aim_run = aim_run
        set_device(self.action_cfg.device)
