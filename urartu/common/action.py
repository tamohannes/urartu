from aim import Run
from omegaconf import DictConfig

from .device import Device


class Action:
    """
    A class to manage and configure actions based on a configuration and an Aim run session.

    This class initializes an action with specific configurations for task execution,
    sets up device configuration for the action, and ties the action to an Aim run session
    for tracking and logging purposes.

    Attributes:
        cfg (DictConfig): The full configuration object, typically containing all settings.
        action_cfg (DictConfig): A subset of the configuration specific to the action.
        task_cfg (dict): Configuration for the task derived from the action configuration.
        aim_run (Run): An Aim run instance for tracking and logging the execution of the action.
    """

    def __init__(self, cfg: DictConfig, aim_run: Run):
        """
        Initializes the Action object with the necessary configuration and Aim session.

        Args:
            cfg (DictConfig): The configuration object providing settings for various components of the action.
            aim_run (Run): The Aim run session to track and manage execution metrics and logs.

        Sets up the device configuration for the action by calling an external method set_device
        from the Device class with the device setting specified in the action configuration.
        """
        self.cfg = cfg
        self.action_cfg = cfg.action_config
        self.task_cfg = self.action_cfg.get("task")
        self.aim_run = aim_run
        Device.set_device(self.action_cfg.device)
