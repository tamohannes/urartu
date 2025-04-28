from aim import Run
from omegaconf import DictConfig

from .device import Device
from urartu.utils.hash import dict_to_8char_hash


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


class ActionDataset(Action):
    """
    A specialized Action class for dataset-related operations.
    
    This class extends the base Action class with functionality specific to dataset handling.
    It automatically generates a unique hash for the dataset based on its name and configuration,
    and sets this information in the Aim run for tracking purposes.
    
    Attributes:
        Inherits all attributes from the Action class.
        
    Note:
        The dataset hash is created using the dataset name combined with an 8-character hash
        derived from the dataset configuration, ensuring uniqueness for tracking.
    """
    
    def __init__(self, cfg: DictConfig, aim_run: Run):
        """
        Initializes the ActionDataset object with the necessary configuration and Aim session.
        
        Args:
            cfg (DictConfig): The configuration object providing settings for various components of the action.
            aim_run (Run): The Aim run session to track and manage execution metrics and logs.
            
        Before initializing the parent Action class, this constructor:
        1. Generates a unique hash for the dataset configuration
        2. Adds this hash to the dataset configuration
        3. Sets the complete configuration in the Aim run
        """
        cfg.action_config.task.dataset["hash"] = f"{cfg.action_config.task.dataset.name}_{dict_to_8char_hash(cfg.action_config.task.dataset)}"
        aim_run.set("cfg", cfg, strict=False)
        super().__init__(cfg, aim_run)
