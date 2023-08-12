from aim import Run
from omegaconf import DictConfig

from urartu.common.action import Action


class Example(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def main(self):
        pass


def main(cfg: DictConfig, aim_run: Run):
    evaluator = Example(cfg, aim_run)
    evaluator.main()
