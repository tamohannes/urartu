from aim import Run, Text
from omegaconf import DictConfig
from tqdm import tqdm

from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.model import Model


class Generate(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def main(self):
        model = Model.get_model(self.task_cfg.model)
        dataset = Dataset.get_dataset(self.task_cfg.dataset)

        for idx, sample in tqdm(enumerate(dataset.dataset)):
            prompt = sample[self.task_cfg.dataset.get("input_key")]
            self.aim_run.track(Text(prompt), name="input")

            output = model.generate(prompt)
            self.aim_run.track(Text(output), name="output")


def main(cfg: DictConfig, aim_run: Run):
    action = Generate(cfg, aim_run)
    action.main()
