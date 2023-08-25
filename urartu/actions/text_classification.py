from aim import Run
from omegaconf import DictConfig
from tqdm import tqdm

from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.metric import Metric
from urartu.common.model import Model


class TextClassifier(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def run(self):
        for task_cfg in self.action_cfg.tasks:
            dataset = Dataset.get_dataset(task_cfg.dataset)
            pipe, tokenizer = Model.get_pipe(task_cfg.model)
            metric = Metric.get_metric(task_cfg.metric)

            for idx, sample in tqdm(enumerate(dataset)):
                premise = sample["text"]
                hypothesis = {"negative": 0, "positive": 1}
                output = pipe(premise, list(hypothesis.keys()))

                model_prediction = output["scores"].index(max(output["scores"]))
                label = dataset["label"][idx]

                metric.add(predictions=model_prediction, references=label)

            final_score = metric.compute()
            self.aim_run.track(
                {"final_score": final_score},
                context={"subset": task_cfg.dataset.get("subset")},
                step=idx,
            )


def main(cfg: DictConfig, aim_run: Run):
    text_classifier = TextClassifier(cfg, aim_run)
    text_classifier.run()
