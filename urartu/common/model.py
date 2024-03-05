from typing import Any, Dict, List

from urartu.common.device import DEVICE


class Model:
    def __init__(self, cfg: List[Dict[str, Any]]):
        self.cfg = cfg
        self.aim_run = None
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        raise NotImplementedError("method '_load_model' is not implemented")

    def generate(self, prompt):
        raise NotImplementedError("method 'generate' is not implemented")


    @staticmethod
    def collate_tokenize(data, tokenizer, dataset_cfg):
        input_batch = []
        for element in data:
            if isinstance(element[dataset_cfg.input_key], list):
                input_text = " ".join(element[dataset_cfg.input_key])
            else:
                input_text = element[dataset_cfg.input_key]
            input_batch.append(input_text)
        tokenized = tokenizer(
            input_batch, padding="longest", truncation=True, return_tensors="pt"
        )
        tokenized.to(DEVICE)
        return tokenized
