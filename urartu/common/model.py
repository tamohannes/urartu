from typing import Any, Dict, List

import hydra

from .device import DEVICE


class Model:
    def __init__(self, cfg: List[Dict[str, Any]]):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self._load_model()

    @staticmethod
    def get_model(cfg):
        return hydra.utils.instantiate(cfg.type, cfg)

    def _load_model(self):
        raise NotImplementedError("method '_load_model' is not implemented")

    # def __call__(self, args):
    #     return self.model(**args)

    def generate(self, prompt):
        raise NotImplementedError("method 'generate' is not implemented")

    @staticmethod
    def collate_tokenize(data, tokenizer, input_key):
        input_batch = []
        for element in data:
            if isinstance(element[input_key], list):
                input_text = " ".join(element[input_key])
            else:
                input_text = element[input_key]
            input_batch.append(input_text)
        tokenized = tokenizer(
            input_batch, padding="longest", truncation=True, return_tensors="pt"
        ).to(DEVICE)
        return tokenized
