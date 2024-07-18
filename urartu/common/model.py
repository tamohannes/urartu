from typing import Any, Dict, List

import hydra

from .device import Device


class Model:
    def __init__(self, cfg: List[Dict[str, Any]]):
        self.cfg = cfg
        self._model = None

    @staticmethod
    def get_model(cfg):
        return hydra.utils.instantiate(cfg.type, cfg)

    @property
    def model(self):
        raise NotImplementedError("property 'model' instantiation is not implemented")

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
        ).to(Device.get_device())
        return tokenized
