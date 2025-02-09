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
