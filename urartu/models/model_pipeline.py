from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from urartu.common.device import Device
from urartu.common.model import Model
from urartu.utils.dtype import eval_dtype


class ModelPipeline(Model):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            clm_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name,
                cache_dir=self.cfg.cache_dir,
                device_map=Device.get_device(),
                torch_dtype=eval_dtype(self.cfg.dtype),
                token=self.cfg.api_token,
            )

            self._model = pipeline(
                "text-generation",
                model=clm_model,
                tokenizer=self.tokenizer,
                torch_dtype=eval_dtype(self.cfg.dtype),
                device_map=Device.get_device(),
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        return self._tokenizer

    def generate(self, prompt: str, generate_cfg):
        output = self.model(prompt, **generate_cfg)
        output_first_resp = output[0]["generated_text"]

        return output_first_resp
