from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..common.device import DEVICE
from ..common.model import Model


class PipelineModel(Model):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.name,
            cache_dir=self.cfg.get("cache_dir"),
            device_map=DEVICE,
            torch_dtype=eval(self.cfg.get("dtype")),
            token=self.cfg.get("api_token"),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)

        self.model = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=eval(self.cfg.get("dtype")),
            device_map=DEVICE,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt: str, generate_cfg):
        output = self.model(prompt, **generate_cfg)
        output_first_resp = output[0]["generated_text"]

        return output_first_resp
