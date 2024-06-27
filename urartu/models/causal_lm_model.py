from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..common.device import DEVICE
from ..common.model import Model


class CausalLMModel(Model):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _load_model(self) -> AutoModelForCausalLM:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.name,
            cache_dir=self.cfg.get("cache_dir"),
            device_map=DEVICE,
            torch_dtype=eval(self.cfg.get("dtype")),
            token=self.cfg.get("api_token"),
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, generate_cfg=None):
        if not generate_cfg:
            generate_cfg = self.cfg.get("generate")
        self.model.eval()

        prompt_tokenized = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        prompt_tensor = prompt_tokenized["input_ids"].to(self.model.device)
        attention_mask = prompt_tokenized["attention_mask"].to(self.model.device)
        with torch.no_grad():
            output_tokenized = self.model.generate(
                input_ids=prompt_tensor, attention_mask=attention_mask, **generate_cfg
            )
        output = self.tokenizer.decode(output_tokenized[0], skip_special_tokens=True)

        return output
