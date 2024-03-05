from typing import Tuple

import torch
from autoeval.common.device import AUTO_DEVICE
from autoeval.common.model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer


class CausalLMModel(Model):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.name,
            cache_dir=self.cfg.cache_dir,
            device_map=AUTO_DEVICE,
            torch_dtype=eval(self.cfg.dtype),
            token=self.cfg.api_token,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        self.model.eval()

    def generate(self, prompt: str, generate_cfg):
        self.model.eval()
        prompt_tokenized = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            output_tokenized = self.model.generate(prompt_tokenized, **generate_cfg)
        output = self.tokenizer.decode(output_tokenized[0], skip_special_tokens=True)

        return output
