from typing import Tuple

from autoeval.common.device import DEVICE
from autoeval.common.model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PipelineModel(Model):
    def __init__(self, cfg, role) -> None:
        super().__init__(cfg, role)

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.name,
            cache_dir=self.cfg.cache_dir,
            device_map=DEVICE,
            torch_dtype=eval(self.cfg.dtype),
            token=self.cfg.api_token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)

        self.model = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=eval(self.cfg.dtype),
            device_map=DEVICE,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt: str, generate_cfg):
        output = self.model(prompt, **generate_cfg)
        output_first_resp = output[0]["generated_text"]

        return output_first_resp
