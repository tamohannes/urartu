import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urartu.common.device import Device
from urartu.common.model import Model
from urartu.utils.dtype import eval_dtype


class ModelCausalLanguage(Model):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._tokenizer = None

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name,
                cache_dir=self.cfg.cache_dir,
                device_map=Device.get_device(),
                torch_dtype=eval_dtype(self.cfg.dtype),
                token=self.cfg.api_token,
                trust_remote_code=self.cfg.get("trust_remote_code"),
            )

            for param in self._model.parameters():
                param.requires_grad = False
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def generate(self, prompt: str, generate_cfg=None):
        if not generate_cfg:
            generate_cfg = self.cfg.get("generate")
        self.model.eval()

        prompt_tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        prompt_tensor = prompt_tokenized["input_ids"].to(self.model.device)
        attention_mask = prompt_tokenized["attention_mask"].to(self.model.device)
        with torch.no_grad():
            output_tokenized = self.model.generate(
                input_ids=prompt_tensor, attention_mask=attention_mask, **generate_cfg
            )
        if "output_scores" in generate_cfg:
            output = self.tokenizer.decode(output_tokenized["sequences"][0], skip_special_tokens=True)
            return output, output_tokenized['scores']
        else:
            return self.tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
