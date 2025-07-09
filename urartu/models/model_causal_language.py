import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urartu.common.device import Device
from urartu.common.model import Model
from urartu.utils.dtype import eval_dtype


class ModelForCausalLM(Model):
    """
    A class for working with a causal language model (CLM) that handles model initialization,
    configuration, and text generation using the model and tokenizer from the Hugging Face
    Transformers library.

    Attributes:
        cfg: An object containing configuration settings such as model name, cache directory,
             API token, and other model-specific settings.

    Methods:
        model: Returns a causal language model with preloaded configurations.
        tokenizer: Returns a tokenizer associated with the causal language model.
        generate: Generates text based on a given prompt and optional generation settings.
    """

    def __init__(self, cfg) -> None:
        """
        Initializes the ModelCausalLanguage class by setting up the base model configuration.

        Args:
            cfg: Configuration object containing necessary parameters like model name,
                 cache directory, device mapping, dtype evaluation, and optional API token.
        """
        super().__init__(cfg)
        self._tokenizer = None

    @property
    def model(self) -> AutoModelForCausalLM:
        """
        Retrieves or instantiates the causal language model specified in the configuration.
        The model is set to evaluation mode with gradients disabled.

        Returns:
            An instance of AutoModelForCausalLM ready for inference.
        """
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name,
                cache_dir=self.cfg.get("cache_dir"),
                device_map=Device.get_device(),
                torch_dtype=(
                    eval_dtype(self.cfg.get("dtype"))
                    if self.cfg.get("dtype") is not None
                    else None
                ),
                token=self.cfg.get("api_token"),
                trust_remote_code=self.cfg.get("trust_remote_code"),
                revision=self.cfg.get("revision"),
            )
            for param in self._model.parameters():
                param.requires_grad = False
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        """
        Retrieves or instantiates the tokenizer associated with the causal language model,
        setting the pad token to be the same as the eos token if not already set.

        Returns:
            An instance of AutoTokenizer.
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def generate(self, prompt: str, generate_cfg=None):
        """
        Generates text based on a provided prompt and optional generation settings.

        Args:
            prompt: A string containing the initial text to prompt the model with.
            generate_cfg: An optional dictionary containing generation parameters such as
                          maximum length, number of beams, etc. Defaults to configurations
                          specified in the model's config.

        Returns:
            A string of generated text. If 'output_scores' is included in the generation
            configuration, it returns a tuple of the generated text and the corresponding
            generation scores.
        """
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
        if "output_scores" in generate_cfg:
            output = self.tokenizer.decode(
                output_tokenized["sequences"][0], skip_special_tokens=True
            )
            scores = output_tokenized["scores"]
            if isinstance(scores, (list, tuple)):
                scores = torch.stack(scores)  # Convert list of tensors to single tensor
            return output, scores
        else:
            return self.tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
