from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from urartu.common.device import Device
from urartu.common.model import Model
from urartu.utils.dtype import eval_dtype


class ModelPipeline(Model):
    """
    A class for configuring and utilizing a text generation pipeline based on a causal
    language model (CLM) from the Hugging Face Transformers library. This class simplifies
    the initialization and application of the model and tokenizer within a pipeline
    for text generation.

    Attributes:
        cfg: Configuration object containing model details such as model name, cache directory,
             device mapping, data type, and API token for model retrieval.

    Methods:
        model: Returns a configured text generation pipeline using the specified CLM and tokenizer.
        tokenizer: Returns a tokenizer compatible with the CLM.
        generate: Generates text based on a given prompt and generation settings.
    """

    def __init__(self, cfg) -> None:
        """
        Initializes the ModelPipeline class with configuration settings for deploying a
        text generation model.

        Args:
            cfg: Configuration object with necessary settings for model deployment.
        """
        super().__init__(cfg)
        self._tokenizer = None

    @property
    def model(self):
        """
        Retrieves or creates a text generation pipeline equipped with a causal language model
        and a tokenizer, configured as specified in the configuration object. The model is set
        to run on the appropriate device with specified data types and any necessary tokens.

        Returns:
            A Hugging Face pipeline object for text generation.
        """
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
        """
        Retrieves or initializes the tokenizer associated with the causal language model,
        necessary for correctly formatting input prompts for the pipeline.

        Returns:
            An instance of AutoTokenizer.
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        return self._tokenizer

    def generate(self, prompt: str, generate_cfg):
        """
        Generates text based on the provided prompt and additional generation configurations
        using the configured text generation pipeline.

        Args:
            prompt: A string containing the text to prompt the model.
            generate_cfg: A dictionary of generation settings such as max_length, num_return_sequences, etc.

        Returns:
            A string containing the generated text as the first response from the pipeline.
        """
        output = self.model(prompt, **generate_cfg)
        output_first_resp = output[0]["generated_text"]

        return output_first_resp
