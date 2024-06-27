from typing import Tuple, Union

import tiktoken
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..common.model import Model


class OpenAIModel(Model):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.model = AzureChatOpenAI(
            deployment_name=self.cfg.name,
            openai_api_type=self.cfg.openai_api_type,
            openai_api_version=self.cfg.openai_api_version,
            azure_endpoint=self.cfg.azure_openai_endpoint,
            openai_api_key=self.cfg.azure_openai_api_key,
        )

    def generate(self, prompt: Union[str, Tuple[str, str]], generate_cfg):
        output = self.model(HumanMessage(content=prompt))

        return output

    def _get_num_tokens(self, string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
