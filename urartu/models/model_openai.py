from typing import Tuple, Union

from urartu.common.model import Model


class ModelOpenAI(Model):
    """
    A class for interacting with an OpenAI model deployed through Azure, facilitating
    the initialization, configuration, and use of the model for text generation.

    Attributes:
        cfg: A configuration object containing model deployment details and API keys for Azure.

    Methods:
        model: Returns an instance of the model ready for interaction.
        generate: Generates responses based on input prompts.
        _get_num_tokens: Calculates the number of tokens in a given string based on the specified model encoding.
    """

    def __init__(self, cfg) -> None:
        """
        Initializes the ModelOpenAI class by setting up the base model configuration.

        Args:
            cfg: Configuration object containing parameters such as the name of the Azure deployment,
                 API types, API versions, endpoint, and API keys.
        """
        super().__init__(cfg)
        self._tiktoken = None
        self._AzureChatOpenAI = None

    @property
    def model(self):
        """
        Lazily imports and creates an instance of AzureChatOpenAI
        """
        if self._model is None:
            from langchain_openai import AzureChatOpenAI

            self._model = AzureChatOpenAI(
                deployment_name=self.cfg.name,
                openai_api_type=self.cfg.openai_api_type,
                openai_api_version=self.cfg.openai_api_version,
                azure_endpoint=self.cfg.azure_openai_endpoint,
                openai_api_key=self.cfg.azure_openai_api_key,
            )
        return self._model

    def generate(self, prompt: Union[str, Tuple[str, str]], generate_cfg):
        """
        Generates a response from the model based on the provided prompt and generation configurations.

        Args:
            prompt: A string or a tuple containing the prompt(s) to be sent to the model.
            generate_cfg: Configuration parameters specifying how the model should generate the response.

        Returns:
            A string containing the generated response from the model.
        """
        # Import only when needed
        from langchain.schema import HumanMessage

        output = self.model(HumanMessage(content=prompt))
        return output

    def _get_num_tokens(self, string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
        """
        Computes the number of tokens that the specified string will be broken into, using a particular model encoding.

        Args:
            string: The text string to be tokenized.
            encoding_name: The name of the encoding model to be used (default is "gpt-3.5-turbo").

        Returns:
            An integer representing the number of tokens.
        """
        # Import only when needed
        import tiktoken

        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
