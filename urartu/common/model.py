from typing import Any, Dict, List

import hydra
from transformers import AutoModelForCausalLM


class Model:
    """
    A class for handling model instantiation and operations within a machine learning framework.
    This class uses configuration-driven instantiation and requires implementation of model-specific
    behavior such as property access and generation methods.

    Attributes:
        cfg (List[Dict[str, Any]]): Configuration list that defines parameters for model instantiation.
        _model (Any): Internal model object, initially set to None until instantiated.

    Methods:
        get_model: Static method to instantiate a model using Hydra's utility based on the configuration.
        model: Property that should provide access to the instantiated model object.
        generate: Method to perform generation based on a given input prompt.
    """

    def __init__(self, cfg: List[Dict[str, Any]]):
        """
        Initializes the Model object with the necessary configuration.

        Args:
            cfg (List[Dict[str, Any]]): A list of dictionaries defining the configuration for the model.
        """
        self.cfg = cfg
        self._model = None

    @staticmethod
    def get_model(cfg):
        """
        Instantiates and returns a model based on the provided configuration using Hydra's instantiation utilities.

        Args:
            cfg (Dict[str, Any]): A dictionary containing configuration details necessary for model instantiation,
                                  including the type of model to instantiate.

        Returns:
            Any: The instantiated model, which type depends on the configuration specifics.
        """
        return hydra.utils.instantiate(cfg.type, cfg)

    @property
    def model(self):
        """
        Property to access the instantiated model.
        """
        return self._model

    @model.setter
    def model(self, value: Any):
        """
        Setter for the model property.

        Args:
            value (Any): The model instance to be set.
        """
        self._model = value

    def generate(self, prompt):
        """
        Abstract method to generate output based on the provided prompt.

        Args:
            prompt: Input prompt to which the model should respond.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass, indicating that the model does not
                                 support generation functionality.
        """
        raise NotImplementedError("method 'generate' is not implemented")
