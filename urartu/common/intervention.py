from typing import Any, Dict, List

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformer_lens.components import (
    Embed,
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    LayerNorm,
    LayerNormPre,
    Unembed,
    GatedMLP,
    MLP,
    MoE,
    Attention
)


class Intervention(nn.Module):
    """
    A class provides a flexible interface for applying targeted interventions to a pretrained language model during inference or training. 
    This class is designed to manipulate model behavior in a controlled manner—such as modifying attention weights, injecting tokens, 
    or altering intermediate representations—to evaluate robustness, interpretability, or ethical behavior. It supports dynamic intervention
    points across different layers or components of the model and can be configured to operate conditionally based on input patterns or 
    predefined triggers. 
    Attributes:
        cfg (List[Dict[str, Any]]): Configuration list that defines parameters for model instantiation.
        _model (Any): Internal model object, initially set to None until instantiated.

    Methods:
        get_model: Static method to instantiate a model using Hydra's utility based on the configuration.
        model: Property that should provide access to the instantiated model object.
        generate: Method to perform generation based on a given input prompt.
    """

    def __init__(self, cfg: List[Dict[str, Any]], embed: Embed, blocks: nn.ModuleList):
        """
        Initializes the Model object with the necessary configuration.

        Args:
            cfg (List[Dict[str, Any]]): A list of dictionaries defining the configuration for the model.
        """
        self.cfg = cfg
        self.embed = embed
        self.blocks = blocks
        self._model = None
        super().__init__()
    
    @classmethod
    def from_pretrained(cls, cfg: Dict[str, Any], tokenizer:AutoTokenizer, dataloader: DataLoader):
        """
        Abstract method to load pretrained model based on the provided configuration.

        Args:
            cfg: (Dict[str, Any]): A dictionary containing configuration details necessary for model instantiation,
                                   including the type of model to instantiate.
            tokenizer: AutoTokenizer: A tokenizer object (typically from Hugging Face’s AutoTokenizer) used for 
                                    text preprocessing.

            dataloader: Dataloader: A dataloader object used for supplying data to the model during training or evaluation

        Raises:
            NotImplementedError: If the method is not implemented in a subclass, indicating that the model does not
                                 support generation functionality.
        """
        raise NotImplementedError("method 'from_pretrained' is not implemented")

    def forward(self, tokens, return_states=False):
        """
        Abstract method to performs a forward pass through the transformer model with optional support for different
        positional embeddings and edge-based output masking.

        Args:
            tokens (torch.Tensor): 
                Input token IDs of shape (batch_size, sequence_length). These are usually integer 
                indices from a vocabulary.
            return_states (bool, optional): 
                If True, returns the final hidden states (residual stream) instead of the logits. 
                Default is False.

        Returns:
            torch.Tensor:
                - If `return_states=True`, returns the final hidden states of shape 
                (batch_size, sequence_length, 1, d_model).
                - If `return_states=False`, returns a list containing the output logits tensor of shape 
                (batch_size, sequence_length, vocab_size).

        Raises:
            ValueError: If an unsupported `positional_embedding_type` is specified in the model configuration.

            NotImplementedError: If the method is not implemented in a subclass, indicating that the model does not
                                support generation functionality.
        """
        raise NotImplementedError("method 'forward' is not implemented")

    @torch.no_grad
    def evaluate(self, dl=None, reverse=False):
        """
        Abstract class to evaluate the model on a dataset with optional structural masking applied, and returns 
        accuracy and interpretability metrics such as KL divergence and faithfulness loss.

        Args:
            dl (torch.utils.data.DataLoader, optional): 
                The dataloader used for evaluation. If None, uses `self.dls.eval` by default.
            reverse (bool, optional): 
                If True, applies the reverse of the weight and edge masks during evaluation. 
                Default is False.

        Returns:
            dict: A dictionary containing the following evaluation metrics:
                - 'acc' (float): Accuracy of the masked model on the evaluation dataset.
                - 'kl' (float): Mean KL divergence between original and masked logits.
                - 'faith_loss' (float): Mean faithfulness loss, measuring how well the masked model 
                aligns with the unmasked model's outputs.
                - 'weight_density' (float or str): Proportion of active weights, or 'na' if weight 
                masks are disabled.
                - 'edge_density' (float or str): Proportion of active edges, or 'na' if edge masks 
                are disabled.
                - 'n_correct' (int): Number of correctly predicted samples.
                - 'total' (int): Total number of evaluation samples.
        Raises:
            NotImplementedError: If the method is not implemented in a subclass, indicating that the model does not
                                 support generation functionality.
        """
        raise NotImplementedError("method 'evaluate' is not implemented")



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
    
    def intervene(self):
        """
        Abstract method to apply a custom intervention function to a specified layer or component during the forward pass.

        This is typically used for mechanistic interpretability tasks, such as modifying 
        activations, patching internal states, or probing model behavior under controlled changes.
        Raises:
            NotImplementedError: If the method is not implemented in a subclass, indicating that the model does not
                                 support generation functionality.
        """
        raise NotImplementedError("method 'intervene' is not implemented")


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
