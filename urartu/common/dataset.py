import logging
from typing import Any, Dict, List

import hydra
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.WARNING)


class Dataset:
    """
    A class for managing datasets within a machine learning context, which supports the dynamic instantiation
    of datasets and configuration of dataloaders with custom collation functions based on tokenizer constraints.

    Attributes:
        cfg (List[Dict[str, Any]]): Configuration list that defines parameters for datasets.
        dataset (Any): Holds the dataset object, which can be of any type depending on the instantiation.

    Methods:
        _get_dataset: Abstract method to be implemented by subclasses to instantiate the dataset.
        get_dataloader: Creates a DataLoader instance configured according to specified parameters and optional tokenizer adjustments.
    """

    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        """
        Initializes the Dataset object with the necessary configuration.

        Args:
            cfg (List[Dict[str, Any]]): A list of dictionaries defining the configuration for the dataset.

        The constructor initializes the dataset attribute to None and calls the `_get_dataset()` method,
        which should be implemented by subclasses to specify how the dataset is instantiated.
        """
        self.cfg = cfg
        self.dataset = None
        self._get_dataset()

    @staticmethod
    def get_dataset(cfg):
        """
        Static method to instantiate a dataset using the configuration provided.

        Args:
            cfg (Dict[str, Any]): A dictionary configuration for creating the dataset.

        Returns:
            An instantiated dataset object as defined by the configuration.

        This method utilizes Hydra's instantiation utility to create a dataset object based on a type
        specification and additional parameters provided in the configuration dictionary.
        """
        return hydra.utils.instantiate(cfg.type, cfg)

    def _get_dataset(self):
        """
        Abstract method to instantiate the dataset based on internal configuration.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("method '_get_dataset' is not implemented")

    def get_dataloader(self, dataloader_cfg: Dict[str, Any], tokenizer, return_attrs: bool = False):
        """
        Creates and returns a DataLoader for the dataset, with optional tokenization and attribute inclusion.

        Args:
            dataloader_cfg (Dict[str, Any]): Configuration dictionary for the DataLoader.
            tokenizer: Tokenizer object used for tokenizing input data.
            return_attrs (bool): Flag to indicate whether to return additional attributes in the output batches.

        Returns:
            DataLoader: A DataLoader object configured according to the provided specifications.

        The method defines a collate function that adjusts batch tokenization according to the tokenizer's
        maximum input size, optionally includes additional attributes, and handles warnings for inputs
        exceeding tokenizer limits.
        """

        def collate_fn(examples):
            max_input_size = max(len(example[dataloader_cfg["input_key"]]) for example in examples)
            max_length = min(max_input_size, tokenizer.model_max_length)

            for example in examples:
                example_length = len(example[dataloader_cfg["input_key"]])
                if example_length > tokenizer.model_max_length:
                    logging.warning(f"Example input length {example_length} exceeds tokenizer.model_max_length {tokenizer.model_max_length}.")

            tokenized = tokenizer(
                [example[dataloader_cfg["input_key"]] for example in examples],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

            if return_attrs:
                for k in examples[0].keys():
                    if k not in tokenized:
                        tokenized[k] = [example[k] for example in examples]

            return tokenized

        return {
            split_name: DataLoader(
                self.dataset[split_name],
                shuffle=dataloader_cfg.get("shuffle", False),
                batch_size=dataloader_cfg.get("batch_size", 8),
                num_workers=dataloader_cfg.get("num_workers", 2),
                pin_memory=True,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
            for split_name in self.dataset.keys()
        }
