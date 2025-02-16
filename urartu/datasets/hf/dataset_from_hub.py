from typing import Any, Dict, List

from datasets import load_dataset

from urartu.common.dataset import Dataset


class DatasetFromHub(Dataset):
    """
    A subclass of the Dataset class that facilitates loading datasets directly from the Hugging Face Hub.
    This class is tailored to load datasets by name, optionally specifying subsets and splits via a configuration object.

    Methods:
        _get_dataset: Overrides the base class method to instantiate the dataset from the Hugging Face Hub based on the provided configuration.
    """

    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        """
        Initializes the DatasetFromHub object with configuration details.

        Args:
            cfg (List[Dict[str, Any]]): A list of dictionaries defining the configuration for the dataset,
                                        which must include a 'name' key specifying the dataset to load from the Hugging Face Hub.

        The constructor initializes the DatasetFromHub instance by calling the constructor of the base Dataset class.
        """
        super().__init__(cfg)

    def _get_dataset(self):
        """
        Instantiates the dataset by loading it from the Hugging Face Hub based on the provided configuration.

        Raises:
            TypeError: If the 'name' key is missing in the configuration, indicating that the dataset name to be loaded is not specified.

        This method checks for the 'name' key in the configuration and uses it along with optional 'subset' and 'split'
        parameters to load the dataset from the Hugging Face Hub. The loaded dataset is then stored in the `self.dataset` attribute.
        """
        if "name" not in self.cfg:
            raise TypeError("Argument 'name' is missing")
        self.dataset = load_dataset(
            self.cfg.name,
            self.cfg.get("subset"),
            split=self.cfg.get("split"),
        )
