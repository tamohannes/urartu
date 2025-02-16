from typing import Any, Dict, List

from datasets import Dataset as HFDataset

from urartu.common.dataset import Dataset


class DatasetFromDict(Dataset):
    """
    A subclass of the Dataset class that specifically handles the creation of a dataset from a dictionary.
    This class leverages the Hugging Face datasets library to convert a dictionary into a dataset suitable
    for use in machine learning and NLP tasks.

    Methods:
        _get_dataset: Overrides the base class method to instantiate the dataset from a dictionary provided in the configuration.
    """

    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        """
        Initializes the DatasetFromDict object with configuration details.

        Args:
            cfg (List[Dict[str, Any]]): A list of dictionaries defining the configuration for the dataset,
                                        which must include a 'data' key with the actual data to be converted into the dataset.

        The constructor initializes the DatasetFromDict instance by calling the constructor of the base Dataset class.
        """
        super().__init__(cfg)

    def _get_dataset(self):
        """
        Instantiates the dataset from the dictionary provided in the configuration.

        Raises:
            TypeError: If the 'data' key is missing in the configuration, indicating that the required data for dataset creation is absent.

        This method checks for the presence of the 'data' key in the configuration dictionary, and uses the Hugging Face
        'datasets.Dataset.from_dict()' method to convert the dictionary into a dataset. The resulting dataset is then
        stored in the `self.dataset` attribute of the class.
        """
        if "data" not in self.cfg:
            raise TypeError("Argument 'data' is missing")

        self.dataset = HFDataset.from_dict(dict(self.cfg.data))
