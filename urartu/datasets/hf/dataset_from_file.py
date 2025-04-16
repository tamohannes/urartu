from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from urartu.common.dataset import Dataset


class DatasetFromFile(Dataset):
    """
    A subclass of the Dataset class that specifically handles the creation of a dataset from files stored in a directory.
    This class utilizes the `datasets` library to load data from specified file formats and paths as part of the dataset configuration.

    Methods:
        _get_dataset: Overrides the base class method to instantiate the dataset from files according to the provided configuration.
    """

    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        """
        Initializes the DatasetFromFile object with configuration details.

        Args:
            cfg (List[Dict[str, Any]]): A list of dictionaries defining the configuration for the dataset,
                                        which must include a 'data_files' key with the path to the data files to be loaded.

        The constructor initializes the DatasetFromFile instance by calling the constructor of the base Dataset class.
        """
        super().__init__(cfg)

    def _get_dataset(self):
        """
        Instantiates the dataset by loading data from files specified in the configuration.

        Raises:
            TypeError: If the 'data_files' key is missing in the configuration, indicating that the path to the data files is not provided.
                       If the specified path is not a valid directory, an error is raised indicating the issue.
            KeyError: If the file format specified is not supported, an error is raised.

        This method verifies the presence of 'data_files' in the configuration and checks if the path is a valid directory.
        It determines the file format based on the 'file_extensions' key, supports 'json' and 'txt' formats, and loads the data accordingly.
        The dataset is then stored in the `self.dataset` attribute of the class.
        """
        if "data_files" not in self.cfg:
            raise TypeError("Argument 'data_files' is missing")
        data_files_path = Path(self.cfg.data_files)
        if not data_files_path.is_dir():
            raise TypeError(f"Path: '{self.cfg.data_files}' is not a valid directory")

        if self.cfg.file_extension.startswith("json"):
            file_format = "json"
        elif self.cfg.file_extension.startswith("txt"):
            file_format = "txt"
        else:
            raise KeyError(
                f"Files in '{self.cfg.file_extension}' format are not supported"
            )

        data_files = [
            str(file)
            for file in data_files_path.rglob("*." + self.cfg.file_extension)
            if not file.name.startswith(".") and not file.name.startswith("_")
        ]
        if "train_size" in self.cfg:
            dataset = load_dataset(file_format, data_files=data_files)["train"]
            train_size = int(self.cfg.train_size * len(dataset))

            self.dataset = dataset.train_test_split(
                train_size=train_size,
                seed=self.cfg.seed if "seed" in self.cfg else 42
            )
        else:
            self.dataset = load_dataset(file_format, data_files=data_files)
        return self.dataset
