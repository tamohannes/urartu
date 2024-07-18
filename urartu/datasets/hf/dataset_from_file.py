from typing import Any, Dict, List

from datasets import load_dataset
from pathlib import Path

from urartu.common.dataset import Dataset


class DatasetFromFile(Dataset):
    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        super().__init__(cfg)

    def _get_dataset(self):
        if "data_files" not in self.cfg:
            raise TypeError("Argument 'data_files' is missing")
        file_extension = Path(self.cfg.data_files).suffix
        if file_extension.startswith(".json"):
            file_format = "json"
        elif file_extension == ".txt":
            file_format = "text"
        else:
            raise KeyError(f"Failed to load data file '{file_extension}'")
        self.dataset = load_dataset(file_format, data_files=self.cfg.data_files)["train"]
