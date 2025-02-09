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
        data_files_path = Path(self.cfg.data_files)
        if not data_files_path.is_dir():
            raise TypeError("Path: '{self.cfg.data_files_path}' is not a valid directory")

        if self.cfg.file_extensions.startswith("json"):
            file_format = "json"
        elif self.cfg.file_extensions.startswith("txt"):
            file_format = "txt"
        else:
            raise KeyError(f"Files in '{self.cfg.file_extensions}' format are not supported")

        data_files = [
            str(file)
            for file in data_files_path.rglob("*.jsonl")
            if not file.name.startswith(".") or file.name.startswith("_")
        ]
        self.dataset = load_dataset(file_format, data_files=data_files)["train"]
