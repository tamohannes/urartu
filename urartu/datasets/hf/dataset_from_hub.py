from typing import Any, Dict, List

from datasets import load_dataset

from urartu.common.dataset import Dataset


class DatasetFromHub(Dataset):
    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        super().__init__(cfg)

    def _get_dataset(self):
        if "name" not in self.cfg:
            raise TypeError("Argument 'name' is missing")
        self.dataset = load_dataset(
            self.cfg.name,
            self.cfg.get("subset"),
            split=self.cfg.get("split"),
        )
