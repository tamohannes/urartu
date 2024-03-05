from typing import Any, Dict, List

from torch.utils.data import DataLoader

from urartu.common.model import Model


class Dataset:
    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        self.cfg = cfg
        self.dataset = None
        self._get_dataset()

    def _get_dataset(self):
        raise NotImplementedError("method '_get_dataset' is not implemented")

    def get_dataloader(
        self,
        tokenizer,
        dataloader_cfg: List[Dict[str, Any]],
    ):
        return DataLoader(
            self.dataset,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            shuffle=dataloader_cfg.shuffle,
            collate_fn=lambda data: Model.collate_tokenize(data, tokenizer, self.cfg),
        )
