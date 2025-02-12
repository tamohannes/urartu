from typing import Any, Dict, List

import hydra
from torch.utils.data import DataLoader
import logging

from transformers import DataCollatorWithPadding

# Configure logging
logging.basicConfig(level=logging.WARNING)


class Dataset:
    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        self.cfg = cfg
        self.dataset = None
        self._get_dataset()

    @staticmethod
    def get_dataset(cfg):
        return hydra.utils.instantiate(cfg.type, cfg)

    def _get_dataset(self):
        raise NotImplementedError("method '_get_dataset' is not implemented")

    def get_dataloader(self, dataloader_cfg: Dict[str, Any], tokenizer):
        def collate_fn(examples):
            max_input_size = max(len(example[dataloader_cfg["input_key"]]) for example in examples)
            max_length = min(max_input_size, tokenizer.model_max_length)

            for example in examples:
                example_length = len(example[dataloader_cfg["input_key"]])
                if example_length > tokenizer.model_max_length:
                    logging.warning(
                        f"Example input length {example_length} exceeds tokenizer.model_max_length {tokenizer.model_max_length}."
                    )

            tokenized = tokenizer(
                [example[dataloader_cfg["input_key"]] for example in examples],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

            for k in examples[0].keys():
                if k not in tokenized:
                    tokenized[k] = [example[k] for example in examples]

            return tokenized

        return DataLoader(
            self.dataset,
            shuffle=dataloader_cfg.get("shuffle", False),
            batch_size=dataloader_cfg.get("batch_size", 8),
            num_workers=dataloader_cfg.get("num_workers", 2),
            pin_memory=True,
            collate_fn=collate_fn,
        )
