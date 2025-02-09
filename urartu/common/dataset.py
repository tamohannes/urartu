from typing import Any, Dict, List

import hydra
from torch.utils.data import DataLoader

from transformers import DataCollatorWithPadding


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
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples[dataloader_cfg["input_key"]],
                truncation=True,
                padding=False,
                max_length=tokenizer.model_max_length,
                return_tensors=None,
            )
            return {**examples, **tokenized}

        def collate_fn(examples):
            tokenizer_inputs = {k: [example[k] for example in examples] for k in ["input_ids", "attention_mask"]}
            padded = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")(tokenizer_inputs)

            for k in examples[0].keys():
                if k not in padded:
                    padded[k] = [example[k] for example in examples]

            return padded

        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)

        return DataLoader(
            tokenized_datasets,
            shuffle=dataloader_cfg.get("shuffle", False),
            batch_size=dataloader_cfg.get("batch_size", 8),
            num_workers=dataloader_cfg.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
        )
