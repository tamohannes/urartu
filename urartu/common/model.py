from typing import Any, Dict, List

from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoModelWithLMHead,
                          AutoTokenizer, pipeline)

from urartu.common.device import AUTO_DEVICE, DEVICE


class Model:
    @staticmethod
    def get_clm(cfg: List[Dict[str, Any]]) -> AutoModelWithLMHead:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.name,
            cache_dir=cfg.cache_dir,
            device_map=AUTO_DEVICE,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.name, padding_side="left")

        return model, tokenizer

    @staticmethod
    def get_pipe(cfg: List[Dict[str, Any]]) -> AutoModelWithLMHead:
        pipe = pipeline(
            "text-generation",
            model=cfg.name,
            torch_dtype=eval(cfg.dtype),
            device_map=AUTO_DEVICE,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.name)

        return pipe, tokenizer

    @staticmethod
    def get_seq2seq_model(cfg: List[Dict[str, Any]]) -> AutoModelWithLMHead:
        tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.name)
        model = model.to(DEVICE)

        return model, tokenizer

    @staticmethod
    def get_model(cfg: List[Dict[str, Any]]) -> AutoModelWithLMHead:
        model = AutoModelWithLMHead.from_pretrained(cfg.name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.name)
        return model, tokenizer

    @staticmethod
    def get_models(cfg: List[Dict[str, AutoModel]]) -> List[AutoModel]:
        models: List[AutoModel] = []
        for model_cfg in cfg:
            models.append(Model.get_model(model_cfg))
        return models

    @staticmethod
    def collate_tokenize(data, tokenizer, dataset_cfg):
        input_batch = []
        for element in data:
            if isinstance(element[dataset_cfg.input_key], list):
                input_text = " ".join(element[dataset_cfg.input_key])
            else:
                input_text = element[dataset_cfg.input_key]
            input_batch.append(input_text)
        tokenized = tokenizer(
            input_batch, padding="longest", truncation=True, return_tensors="pt"
        )
        tokenized.to(DEVICE)
        return tokenized
