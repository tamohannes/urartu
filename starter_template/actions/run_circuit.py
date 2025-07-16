""" import os
os.environ["HF_HOME"] = "/mnt/beegfs/work/truong/urartu/hf_cache" """
from aim import Run, Text
from omegaconf import DictConfig
from tqdm import tqdm

from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.model import Model

import gc

import torch
from disco_gp.circuit_lm import CircuitTransformer
from disco_gp.configs import Config
from pathlib import Path

class RunCircuit(Action):
    def _init_(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def setup_config(self):
        model_cfg = Config.from_tl('gpt2', dtype=torch.bfloat16, cache_dir=self.cfg.get("cache_dir"), use_auth_token=self.cfg.get("api_token"))
        weight_cfg = Config(**self.cfg.action_config.weight_hparams)
        edge_cfg = Config(**self.cfg.action_config.edge_hparams)
        task_cfg = Config(**self.cfg.action_config.task_cfg)
        exp_cfg = Config(**self.cfg.action_config.exp_cfg)
        circuit_cfg = Config.from_configs(
            weight = weight_cfg,
            edge = edge_cfg,
            task = task_cfg,
            model = model_cfg,
            exp = exp_cfg,
        )
        return circuit_cfg

    def main(self):
        circuit_cfg = self.setup_config()
        model = CircuitTransformer.from_pretrained(circuit_cfg)
        model.prepare_origin_output(model.dls.eval)
        model.cfg.weight.train_epochs = 1
        model.search_circuit()

def main(cfg: DictConfig, aim_run: Run):
    action = RunCircuit(cfg, aim_run)
    action.main()
