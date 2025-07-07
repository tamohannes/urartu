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

weight_hparams = Config(
    use_weight_masks=True,
    gs_temp_weight=0.01,
    logits_w_init=1.0,
    lr=0.1,
    lambda_sparse_init=1.0,
    lambda_complete_init=1.0,
    min_times_lambda_sparse=1.,
    max_times_lambda_sparse=1000.,

    train_epochs=300,
    n_epoch_warmup_lambda_sparse=500,
    n_epoch_cooldown_lambda_sparse=1,
)

edge_hparams = Config(
    use_edge_masks=True,
    gs_temp_edge=0.01,
    logits_e_init=1.0,
    lr=0.1,
    lambda_sparse_init=1.0,
    lambda_complete_init=1.0,
    min_times_lambda_sparse=1.,
    max_times_lambda_sparse=1000.,

    train_epochs=100,
    n_epoch_warmup_lambda_sparse=500,
    n_epoch_cooldown_lambda_sparse=1,
)

task_cfg = Config(
    # task_type='pararel',
    # task = 'P36',

    task_type='blimp',
    task = 'anaphor_number_agreement',
    batch_size=6,
)

exp_cfg = Config(
    evaluate_every=1,
)

model_cfg = Config.from_tl('gpt2', dtype=torch.bfloat16)

cfg = Config.from_configs(
    weight = weight_hparams,
    edge = edge_hparams,
    task = task_cfg,
    model = model_cfg,
    exp = exp_cfg,
)

class RunCircuit(Action):
    def _init_(self, cfg: DictConfig, aim_run: Run) -> None:
        model_cfg = Config.from_tl('gpt2', dtype=torch.bfloat16)
        cfg = Config.from_configs(
            weight = cfg.action_config.weight_hparams,
            edge = cfg.action_config.edge_hparams,
            task = cfg.action_config.task_cfg,
            model = model_cfg,
            exp = cfg.action_config.exp_cfg
        )
        super().__init__(cfg, aim_run)

    def main(self):
        model = CircuitTransformer.from_pretrained(cfg)
        model.prepare_origin_output(model.dls.eval)
        model.cfg.weight.train_epochs = 1
        model.search_circuit()

def main(cfg: DictConfig, aim_run: Run):
    action = RunCircuit(cfg, aim_run)
    action.main()
