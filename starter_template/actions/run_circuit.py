from aim import Run, Text
from omegaconf import DictConfig
from tqdm import tqdm
from urartu.utils.dtype import eval_dtype
import pandas as pd
from datetime import datetime
from argparse import Namespace

from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.configs import Config

from disco_gp.data import get_data_as_dict
from disco_gp.circuit_lm import CircuitTransformer
from transformers import AutoTokenizer
import optuna
import os


class RunCircuit(Action):
    def _init_(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def setup_config(self):
        model_cfg = Config.from_tl(self.cfg.action_config.task.model.name, dtype=eval_dtype(self.cfg.action_config.task.model.dtype))
        weight_cfg = Config(**self.cfg.action_config.weight_hparams)
        edge_cfg = Config(**self.cfg.action_config.edge_hparams)
        task_cfg = Config(**self.cfg.action_config.task.dataset)
        exp_cfg = Config(**self.cfg.action_config.exp_cfg)
        output_dir_path = Config(**{'run_dir': self.cfg.run_dir})
        circuit_cfg = Config.from_configs(
            weight = weight_cfg,
            edge = edge_cfg,
            task = task_cfg,
            model = model_cfg,
            exp = exp_cfg,
            output_dir_path = output_dir_path
        )
        #circuit_cfg['output_dir_path'] = self.cfg.run_dir
        return circuit_cfg
    
    def choose_best_model(self, epoch_results):
        df = pd.DataFrame(epoch_results)
        df.insert(0, 'epoch', range(0, len(df)))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_excel(os.path.join(self.cfg.run_dir, f'train_results_{timestamp}.xlsx'), index=False)
        df['composite_score'] = df['acc'] * (1- df['edge_density']) * (1 - df['weight_density'])
        best_epoch = df['composite_score'].idxmax()
        best_result = df.iloc[best_epoch]
        print(best_result)

    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def set_up_range_param(self, trial, param_name, param_range, log=False):
        min_value = param_range[0]
        max_value = param_range[1]
        return trial.suggest_float(param_name, min_value, max_value, log=log)
    
    def objective(self, trial):
        #weight_lr = trial.suggest_float("weight_lr", 0.4, 1, log=True)
        #weight_lambda_sparse_init = trial.suggest_float("weight_lambda_sparse_init", 0, 1)
        #weight_lambda_complete_init = trial.suggest_float("weight_lambda_complete_init", 1, 10)
        #edge_lr   = trial.suggest_float("edge_lr", 0.05, 0.5, log=True)
        #edge_lambda_sparse_init = trial.suggest_float("edge_lambda_sparse_init", 0, 1)
        #edge_lambda_complete_init = trial.suggest_float("edge_lambda_complete_init", 5, 20)
        circuit_cfg = self.setup_config()
        circuit_cfg.weight.lr = self.set_up_range_param(trial, "weight_lr", self.cfg.action_config.optuna.weight_lr, log = True)
        circuit_cfg.weight.lambda_sparse_init = self.set_up_range_param(trial, "weight_lambda_sparse_init", self.cfg.action_config.optuna.weight_lambda_sparse_init)
        circuit_cfg.weight.lambda_complete_init = self.set_up_range_param(trial, "weight_lambda_complete_init", self.cfg.action_config.optuna.weight_lambda_complete_init)
        circuit_cfg.edge.lr = self.set_up_range_param(trial, "edge_lr", self.cfg.action_config.optuna.edge_lr, log = True)
        circuit_cfg.edge.lambda_sparse_init = self.set_up_range_param(trial, "edge_lambda_sparse_init", self.cfg.action_config.optuna.edge_lambda_sparse_init)
        circuit_cfg.edge.lambda_complete_init = self.set_up_range_param(trial, "edge_lambda_complete_init", self.cfg.action_config.optuna.edge_lambda_complete_init)
        model = CircuitTransformer.from_pretrained(circuit_cfg)
        model.prepare_origin_output(model.dls.eval)
        results = model.search_circuit(trial = trial)
        result = results[-1]
        acc = result['acc']
        weight_density = result['weight_density']
        edge_density = result['edge_density']
        return acc * (1 - weight_density) * (1 - edge_density)
    
    def run_circuit(self):
        circuit_cfg = self.setup_config()
        print(f'weight_lr: {circuit_cfg.weight.lr}')
        print(f'edge_lr: {circuit_cfg.edge.lr}')
        model = CircuitTransformer.from_pretrained(circuit_cfg)
        model.prepare_origin_output(model.dls.eval)
        result = model.evaluate()
        print('Result after model evaluate:')
        print(result)
        epoch_results = model.search_circuit()
        self.choose_best_model(epoch_results)

    def search_hyperparameters(self):
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=100)
        study = optuna.create_study(directions=["maximize"], pruner=pruner)
        study.optimize(self.objective, n_trials=self.cfg.action_config.optuna.number_of_trial)
        print("Best Trial:")
        trial = study.best_trial
        print("  Value:", trial.value)
        print("  Params:", trial.params)


    def main(self):
        if self.cfg.action_config.optuna.use_optuna:
            print('Search hyperparameters')
            self.search_hyperparameters()
        else:
            print('Search circuit')
            self.run_circuit()
        

def main(cfg: DictConfig, aim_run: Run):
    action = RunCircuit(cfg, aim_run)
    action.main()

