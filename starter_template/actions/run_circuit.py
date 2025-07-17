from aim import Run, Text
from omegaconf import DictConfig
from tqdm import tqdm
from urartu.utils.dtype import eval_dtype
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.model import Model
from urartu.common.configs import Config

import gc
from disco_gp.circuit_lm import CircuitTransformer


class RunCircuit(Action):
    def _init_(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def setup_config(self):
        model_cfg = Config.from_tl(self.cfg.action_config.task.model.name, dtype=eval_dtype(self.cfg.action_config.task.model.dtype))
        weight_cfg = Config(**self.cfg.action_config.weight_hparams)
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
    
    def choose_best_model(self, epoch_results):
        df = pd.DataFrame(epoch_results)
        df.insert(0, 'epoch', range(0, len(df)))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.to_excel(f'/mnt/beegfs/work/truong/urartu/train_results_{timestamp}.xlsx', index=False)
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(scaler.fit_transform(df[['acc', 'edge_density', 'weight_density']]),
                          columns=['acc', 'edge_density', 'weight_density'])
        normalized['acc_score'] = normalized['acc']  # maximize
        normalized['edge_score'] = 1 - normalized['edge_density']  # minimize
        normalized['weight_score'] = 1 - normalized['weight_density']  # minimize
        normalized['composite_score'] = (
            0.5 * normalized['acc_score'] +
            0.25 * normalized['edge_score'] +
            0.25 * normalized['weight_score']
        )
        best_epoch = normalized['composite_score'].idxmax()
        best_result = df.iloc[best_epoch]
        print(best_result)



    def main(self):
        circuit_cfg = self.setup_config()
        model = CircuitTransformer.from_pretrained(circuit_cfg)
        model.prepare_origin_output(model.dls.eval)
        epoch_results = model.search_circuit()
        self.choose_best_model(epoch_results)

def main(cfg: DictConfig, aim_run: Run):
    action = RunCircuit(cfg, aim_run)
    action.main()
