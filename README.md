<!--- BADGES: START --->

[![PyPI - Package Version](https://img.shields.io/pypi/v/urartu?logo=pypi&style=flat&color=orange)](https://pypi.org/project/urartu/)
[![PyPI - Python Version](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub - License](https://img.shields.io/github/license/tamohannes/urartu)](https://opensource.org/licenses/Apache-2.0)

<!--- BADGES: END --->

# Urartu

Urartu is an ML workflow runner built around **Pipelines** (orchestrators) and **Actions** (reusable steps) with **automatic caching** and **dependency injection**.

## Installation

```bash
pip install urartu
```

From source:

```bash
git clone git@github.com:tamohannes/urartu.git
cd urartu
pip install -e .
```

## Project layout (recommended)

Run the CLI from a project root that contains:

```
my_project/
├── __init__.py
├── actions/
│   ├── __init__.py
│   └── my_action.py
├── pipelines/
│   ├── __init__.py
│   └── my_pipeline.py
└── configs/
    └── pipeline/
        └── my_pipeline.yaml
```

Optional (per-user configs):

```
my_project/
└── configs_<username>/
    ├── aim/
    ├── machine/
    └── slurm/
```

## Quickstart

Create a pipeline config:

```yaml
# configs/pipeline/my_pipeline.yaml
pipeline_name: my_pipeline
debug: false

pipeline:
  experiment_name: "My pipeline"
  device: auto
  seed: 42

  # Pipeline-level cache policy (propagates to actions)
  cache_enabled: true
  force_rerun: false
  cache_max_age_days: 7

  actions:
    - action_name: my_action
      # Action-specific config (merged with pipeline-level common settings)
      some_param: 123
```

Create the pipeline file:

```python
from aim import Run
from omegaconf import DictConfig
from urartu.common import Pipeline


class MyPipeline(Pipeline):
    pass


def main(cfg: DictConfig, aim_run: Run):
    MyPipeline(cfg, aim_run).main()
```

Create an action:

```python
from omegaconf import DictConfig
from aim import Run
from urartu.common import Action


class MyAction(Action):
    def run(self):
        cache_dir = self.get_cache_entry_dir()
        run_dir = self.get_run_dir()
        # ... compute, write machine-readable artifacts to cache_dir ...
        # ... write plots/reports to run_dir ...

    def get_outputs(self):
        return {
            "cache_dir": str(self.get_cache_entry_dir()),
            "run_dir": str(self.get_run_dir()),
        }
```

Run it:

```bash
urartu my_pipeline
```

## CLI overrides and config groups

- **Overrides**: `pipeline.seed=123`, `pipeline.device=cuda`, `descr="my run"`.
- **Config-group selectors** (unquoted) load `*.yaml` files from:
  - `configs_<username>/<group>/<selector>.yaml`
  - `configs/<group>/<selector>.yaml`
  - built-in defaults in the Urartu package

Examples:

```bash
# Select config files (unquoted values)
urartu my_pipeline machine=local slurm=no_slurm aim=no_aim

# Set literal strings (quoted values)
urartu my_pipeline descr="experiment 001" machine="local"
```

## Notes on outputs and caching

- **Cached, machine-readable artifacts**: write under `self.get_cache_entry_dir(...)` (shared across runs).
- **Run artifacts (plots/reports/logs)**: write under `self.get_run_dir(...)` (unique per run).

## Citation

If you find Urartu helpful in your research, please cite it:

```bibtex
@software{Tamoyan_Urartu_2023,
  author = {Hovhannes Tamoyan},
  license = {Apache-2.0},
  month = {8},
  title = {{Urartu}},
  url = {https://github.com/tamohannes/urartu},
  year = {2023}
}
```
