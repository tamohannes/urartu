<!--- BADGES: START --->

[![PyPI - Package Version](https://img.shields.io/pypi/v/urartu?logo=pypi&style=flat&color=orange)](https://pypi.org/project/urartu/)
[![PyPI - Python Version](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub - License](https://img.shields.io/github/license/tamohannes/urartu)](https://opensource.org/licenses/Apache-2.0)

<!--- BADGES: END --->

# **Urartu 🦁**

**The intelligent ML Pipeline Framework that chains actions into powerful workflows!**

Urartu is a framework for building machine learning workflows by chaining **Actions** into **Pipelines**. Each Action is a self-contained, reusable component with built-in caching, and Pipelines orchestrate multiple Actions with automatic data flow.

# **Installation**

```bash
pip install urartu
```

Or from source:
```bash
git clone git@github.com:tamohannes/urartu.git
cd urartu
pip install -e .
```

# **Quick Start**

## **Running Actions and Pipelines**

```bash
# Run an action
urartu action=my_action

# Run a pipeline
urartu pipeline=my_pipeline

# With options
urartu pipeline=my_pipeline aim=local slurm=no_slurm machine=local
```

## **Project Structure**

```
my_project/
├── actions/              # Action implementations
│   └── my_action.py
├── pipelines/            # Pipeline implementations
│   └── my_pipeline.py
└── configs/
    ├── action/           # Action configurations
    │   └── my_action.yaml
    └── pipeline/         # Pipeline configurations
        └── my_pipeline.yaml
```

# **Core Concepts**

## **Actions**

Actions are self-contained components that perform specific ML tasks:

```python
from urartu.common import Action

class MyAction(Action):
    def run(self):
        # Your ML task here
        data = self.load_data()
        results = self.process(data)
        
        # Save to cache using unified API
        cache_dir = self.get_cache_entry_dir("my_data")
        # Save machine-readable data to cache
        
        # Save plots to run directory (always regenerated)
        plots_dir = self.get_run_dir("plots")
        # Save human-readable outputs here
    
    def get_outputs(self):
        return {
            "results_path": str(self.get_cache_entry_dir("results")),
            "run_dir": str(self.get_run_dir())
        }
```

## **Pipelines**

Pipelines chain Actions together with automatic dependency resolution:

```yaml
# configs/pipeline/my_pipeline.yaml
pipeline_name: my_pipeline

pipeline:
  device: cuda
  seed: 42
  actions:
    - action_name: data_preprocessing
      dataset:
        source: "data.csv"
    
    - action_name: model_training
      depends_on:
        data_preprocessing:
          processed_data: dataset.data_path
      model:
        architecture: "transformer"
```

## **Configuration**

### **Action Config**
```yaml
# configs/action/my_action.yaml
action_name: my_action

action:
  experiment_name: "My Experiment"
  device: cuda
  dataset:
    source: "data.csv"
```

### **Pipeline Config**
```yaml
# configs/pipeline/my_pipeline.yaml
pipeline_name: my_pipeline

pipeline:
  experiment_name: "My Pipeline"
  device: cuda
  actions:
    - action_name: action1
    - action_name: action2
```

# **Key Features**

## **Unified Caching**

Actions automatically cache results. Use the unified APIs:

```python
# For machine-readable cached data
cache_dir = self.get_cache_entry_dir("subdirectory")
# Structure: cache/{action_name}/{cache_hash}/{subdirectory}/

# For human-readable outputs (plots, reports)
run_dir = self.get_run_dir("plots")
# Structure: .runs/{pipeline_name}/{timestamp}/{subdirectory}/
```

**Important**: Plots should always be saved to `run_dir` and regenerated from cached data.

## **Dependency Resolution**

Pipelines automatically inject outputs from previous actions:

```yaml
- action_name: model_training
  depends_on:
    data_preprocessing:
      processed_data: dataset.data_path
      stats: model.feature_stats
```

## **Caching Configuration**

```yaml
action:
  cache_enabled: true
  force_rerun: false
  cache_max_age_days: 7

pipeline:
  cache_enabled: true
  force_rerun: false
  cache_max_age_days: 7
```

# **Advanced Usage**

## **Remote Execution**

Execute workflows on remote machines:

```yaml
# configs_tamoyan/machine/remote.yaml
type: remote
host: "cluster.example.com"
username: "user"
ssh_key: "~/.ssh/id_rsa"
remote_workdir: "/path/to/workspace"
project_name: "my_project"
```

```bash
urartu pipeline=my_pipeline machine=remote slurm=slurm
```

## **Multi-run**

```bash
urartu --multirun pipeline=my_pipeline pipeline.actions[0].learning_rate=1e-3,1e-4,1e-5
```

# **Citation**

If you find Urartu helpful in your research, please cite it:

```bibtex
@software{Tamoyan_Urartu_2023,
  author = {Hovhannes Tamoyan},
  license = {Apache-2.0},
  month = {11},
  title = {{Urartu}},
  url = {https://github.com/tamohannes/urartu},
  year = {2023}
}
```
