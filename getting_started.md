# Getting Started with UrarTU

This guide will lead you through the essential steps to initiate your project with `UrarTU`. We'll employ a simple text generation project as an example and walk you through it from start to finish.

For installation instructions and additional information, please visit the project's GitHub page: **[UrarTU GitHub Repository](https://github.com/tamohannes/urartu)**.

![Figure 1: Schematic Layout of the UrarTU.](https://github.com/tamohannes/urartu/assets/23078323/33bd1271-d3a5-4f07-b206-f45c711ca0d9)

Figure 1: Schematic Layout of the UrarTU.


## **Instantiating the Project**
The first step is to create a structure similar to `UrarTU`, here is the structure of the `example` project we are trying to achieve:

```
example
├── __init__.py
├── actions
│   ├── __init__.py
│   └── generate.py
├── configs
│   ├── __init__.py
│   └── action_config
│       └── generate.yaml
└── configs_tamoyan
    ├── __init__.py
    ├── aim.yaml
    └── slurm.yaml
```

Simply copy this structure in your `example` project



## **Setting Up Your Configuration**

Create a configuration template.

Here's a basic structure for `generate.yaml` configuration file:

```yaml
# @package _global_
action_name: generate

aim:
  repo: ./

action_config:
  workdir: ./
  experiment_name: "Example - next token prediction"
  device: "cpu" # auto, cuda, cpu (default) 

  task:
    model:
      type:
        _target_: urartu.models.causal_lm_model.CausalLMModel
      name: gpt2
      dtype: torch.float32
      cache_dir: ""
      
      generate:
        max_length: 100
        num_beams: 5
        no_repeat_ngram_size: 2

    dataset:
      type:
        _target_: urartu.datasets.hf_datasets.HFDatasets
      name: truthfulqa/truthful_qa
      subset: generation
      split: validation
      input_key: "question"
```

Don’t worry about the configs inside of the `task` argument for now. Their purpose will become evident in the upcoming sections.
However pay attention to their `_target_` argument which are fundamental classes from `urartu` package.

This is a general configuration file for the next token prediction project. However, if multiple team members are working on the same project and have their specific configurations, follow these steps:

1. Create a directory at the same level as the `configs` directory and name it `configs_{username}`, where `{username}` is your OS username.
2. Copy the content of the general configuration file and paste it into the `configs_{username}` directory.
3. Customize the specific settings as needed. Suppose I prefer my Aim repository to be a remote URL rather than a local path.

 To achieve this, I've created a custom configuration that's unique to my setup:

```yaml
# @package _global_

repo: aim://0.0.0.0:43800
log_system_params: true
```

## Enabling slurm

With just a few straightforward slurm configuration parameters, we can seamlessly submit our action to the slurm system. To achieve this, override the pre-defined `slurm` configuration within the `urartu/urartu/config/main.yaml` file in `generate.yaml` or in your user specific configs:

Setting the `use_slurm` argument to `true` activates slurm job submission. The other arguments align with familiar `sbatch` command options.

## **Creating the Action File**

Next, let's create the action file that will use the parsed configuration to kickstart your work.

Inside `generate.py`, define a main method with the following arguments:

```python
from aim import Run, Text
from omegaconf import DictConfig

def main(cfg: DictConfig, aim_run: Run):
    example = Generate(cfg, aim_run)
    example.main()
```

The `cfg` parameter will contain overridden parameters, and `aim_run` is an instance of our Aim run for tracking progress.

## **Implementing the `Generate` Class**

Now, let's create the `Generate` class:

```python
from urartu.common.action import Action

class Generate(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def main(self):
        # Your code goes here
```

Ensure that `Generate` inherits from the abstract `Action` class. From this point forward, you have full control to implement your text generation logic, here is the final script:

```python
from omegaconf import DictConfig
from aim import Run, Text

from tqdm import tqdm

from urartu.common.action import Action
from urartu.common.dataset import Dataset
from urartu.common.model import Model


class Generate(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def main(self):
        model = Model.get_model(self.task_cfg.model)
        dataset = Dataset.get_dataset(self.task_cfg.dataset)

        for idx, sample in tqdm(enumerate(dataset.dataset)):
            prompt = sample[self.task_cfg.dataset.get("input_key")]
            self.aim_run.track(Text(prompt), name="input")

            output = model.generate(prompt)
            self.aim_run.track(Text(output), name="output")


def main(cfg: DictConfig, aim_run: Run):
    example = Generate(cfg, aim_run)
    example.main()
```

Here, we utilize the HuggingFace Casual Language Model to continue a given token sequence from `truthful_qa`. We then track the inputs and outputs of the model.

Once you've completed these steps, you can register the project in `UrarTU`:

```bash
urartu register --name=example --path=PATH_TO_EXAMPLE_MODULE
```

After which you can easily run the `generate` action from the command line by specifying `generate` as the `action_config`:

```bash
urartu launch --name=example action_config=generate
```

### Batch Execution with Multiple Configurations

You can streamline your experimentation by using Hydra's `--multirun` flag, allowing you to submit multiple runs with different parameters all at once. For example, if you need to run the same script with various model `dtype`s, follow these steps:

1. Add a Hydra sweeper configuration at the end of your config file:
    
```yaml
hydra:
  sweeper:
    params:
      ++action_config.tasks.0.model.dtype: torch.float32, torch.bfloat16
```
    
The double plus sign (`++`) will append this configuration to the existing one, resulting in three runs with `action_config.tasks.0.model.dtype` set to `torch.float16`, `torch.float32`, and `torch.bfloat16`.
    
2. Execute the following command to start the batch runs:
    
```bash
urartu launch --name=example --multirun action_config=generate
```

This approach simplifies the process of running experiments with various configurations, making it easier to explore and optimize your models.

## Monitoring the progress of the run

To monitor your experiment's progress and view tracked metadata, simply initiate Aim with the following command:

```bash
aim up
```

You can expect a similar experience as demonstrated in the following image:

https://github.com/tamohannes/urartu/assets/23078323/11705f35-e3df-41f0-b0d1-42eb846a5921


## **Resources**

`UrarTU` is built upon a straightforward combination of three widely recognized libraries. For more in-depth information on how each of these libraries operates, please consult their respective GitHub repositories:

- **Hydra**: **[GitHub Repository](https://github.com/facebookresearch/hydra), [Getting started | Hydra](https://hydra.cc/docs/1.3/intro/)**
- **Submit**: **[GitHub Repository](https://github.com/facebookincubator/submitit)**
- **Aim**: **[GitHub Repository](https://github.com/aimhubio/aim)**

These repositories provide detailed insights into the inner workings of each library.
