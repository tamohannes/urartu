# UrarTU 🦁

Harness the power of UrarTU, a versatile ML framework meticulously designed to provide intuitive abstractions of familiar pipeline checkpoints. With a .yaml file-based configuration system, and the added convenience of seamless Slurm job submission on clusters, UrarTU takes the complexity out of machine learning, so you can focus on what truly matters! 🚀


## Installation

Getting Started with UrarTU is a Breeze! 🌀 Simply follow these steps to set up the essential packages and create a local package named 'urartu':

- Clone the repository: `git clone git@github.com:tmynn/urartu.git`
- Navigate to the project directory: `cd urartu`
- Execute the magic command: `pip install .`


Adding a Dash of Convenience! 🎉 Once you've executed the previous command, you'll not only have UrarTU ready to roll, but we've also sprinkled in a touch of magic for you ✨. An alias will be conjured, granting you easy access to UrarTU from any directory within your operating system:
```bash
urartu --help
```


> **Note for Usage on Slurm System**
> For an enhanced experience with the Slurm job cancellation process, it is recommended to utilize a specific fork of the `submitit` package available at: [https://github.com/tmynn/submitit](https://github.com/tmynn/submitit). This fork includes the `ResumableSlurmJob.on_job_fail` callback, which allows the incorporation of additional functionality within this callback to ensure a graceful job termination.

## Example Usage

Running an action with Urartu is as easy as waving a wand. Just provide the name of the configuration file containing the action, followed by the action name itself. 🪄 For instance, let's say you want to ignite the `example` action – an action that's a bit shy on functionality for now.

Simply execute the following command in your terminal:
```bash
urartu action_config=example
```

## Exploring the Experiments
Unveiling Insights with Ease! 🔍 Urartu, pairs up with [Aim](https://github.com/aimhubio/aim), a remarkable open-source AI metadata tracker designed to be both intuitive and potent. To dive into the wealth of metrics that Aim effortlessly captures, simply follow these steps:
- Navigate to the directory housing the .aim repository.
- Execute the command that sparks the magic:
```bash
aim up
```
Behold as your experiments come to life with clarity and depth! Aim brings your data to the forefront, and with it, the power to make informed decisions and chart new territories in the realm of machine learning. 📈

## Navigating the UrarTU Architecture

Within UrarTU lies a well-organized structure that simplifies your interaction with machine learning components.

### Configs: Tailoring Your Setup

- `urartu/config/main.yaml`: This core configuration file sets the foundation for default settings, covering all available keys within the system.
- `urartu/config/action_config` Directory: A designated space for specific action configurations.


#### Crafting Customizations

Adapting configurations is straightforward through two approaches:

1. CLI Approach: Enhance your commands with specific key-value pairs right in the CLI. For example, altering your workdir path is as simple as:
```bash
urartu action_config=example action_config.workdir=PATH_TO_WORKDIR
```

2. Custom Config Files: For a more streamlined experience, custom config files come to the fore. The `configs_{os_username}` directory hosts personalized configurations that seamlessly integrate with hydra’s search path. This directory structure echoes that of `urartu/config`. As an illustration, a directory named `configs_tamoyan` could house an `example.yaml` file within `action_config` directory with tailored settings. No extra steps are required to implement these changes! Your customizations seamlessly load and override the configs, making the process smooth and hassle-free. ✨


### Actions: Shaping Functionality

Central to UrarTU's architecture is the `Action` class. This foundational script governs all actions and their behavior. From loading CLI arguments to orchestrating the `main` function of a chosen action, the `action_name` parameter plays the pivotal role in this functionality.


### Effortless Launch

With UrarTU, launching actions becomes a breeze, offering you two distinctive pathways. 🚀

- Local Marvel: The first route lets you run jobs on your local machine – the very platform where the script takes flight.
- Cluster Voyage: The second option invites you to embark on a journey to the slurm cluster. By setting the `slurm.use_slurm` configuration in `config/main.yaml` which takes boolean values, you can toggle between these options effortlessly.

Experience the freedom to choose your launch adventure, tailored to your needs and aspirations!


And just like that, you're all set to embark on your machine learning journey with Urartu! 🌟
If you run into any hiccups along the way or have any suggestions, don't hesitate to open an issue for assistance.