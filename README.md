<!--- BADGES: START --->

[![PyPI - Package Version](https://img.shields.io/pypi/v/urartu?logo=pypi&style=flat&color=orange)](https://pypi.org/project/urartu/)
[![PyPI - Python Version](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub - License](https://img.shields.io/github/license/tamohannes/urartu)](https://opensource.org/licenses/Apache-2.0)

<!--- BADGES: END --->

🚀 New Release Available!
We've just rolled out a new version of this project! 🎉 While the README.md is still catching up with all the exciting changes, feel free to explore the release notes and dive in. Your feedback is always welcome! ❤️

# UrarTU 🦁

Harness the power of UrarTU, a versatile ML framework meticulously designed to provide intuitive abstractions of familiar pipeline components. With a `.yaml` file-based configuration system, and the added convenience of seamless `slurm` job submission on clusters, `UrarTU` takes the complexity out of machine learning, so you can focus on what truly matters! 🚀

![urartu_schema drawio](https://github.com/tamohannes/urartu/assets/23078323/9d747c2d-9856-4dbe-85ab-74a595f86603)

## Installation

Getting Started with `UrarTU` is a Breeze! 🌀 Simply do
```bash
pip install urartu
```



<!-- Or follow these steps to install from the source:

- Clone the repository: `git clone git@github.com:tamohannes/urartu.git`
- Navigate to the project directory: `cd urartu`
- Execute the magic command: `pip install -e .`


Adding a dash of convenience! ✨ Once you've executed the previous command, you'll also have an alias conjured, granting you easy access to `UrarTU` from any directory within your operating system:
```bash
urartu --help
```

## The Structure
Think of `UrarTU` as a project insantiator. It simplifies project creation by offering high-level abstractions, `yaml`-based file configuration, and `slurm` job management. It also includes essential features for machine learning pipelines, such as dataset reading, model loading, device handling and more.

`Urartu` features a registry where you can manage your project (a Python module) and execute its actions using the `urartu` command.
To register a project simply run:

```bash
urartu register --name=example --path=PATH_TO_EXAMPLE_MODULE
```

Here, we register a module named `example` located at `PATH_TO_EXAMPLE_MODULE`.

To remove a module from the registry, simply run:
```bash
urartu unregister --name=example
```

To see the available modules, use the command `urartu -h`. Under the launch command, you’ll find a list of all registered modules.


## Navigating the UrarTU Architecture

Within `UrarTU` lies a well-organized structure that simplifies your interaction with machine learning components.

### Configs: Tailoring Your Setup

The default configs which shape the way of configs are defined under `urartu/config` directory:
- `urartu/config/main.yaml`: This core configuration file sets the foundation for default settings, covering all available keys within the system.
- `urartu/config/action_config` Directory: A designated space for specific action configurations.


### Crafting Customizations

Tailoring configurations to your needs is a breeze with UrarTU. You have two flexible options:

1. **Custom Config Files**: To simplify configuration adjustments, UrarTU will read the content of your inherited module in `configs` directory where you can store personalized configuration files. These files seamlessly integrate with Hydra's search path. The directory structure mirrors that of `urartu/config`. You can define project-specific configurations in specially named files. For instance, an `example.yaml` file within the `configs` directory can house all the configurations specific to your 'example' project, with customized settings.

    - **Personalized User Configs**: To further tailor configurations for individual users, create a directory named `configs_{username}` at the same level as the `configs` directory, where `{username}` represents your operating system username. The beauty of this approach is that there are no additional steps required. Your customizations will smoothly load and override the default configurations, ensuring a seamless and hassle-free experience. ✨

    The order of precedence for configuration overrides is as follows: `urartu/config`, `configs`, `configs_{username}`, giving priority to user-specific configurations.

2. **CLI Approach**: For those who prefer a command-line interface (CLI) approach, UrarTU offers a convenient method. You can enhance your commands with specific key-value pairs directly in the CLI. For example, modifying your working directory path is as simple as:

    ```bash
    urartu action_config=example action_config.experiment_name=NAME_OF_EXPERIMENT
    ```

Choose the method that suits your workflow best and enjoy the flexibility UrarTU provides for crafting custom configurations.


### Actions: Shaping Functionality

Central to UrarTU's architecture is the `Action` class. This foundational script governs all actions and their behavior. From loading CLI arguments to orchestrating the `main` function of a chosen action, the `action_name` parameter plays the pivotal role in this functionality.

### Effortless Launch

With UrarTU, launching actions becomes a breeze, offering you two distinctive pathways. 🚀

- Local Marvel: The first route lets you run jobs on your local machine – the very platform where the script takes flight.
- Cluster Voyage: The second option invites you to embark on a journey to the slurm cluster. By setting the `slurm.use_slurm` configuration in `config/main.yaml` which takes boolean values, you can toggle between these options effortlessly.

Experience the freedom to choose your launch adventure, tailored to your needs and aspirations!


And just like that, you're all set to embark on your machine learning journey with UrarTU! 🌟
If you run into any hiccups along the way or have any suggestions, don't hesitate to open an issue for assistance.
















## Example Project

For a sample project see [Getting Started Guide](./getting_started.md)


## Exploring the Experiments
Unveiling Insights with Ease! 🔍 UrarTU, pairs up with [Aim](https://github.com/aimhubio/aim), a remarkable open-source AI metadata tracker designed to be both intuitive and potent. To dive into the wealth of metrics that Aim effortlessly captures, simply follow these steps:
- Navigate to the directory housing the .aim repository.
- Execute the command that sparks the magic:
```bash
aim up
```
Behold as your experiments come to life with clarity and depth! Aim brings your data to the forefront, and with it, the power to make informed decisions and chart new territories in the realm of machine learning. 📈 -->
