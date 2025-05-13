<!--- BADGES: START --->

[![PyPI - Package Version](https://img.shields.io/pypi/v/urartu?logo=pypi&style=flat&color=orange)](https://pypi.org/project/urartu/)
[![PyPI - Python Version](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub - License](https://img.shields.io/github/license/tamohannes/urartu)](https://opensource.org/licenses/Apache-2.0)

<!--- BADGES: END --->

🚀 New Release Available!
We're excited to announce a new version of our project! 🎉 Our README.md is currently being updated to reflect all the fantastic changes. In the meantime, please explore the release notes and dive in. We'd love to hear your feedback! ❤️

# **Urartu 🦁**

Welcome to Urartu, your go-to NLP framework designed to simplify your work. Urartu provides easy-to-use abstractions for common pipeline components, making your machine learning journey smoother.
With a `.yaml` file-based configuration system and seamless `slurm` job submission capabilities on clusters, Urartu removes the technical hassle so you can focus on making impactful NLP work! 🚀

![urartu_schema drawio](https://github.com/tamohannes/urartu/assets/23078323/9d747c2d-9856-4dbe-85ab-74a595f86603)

# **Installation**

Getting started with Urartu is super easy! 🌀 Just run:
```bash
pip install urartu
```

Or, if you prefer to install directly from the source:

- Clone the repository:
    ```bash
    git clone git@github.com:tamohannes/urartu.git`
    ```
- Navigate to the project directory:
    ```bash
    cd urartu
    ```
- Execute the magic command:
    ```bash
    pip install -e .
    ```

And just like that, you're all set! ✨ Use the following command anywhere in your system to access Urartu:

```bash
urartu --help
```

# **Getting started**

To jump right in with Urartu, start with our `starter_template`. You can copy this to begin your project or check out the steps in our [Starter Template Setup](./starter_template_setup.md). Your setup will mirror what's found in this directory.

Think of Urartu as the foundational framework for your projects, similar to an abstract class in object-oriented programming (OOP).
Your project acts as the concrete implementation, where Urartu provides the foundational scaffolding.
It includes high-level abstractions, configuration through `.yaml` files powered by [Hydra](https://github.com/facebookresearch/hydra), and `slurm` job management utilizing the [Submitit](https://github.com/facebookincubator/submitit) library.
This setup ensures your projects are both flexible and robust, making your machine learning workflow efficient and scalable.
It also includes key NLP features such as dataset readers, model loaders, and device handlers.

Here's how to get started:
1. Extend Urartu: Inherit the structure of Urartu and customize it by writing your own actions and configurations, akin to implementing methods from an abstract class in OOP.
2. Utilize Core Functionalities: Jumpstart your project with pre-defined functionalities:
    - Datasets:
        - Load a HF (Hugging Face) dataset from a dictionary, a file, or directly from the HF hub.
    - Models:
        - Use a HF model as a causal language model or integrate it into a pipeline.
        - Incorporate the OpenAI API for advanced modeling.
3. Customize Further: Develop and place your own classes within the corresponding directories of your project to meet your specific needs.

By following these steps, you can efficiently set up and customize your machine learning projects with Urartu.

# **Firing Up 🔥**

Once you've cloned the `starter_template`, head over to that directory in your terminal:
```bash
cd starter_template
```

To launch a single run with predefined configurations, execute the following command:
```bash
urartu action_config=generate aim=aim slurm=slurm
```

If you're looking to perform multiple runs, simply use the `--multirun` flag. To configure multiple runs, add a sweeper at the end of your `generate.yaml` config file like this:

```yaml
...

hydra:
  sweeper:
    params:
      action_config.task.model.generate.num_beams: 1,5,10
```
This setup initiates 3 separate runs, each utilizing different `num_beams` settings to adjust the model's behavior.

Then, start your multi-run session with the same command:

```bash
urartu action_config=generate aim=aim slurm=slurm
```

With these steps, you can effortlessly kickstart your machine learning experiments with Urartu, whether for a single test or comprehensive multi-run analyses!

# **Navigating the Urartu Architecture**

Dive into the structured world of Urartu, where managing NLP components becomes straightforward and intuitive.

## **Configs: Tailoring Your Setup**

Set up your environment effortlessly with our configuration templates found in the `urartu/config` directory:
- `urartu/config/main.yaml`: This primary configuration file lays the groundwork with default settings for all system keys.
- `urartu/config/action_config` This space is dedicated to configurations specific to various actions.

## **Crafting Customizations**

Configuring Urartu to meet your specific needs is straightforward. You have two easy options:

1. **Custom Config Files**: Store your custom configuration files in the configs directory to adjust the settings. This directory aligns with `urartu/config`, allowing you to maintain project-specific settings in files like `generate.yaml` for your `starter_template` project.

    - **Personalized User Configs**: For an even more tailored experience, create a `configs_{username}` directory at the same level as configs, replacing `{username}` with your system username. This setup automatically loads and overrides default settings without extra steps. ✨

Configuration files are prioritized in the following order: `urartu/config`, `starter_template/configs`, `starter_template/configs_{username}`, ensuring your custom settings take precedence.

2. **CLI Approach**: If you prefer using the command-line interface (CLI), Urartu supports enhancing commands with key-value pairs directly in the CLI, such as:

    ```bash
    urartu action_config=example action_config.experiment_name=NAME_OF_EXPERIMENT
    ```

Select the approach that best fits your workflow and enjoy the customizability that Urartu offers.

## **Actions: Shaping Functionality**

At the heart of Urartu is the `Action` class, which orchestrates all operations. This script manages everything from CLI arguments to the execution of the main function based on the `action_name` parameter.

## **Logging: Capture Every Detail**

Urartu is equipped with a comprehensive logging system to ensure no detail of your project's execution is missed. Here's how it works:
- Standard Runs: Every execution is meticulously logged and stored in a structured directory within your current working directory. The path format is:
`.runs/${action_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}`
- Debug Mode: If the debug flag is enabled, logs are saved under: `.runs/debug/${action_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}`
- Multi-run Sessions: For runs involving multiple configurations or tests, logs are appended with a `.runs/debug/${action_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}_multirun` suffix to differentiate them.

Each run directory is organized to contain essential files such as:
- output.log: Captures all output from the run.
- notes.md: Allows for manual annotations and observations.
- cfg.yaml: Stores the configuration used for the run.

Additional files may be included depending on the type of run, ensuring you have all the data you need at your fingertips.

## **Effortless Launch**

Launching with Urartu is a breeze, offering you two launch options:

- Local Marvel: Execute jobs right on your local machine.
- Cluster Voyage: Set sail to the slurm cluster by toggling the `slurm.use_slurm` in `config_{username}/slurm/slurm.yaml` to switch between local and cluster executions.

Choose your adventure and launch your projects with ease! 🚀

Encountered any issues or have suggestions? Feel free to open an issue for support.

# **Exploring the Experiments**
Unveil insights with ease using Urartu in partnership with [Aim](https://github.com/aimhubio/aim), the intuitive and powerful open-source AI metadata tracker. To access a rich trove of metrics captured by Aim, simply:
- Navigate to the directory containing the .aim repository.
- Fire up the magic with:
```bash
aim up
```
Watch as Aim brings your experiments into sharp relief, providing the clarity needed to drive informed decisions and pioneering efforts in machine learning. 📈
