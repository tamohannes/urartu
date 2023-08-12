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
