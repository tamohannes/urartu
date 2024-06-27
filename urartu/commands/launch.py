import argparse
import logging
import re

from aim import Run
from hydra import compose, initialize

from .command import Command

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from ..utils.launcher import launch, launch_on_slurm
from ..utils.slurm import is_submitit_available
from ..utils.registry import Registry


@Command.register("launch")
class Launch(Command):
    """
    Launches an action from a specific module
    """

    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """urartu: launcher"""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Launch an action from a given module",
        )

        subparser.add_argument("--name", type=str, help="name of the project/module")
        subparser.add_argument(
            "module_args", nargs=argparse.REMAINDER, help="module arguments"
        )

        subparser.set_defaults(fire=self._launch)

        return subparser

    def _launch(self, args: argparse.Namespace):
        module_name = re.sub(r"[^A-Za-z0-9]+", "", args.name)
        module_path = Registry.get_module_path_by_name(module_name)
 
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="main", overrides=args.module_args)

        aim_run = Run(
            repo=cfg.aim.repo,
            experiment=cfg.action_config.experiment_name,
        )
        aim_run.set("cfg", cfg, strict=False)

        if cfg.slurm.use_slurm:
            assert (
                is_submitit_available()
            ), "Please 'pip install submitit' to schedule jobs on SLURM"

            launch_on_slurm(
                module=module_path,
                action_name=cfg.action_name,
                cfg=cfg,
                aim_run=aim_run,
            )
        else:
            launch(
                module=module_path,
                action_name=cfg.action_name,
                cfg=cfg,
                aim_run=aim_run,
            )

        if aim_run.active:
            aim_run.close()
