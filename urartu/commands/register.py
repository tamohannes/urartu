import argparse
import logging
import re
from pathlib import Path

from urartu.commands.command import Command
from urartu.utils.registry import Registry


@Command.register("register")
class Register(Command):
    """
    Registers a module by given location.
    """

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """urartu: register"""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Register a new project in the registry",
        )

        subparser.add_argument("--name", type=str, help="name of the project/module")
        subparser.add_argument("--path", type=str, help="path of the root dir of the project/module")

        subparser.set_defaults(fire=self._register)

        return subparser

    def _register(self, args: argparse.Namespace):
        module_root_dir = Path(args.path)
        module_name = re.sub(r"[^A-Za-z0-9]+", "", args.name)

        if Registry.add_entry(module_name, module_root_dir):
            logging.info(f"Module '{module_name}' is registered successfully under path:'{module_root_dir}'")
