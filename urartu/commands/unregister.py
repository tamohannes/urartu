import argparse
import logging
import re

from .command import Command
from ..utils.registry import Registry


@Command.register("unregister")
class Unregister(Command):
    """
    Unregisters/deletes a module by given name.
    """

    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """urartu: unregister"""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Unregister a project from the registry",
        )

        subparser.add_argument("--name", type=str, help="name of the project/module")

        subparser.set_defaults(fire=self._unregister)

        return subparser

    def _unregister(self, args: argparse.Namespace):
        module_name = re.sub(r"[^A-Za-z0-9]+", "", args.name)

        if Registry.remove_entry(module_name):
            logging.info(f"Module '{module_name}' is unregistered successfully")
