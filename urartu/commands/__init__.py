import argparse
from typing import Tuple

from .command import Command

from .launch import Launch
from .register import Register


def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Creates the argument parser for the main program and uses it to parse the args.
    """
    parser = argparse.ArgumentParser(description="Run urartu")

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    def add_subcommands():
        for subcommand_name in sorted(Command.list_available()):
            subcommand_class = Command.by_name(subcommand_name)
            subcommand = subcommand_class()
            subcommand.add_subparser(subparsers)

    add_subcommands()
    args = parser.parse_args()

    return parser, args


def main():
    parser, args = parse_args()
    if "fire" in dir(args):
        args.fire(args)
    else:
        parser.print_help()
