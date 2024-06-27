import argparse
from typing import Callable, Dict, Optional, Type, TypeVar

from overrides import overrides

from ..utils.registrable import Registrable

T = TypeVar("T", bound="Command")


class Command(Registrable):

    _reverse_registry: Dict[Type, str] = {}

    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        raise NotImplementedError

    @classmethod
    @overrides
    def register(
        cls: Type[T],
        name: str,
        constructor: Optional[str] = None,
        exist_ok: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        super_register_fn = super().register(
            name, constructor=constructor, exist_ok=exist_ok
        )

        def add_name_to_reverse_registry(subclass: Type[T]) -> Type[T]:
            subclass = super_register_fn(subclass)
            # Don't need to check `exist_ok`, as it's done by super.
            # Also, don't need to delete previous entries if overridden, they can just stay there.
            cls._reverse_registry[subclass] = name
            return subclass

        return add_name_to_reverse_registry

    @property
    def name(self) -> str:
        return self._reverse_registry[self.__class__]
