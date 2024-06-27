from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from urartu.utils.registry import Registry
from urartu.utils.user import get_current_user

current_user = get_current_user()
registry_paths = Registry.get_module_paths()


class UrartuPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for registry_path in registry_paths:
            search_path.prepend(provider="urartu", path=f"{registry_path}//configs")
            search_path.prepend(
                provider="urartu", path=f"{registry_path}_{current_user}//configs"
            )
            search_path.prepend(
                provider="urartu", path=f"pkg://{registry_path}//configs"
            )
            search_path.prepend(
                provider="urartu",
                path=f"file://{registry_path}_{current_user}//configs",
            )
