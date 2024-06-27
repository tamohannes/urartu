from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from urartu.utils.registry import Registry
from urartu.utils.user import get_current_user
import logging

logger = logging.getLogger(__name__)

current_user = get_current_user()
registry_paths = Registry.get_module_paths()

class UrartuPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if registry_paths:
            for registry_path in registry_paths:
                paths = [
                    (f"{registry_path}/configs", "file"),
                    (f"{registry_path}/configs_{current_user}", "file"),
                    (f"pkg://{registry_path}/configs", "pkg"),
                    (f"file://{registry_path}/configs_{current_user}", "file")
                ]
                for path, type in paths:
                    search_path.prepend(provider="urartu", path=path)
                    logger.debug(f"Prepended {type} path: {path}")
        else:
            logger.warning("No registry paths found for UrartuPlugin.")