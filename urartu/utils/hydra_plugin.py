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
                search_path.append(provider="urartu", path=f"file://{registry_path}/configs_{current_user}")
                search_path.append(provider="urartu", path=f"pkg://{registry_path}/configs_{current_user}")
                search_path.append(provider="urartu", path=f"file://{registry_path}/configs")
                search_path.append(provider="urartu", path=f"pkg://{registry_path}/configs")
        else:
            logger.warning("No registry paths found for UrartuPlugin.")