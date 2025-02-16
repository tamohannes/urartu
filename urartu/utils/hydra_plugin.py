import logging
import os
import pwd
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

logger = logging.getLogger(__name__)

current_user = pwd.getpwuid(os.getuid()).pw_name
cwd = Path.cwd()


class UrartuPlugin(SearchPathPlugin):
    """
    A plugin for Hydra that manipulates the configuration search path by adding custom
    paths. This plugin dynamically adds both local and packaged configuration directories
    based on the current user and the working directory, allowing for user-specific and
    generic configurations to be loaded.

    Methods:
        manipulate_search_path: Appends user-specific and generic configuration paths to the search path.
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """
        Modifies the provided Hydra configuration search path by appending new paths that
        point to configuration directories both locally and in packages. This setup allows
        configurations to be personalized per user or provided generally across users.

        Args:
            search_path (ConfigSearchPath): The Hydra config search path instance that will be manipulated.

        Effects:
            Appends four paths to the `search_path`:
            - A local file system path for user-specific configurations.
            - A packaged path for user-specific configurations within installed packages.
            - A local file system path for general configurations.
            - A packaged path for general configurations within installed packages.
        """
        search_path.append(
            provider="urartu", path=f"file://{cwd}/configs_{current_user}"
        )
        search_path.append(
            provider="urartu", path=f"pkg://{cwd}/configs_{current_user}"
        )
        search_path.append(provider="urartu", path=f"file://{cwd}/configs")
        search_path.append(provider="urartu", path=f"pkg://{cwd}/configs")
