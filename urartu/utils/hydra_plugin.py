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
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="urartu", path=f"file://{cwd}/configs_{current_user}")
        search_path.append(provider="urartu", path=f"pkg://{cwd}/configs_{current_user}")
        search_path.append(provider="urartu", path=f"file://{cwd}/configs")
        search_path.append(provider="urartu", path=f"pkg://{cwd}/configs")
