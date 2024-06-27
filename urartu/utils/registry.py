import yaml
import os
from pathlib import Path

registry_file_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.joinpath("registry.yaml")


class Registry:
    REGISTRY_FILE_PATH = registry_file_path

    @staticmethod
    def add_entry(module_name, module_root_dir):
        file_content = Registry.load_file_content()

        if module_name in file_content:
            raise ValueError(f"Module with name '{module_name}' already exists in the registery, use a different name")
        if str(module_root_dir) in file_content.values():
            raise ValueError(
                f"Module with path '{str(module_root_dir)}' already exists in the registery, use a different path"
            )
        if not module_root_dir.exists() or module_root_dir.is_file():
            raise FileNotFoundError(f"Couldn't find directory: {module_root_dir}")
        else:
            file_content[module_name] = str(module_root_dir)

            with open(Registry.REGISTRY_FILE_PATH, "a") as file:
                yaml.dump(file_content, file)

            return True

    @staticmethod
    def load_file_content():
        if not Registry.REGISTRY_FILE_PATH.exists():
            Registry.REGISTRY_FILE_PATH.touch()
            file_content = None
        else:
            with open(Registry.REGISTRY_FILE_PATH, "r") as file:
                file_content = yaml.safe_load(file)
        
        if file_content is None:
            return {}
            # raise RuntimeError(
            #     f"Registery is empty, register a module using `urartu register --name='NAME' --path='PATH'` command."
            # )

        return file_content

    @staticmethod
    def get_module_paths():
        file_content = Registry.load_file_content()

        return list(file_content.values())

    @staticmethod
    def get_module_path_by_name(module_name):
        file_content = Registry.load_file_content()

        if module_name not in file_content:
            raise KeyError(
                f"Registery with name '{module_name}' is not found, start with registering a module using `urartu register --name='NAME' --path='PATH'` command."
            )
        return file_content[module_name]
