from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Union, Mapping
import yaml

import transformer_lens.loading_from_pretrained as loading

class Config:
    def __init__(self, **kwargs):
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_sections", {})
        for k, v in kwargs.items():                     # store leaves vs sub-configs
            (self._sections if isinstance(v, Config) else self._values)[k] = v

    def __getattr__(self, key):
        if key in self._values:        # ❶ leaf value
            return self._values[key]
        if key in self._sections:      # ❷ return the sub-config itself  ← FIX
            return self._sections[key]
        for sub in self._sections.values():            # ❸ deep fallback
            try:
                return getattr(sub, key)
            except AttributeError:
                pass
        raise AttributeError(key)

    def __setattr__(self, key, value):
        (self._sections if isinstance(value, Config) else self._values)[key] = value

    def __repr__(self):
        body = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"Config({body})" if body else "Config()"

    @classmethod
    def from_configs(cls, **sections):
        root = cls(**sections)               # installs named sub-configs
        merged = {}
        for cfg in sections.values():        # left→right precedence
            merged.update(cfg._values)
        root._values.update(merged)
        return root

    @classmethod
    def from_yaml(cls, src: Union[str, Path]) -> "Config":
        """
        Load a YAML file (or string path/Path) and convert it to a Config.
        Nested mappings → nested Config objects.
        """
        path = Path(src) if not isinstance(src, Path) else src
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, Mapping):
            raise TypeError("Top-level YAML must be a mapping/object")

        def _to_cfg(node):
            if isinstance(node, Mapping):
                return cls(**{k: _to_cfg(v) for k, v in node.items()})
            return node                        # leaf value

        return _to_cfg(raw)

    @classmethod
    def from_tl(self, model_name, **kwargs) -> "Config":
        """
        Load a model from the TL library and convert it to a Config.
        This is a convenience wrapper around `loading.from_pretrained`.
        """
        official_model_name = loading.get_official_model_name(model_name)
        tl_config = loading.get_pretrained_model_config(official_model_name, **kwargs)
        cfg = Config(**tl_config.to_dict())
        return cfg

    def add(self, **sections: "Config") -> "Config":
        """
        Attach one or more *named* sub-configs to this config.

        >>> cfg.add(data=data_cfg, model=model_cfg)

        * Each kwarg value must be a Config instance.
        * If a section name already exists, we deep-merge via `update`.
        * Leaf keys of each added subsection are also promoted to the
          parent's flat lookup space so that `cfg.foo` still works.

        Returns self so you can chain calls.
        """
        for name, sub in sections.items():
            if not isinstance(sub, Config):
                raise TypeError(f"{name} must be a Config, got {type(sub)}")

            if name in self._sections:          # merge if already present
                self._sections[name].update(sub)
            else:                               # otherwise just attach
                self._sections[name] = sub

            # expose the subsection's leaves for flat access
            self._values.update(sub._values)

        return self

    def is_layer_norm_activation(self) -> bool:
        return self.act_fn is not None and self.act_fn.endswith("_ln")