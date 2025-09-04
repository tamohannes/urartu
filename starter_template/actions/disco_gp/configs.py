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

    def has(self, *keys) -> bool:
        """
        Check if this Config or any of its sub-configs contain the key(s).
        If a list of keys is provided, returns True only if all keys are present.
        """
        return all(self._has(k) for k in keys)

    def _has(self, key) -> bool:
        """
        Check if this Config or any of its sub-configs contain the key.
        """
        if key in self._values or key in self._sections:
            return True
        for sub in self._sections.values():
            if sub._has(key):
                return True
        return False

    def get(self, key, default=None):
        """
        Like dict.get: return the value for key if present, else default.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def to_dict(self, collapse=False, concat_name=True) -> Dict[str, Any]:
        """
        Convert this Config (and all sub-configs) to a nested dict.

        If collapse is True, return a flattened dict.
          - If concat_name is True, parent/child names are joined with '--'.
          - If concat_name is a str, that string is used as the separator.
          - If concat_name is False, parent names are not included in flattened keys.
        Nested lists are handled by enumerating their items and using the index
        as a path component in flattened keys.
        """
        # Keys that were promoted from immediate subsections (to support flat getattr)
        promoted_from_sections = set().union(
            *[set(sub._values.keys()) for sub in self._sections.values()] or [set()]
        )

        def _materialize(value):
            if isinstance(value, Config):
                return value.to_dict(collapse=False, concat_name=concat_name)
            if isinstance(value, list):
                return [_materialize(v) for v in value]
            return value

        if not collapse:
            # Nested dict: omit promoted duplicates at the top level
            out: Dict[str, Any] = {
                k: _materialize(v)
                for k, v in self._values.items()
                if k not in promoted_from_sections
            }
            for name, sub in self._sections.items():
                out[name] = sub.to_dict(collapse=False, concat_name=concat_name)
            return out

        # Collapsed / flattened representation
        sep = "--" if concat_name is True else (concat_name if isinstance(concat_name, str) else None)
        flat: Dict[str, Any] = {}

        def _emit(path_parts, value):
            if isinstance(value, Config):
                # Emit this config's own (non-promoted) leaves
                for k, v in value._values.items():
                    if path_parts or k not in promoted_from_sections:
                        # At root, skip promoted duplicates; below root, always include
                        _emit(path_parts + [k], v)
                # Recurse into named subsections
                for name, sub in value._sections.items():
                    _emit(path_parts + [name], sub)
                return
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    _emit(path_parts + [str(idx)], item)
                return

            # Make key according to policy
            if sep is None:
                key = str(path_parts[-1]) if path_parts else ""
            else:
                key = sep.join(map(str, path_parts))
            flat[key] = value

        _emit([], self)
        return flat

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
    def from_tl(self, full_model_name, **kwargs) -> "Config":
        """
        Load a model from the TL library and convert it to a Config.
        This is a convenience wrapper around `loading.from_pretrained`.
        """
        official_model_name = loading.get_official_model_name(full_model_name)
        tl_config = loading.get_pretrained_model_config(official_model_name, **kwargs)
        cfg = Config(**tl_config.to_dict())
        cfg.full_model_name = full_model_name
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
