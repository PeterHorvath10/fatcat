"""Settings for every script come from TOML files in config/ (no setup values in the code).

The shipped files leave every setup-specific value empty (""), so nothing silently runs with someone else's
addresses, delays or thresholds. A script asks for what it needs with cfg.get("section.key"); an empty or
missing value stops it with the file and key to fill in.

    cfg = load("host.toml", "experiments/thor_fixed_time.toml")   # later files override earlier ones
    port = cfg.get("boards.thor.target_port", int)
    start_ns = cfg.get("glitch.window_start_ns", float)

Environment overrides: EMFI_<SECTION>__<KEY>=value (e.g. EMFI_GLITCH__WINDOW_START_NS=7450000).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    import tomli as tomllib

CONFIG_DIR = Path(os.environ.get("EMFI_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))
ENV_PREFIX = "EMFI_"
_UNSET = object()


class ConfigError(RuntimeError):
    pass


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _empty(v: Any) -> bool:
    return v is None or v == "" or v == []


class Config:
    def __init__(self, data: dict, sources: list[str]):
        self.data, self.sources = data, sources

    def _lookup(self, key: str) -> Any:
        env = os.environ.get(ENV_PREFIX + key.upper().replace(".", "__"))
        if env is not None:
            return env
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return _UNSET
            node = node[part]
        return node

    def get(self, key: str, cast: Callable[[Any], Any] | None = None, default: Any = _UNSET) -> Any:
        """The value of 'section.key'. Empty/missing raises ConfigError unless a default is given."""
        v = self._lookup(key)
        if v is _UNSET or _empty(v):
            if default is not _UNSET:
                return default
            raise ConfigError(f"setting '{key}' is empty: set it in one of {self.sources} "
                              f"or as {ENV_PREFIX}{key.upper().replace('.', '__')}")
        if cast is None:
            return v
        if isinstance(v, list):
            return [cast(x) for x in v]
        if cast is bool and isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return cast(v)

    def has(self, key: str) -> bool:
        v = self._lookup(key)
        return v is not _UNSET and not _empty(v)

    def section(self, key: str) -> dict:
        v = self._lookup(key)
        return v if isinstance(v, dict) else {}


def load(*names: str | Path) -> Config:
    """Merge the given TOML files (relative to config/ unless absolute); later files win."""
    data: dict = {}
    sources = []
    for name in names:
        path = Path(name) if Path(name).is_absolute() or Path(name).exists() else CONFIG_DIR / name
        if not path.exists():
            raise ConfigError(f"config file {path} not found")
        with open(path, "rb") as f:
            _merge(data, tomllib.load(f))
        sources.append(str(path))
    return Config(data, sources)
