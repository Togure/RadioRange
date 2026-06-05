from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return _load_config_recursive(config_path, seen=set())


def _load_config_recursive(config_path: Path, seen: set[Path]) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"Circular config include detected at {config_path}.")
    seen.add(config_path)

    text = config_path.read_text(encoding="utf-8")

    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ImportError:
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping.")

    includes = loaded.pop("include", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(f"Config include field in {config_path} must be a string or list.")

    merged: dict[str, Any] = {}
    for include_path in includes:
        child_path = (config_path.parent / str(include_path)).resolve()
        merged = deep_merge(merged, _load_config_recursive(child_path, seen))

    seen.remove(config_path)
    return deep_merge(merged, loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
