from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.models import ChannelTruth
from environments.custom_scene import generate_custom_scene_truths
from environments.floorplan import generate_floorplan_truths
from environments.manual_paths import generate_manual_path_truths
from environments.simple_room import generate_simple_room_truths
from environments.sionna_builtin import generate_sionna_builtin_truths
from environments.statistical import generate_tdl_truths


TruthGenerator = Callable[..., list[ChannelTruth]]


def generate_truths(
    config: dict[str, Any],
    rng: np.random.Generator,
    carrier_frequency_hz: float | None = None,
) -> list[ChannelTruth]:
    env_type = str(config.get("environment", {}).get("type", "standard_tdl"))
    generators: dict[str, TruthGenerator] = {
        "standard_tdl": generate_tdl_truths,
        "manual_paths": generate_manual_path_truths,
        "simple_room": generate_simple_room_truths,
        "floorplan": generate_floorplan_truths,
        "custom_scene": generate_custom_scene_truths,
        "sionna_builtin_scene": generate_sionna_builtin_truths,
    }
    if env_type not in generators:
        raise ValueError(f"Unknown environment.type: {env_type}")
    return generators[env_type](config, rng, carrier_frequency_hz=carrier_frequency_hz)
