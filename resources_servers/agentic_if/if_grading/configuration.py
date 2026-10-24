# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned judging configuration; no Gym imports or implicit credential discovery."""

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files

import yaml


@lru_cache(maxsize=1)
def _default_config() -> dict:
    return yaml.safe_load(
        files("resources_servers.agentic_if.if_grading").joinpath("data/judge.yaml").read_text(encoding="utf-8")
    )


def load_config() -> dict:
    """Return an independent copy of the checked-in, calibrated judge settings."""
    return deepcopy(_default_config())


def votes_for(config: dict, context: str, n_steps: int | None = None) -> int:
    """Apply calibrated context voting, including the after-event length threshold."""
    settings = config["contexts"].get(context) or {}
    default = config["model"].get("votes", 1)
    if n_steps is not None and n_steps < settings.get("votes_from_steps", 0):
        return default
    return settings.get("votes", default)


def config_hash(config: dict) -> str:
    """Keep the recorded identity of unchanged calibrated prompts and model settings."""
    body = {key: value for key, value in config.items() if key not in {"version", "changed", "note"}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
