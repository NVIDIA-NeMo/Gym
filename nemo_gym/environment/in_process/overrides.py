# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The config overrides a Python caller passes to planning and serving."""

from collections.abc import Mapping
from typing import Any, Optional

from nemo_gym.config.config_types import ConfigError


def _checked_overrides(overrides: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The overrides as a dict, refusing a dotted key at any depth.

    Merged as a dict, `"judge.model"` would be one key with a dot in its name, while `++judge.model=x` on the
    command line sets `model` inside `judge`. Refusing it keeps the dict and the command line from disagreeing.
    """
    checked = dict(overrides or {})
    _refuse_dotted_keys(checked)
    return checked


def _refuse_dotted_keys(node: Any) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if "." in str(key):
                raise ConfigError(f"Override key {key!r} is dotted: nest the keys as dicts instead.")
            _refuse_dotted_keys(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _refuse_dotted_keys(item)
