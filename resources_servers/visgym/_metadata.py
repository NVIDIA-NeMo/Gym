# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image


logger = logging.getLogger(__name__)

_DROP = object()
_WARNED_MESSAGES: set[str] = set()


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    logger.warning(message)


def _is_numpy_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("numpy") and hasattr(value, "tolist")


def _json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _sanitize_sequence(values: Any) -> list[Any]:
    sanitized = []
    for item in values:
        sanitized_item = sanitize_metadata(item)
        if sanitized_item is not _DROP:
            sanitized.append(sanitized_item)
    return sanitized


def sanitize_metadata(value: Any) -> Any:
    """Convert VisGym env info into JSON-serializable values."""

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            sanitized_item = sanitize_metadata(item)
            if sanitized_item is not _DROP:
                sanitized[str(key)] = sanitized_item
        return sanitized

    if isinstance(value, tuple):
        return _sanitize_sequence(value)

    if isinstance(value, (set, frozenset)):
        return _sanitize_sequence(sorted(value, key=repr))

    if isinstance(value, list):
        return _sanitize_sequence(value)

    if _is_numpy_array(value):
        return sanitize_metadata(value.tolist())

    if isinstance(value, Image.Image):
        _warn_once("Dropping PIL image from VisGym metadata; images belong in observations.")
        return _DROP

    if isinstance(value, Path):
        return str(value)

    if _json_serializable(value):
        return value

    _warn_once(
        f"Converting non-JSON-serializable VisGym metadata value to repr: {type(value).__name__}"
    )
    return repr(value)
