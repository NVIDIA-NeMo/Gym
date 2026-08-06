# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identifiers shared by catalog and composition-lock code."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LOCK_SCHEMA_VERSION = 1
LOCK_RELATIVE_PATH = Path("nemo_gym/resources/environment-composition-locks.json")


def environment_version_key(manifest: Any) -> str:
    """Return the stable semver-qualified composition-lock key."""

    kind = getattr(manifest.kind, "value", manifest.kind)
    return f"{kind}:{manifest.name}@{manifest.version}"


__all__ = [
    "LOCK_RELATIVE_PATH",
    "LOCK_SCHEMA_VERSION",
    "environment_version_key",
]
