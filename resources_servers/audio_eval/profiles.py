# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark profile files (``profiles.json``) shared by the audio resources server and benchmark preparation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """Benchmark configs name files relative to the repository root, but servers run from their own directory."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() or candidate.exists() else REPO_ROOT / candidate


def _resolve_profile(profile: dict, defaults: dict, templates: dict) -> dict:
    """Apply ``defaults`` under ``profile``: a key both define as objects merges one level deep, else the profile wins."""
    resolved = deepcopy(defaults)
    for key, value in profile.items():
        base = resolved.get(key)
        resolved[key] = {**base, **value} if isinstance(base, dict) and isinstance(value, dict) else value
    template = resolved.get("judge_prompt_template")
    if isinstance(template, str):
        resolved["judge_prompt_template"] = templates[template]
    return resolved


@lru_cache(maxsize=None)
def _read_profiles(path: str, version: tuple[int, int]) -> dict:
    document = json.loads(Path(path).read_text())
    if not document.get("profiles"):
        raise ValueError(f"Invalid audio benchmark profiles: {path}")
    defaults, templates = document.get("defaults", {}), document.get("judge_prompt_templates", {})
    document["profiles"] = {
        name: _resolve_profile(profile, defaults, templates) for name, profile in document["profiles"].items()
    }
    return document


def load_profiles(path: str | Path) -> dict:
    """Read ``profiles.json`` (cached; treat the result as read-only). Profiles live under ``["profiles"]``.

    Optional top-level keys remove repetition: ``defaults`` holds contract fields every profile inherits (an object
    field merges one level deep with the profile's own, anything else is replaced by the profile's), and
    ``judge_prompt_templates`` names judge templates a profile selects with ``"judge_prompt_template": "<name>"``.
    The returned profiles are fully resolved.
    """
    path = Path(path).resolve()
    stat = path.stat()
    return _read_profiles(str(path), (stat.st_mtime_ns, stat.st_size))


def source_hash(row: dict) -> str:
    """Digest binding a verify request to its immutable source row (``ng_source_row_sha256``)."""
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
