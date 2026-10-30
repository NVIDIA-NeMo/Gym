# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content identities of the installed scorer and of each graded check."""

import hashlib
import json
from collections.abc import Iterator
from functools import lru_cache
from importlib.resources import files
from importlib.resources.abc import Traversable

from pydantic import JsonValue


def fingerprint(value: JsonValue) -> str:
    """SHA-256 of a JSON value, independent of mapping order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _source_files(root: Traversable, prefix: str = "") -> Iterator[tuple[str, str]]:
    for entry in sorted(root.iterdir(), key=lambda entry: entry.name):
        name = prefix + entry.name
        if entry.is_dir() and entry.name != "__pycache__":
            yield from _source_files(entry, name + "/")
        elif entry.is_file() and name.endswith((".py", ".yaml")):
            yield name, hashlib.sha256(entry.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def scorer_identity() -> dict[str, str]:
    """The scorer's module and the SHA-256 of its source and judge configuration, computed once per process.

    It works from an installed wheel as from a checkout; Gym stamps runs with the checkout's commit.
    """
    return {
        "module": "resources_servers.agentic_if.if_grading",
        "source_sha256": fingerprint(dict(_source_files(files("resources_servers.agentic_if.if_grading")))),
    }
