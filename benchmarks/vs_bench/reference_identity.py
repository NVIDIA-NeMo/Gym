# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a clean, revision-pinned checkout of the third-party benchmark."""

import hashlib
import json
from pathlib import Path


AUTHOR_REVISION = "2c3531cdf91d1b620b4a9015126a242eb305a4d0"
# Explicitly recognized local bugfix to the pinned revision; original source is
# still accepted. Keep this hash distinct so the modified scorer is auditable.
KAZ_ARRAY_FIX_SHA256 = "c7ac90b00c7f7e901667c997417f2d5b93bca8b54ecefc03e1c16c442a198a2d"


def validate_author_checkout(author_root: str, author_python: str) -> str:
    root, python = Path(author_root), Path(author_python)
    if not root.is_absolute() or not python.is_absolute() or not python.is_file():
        raise ValueError("Author checkout and Python executable must be explicit paths")
    root = root.resolve()
    manifest = json.loads(Path(__file__).with_name("reference_source_manifest.json").read_text())
    if manifest["revision"] != AUTHOR_REVISION:
        raise ValueError("Incorrect source manifest revision")
    variant = "upstream"
    for name, expected in manifest["sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe author file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        accepted = {expected}
        if name == "envs/knights_archers_zombies.py":
            accepted.add(KAZ_ARRAY_FIX_SHA256)
            if actual == KAZ_ARRAY_FIX_SHA256:
                variant = "kaz_numpy_labels_v1"
        if actual not in accepted:
            raise ValueError(f"Author source differs from pinned reference: {name}")
    unexpected = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if str(path.relative_to(root)) not in manifest["sha256"]
    ]
    if unexpected:
        raise ValueError("Author checkout contains unexpected Python files")
    return variant
