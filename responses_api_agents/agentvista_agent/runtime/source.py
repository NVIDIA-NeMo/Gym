# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the public upstream execution/tool/judge dependency before an episode."""

import hashlib
import json
from pathlib import Path


def source_pin() -> dict:
    return json.loads(Path(__file__).with_name("source-pin.json").read_text())


def verify_source(source: Path) -> dict:
    """Reject missing or modified upstream files without modifying the checkout."""
    pin = source_pin()
    for name, expected in pin["files"].items():
        try:
            actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"Missing pinned AgentVista source file: {name}") from exc
        if actual != expected:
            raise ValueError(f"AgentVista source differs from pinned revision: {name}")
    return {"repository": pin["repository"], "revision": pin["revision"], "files_verified": len(pin["files"])}
