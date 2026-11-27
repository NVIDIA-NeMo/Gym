# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the ui_generation_agentic stack lockfiles in step with the manifests embedded in their Dockerfiles."""

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "resources_servers" / "ui_generation_agentic" / "docker"
# The manifest only exists as a heredoc in docker/<stack>.Dockerfile; docker/regen_locks.sh derives the lock from it.
HEREDOC = re.compile(r"<<'EOF' /app/package\.json\n(.*?)\nEOF\n", re.S)


@pytest.mark.parametrize("stack", ["ui-next", "ui-vite"])
def test_lock_mirrors_dockerfile_manifest(stack: str) -> None:
    match = HEREDOC.search((DOCKER / f"{stack}.Dockerfile").read_text())
    assert match, f"{stack}.Dockerfile has no package.json heredoc"
    manifest = json.loads(match.group(1))
    lock = json.loads((DOCKER / stack / "package-lock.json").read_text())

    assert lock["lockfileVersion"] == 3
    assert lock["packages"][""]["dependencies"] == manifest["dependencies"]
    # Every pin is exact, so the hoisted copy must resolve to exactly the pinned version.
    resolved = {name: lock["packages"][f"node_modules/{name}"]["version"] for name in manifest["dependencies"]}
    assert resolved == manifest["dependencies"]
    for name, version in manifest.get("overrides", {}).items():
        assert lock["packages"][f"node_modules/{name}"]["version"] == version, f"override {name} not applied"
