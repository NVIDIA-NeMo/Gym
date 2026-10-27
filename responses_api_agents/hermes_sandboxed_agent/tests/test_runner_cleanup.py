# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from responses_api_agents.hermes_sandboxed_agent.runner import validate_runtime


def test_runtime_checkout_must_match_manifest(tmp_path):
    source = tmp_path / "hermes-src"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    git("init")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "pin")
    commit = git("rev-parse", "HEAD")
    manifest = tmp_path / "hermes-runtime.json"
    manifest.write_text(json.dumps({"hermes_commit": commit}))
    assert validate_runtime(tmp_path)["hermes_commit"] == commit
    manifest.write_text(json.dumps({"hermes_commit": "0" * 40}))
    with pytest.raises(ValueError, match="does not match"):
        validate_runtime(tmp_path)
