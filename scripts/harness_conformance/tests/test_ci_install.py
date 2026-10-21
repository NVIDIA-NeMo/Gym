# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that CI syncs each registered harness from its committed project and lock."""

import subprocess
import sys

import pytest
from scripts.harness_conformance.ci import ROOT, install_runtime
from scripts.harness_conformance.registry import HARNESSES


@pytest.mark.parametrize("harness", HARNESSES)
def test_install_uses_locked_adapter_project(monkeypatch, harness):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

    install_runtime(harness)

    command, kwargs = calls[0]
    adapter = ROOT / "responses_api_agents" / f"{harness}_agent"
    assert command == ["uv", "sync", "--locked", "--project", str(adapter)]
    assert kwargs["cwd"] == adapter
    assert kwargs["check"] is True
    assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == sys.prefix
    assert (adapter / "pyproject.toml").is_file()
    assert (adapter / "uv.lock").is_file()
