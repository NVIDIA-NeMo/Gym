# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check how CI installs each registered harness's requirements, against the real adapters."""

import subprocess

import pytest
from scripts.harness_conformance.ci import ROOT, install_runtime
from scripts.harness_conformance.registry import HARNESSES


@pytest.mark.parametrize("harness", HARNESSES)
def test_install_uses_adapter_dir_and_overrides(monkeypatch, harness):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs.get("cwd"))))

    install_runtime(harness)

    command, cwd = calls[0]
    adapter = ROOT / "responses_api_agents" / f"{harness}_agent"
    assert cwd == adapter
    assert command[command.index("-r") + 1] == "requirements.txt"
    ships_overrides = (adapter / "overrides.txt").is_file()
    assert ("--override" in command) is ships_overrides
    assert not ships_overrides or command[command.index("--override") + 1] == "overrides.txt"
