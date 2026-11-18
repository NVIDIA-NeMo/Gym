# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that CI syncs each registered harness from its committed project and lock."""

import subprocess
import sys

import pytest
from scripts.harness_conformance import ci
from scripts.harness_conformance.ci import ROOT, install_runtime
from scripts.harness_conformance.registry import HARNESSES


@pytest.mark.parametrize("harness", HARNESSES)
def test_install_uses_locked_adapter_project(monkeypatch, harness):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(ci, "_build_crewai_sandbox_runtime", lambda: None)

    install_runtime(harness)

    command, kwargs = calls[0]
    adapter = ROOT / "responses_api_agents" / f"{harness}_agent"
    assert command == ["uv", "sync", "--locked", "--inexact", "--no-dev", "--project", str(adapter)]
    assert kwargs["cwd"] == adapter
    assert kwargs["check"] is True
    assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == sys.prefix
    assert (adapter / "pyproject.toml").is_file()
    assert (adapter / "uv.lock").is_file()


def test_openclaw_install_leaves_runtime_to_session(monkeypatch):
    import importlib

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    def unexpected_setup(name):
        raise AssertionError(f"Host runtime setup must not be imported: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_setup)
    install_runtime("openclaw")


def test_crewai_install_syncs_its_locked_project_and_builds_its_sandbox_runtime(monkeypatch):
    import importlib

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(ci, "_build_crewai_sandbox_runtime", lambda: calls.append("sandbox runtime"))

    def unexpected_setup(name):
        raise AssertionError(f"CrewAI has no runtime setup module: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_setup)
    install_runtime("crewai")

    # The probes' first seed finds the runtime built, instead of building it within one episode's time limit.
    assert calls[1:] == ["sandbox runtime"]
