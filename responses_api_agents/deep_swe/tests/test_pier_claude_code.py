# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def npm_agent_class(monkeypatch: pytest.MonkeyPatch):
    class FakeClaudeCode:
        def __init__(self, version: str | None) -> None:
            self._version = version

        @staticmethod
        def name() -> str:
            return "claude-code"

        @staticmethod
        def get_version_command() -> str:
            return 'export PATH="$HOME/.local/bin:$PATH"; claude --version'

    class FakeInstallStep(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class FakeAgentInstallSpec(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    modules = {
        "pier": ModuleType("pier"),
        "pier.agents": ModuleType("pier.agents"),
        "pier.agents.installed": ModuleType("pier.agents.installed"),
        "pier.agents.installed.claude_code": ModuleType("pier.agents.installed.claude_code"),
        "pier.models": ModuleType("pier.models"),
        "pier.models.agent": ModuleType("pier.models.agent"),
        "pier.models.agent.install": ModuleType("pier.models.agent.install"),
    }
    modules["pier.agents.installed.claude_code"].ClaudeCode = FakeClaudeCode
    modules["pier.models.agent.install"].AgentInstallSpec = FakeAgentInstallSpec
    modules["pier.models.agent.install"].InstallStep = FakeInstallStep
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).parents[1] / "pier_claude_code.py"
    spec = importlib.util.spec_from_file_location("deep_swe_test_npm_agent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ClaudeCodeNpmInstall


def test_npm_install_spec_pins_version(npm_agent_class) -> None:
    spec = npm_agent_class("2.1.153").install_spec()
    assert spec.agent_name == "claude-code"
    assert spec.version == "2.1.153"
    assert spec.verification_command == 'export PATH="$HOME/.local/bin:$PATH"; claude --version'
    assert spec.steps[0].user == "root"
    assert "nodejs npm" in spec.steps[0].run
    assert spec.steps[1].user == "agent"
    assert "@anthropic-ai/claude-code@2.1.153" in spec.steps[1].run


def test_npm_install_spec_allows_unpinned_diagnostic(npm_agent_class) -> None:
    spec = npm_agent_class(None).install_spec()
    assert "@anthropic-ai/claude-code;" in spec.steps[1].run
