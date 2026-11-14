# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required search tooling is available before either replay entrypoint infers."""

import os
import shutil
import subprocess

import pytest

from responses_api_agents.ngc_recorded_context import setup_ripgrep


def test_startup_adopts_installed_component_binary_with_empty_path(monkeypatch):
    monkeypatch.setenv("PATH", "")
    executable = setup_ripgrep.ensure_ripgrep()
    assert shutil.which("rg") == executable
    assert subprocess.check_output([executable, "--version"], text=True).startswith("ripgrep ")


def test_existing_binary_is_reused_without_changing_path(monkeypatch):
    previous = os.environ["PATH"]
    executable = shutil.which("rg")
    assert setup_ripgrep.ensure_ripgrep() == executable
    assert os.environ["PATH"] == previous


@pytest.mark.parametrize("entrypoint", ["native", "audit"])
def test_entrypoints_expose_search_before_inference(entrypoint, tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from benchmarks.partner_if.tests.test_agent_sessions import make_agent
    from nemo_gym.client import ServerClient
    from responses_api_agents.ngc_recorded_context.app import ReplayAgent, ReplayAgentConfig
    from responses_api_agents.ngc_recorded_context.readonly_tools import ReadonlyTools
    from responses_api_agents.ngc_recorded_context.replay import ReplayConfig

    monkeypatch.setenv("PATH", "")
    if entrypoint == "native":
        agent = make_agent(tmp_path)
    else:
        agent = ReplayAgent(
            config=ReplayAgentConfig(
                name="audit",
                entrypoint="app.py",
                host="127.0.0.1",
                port=9999,
                num_workers=1,
                model="synthetic",
                inference_api_key="synthetic",
                resources_server={"type": "resources_servers", "name": "partner_if"},
                replay=ReplayConfig(package_dir=tmp_path, artifacts_dir=tmp_path),
            ),
            server_client=MagicMock(spec=ServerClient),
        )
    stdout, stderr, code = ReadonlyTools({"/repo/a.py": "needle"}, {}).search(["needle", "/repo"], "")
    assert code == 0 and "needle" in stdout and not stderr
    agent.server_client.assert_not_called()


def test_missing_component_install_fails_before_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(setup_ripgrep.sysconfig, "get_path", lambda _: str(tmp_path))
    with pytest.raises(RuntimeError, match="uv sync --locked"):
        setup_ripgrep.ensure_ripgrep()
