# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime bundle stays out of the coding agent's PATH, and the runner's Python is isolated."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from responses_api_agents.anyswe_agent.agent_runner import _agent_bin_dir
from responses_api_agents.anyswe_agent.app import AnySweAgent, AnySweInstanceConfig
from responses_api_agents.anyswe_agent.tests.test_app import _config


class TestRuntimeIsolation:
    @staticmethod
    def _bundle(tmp_path: Path, *, claude_is_node_script: bool) -> Path:
        bin_dir = tmp_path / "deps" / "bin"
        bin_dir.mkdir(parents=True)
        for name in ("python", "python3", "pip", "node", "npm"):
            (bin_dir / name).write_text("#!/bin/sh\n")
        (bin_dir / "claude").write_text("#!/usr/bin/env node\n" if claude_is_node_script else "\x7fELF")
        return tmp_path / "deps"

    def test_unknown_harness_keeps_the_whole_bundle_bin(self, tmp_path: Path) -> None:
        deps = self._bundle(tmp_path, claude_is_node_script=False)
        assert _agent_bin_dir(str(deps), "responses_api_agents.hermes_agent.app") == f"{deps}/bin"

    def test_native_cli_hides_the_bundle_python_pip_and_node(self, tmp_path: Path) -> None:
        deps = self._bundle(tmp_path, claude_is_node_script=False)
        shim = Path(_agent_bin_dir(str(deps), "responses_api_agents.claude_code_agent.app"))
        assert sorted(path.name for path in shim.iterdir()) == ["claude"]
        assert (shim / "claude").resolve() == (deps / "bin" / "claude").resolve()

    def test_node_script_cli_also_gets_node(self, tmp_path: Path) -> None:
        deps = self._bundle(tmp_path, claude_is_node_script=True)
        shim = Path(_agent_bin_dir(str(deps), "responses_api_agents.claude_code_agent.app"))
        assert sorted(path.name for path in shim.iterdir()) == ["claude", "node"]

    def test_runner_python_ignores_the_task_images_python_environment(self, tmp_path: Path) -> None:
        commands: list[str] = []

        class _FakeSandbox:
            def __init__(self, provider, spec) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def exec(self, command: str, **kwargs):
                commands.append(command)
                return SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)

            async def download(self, remote: str, local: Path) -> None:
                pass

        params = AnySweInstanceConfig(
            **_config().model_dump(),
            run_session_id="s",
            base_results_dir=tmp_path,
            model_server_url="http://policy:8000",
            resolved_sandbox_provider={"opensandbox": {}},
            sandbox_default_metadata={},
            problem_info={"instance_id": "astropy__astropy-12907"},
            body={"input": "fix it", "model": "policy"},
            persistent_dir=tmp_path,
            metrics_fpath=tmp_path / "metrics.json",
            container="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        )
        params.metrics_fpath.write_text("{}")
        (tmp_path / "instruction.txt").write_text("fix it")
        (tmp_path / "agent_runner.py").write_text("")

        agent = AnySweAgent.__new__(AnySweAgent)
        object.__setattr__(
            agent, "__dict__", {"config": params, "server_client": SimpleNamespace(global_config_dict={})}
        )

        with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", _FakeSandbox):
            asyncio.run(agent._run_agent_in_sandbox(params))

        runner = [command for command in commands if "agent_runner.py" in command]
        assert runner == [
            "HOME=/sandbox/home TMPDIR=/sandbox/tmp /agent_deps_mount/bin/python -I /trajectories_mount/agent_runner.py"
        ]
