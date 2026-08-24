# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_agents.vibench_agent.app import PRD_FILENAME, VibenchAgent, VibenchAgentConfig


def make_agent(tmp_path: Path, **overrides) -> VibenchAgent:
    config = VibenchAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vibench_opencode_agent",
        resources_server={"type": "resources_servers", "name": "vibench_resources_server"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        opencode_version="1.17.11",
        opencode_max_context_window=262144,
        sandbox_provider="sandbox",
        sandbox_config={"ttl_s": 100, "resources": {"cpu": 2}},
        sandbox_timeout=60.0,
        build_image="app-bench-base:latest",
        artifact_dir=str(tmp_path / "artifacts"),
        **overrides,
    )
    return VibenchAgent(config=config, server_client=MagicMock(spec=ServerClient))


class _FakeExec:
    def __init__(self, return_code=0, stdout="", stderr=""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _FakeSandbox:
    """Records what the agent asks the sandbox to do."""

    def __init__(self, exec_result=None, download_ok=True):
        self.execs: list[str] = []
        self.uploads: list[tuple] = []
        self.downloads: list[tuple] = []
        self._exec_result = exec_result or _FakeExec()
        self._download_ok = download_ok
        self.stopped = False

    async def exec(self, command, **kwargs):
        self.execs.append(command)
        return self._exec_result

    async def upload(self, local, remote):
        self.uploads.append((str(local), remote))

    async def download(self, remote, local):
        if not self._download_ok:
            raise RuntimeError("download failed")
        Path(local).write_bytes(b"tar-bytes")
        self.downloads.append((remote, str(local)))

    async def stop(self):
        self.stopped = True


class TestHarvest:
    @pytest.mark.asyncio
    async def test_writes_a_tarball_into_the_shared_artifact_dir(self, tmp_path):
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox()

        path = await agent._harvest_app(sandbox, "sess-1")

        assert path is not None
        assert Path(path).exists()
        # The resources server refuses anything outside artifact_dir.
        assert Path(path).parent == Path(agent.config.artifact_dir).expanduser()

    @pytest.mark.asyncio
    async def test_excludes_node_modules_and_the_prd(self, tmp_path):
        """node_modules is huge and platform-specific; the grader supplies its own PRD."""
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox()

        await agent._harvest_app(sandbox, "sess-1")

        cmd = sandbox.execs[0]
        assert "--exclude=./node_modules" in cmd
        assert "--exclude=./.git" in cmd
        assert f"--exclude=./{PRD_FILENAME}" in cmd

    @pytest.mark.asyncio
    async def test_tar_failure_yields_no_artifact(self, tmp_path):
        """None is how the resources server learns to score this a build failure."""
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox(exec_result=_FakeExec(return_code=2, stderr="tar: no such dir"))

        assert await agent._harvest_app(sandbox, "sess-1") is None

    @pytest.mark.asyncio
    async def test_download_failure_yields_no_artifact(self, tmp_path):
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox(download_ok=False)

        assert await agent._harvest_app(sandbox, "sess-1") is None

    @pytest.mark.asyncio
    async def test_artifact_names_do_not_collide_across_rollouts(self, tmp_path):
        agent = make_agent(tmp_path)

        a = await agent._harvest_app(_FakeSandbox(), "sess-1")
        b = await agent._harvest_app(_FakeSandbox(), "sess-1")

        assert a != b, "concurrent rollouts would overwrite each other's app"


class TestBuildSandbox:
    @pytest.mark.asyncio
    async def test_stages_the_prd_and_assets_without_leaking_test_assets(self, tmp_path, monkeypatch):
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox()
        captured = {}

        class _Async:
            def __init__(self, provider, spec):
                captured["spec"] = spec

            async def start(self):
                return None

            def __getattr__(self, item):
                return getattr(sandbox, item)

        monkeypatch.setattr("responses_api_agents.vibench_agent.app.create_provider", lambda c: MagicMock())
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.resolve_provider_config", lambda *a: {})
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.resolve_provider_metadata", lambda *a: {})
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.AsyncSandbox", _Async)

        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "data.csv").write_text("a,b")

        await agent._create_build_sandbox("MY PRD TEXT", [str(assets)])

        spec = captured["spec"]
        assert spec.image == "app-bench-base:latest"
        assert spec.workdir == "/app"
        # Staged via SandboxSpec.files so it exists before the harness starts.
        assert spec.files[f"/app/{PRD_FILENAME}"] == "MY PRD TEXT"
        assert sandbox.uploads == [(str(assets / "data.csv"), "/app/assets/data.csv")]

    @pytest.mark.asyncio
    async def test_absent_asset_dir_is_skipped(self, tmp_path, monkeypatch):
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox()

        class _Async:
            def __init__(self, provider, spec):
                pass

            async def start(self):
                return None

            def __getattr__(self, item):
                return getattr(sandbox, item)

        monkeypatch.setattr("responses_api_agents.vibench_agent.app.create_provider", lambda c: MagicMock())
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.resolve_provider_config", lambda *a: {})
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.resolve_provider_metadata", lambda *a: {})
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("responses_api_agents.vibench_agent.app.AsyncSandbox", _Async)

        await agent._create_build_sandbox("PRD", [str(tmp_path / "missing")])

        assert sandbox.uploads == []
