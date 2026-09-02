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
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_agents.vibench_agent.app import (
    EXPORT_FILENAME,
    PRD_FILENAME,
    VibenchAgent,
    VibenchAgentConfig,
    rewrite_loopback_url_for_docker,
)


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


class _FakeRequest:
    """Minimal stand-in: the parent only awaits request.json()."""

    async def json(self):
        return {}


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
    async def test_excludes_dependency_trees_and_the_prd(self, tmp_path):
        """Dependency trees are machine-specific and setup-environment.sh rebuilds them."""
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox()

        await agent._harvest_app(sandbox, "sess-1")

        cmd = sandbox.execs[0]
        assert "--exclude=node_modules" in cmd
        assert "--exclude=.git" in cmd
        assert f"--exclude={PRD_FILENAME}" in cmd
        # The harness writes its own transcript beside the app; it must not be graded.
        assert f"--exclude={EXPORT_FILENAME}" in cmd

    @pytest.mark.asyncio
    async def test_excludes_virtualenvs(self, tmp_path):
        """A venv's bin/python symlinks outside the app dir, so harvesting one makes the
        verifier refuse the whole tarball and score a working app as a build failure."""
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox()

        await agent._harvest_app(sandbox, "sess-1")

        cmd = sandbox.execs[0]
        for name in (".venv", "venv", "__pycache__"):
            assert f"--exclude={name}" in cmd, f"{name} would be harvested"
        # Deliberately unanchored: GNU tar applies a ./-anchored pattern only at the top
        # level, so nested node_modules/.venv would still be tarred -- and a nested venv
        # symlink makes the verifier reject the whole artifact.
        assert "--exclude=./" not in cmd, "anchored patterns miss nested trees"

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
    async def test_failed_asset_upload_stops_the_started_sandbox(self, tmp_path, monkeypatch):
        """Past start() a container exists; the caller only registers cleanup once this
        returns, so a raise here would leak it until TTL."""
        agent = make_agent(tmp_path)
        sandbox = _FakeSandbox()

        async def boom(local, remote):
            raise RuntimeError("upload failed")

        sandbox.upload = boom

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

        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "data.csv").write_text("a,b")

        with pytest.raises(RuntimeError, match="upload failed"):
            await agent._create_build_sandbox("PRD", [str(assets)])

        assert sandbox.stopped is True, "sandbox leaked after a failed upload"

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


class TestAggregateMetricsProxy:
    @pytest.mark.asyncio
    async def test_forwards_to_the_resources_server(self, tmp_path):
        """Rollout collection POSTs /aggregate_metrics to the agent; without this proxy the
        resources server's failure-breakdown metrics never run."""
        agent = make_agent(tmp_path)
        posted = {}

        class _Resp:
            ok = True
            status = 200

            async def read(self):
                return json.dumps({"agent_metrics": {"mean_reward": 0.5}}).encode()

        async def fake_post(server_name, url_path, json, **kwargs):
            posted["server"] = server_name
            posted["path"] = url_path
            return _Resp()

        agent.server_client.post = fake_post

        from nemo_gym.config_types import AggregateMetricsRequest

        result = await agent.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))

        assert result.agent_metrics == {"mean_reward": 0.5}
        assert posted["server"] == "vibench_resources_server"
        assert posted["path"] == "/aggregate_metrics"


class TestSandboxModelUrl:
    def test_rewrites_loopback_to_the_docker_host_gateway(self):
        assert rewrite_loopback_url_for_docker("http://127.0.0.1:9000") == "http://host.docker.internal:9000"
        assert rewrite_loopback_url_for_docker("http://localhost:9000/v1") == "http://host.docker.internal:9000"
        assert rewrite_loopback_url_for_docker("http://0.0.0.0:9000") == "http://host.docker.internal:9000"

    def test_leaves_a_routable_host_alone(self):
        assert rewrite_loopback_url_for_docker("http://10.0.0.8:9000") == "http://10.0.0.8:9000"

    @pytest.mark.asyncio
    async def test_opencode_config_rewrites_only_the_host(self, tmp_path, monkeypatch):
        """Loopback is the container itself inside a bridged sandbox."""
        agent = make_agent(tmp_path)
        monkeypatch.setattr(agent, "_uses_docker_provider", lambda: True)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.get_server_url",
            lambda name: "http://127.0.0.1:9000",
        )

        config = await agent._create_opencode_config(_FakeRequest())

        assert config["provider"]["nemo_gym"]["options"]["baseURL"] == "http://host.docker.internal:9000/v1"

    @pytest.mark.asyncio
    async def test_the_runs_capture_path_survives_the_rewrite(self, tmp_path, monkeypatch):
        """The parent builds this through base_url_for_run, so the URL carries the rollout
        prefix. Rebuilding it from get_server_url would silently disable token capture."""
        agent = make_agent(tmp_path)
        monkeypatch.setattr(agent, "_uses_docker_provider", lambda: True)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.get_server_url",
            lambda name: "http://127.0.0.1:9000",
        )
        monkeypatch.setattr(
            type(agent), "base_url_for_run", lambda self, base_url, body: f"{base_url}/ng-rollout/abc123"
        )

        config = await agent._create_opencode_config(_FakeRequest())

        assert (
            config["provider"]["nemo_gym"]["options"]["baseURL"]
            == "http://host.docker.internal:9000/ng-rollout/abc123/v1"
        )

    @pytest.mark.asyncio
    async def test_explicit_override_replaces_the_origin_but_keeps_the_path(self, tmp_path, monkeypatch):
        """sandbox_model_base_url is for providers whose boxes have their own address."""
        agent = make_agent(tmp_path, sandbox_model_base_url="http://sandbox-gw:7000/v1")
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.get_server_url",
            lambda name: "http://127.0.0.1:9000",
        )
        monkeypatch.setattr(type(agent), "base_url_for_run", lambda self, base_url, body: f"{base_url}/ng-rollout/xyz")

        config = await agent._create_opencode_config(_FakeRequest())

        assert config["provider"]["nemo_gym"]["options"]["baseURL"] == "http://sandbox-gw:7000/ng-rollout/xyz/v1"

    @pytest.mark.asyncio
    async def test_non_docker_provider_leaves_the_url_alone(self, tmp_path, monkeypatch):
        agent = make_agent(tmp_path)
        monkeypatch.setattr(agent, "_uses_docker_provider", lambda: False)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.get_server_url",
            lambda name: "http://10.0.0.5:9000",
        )

        config = await agent._create_opencode_config(_FakeRequest())

        assert config["provider"]["nemo_gym"]["options"]["baseURL"] == "http://10.0.0.5:9000/v1"

    def test_signature_matches_the_parent(self, tmp_path):
        """This override broke once when the parent gained a request argument; a mismatch
        means the URL rewrite silently never runs and the harness talks to itself."""
        import inspect

        from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent

        mine = inspect.signature(VibenchAgent._create_opencode_config)
        base = inspect.signature(OpenCodeSandboxedAgent._create_opencode_config)
        assert list(mine.parameters) == list(base.parameters)
        assert inspect.iscoroutinefunction(VibenchAgent._create_opencode_config)
