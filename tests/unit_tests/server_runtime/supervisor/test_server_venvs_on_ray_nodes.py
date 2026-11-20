# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.server_runtime.supervisor import server_supervisor
from nemo_gym.server_runtime.supervisor.server_supervisor import (
    ServerSpinupTimeoutError,
    ServerStartFailedError,
    _ConfiguredServer,
    _setup_server_venvs_on_other_nodes,
    _setup_server_venvs_on_this_node,
)


class _RayError(Exception):
    pass


class _TaskUnschedulableError(_RayError):
    pass


class _FakeRay:
    """Records submitted setup tasks; `outcomes` maps a node id to "ok", "pending", "unschedulable" or "failed"."""

    def __init__(self, node_ids: list[str], outcomes: dict[str, str]) -> None:
        self.nodes_list = [
            {"NodeID": n, "NodeManagerAddress": f"10.0.0.{i}", "Alive": True} for i, n in enumerate(node_ids)
        ]
        self.outcomes = outcomes
        self.submitted: list[dict] = []
        self.cancelled: list[str] = []
        self.wait_timeout = "unset"
        self.markers_seen: list[bool] = []

    def get_runtime_context(self):
        return types.SimpleNamespace(get_node_id=lambda: "this")

    def nodes(self):
        return self.nodes_list

    def remote(self, num_cpus):
        fake = self

        class _Task:
            def options(self, scheduling_strategy, runtime_env):
                def remote(servers, global_config_dict, env_vars):
                    node_id = scheduling_strategy.node_id
                    fake.markers_seen.extend(m is not None and m.exists() for _, _, m in servers)
                    fake.submitted.append(
                        {
                            "node": node_id,
                            "soft": scheduling_strategy.soft,
                            "runtime_env": runtime_env,
                            "servers": servers,
                            "env": env_vars,
                        }
                    )
                    return node_id

                return types.SimpleNamespace(remote=remote)

        return lambda fn: _Task()

    def wait(self, refs, num_returns, timeout):
        self.wait_timeout = timeout
        pending = [r for r in refs if self.outcomes.get(r) == "pending"]
        return [r for r in refs if r not in pending], pending

    def get(self, ref):
        outcome = self.outcomes.get(ref, "ok")
        if outcome == "failed":
            raise _RayError("Setting up the venv of `agent@10.0.0.1` failed with exit code 1.")
        if outcome == "unschedulable":
            raise _TaskUnschedulableError()

    def cancel(self, ref, force):
        self.cancelled.append(ref)


@pytest.fixture
def install_fake_ray(monkeypatch):
    def install(fake: _FakeRay) -> _FakeRay:
        ray_module = types.ModuleType("ray")
        for name in ("get_runtime_context", "nodes", "remote", "wait", "get", "cancel"):
            setattr(ray_module, name, getattr(fake, name))
        ray_module.util = types.SimpleNamespace(get_node_ip_address=lambda: "10.0.0.9")
        exceptions = types.ModuleType("ray.exceptions")
        exceptions.RayError = _RayError
        exceptions.TaskUnschedulableError = _TaskUnschedulableError
        strategies = types.ModuleType("ray.util.scheduling_strategies")
        strategies.NodeAffinitySchedulingStrategy = lambda node_id, soft: types.SimpleNamespace(
            node_id=node_id, soft=soft
        )
        monkeypatch.setitem(sys.modules, "ray", ray_module)
        monkeypatch.setitem(sys.modules, "ray.exceptions", exceptions)
        monkeypatch.setitem(sys.modules, "ray.util", types.ModuleType("ray.util"))
        monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
        return fake

    return install


def _servers(tmp_path: Path) -> tuple[list[_ConfiguredServer], dict]:
    server_dir = tmp_path / "repo" / "responses_api_agents" / "agent"
    server_dir.mkdir(parents=True)
    servers = [
        _ConfiguredServer("agent", "responses_api_agents", "agent", OmegaConf.create({}), Path("app.py"), server_dir),
        _ConfiguredServer("other", "resources_servers", "other", OmegaConf.create({}), Path("app.py"), server_dir),
    ]
    return servers, OmegaConf.create({"uv_venv_dir": str(tmp_path / "venvs")})


def test_a_single_node_cluster_submits_nothing(tmp_path, install_fake_ray):
    fake = install_fake_ray(_FakeRay([], {}))
    servers, config = _servers(tmp_path)

    _setup_server_venvs_on_other_nodes(servers, ["agent"], config, 600)

    assert fake.submitted == []
    assert not (tmp_path / "venvs").exists()


def test_each_other_node_gets_one_pinned_task_with_only_forwarded_env(tmp_path, install_fake_ray, monkeypatch):
    fake = install_fake_ray(_FakeRay(["a", "b"], {}))
    servers, config = _servers(tmp_path)
    monkeypatch.setenv("UV_INDEX_URL", "https://index")
    monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
    monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "secret")

    _setup_server_venvs_on_other_nodes(servers, ["agent", "other"], config, 600)

    assert [t["node"] for t in fake.submitted] == ["a", "b"]
    assert all(t["soft"] is False for t in fake.submitted)
    assert all(t["runtime_env"] == {"py_executable": sys.executable} for t in fake.submitted)
    for task in fake.submitted:
        assert task["env"]["UV_INDEX_URL"] == "https://index"
        assert task["env"]["NEMO_GYM_ALLOW_PRERELEASE"] == "true"
        assert "NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN" not in task["env"]
    # Both Ray servers share one server dir, so its venv is set up once.
    assert [name for name, _, _ in fake.submitted[0]["servers"]] == ["agent"]
    assert fake.wait_timeout == 600
    assert all(fake.markers_seen)
    assert not list((tmp_path / "venvs").rglob("*.shared-storage-marker"))


def test_a_timeout_names_the_node_cancels_and_removes_markers(tmp_path, install_fake_ray):
    fake = install_fake_ray(_FakeRay(["a", "b"], {"b": "pending"}))
    servers, config = _servers(tmp_path)

    with pytest.raises(
        ServerSpinupTimeoutError, match=r"(?s)agent on Ray node\(s\) 10\.0\.0\.1.*server_spinup_timeout_seconds"
    ):
        _setup_server_venvs_on_other_nodes(servers, ["agent"], config, 30)

    assert fake.cancelled == ["a", "b"]
    assert not list((tmp_path / "venvs").rglob("*.shared-storage-marker"))


def test_a_failed_node_raises_a_start_failure_naming_it(tmp_path, install_fake_ray):
    fake = install_fake_ray(_FakeRay(["a", "b"], {"b": "failed"}))
    servers, config = _servers(tmp_path)

    with pytest.raises(ServerStartFailedError, match=r"agent failed on Ray node 10\.0\.0\.1"):
        _setup_server_venvs_on_other_nodes(servers, ["agent"], config, 0)

    assert fake.wait_timeout is None
    assert fake.cancelled == ["a", "b"]


def test_a_node_that_can_no_longer_run_tasks_is_skipped(tmp_path, install_fake_ray):
    fake = install_fake_ray(_FakeRay(["a", "b"], {"b": "unschedulable"}))
    servers, config = _servers(tmp_path)

    _setup_server_venvs_on_other_nodes(servers, ["agent"], config, 600)

    assert fake.cancelled == []


def test_read_only_venv_storage_sends_no_marker(tmp_path, install_fake_ray, monkeypatch):
    fake = install_fake_ray(_FakeRay(["a"], {}))
    servers, config = _servers(tmp_path)

    def read_only(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "touch", read_only)

    _setup_server_venvs_on_other_nodes(servers, ["agent"], config, 600)

    assert fake.submitted[0]["servers"][0][2] is None


def test_a_node_sharing_the_venv_storage_skips_setup(tmp_path, install_fake_ray, monkeypatch):
    install_fake_ray(_FakeRay([], {}))
    shared, local = tmp_path / "shared", tmp_path / "local"
    marker = tmp_path / "marker"
    marker.touch()
    ran = []
    monkeypatch.setattr(server_supervisor, "setup_env_command", lambda dir_path, config, label: f"setup {label}")
    monkeypatch.setattr(
        server_supervisor,
        "run_command",
        lambda command, dir_path, label, global_config_dict: ran.append(command)
        or types.SimpleNamespace(wait=lambda: 0, returncode=0),
    )
    # Set through monkeypatch so the value the function writes is undone after the test
    monkeypatch.setenv("UV_INDEX_URL", "https://before")

    _setup_server_venvs_on_this_node(
        [("shared", shared, marker), ("local", local, tmp_path / "absent"), ("read_only", local, None)],
        OmegaConf.create({}),
        {"UV_INDEX_URL": "https://index"},
    )

    assert ran == ["setup local@10.0.0.9", "setup read_only@10.0.0.9"]
    assert server_supervisor.os.environ["UV_INDEX_URL"] == "https://index"


def test_a_failed_setup_command_names_the_server_and_node(tmp_path, install_fake_ray, monkeypatch):
    install_fake_ray(_FakeRay([], {}))
    monkeypatch.setattr(server_supervisor, "setup_env_command", lambda dir_path, config, label: "setup")
    monkeypatch.setattr(
        server_supervisor,
        "run_command",
        lambda command, dir_path, label, global_config_dict: types.SimpleNamespace(wait=lambda: 1, returncode=1),
    )

    with pytest.raises(RuntimeError, match=r"agent@10\.0\.0\.9"):
        _setup_server_venvs_on_this_node([("agent", tmp_path, None)], OmegaConf.create({}), {})
