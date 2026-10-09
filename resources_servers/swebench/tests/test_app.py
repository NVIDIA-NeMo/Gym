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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.sandbox.utils import CPU_CAP_ENV_VARS
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.swebench.app import (
    DockerContainer,
    SwebenchResourcesServer,
    SwebenchResourcesServerConfig,
    SWEBenchSeedSessionRequest,
    SWEBenchSeedSessionResponse,
    SWEBenchVerifyResponse,
)


def make_sandbox(
    *,
    exec_result: SandboxExecResult | None = None,
    exec_error: Exception | None = None,
    upload_error: Exception | None = None,
    stop_error: Exception | None = None,
) -> MagicMock:
    sandbox = MagicMock()
    sandbox._handle = SandboxHandle(sandbox_id="sandbox-123", provider_name="test-provider", raw=None)
    sandbox.exec = AsyncMock(return_value=exec_result, side_effect=exec_error)
    sandbox.upload = AsyncMock(side_effect=upload_error)
    sandbox.stop = AsyncMock(side_effect=stop_error)
    return sandbox


class TestApp:
    def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        config = SwebenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            sandbox_provider="test",
            sandbox_config=dict(),
            is_verifying_golden_patch=True,
        )
        server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()

        client = TestClient(app)

        eval_sandbox = make_sandbox()
        monkeypatch.setattr(
            "resources_servers.swebench.app.SwebenchResourcesServer._create_sandbox",
            AsyncMock(return_value=eval_sandbox),
        )
        monkeypatch.setattr(
            "resources_servers.swebench.app.run_instance",
            AsyncMock(return_value=dict(resolved=True, completed=True)),
        )

        res = client.post(
            "/verify",
            json={
                "repo": "astropy/astropy",
                "instance_id": "my instance_id",
                "base_commit": "my base_commit",
                "patch": "my patch",
                "test_patch": "my test_patch",
                "problem_statement": "my problem_statement",
                "hints_text": "",
                "created_at": "my created_at",
                "version": "4.3",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
                "environment_setup_commit": "my environment_setup_commit",
                "difficulty": "my difficulty",
                "responses_create_params": {"input": []},
                "response": {
                    "output": [],
                    "id": "",
                    "created_at": 0,
                    "model": "",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
                "subset": "my subset",
                "split": "my split",
            },
        )
        assert res.status_code == 200
        observation = res.json()["verifier_sandbox_observation"]
        assert observation.pop("wall_time_s") >= 0
        assert observation == {
            "kind": "sandbox",
            "role": "verifier",
            "provider": "test-provider",
            "sandbox_id": "sandbox-123",
            "outcome": "completed",
            "exit_code": None,
            "cpu_time_s": None,
            "peak_memory_mib": None,
            "resource_usage_source": None,
            "error_type": None,
        }

    async def test_create_sandbox_derives_cpu_cap_env_from_cpu_limit(self, monkeypatch: MonkeyPatch) -> None:
        sandbox = MagicMock()
        sandbox.start_with_setup = AsyncMock(side_effect=lambda spec, setup: sandbox)
        monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: MagicMock())
        monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", MagicMock(return_value=sandbox))
        monkeypatch.setattr("resources_servers.swebench.app.patch_swebench_multilingual_sandbox", AsyncMock())
        test_spec = SimpleNamespace(
            instance_image_key="img:key", instance_id="astropy__astropy-12907", repo="astropy/astropy"
        )

        async def created_spec(sandbox_config: dict[str, Any]) -> Any:
            config = SwebenchResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                sandbox_provider="test",
                sandbox_config=sandbox_config,
                is_verifying_golden_patch=True,
            )
            server = SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
            await server._create_sandbox(test_spec)
            return sandbox.start_with_setup.await_args.args[0]

        # Floored to whole cores; explicit sandbox_config.env keys win over the derived caps.
        spec = await created_spec({"resources": {"cpu": 2.7}, "env": {"OMP_NUM_THREADS": "16"}})
        assert spec.env["OMP_NUM_THREADS"] == "16"
        derived = [name for name in CPU_CAP_ENV_VARS if name != "OMP_NUM_THREADS"]
        assert {name: spec.env[name] for name in derived} == {name: "2" for name in derived}

        # Opt-out and no-cpu-limit paths omit only the derived caps; the
        # SWE-bench Verified Python environment is applied independently.
        for sandbox_config in (
            {"resources": {"cpu": 2}, "derive_cpu_env": False},
            {"resources": {"memory_mib": 1024}},
        ):
            spec = await created_spec(sandbox_config)
            assert not set(CPU_CAP_ENV_VARS) & spec.env.keys()
            assert spec.env["CONDA_DEFAULT_ENV"] == "testbed"

    def test_unobserved_response_omits_optional_field(self) -> None:
        response = SWEBenchVerifyResponse.model_construct(verifier_sandbox_observation=None)

        assert "verifier_sandbox_observation" not in response.model_dump()

    async def test_eval_exit_code_is_observed_without_treating_failed_tests_as_sandbox_failure(self) -> None:
        sandbox = make_sandbox(exec_result=SandboxExecResult(stdout="test output", stderr=None, return_code=7))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        test_output, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=3.5, evaluation_completed=True)

        assert test_output == "test output"
        assert timed_out is False
        assert observation.outcome == "completed"
        assert observation.exit_code == 7
        assert observation.wall_time_s == 3.5

    async def test_timeout_is_observed_without_changing_harness_timeout_behavior(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(
                stdout=None,
                stderr="backend failed",
                return_code=125,
                error_type="sandbox",
            )
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.exec_run("git apply patch.diff")
        sandbox.exec.side_effect = TimeoutError("timed out")
        test_output, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=60.0, evaluation_completed=False)

        assert test_output == ""
        assert timed_out is True
        assert observation.outcome == "timeout"
        assert observation.exit_code is None
        assert observation.error_type == "TimeoutError"

    async def test_runtime_error_is_observed_and_still_propagates(self) -> None:
        sandbox = make_sandbox(exec_error=RuntimeError("Sandbox was OOM-killed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        with pytest.raises(RuntimeError, match="OOM-killed"):
            await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)

        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)
        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"
        assert observation.exit_code is None

    @pytest.mark.parametrize(
        ("error_type", "expected_outcome"),
        [("sandbox", "sandbox_error"), ("TimeoutError", "timeout")],
    )
    async def test_provider_error_does_not_report_sentinel_as_process_exit_code(
        self, error_type: str, expected_outcome: str
    ) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(stdout=None, stderr="backend failed", return_code=125, error_type=error_type)
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        _, timed_out, _ = await container.exec_run_with_timeout("/bin/bash /eval.sh", timeout=60)
        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)

        assert timed_out is False
        assert observation.outcome == expected_outcome
        assert observation.error_type == error_type
        assert observation.exit_code is None

    async def test_pre_eval_provider_error_is_observed(self) -> None:
        sandbox = make_sandbox(
            exec_result=SandboxExecResult(stdout=None, stderr="backend failed", return_code=125, error_type="sandbox")
        )
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.exec_run("git apply patch.diff")
        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)

        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "sandbox"
        assert observation.exit_code is None

    async def test_upload_error_is_observed(self, tmp_path: Path) -> None:
        sandbox = make_sandbox(upload_error=RuntimeError("upload failed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        with pytest.raises(RuntimeError, match="upload failed"):
            await container.copy(tmp_path / "patch.diff", Path("/tmp/patch.diff"))

        observation = container.observation(wall_time_s=1.0, evaluation_completed=False)
        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"

    async def test_cleanup_error_is_fail_open_and_observed(self) -> None:
        sandbox = make_sandbox(stop_error=RuntimeError("stop failed"))
        container = DockerContainer(id="run-id", instance_id="instance-id")
        container._inner_container = sandbox

        await container.cleanup()
        observation = container.observation(wall_time_s=2.0, evaluation_completed=True)

        assert observation.outcome == "sandbox_error"
        assert observation.error_type == "RuntimeError"


_INSTANCE = {
    "repo": "astropy/astropy",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "base",
    "patch": "gold patch",
    "test_patch": "test patch",
    "problem_statement": "Fix it",
    "hints_text": "",
    "created_at": "2022-01-01",
    "version": "4.3",
    "FAIL_TO_PASS": "[]",
    "PASS_TO_PASS": "[]",
    "environment_setup_commit": "setup",
    "difficulty": "easy",
    "subset": "verified",
    "split": "test",
}
_RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
_TEST_SPEC = SimpleNamespace(
    instance_image_key="img:key", instance_id="astropy__astropy-12907", repo="astropy/astropy"
)


def _session_server(monkeypatch: MonkeyPatch) -> tuple[SwebenchResourcesServer, MagicMock]:
    """A server whose every sandbox is one mock that runs in the ``/testbed`` image WORKDIR."""
    sandbox = MagicMock()
    sandbox.start_with_setup = AsyncMock()
    sandbox._handle = SandboxHandle(sandbox_id="sb-1", provider_name="test-provider", raw=None)
    sandbox.exec = AsyncMock(
        side_effect=lambda command, **kwargs: SandboxExecResult(
            return_code=0, stdout="/testbed\n" if command == "pwd" else "diff --git a/x b/x\n", stderr=""
        )
    )
    sandbox.serialize = AsyncMock(return_value={"sandbox_id": "sb-1", "provider": "test-provider"})
    sandbox.stop = AsyncMock()
    monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: MagicMock())
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr("resources_servers.swebench.app.AsyncSandbox", MagicMock(return_value=sandbox))
    monkeypatch.setattr(SwebenchResourcesServer, "_make_test_spec", lambda self, body: _TEST_SPEC)
    config = SwebenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        sandbox_provider="test",
        sandbox_config={},
        apply_anti_cheating=False,
    )
    return SwebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)), sandbox


def _typed_seed(
    session_id: str = "resources-session", task_id: str = "astropy__astropy-12907"
) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="swebench_verified", task_id=task_id),
        task_data=_INSTANCE | {"responses_create_params": {"input": "Fix it"}},
    )


def _close_body(seed: ResourcesSeedSessionRequest) -> dict[str, Any]:
    return {"resources_session_id": seed.resources_session_id, "episode_id": seed.episode_id.model_dump(mode="json")}


def test_typed_seed_hands_the_agent_the_task_sandbox_and_close_stops_it_once(monkeypatch: MonkeyPatch) -> None:
    server, sandbox = _session_server(monkeypatch)
    seed = _typed_seed()

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {
            "resources_session_id": "resources-session",
            "resources_tools": None,
            "sandbox_access": {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "test",
                    "descriptor": {"sandbox_id": "sb-1", "provider": "test-provider"},
                },
                "workdir": "/testbed",
            },
        }
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).json() == seeded.json()
        assert sandbox.start_with_setup.await_count == 1

        for _ in range(2):
            closed = client.post("/close_session", json=_close_body(seed))
            assert closed.json() == {"resources_session_id": "resources-session"}
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 500
        assert sandbox.start_with_setup.await_count == 1

    sandbox.stop.assert_awaited_once()


def test_typed_verify_consumes_the_sandbox_and_close_does_not_stop_it_again(monkeypatch: MonkeyPatch) -> None:
    server, sandbox = _session_server(monkeypatch)
    run_instance = AsyncMock(return_value=dict(resolved=True, completed=True))
    monkeypatch.setattr("resources_servers.swebench.app.run_instance", run_instance)
    seed = _typed_seed()
    verify = _INSTANCE | {"responses_create_params": {"input": "Fix it"}, "response": _RESPONSE}

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
        verified = client.post("/verify", json=verify)
        assert verified.status_code == 200, verified.text
        assert verified.json()["model_patch"] == "diff --git a/x b/x\n"
        assert run_instance.await_args.kwargs["pred"]["model_patch"] == "diff --git a/x b/x\n"
        # A second verify or seed would hand out a sandbox that no longer exists.
        assert client.post("/verify", json=verify).status_code == 500
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 500
        assert client.post("/close_session", json=_close_body(seed)).status_code == 200

    # Verify stops the task sandbox after extracting the patch; close finds nothing left to stop.
    sandbox.stop.assert_awaited_once()


def test_close_retries_a_stop_that_failed_during_typed_verify(monkeypatch: MonkeyPatch) -> None:
    server, sandbox = _session_server(monkeypatch)
    monkeypatch.setattr(
        "resources_servers.swebench.app.run_instance", AsyncMock(return_value=dict(resolved=True, completed=True))
    )
    sandbox.stop = AsyncMock(side_effect=[RuntimeError("stop failed"), None])
    seed = _typed_seed()
    verify = _INSTANCE | {"responses_create_params": {"input": "Fix it"}, "response": _RESPONSE}

    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
        assert client.post("/verify", json=verify).status_code == 200
        assert client.post("/close_session", json=_close_body(seed)).status_code == 200

    assert sandbox.stop.await_count == 2
    assert not server._session_id_to_sandbox


async def test_typed_seed_rejects_a_task_id_that_names_another_instance(monkeypatch: MonkeyPatch) -> None:
    server, sandbox = _session_server(monkeypatch)
    with pytest.raises(ValueError, match="does not match the task row"):
        await server.seed_session(MagicMock(session={}), _typed_seed(task_id="other"))
    sandbox.start_with_setup.assert_not_awaited()


@pytest.mark.parametrize("failure", ["serialize", "pwd"])
async def test_typed_seed_stops_the_sandbox_when_the_handoff_fails(monkeypatch: MonkeyPatch, failure: str) -> None:
    server, sandbox = _session_server(monkeypatch)
    if failure == "serialize":
        sandbox.serialize = AsyncMock(side_effect=RuntimeError("cannot serialize"))
        expected = "cannot serialize"
    else:
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=1, stdout="", stderr="no shell"))
        expected = "Could not resolve the task sandbox workdir: no shell"

    with pytest.raises(RuntimeError, match=expected):
        await server.seed_session(MagicMock(session={}), _typed_seed())

    sandbox.stop.assert_awaited_once()
    assert not server._session_id_to_sandbox
    assert not server._task_sessions


async def test_typed_sessions_require_a_single_worker_but_legacy_seeds_do_not(monkeypatch: MonkeyPatch) -> None:
    server, sandbox = _session_server(monkeypatch)
    server.config.num_workers = 2
    with pytest.raises(ValueError, match="num_workers=1"):
        await server.seed_session(MagicMock(session={}), _typed_seed())
    sandbox.start_with_setup.assert_not_awaited()

    legacy = await server.seed_session(
        MagicMock(session={SESSION_ID_KEY: "cookie"}), SWEBenchSeedSessionRequest(**_INSTANCE)
    )
    assert legacy == SWEBenchSeedSessionResponse(sandbox_handle="sb-1")


def test_legacy_seed_body_still_returns_the_sandbox_handle_over_http(monkeypatch: MonkeyPatch) -> None:
    server, sandbox = _session_server(monkeypatch)
    with TestClient(server.setup_webserver()) as client:
        seeded = client.post("/seed_session", json=_INSTANCE | {"responses_create_params": {"input": "Fix it"}})
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {"sandbox_handle": "sb-1"}
        sandbox.serialize.assert_not_awaited()
    # Shutdown stops the legacy session's sandbox, which no verify released.
    sandbox.stop.assert_awaited_once()


async def test_shutdown_stops_sandboxes_no_close_released(monkeypatch: MonkeyPatch) -> None:
    server, sandbox = _session_server(monkeypatch)
    await server.seed_session(MagicMock(session={}), _typed_seed())
    await server.shutdown()
    sandbox.stop.assert_awaited_once()
    assert not server._session_id_to_sandbox


def test_typed_sessions_meet_the_resources_session_contract(monkeypatch: MonkeyPatch) -> None:
    server, _ = _session_server(monkeypatch)
    check_resources_session_contract(server.setup_webserver(), _typed_seed("contract-session"), keeps_state=True)
