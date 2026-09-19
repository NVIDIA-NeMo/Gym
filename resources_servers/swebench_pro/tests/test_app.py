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

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.single_agent_episode_types import ResponsesResourcesVerifyRequest, ResponsesVerificationInput
from resources_servers.swebench_pro.app import (
    SWEBenchProInstanceRequest,
    SWEBenchProResourcesServer,
    SWEBenchProResourcesServerConfig,
    SWEBenchProSeedSessionRequest,
    _attempt_budget,
    _budget_spent,
)
from resources_servers.swebench_pro.verification import VerificationResult


def request_body() -> dict:
    return {
        "repo": "example/repo",
        "instance_id": "instance_example",
        "base_commit": "abc123",
        "patch": "gold patch",
        "test_patch": "",
        "problem_statement": "Fix it",
        "fail_to_pass": '["new_test"]',
        "pass_to_pass": '["old_test"]',
        "before_repo_set_cmd": "",
        "selected_test_files_to_run": '["tests"]',
        "dockerhub_tag": "example-tag",
        "run_script": "#!/bin/bash\nexit 0\n",
        "parser_script": "print('{}')\n",
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
    }


def make_server(
    *,
    golden: bool,
    apply_anti_cheating: bool = True,
    **overrides: object,
) -> SWEBenchProResourcesServer:
    config = SWEBenchProResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swebench_pro_resources_server",
        sandbox_provider="test",
        sandbox_config={},
        is_verifying_golden_patch=golden,
        apply_anti_cheating=apply_anti_cheating,
        prefetch_go_modules=True,
        **overrides,
    )
    return SWEBenchProResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_golden_patch_verify_and_cleanup(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=True)
    sandbox = SimpleNamespace(stop=AsyncMock())
    create = AsyncMock(return_value=sandbox)
    verify = AsyncMock(
        return_value=VerificationResult(
            completed=True,
            resolved=True,
            patch_applied=True,
            test_results={
                "tests": [{"name": "new_test", "status": "PASSED"}, {"name": "old_test", "status": "PASSED"}]
            },
        )
    )
    monkeypatch.setattr(server, "_create_sandbox", create)
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert response.json()["reward"] == 1.0
    assert response.json()["model_patch"] == "gold patch"
    assert response.json()["resolved"] is True
    assert verify.await_args.kwargs["inputs"].prefetch_go_modules is True
    create.assert_awaited_once()
    sandbox.stop.assert_awaited_once()


def test_normal_verify_extracts_agent_patch(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False)
    sandbox = SimpleNamespace(stop=AsyncMock())
    monkeypatch.setattr(server, "_extract_model_patch", AsyncMock(return_value="agent patch"))
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    verify = AsyncMock(
        return_value=VerificationResult(
            completed=True,
            resolved=False,
            patch_applied=True,
            test_results={
                "tests": [{"name": "new_test", "status": "FAILED"}, {"name": "old_test", "status": "PASSED"}]
            },
            test_output="test run output",
        )
    )
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert response.json()["model_patch"] == "agent patch"
    assert response.json()["reward"] == 0.0
    assert response.json()["test_output"] == "test run output"


def test_verify_reports_sandbox_failure(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=True)
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=RuntimeError("sandbox unavailable")))

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert response.json()["evaluation_completed"] is False
    assert response.json()["reward"] == 0.0
    assert "sandbox unavailable" in response.json()["error"]


def test_schema_rejects_missing_evaluator_asset() -> None:
    body = request_body()
    del body["parser_script"]

    response = TestClient(make_server(golden=True).setup_webserver()).post("/verify", json=body)

    assert response.status_code == 422


def test_image_digest_avoids_case_sensitive_tag_rewriting() -> None:
    body = request_body()
    body["image_digest"] = "sha256:abc123"
    instance = SWEBenchProInstanceRequest.model_validate(body)

    assert make_server(golden=True)._image(instance) == "docker.io/jefzda/sweap-images@sha256:abc123"


@pytest.mark.asyncio
async def test_seed_session_applies_shared_anti_cheat_setup(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        upload=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    body = SWEBenchProSeedSessionRequest.model_validate(request_body())

    response = await server.seed_session(request, body)

    expected_script = Path(__file__).parents[2] / "swebench" / "anti_cheat_setup.sh"
    sandbox.upload.assert_awaited_once_with(expected_script, "/app/anti_cheat_setup.sh")
    # anti-cheat first, then normalize the container, then snapshot its untracked files
    assert sandbox.exec.await_count == 3
    assert sandbox.exec.await_args_list[0].args[0] == (
        "git reset --hard && WORKING_DIRECTORY=/app bash anti_cheat_setup.sh && rm anti_cheat_setup.sh"
    )
    assert sandbox.exec.await_args_list[0].kwargs["timeout_s"] == 600
    assert response.sandbox_handle == "sandbox-id"
    assert server._session_id_to_sandbox["session"] is sandbox


@pytest.mark.asyncio
async def test_seed_session_can_skip_anti_cheat_setup(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False, apply_anti_cheating=False)
    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
        upload=AsyncMock(),
        exec=AsyncMock(),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    body = SWEBenchProSeedSessionRequest.model_validate(request_body())

    await server.seed_session(request, body)

    sandbox.upload.assert_not_awaited()
    # anti-cheat is skipped, but the container is still normalized and snapshotted
    assert sandbox.exec.await_count == 2


@pytest.mark.asyncio
async def test_episode_seed_returns_direct_access_and_resources_close_owns_stop(
    monkeypatch: MonkeyPatch,
) -> None:
    server = make_server(golden=False, apply_anti_cheating=False)
    sandbox = SimpleNamespace(
        exec=AsyncMock(),
        serialize=AsyncMock(return_value={"sandbox_id": "sandbox-id"}),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.resolve_provider_config",
        lambda name, config: {"opensandbox": {}},
    )
    monkeypatch.setattr("resources_servers.swebench_pro.app.get_global_config_dict", lambda: {})
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    task_data = request_body()
    task_data.pop("responses_create_params")
    task_data.pop("response")

    response = await server.seed_session(
        request,
        ResourcesSeedSessionRequest(
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="swebench_pro", task_id="instance_example"),
            task_data=task_data,
        ),
    )

    assert response.resources_session_id == "session"
    assert response.sandbox_access.connection.provider_config_ref == server.config.sandbox_provider
    assert response.sandbox_access.connection.descriptor == {"sandbox_id": "sandbox-id"}
    assert server._session_id_to_identity["session"] == (
        EpisodeId(rollout_id="rollout"),
        TaskId(taskset="swebench_pro", task_id="instance_example"),
    )
    sandbox.stop.assert_not_awaited()

    with pytest.raises(ValueError, match="Verification identity"):
        await server.verify(
            request,
            ResponsesResourcesVerifyRequest(
                episode_id=EpisodeId(rollout_id="different"),
                task_id=TaskId(taskset="swebench_pro", task_id="instance_example"),
                verification_input=ResponsesVerificationInput(
                    responses_create_params={"input": "task"},
                    response=NeMoGymResponse.model_construct(id="response", output=[]),
                ),
            ),
        )

    with pytest.raises(ValueError, match="episode_id does not match"):
        await server.close_session(
            request,
            ResourcesCloseSessionRequest(
                resources_session_id="session",
                episode_id=EpisodeId(rollout_id="different"),
            ),
        )
    sandbox.stop.assert_not_awaited()

    await server.close_session(
        request,
        ResourcesCloseSessionRequest(
            resources_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
        ),
    )
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_episode_seed_rolls_back_sandbox_when_handoff_fails(monkeypatch: MonkeyPatch) -> None:
    server = make_server(golden=False, apply_anti_cheating=False)
    sandbox = SimpleNamespace(
        exec=AsyncMock(),
        serialize=AsyncMock(side_effect=RuntimeError("cannot serialize")),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(
        "resources_servers.swebench_pro.app.resolve_provider_config",
        lambda name, config: {"opensandbox": {}},
    )
    monkeypatch.setattr("resources_servers.swebench_pro.app.get_global_config_dict", lambda: {})
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    task_data = request_body()
    task_data.pop("responses_create_params")
    task_data.pop("response")

    with pytest.raises(RuntimeError, match="cannot serialize"):
        await server.seed_session(
            request,
            ResourcesSeedSessionRequest(
                episode_id=EpisodeId(rollout_id="rollout"),
                task_id=TaskId(taskset="swebench_pro", task_id="instance_example"),
                task_data=task_data,
            ),
        )

    sandbox.stop.assert_awaited_once()
    assert "session" not in server._session_id_to_sandbox
    assert "session" not in server._session_id_to_task


@pytest.mark.asyncio
async def test_episode_close_retains_state_when_sandbox_stop_fails() -> None:
    server = make_server(golden=False)
    sandbox = SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("stop failed")))
    identity = (
        EpisodeId(rollout_id="rollout"),
        TaskId(taskset="swebench_pro", task_id="instance_example"),
    )
    server._session_id_to_sandbox["session"] = sandbox
    server._session_id_to_task["session"] = SWEBenchProInstanceRequest.model_validate(request_body())
    server._session_id_to_identity["session"] = identity
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    with pytest.raises(RuntimeError, match="stop failed"):
        await server.close_session(
            request,
            ResourcesCloseSessionRequest(
                resources_session_id="session",
                episode_id=identity[0],
            ),
        )

    assert server._session_id_to_sandbox["session"] is sandbox
    assert server._session_id_to_identity["session"] == identity


@pytest.mark.parametrize("verdict", ["resolved", "unresolved", "infrastructure_failure"])
def test_native_episode_http_lifecycle_preserves_verdict_and_private_task_data(
    monkeypatch: MonkeyPatch, verdict: str
) -> None:
    server = make_server(golden=False, apply_anti_cheating=False, inconclusive_verification_retries=0)
    events: list[str] = []

    async def task_exec(command: str, **kwargs: object) -> SimpleNamespace:
        if "--no-pager diff" in command:
            events.append("extract-patch")
            return SimpleNamespace(return_code=0, stdout="agent patch", stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def stop_task() -> None:
        events.append("stop-task")

    async def stop_verifier() -> None:
        events.append("stop-verifier")

    task_sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=task_exec),
        serialize=AsyncMock(return_value={"sandbox_id": "task-sandbox"}),
        stop=AsyncMock(side_effect=stop_task),
        pty=SimpleNamespace(create=AsyncMock()),
    )
    verifier_sandbox = SimpleNamespace(stop=AsyncMock(side_effect=stop_verifier))

    async def create_sandbox(body: SWEBenchProInstanceRequest, files: dict | None = None) -> SimpleNamespace:
        if files is None:
            events.append("create-task")
            return task_sandbox
        events.append("create-verifier")
        assert body.patch == "gold patch"
        assert body.run_script == request_body()["run_script"]
        return verifier_sandbox

    completed = verdict != "infrastructure_failure"
    resolved = verdict == "resolved"
    test_results = {
        "tests": [
            {"name": "new_test", "status": "PASSED" if resolved else "FAILED"},
            {"name": "old_test", "status": "PASSED"},
        ]
    }

    async def verify(**kwargs: object) -> VerificationResult:
        events.append("verify")
        assert kwargs["sandbox"] is verifier_sandbox
        assert kwargs["inputs"].patch == "agent patch"
        return VerificationResult(
            completed=completed,
            resolved=resolved,
            patch_applied=completed,
            test_results=test_results if completed else None,
            test_output="verifier output",
            error=None if completed else "sandbox unavailable",
        )

    monkeypatch.setattr(server, "_create_sandbox", create_sandbox)
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)
    task_data = request_body()
    responses_create_params = task_data.pop("responses_create_params")
    agent_response = task_data.pop("response")
    episode_id = {"rollout_id": "rollout", "attempt": 1}
    task_id = {"taskset": "swebench_pro", "task_id": "instance_example"}

    with TestClient(server.setup_webserver()) as client:
        seed = client.post(
            "/seed_session", json={"episode_id": episode_id, "task_id": task_id, "task_data": task_data}
        )
        assert seed.status_code == 200
        session_id = seed.json()["resources_session_id"]
        assert client.cookies
        assert seed.json()["sandbox_access"] == {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "test",
                "descriptor": {"sandbox_id": "task-sandbox"},
            },
            "workdir": "/app",
        }
        assert "gold patch" not in seed.text
        assert "run_script" not in seed.json()
        task_sandbox.pty.create.assert_not_awaited()
        task_sandbox.stop.assert_not_awaited()

        # Verification receives only identity and the agent response; private assets stay in the resources session.
        response = client.post(
            "/verify",
            json={
                "episode_id": episode_id,
                "task_id": task_id,
                "verification_input": {"responses_create_params": responses_create_params, "response": agent_response},
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["reward"] == float(resolved)
        # The shared response now includes a default mask, but SWE Pro still
        # reports verification completion separately rather than deriving it.
        assert result["mask_sample"] is False
        assert result["evaluation_completed"] is completed
        assert result["resolved"] is resolved
        assert result["model_patch"] == "agent patch"
        assert result["test_results"] == (test_results if completed else None)
        assert result["test_output"] == "verifier output"
        assert result["error"] == (None if completed else "sandbox unavailable")
        assert result["response"] == NeMoGymResponse.model_validate(agent_response).model_dump(mode="json")

        close_body = {"resources_session_id": session_id, "episode_id": episode_id}
        close = client.post("/close_session", json=close_body)
        assert close.status_code == 200
        # Upstream rejects unknown sessions after successful close removes identity.
        # A repeated request must not stop an already released sandbox again.
        with pytest.raises(ValueError, match="Unknown resources session"):
            client.post("/close_session", json=close_body)
        assert session_id not in server._session_id_to_task
        assert session_id not in server._session_id_to_identity
        assert session_id not in server._session_id_to_sandbox

    assert events == ["create-task", "extract-patch", "stop-task", "create-verifier", "verify", "stop-verifier"]
    task_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_model_patch_includes_commits_and_untracked_files() -> None:
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="complete patch", stderr="")),
        stop=AsyncMock(),
    )
    server._session_id_to_sandbox["session"] = sandbox

    patch = await server._extract_model_patch("session", "abc123")

    assert patch == "complete patch"
    command = sandbox.exec.await_args.args[0]
    assert "git -C /app add -N ." in command
    assert "git -C /app --no-pager diff abc123" in command
    sandbox.stop.assert_awaited_once()
    assert "session" not in server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_extract_model_patch_drops_untracked_files_the_image_already_shipped() -> None:
    """Artifacts the task image ships are not the agent's work and break `git apply`."""
    artifact = (
        "diff --git a/dump.rdb b/dump.rdb\nnew file mode 100644\n--- /dev/null\n+++ b/dump.rdb\n@@ -0,0 +1 @@\n+x\n"
    )
    fix = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    server = make_server(golden=False)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout=artifact + fix, stderr="")),
        stop=AsyncMock(),
    )
    server._session_id_to_sandbox["session"] = sandbox
    server._session_id_to_pristine_untracked["session"] = frozenset({"dump.rdb"})

    patch = await server._extract_model_patch("session", "abc123")

    assert patch == fix
    assert "session" not in server._session_id_to_pristine_untracked


@pytest.mark.asyncio
async def test_seed_session_normalizes_the_agent_environment_before_snapshotting() -> None:
    """The agent runs the same suites, so it needs the same repaired container the verifier gets.

    Order matters: normalization deletes stale files, so the untracked baseline must be
    taken afterwards or it records paths that no longer exist.
    """
    server = make_server(golden=False)
    calls: list[str] = []

    async def record(command, *args, **kwargs):
        calls.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox = SimpleNamespace(
        exec=record,
        upload=AsyncMock(),
        stop=AsyncMock(),
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
    )
    server._create_sandbox = AsyncMock(return_value=sandbox)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    await server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body()))

    normalize = next(i for i, c in enumerate(calls) if "Xvfb" in c)
    snapshot = next(i for i, c in enumerate(calls) if "ls-files --others" in c)
    assert normalize < snapshot, calls


@pytest.mark.asyncio
async def test_seed_session_survives_a_container_it_cannot_normalize() -> None:
    server = make_server(golden=False)

    async def boom(command, *args, **kwargs):
        if "Xvfb" in command:
            raise RuntimeError("exec failed")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    sandbox = SimpleNamespace(
        exec=boom,
        upload=AsyncMock(),
        stop=AsyncMock(),
        _handle=SimpleNamespace(sandbox_id="sandbox-id"),
    )
    server._create_sandbox = AsyncMock(return_value=sandbox)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})

    # A container that cannot be normalized is still worth running.
    await server.seed_session(request, SWEBenchProSeedSessionRequest.model_validate(request_body()))
    assert server._session_id_to_sandbox["session"] is sandbox


@pytest.mark.asyncio
async def test_pristine_untracked_files_lists_and_tolerates_failure() -> None:
    server = make_server(golden=False)
    listing = SimpleNamespace(return_code=0, stdout="dump.rdb\nappendonlydir/appendonly.aof.manifest\n\n", stderr="")
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=listing))

    assert await server.pristine_untracked_files(sandbox) == frozenset(
        {"dump.rdb", "appendonlydir/appendonly.aof.manifest"}
    )
    assert "ls-files --others --exclude-standard" in sandbox.exec.await_args.args[0]

    failing = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="", stderr="boom")))
    assert await server.pristine_untracked_files(failing) == frozenset()


@pytest.mark.asyncio
async def test_shutdown_stops_abandoned_session_sandboxes() -> None:
    server = make_server(golden=False)
    first = SimpleNamespace(stop=AsyncMock())
    second = SimpleNamespace(stop=AsyncMock())
    server._session_id_to_sandbox = {"first": first, "second": second}

    await server.shutdown()

    first.stop.assert_awaited_once()
    second.stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {}


def inconclusive_result() -> VerificationResult:
    """A verdict-less run: `inconclusive_reason` reports "no usable output"."""
    return VerificationResult(completed=True, resolved=False, patch_applied=True, test_results=None)


def test_verify_bounds_an_attempt_that_never_returns(monkeypatch: MonkeyPatch) -> None:
    """A hung attempt must fail the rollout, not hold the run open until the wall clock."""
    server = make_server(golden=True, verification_attempt_timeout=0.05, inconclusive_verification_retries=0)

    async def _hang(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(server, "_create_sandbox", _hang)

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert response.json()["evaluation_completed"] is False
    assert response.json()["reward"] == 0.0
    assert "Verification failed" in response.json()["error"]


def test_verify_stops_retrying_once_the_rollout_budget_is_spent(monkeypatch: MonkeyPatch) -> None:
    """The retry sequence is bounded in aggregate, not just per attempt."""
    clock = {"t": 1000.0}
    monkeypatch.setattr("resources_servers.swebench_pro.app.time", lambda: clock["t"])
    server = make_server(golden=True, verification_total_timeout=1500.0)
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=SimpleNamespace(stop=AsyncMock())))

    async def _verify(**kwargs: object) -> VerificationResult:
        clock["t"] += 1000.0
        return inconclusive_result()

    verify = AsyncMock(side_effect=_verify)
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    # Budget is 1500s and each attempt burns 1000s: the second attempt still
    # starts (500s left, which is what its own ceiling is clamped to) and the
    # third never does.
    assert verify.await_count == 2


def test_verify_uses_every_attempt_when_no_budget_is_set(monkeypatch: MonkeyPatch) -> None:
    """Leaving both ceilings unset preserves the previous unbounded behaviour."""
    server = make_server(golden=True, verification_attempt_timeout=None, verification_total_timeout=None)
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=SimpleNamespace(stop=AsyncMock())))
    verify = AsyncMock(return_value=inconclusive_result())
    monkeypatch.setattr("resources_servers.swebench_pro.app.run_verification", verify)

    response = TestClient(server.setup_webserver()).post("/verify", json=request_body())

    assert response.status_code == 200
    assert verify.await_count == 3


def test_attempt_budget_takes_the_smaller_of_the_two_ceilings(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("resources_servers.swebench_pro.app.time", lambda: 100.0)
    assert _attempt_budget(None, None) is None
    assert _attempt_budget(30.0, None) == 30.0
    assert _attempt_budget(None, 150.0) == 50.0
    assert _attempt_budget(30.0, 150.0) == 30.0
    assert _attempt_budget(80.0, 150.0) == 50.0
    # A spent budget yields zero rather than a negative timeout.
    assert _attempt_budget(30.0, 90.0) == 0.0
    assert _budget_spent(None) is False
    assert _budget_spent(90.0) is True
    assert _budget_spent(150.0) is False
