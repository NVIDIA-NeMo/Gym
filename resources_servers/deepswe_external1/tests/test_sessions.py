# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode sessions and the agent-run seed, through the server's HTTP routes with fake sandboxes."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract
from resources_servers.deepswe.app import VerifierResult
from resources_servers.deepswe.validate_golden import _empty_response
from resources_servers.deepswe_external1.app import (
    DeepsweExternal1ResourcesServer,
    DeepsweExternal1ResourcesServerConfig,
    DeepsweExternal1VerifyRequest,
)
from resources_servers.deepswe_external1.inline_task import InlineTask
from resources_servers.deepswe_external1.prepare_examples import task_row


CLOSE = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}


def make_server(*, mode: str = "agent") -> tuple[DeepsweExternal1ResourcesServer, list[AsyncMock], list[str]]:
    """A server whose sandboxes are fakes, with patch collection and grading stubbed."""
    config = DeepsweExternal1ResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="deepswe_external1",
        is_verifying_golden_patch=mode == "golden",
        is_verifying_null_patch=mode == "null",
        sandbox_provider="sandbox",
        sandbox_config={},
        logs_dir=Path.cwd() / "logs",
    )
    server = DeepsweExternal1ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandboxes: list[AsyncMock] = []
    phases: list[str] = []

    async def create(
        _task: InlineTask, *, phase: str, files=None, session_id=None, verifier_session_id=None
    ) -> AsyncMock:
        box = AsyncMock()
        box.serialize.return_value = {"sandbox_id": f"sb-{len(sandboxes) + 1}"}
        sandboxes.append(box)
        phases.append(phase)
        if session_id is not None:
            server._session_id_to_sandbox[session_id] = box
        if verifier_session_id is not None:
            server._verifier_sandboxes[verifier_session_id] = box
        return box

    server._create_sandbox = AsyncMock(side_effect=create)
    server._collect_model_patch = AsyncMock(return_value=b"committed patch")
    server._run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    return server, sandboxes, phases


def seed(task: InlineTask, session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="deepswe_external1", task_id=task.data.task_id),
        task_data=task_row(task.data, "original instruction\n"),
    )


def verify_body(task: InlineTask) -> dict:
    return task_row(task.data, "original instruction\n") | {"response": _empty_response()}


def test_episode_session_lends_the_agent_sandbox_grades_it_once_and_closes(task: InlineTask) -> None:
    server, sandboxes, phases = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed(task).model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_access"] == {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "sb-1"}},
            "workdir": "/app",
        }
        # The agent gets a lease that cannot destroy the sandbox verify still has to collect from.
        sandboxes[0].serialize.assert_any_await(scope="operate")
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).json() == seeded.json()
        assert len(sandboxes) == 1, "a repeated seed returns the same sandbox"
        assert client.post("/seed_session", json=task_row(task.data, "x")).status_code == 409

        verified = client.post("/verify", json=verify_body(task))
        assert verified.status_code == 200, verified.text
        result = verified.json()
        assert result["reward"] == 1.0 and result["evaluation_completed"] and not result["mask_sample"]
        assert result["agent_sandbox_id"] == "sb-1" and result["verifier_sandbox_id"] == "sb-2"
        assert result["model_patch"] == "committed patch" and result["cleanup_errors"] == []
        assert phases == ["agent", "verifier"]
        sandboxes[0].stop.assert_awaited_once()
        sandboxes[1].stop.assert_awaited_once()

        repeated = client.post("/verify", json=verify_body(task))
        assert repeated.status_code == 200 and repeated.json() == result
        assert len(sandboxes) == 2, "an identical retry replays the verdict"

        assert client.post("/close_session", json=CLOSE).status_code == 200
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 409
        late = client.post("/verify", json=verify_body(task)).json()
        assert late["mask_sample"] and late["failure_kind"] == "session_lost" and late["reward"] == 0

    sandboxes[0].stop.assert_awaited_once()
    assert not server._session_id_to_sandbox and not server._seeded_agent_sandboxes
    assert not server._verifier_sandboxes


def test_close_without_verify_stops_the_agent_sandbox_and_a_failed_stop_is_retried(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("delete unavailable"), None]
        assert client.post("/close_session", json=CLOSE).status_code == 500
        assert server._session_id_to_sandbox, "a failed stop keeps the sandbox for the retried close"
        assert client.post("/close_session", json=CLOSE).status_code == 200
    assert sandboxes[0].stop.await_count == 2
    assert not server._session_id_to_sandbox and not server._seeded_agent_sandboxes


def test_verify_of_another_task_is_refused_and_keeps_the_session(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    other = verify_body(task) | {"image": "public.example/tasks/other@sha256:" + "c" * 64}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        assert client.post("/verify", json=other).status_code == 409
        sandboxes[0].stop.assert_not_awaited()
        assert client.post("/verify", json=verify_body(task)).json()["reward"] == 1.0


def test_transient_agent_cleanup_failure_retries_and_proceeds_with_verification(task: InlineTask) -> None:
    server, sandboxes, phases = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("delete unavailable"), None]
        result = client.post("/verify", json=verify_body(task)).json()
        assert result["reward"] == 1.0 and result["evaluation_completed"] and not result["mask_sample"]
        assert result["cleanup_errors"] == []
        assert phases == ["agent", "verifier"]
        assert sandboxes[0].stop.await_count == 2
        assert not server._session_id_to_sandbox
        assert client.post("/verify", json=verify_body(task)).json() == result
        assert server._run_verifier.await_count == 1
        assert client.post("/close_session", json=CLOSE).status_code == 200


def test_failed_agent_cleanup_masks_the_sample_and_close_retries_the_stop(task: InlineTask) -> None:
    server, sandboxes, phases = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("delete unavailable")] * 2 + [None]
        result = client.post("/verify", json=verify_body(task)).json()
        assert result["mask_sample"] and not result["evaluation_completed"]
        assert result["failure_stage"] == "agent_cleanup" and result["cleanup_errors"] == ["agent"]
        assert phases == ["agent"], "no verifier starts while the agent sandbox is still held"
        assert client.post("/close_session", json=CLOSE).status_code == 200
    assert sandboxes[0].stop.await_count == 3 and not server._session_id_to_sandbox


def _fail_verifier_stops(server: DeepsweExternal1ResourcesServer, outcomes: list) -> None:
    create = server._create_sandbox.side_effect

    async def create_with_failing_verifier_stop(*args, **kwargs) -> AsyncMock:
        box = await create(*args, **kwargs)
        if kwargs["phase"] == "verifier":
            box.stop.side_effect = outcomes
        return box

    server._create_sandbox.side_effect = create_with_failing_verifier_stop


@pytest.mark.parametrize("failed_stops", [1, 2])
def test_verifier_released_before_return_has_consistent_diagnostics(task: InlineTask, failed_stops: int) -> None:
    server, sandboxes, _ = make_server()
    _fail_verifier_stops(server, [RuntimeError("delete unavailable")] * failed_stops + [None])
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        result = client.post("/verify", json=verify_body(task)).json()
        # Include success in the shared wrapper, after _verify_task has persisted its verdict.
        assert result["reward"] == 1.0 and not result["mask_sample"]
        assert result["cleanup_errors"] == []
        assert sandboxes[1].stop.await_count == failed_stops + 1
        assert not server._verifier_sandboxes
        assert client.post("/verify", json=verify_body(task)).json() == result
        assert json.loads((Path(result["log_dir"]) / "result.json").read_text()) == result


def test_verifier_whose_stops_keep_failing_stays_owned_until_close(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    _fail_verifier_stops(server, [RuntimeError("down")] * 3 + [None])
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        result = client.post("/verify", json=verify_body(task)).json()
        assert result["reward"] == 1.0 and not result["mask_sample"]
        assert result["cleanup_errors"] == ["verifier"]
        assert client.post("/verify", json=verify_body(task)).json() == result
        assert json.loads((Path(result["log_dir"]) / "result.json").read_text()) == result
        # Two stops in the request and one in the verify wrapper failed; the session still owns the verifier.
        assert "resources-session" in server._verifier_sandboxes
        assert client.post("/close_session", json=CLOSE).status_code == 200
    assert sandboxes[1].stop.await_count == 4
    assert not server._verifier_sandboxes and not server._task_sessions


@pytest.mark.asyncio
async def test_cancelled_verify_stops_both_sandboxes_and_a_retry_grades_the_saved_patch(task: InlineTask) -> None:
    server, sandboxes, phases = make_server()
    stopped: list[str] = []
    create = server._create_sandbox.side_effect

    async def create_with_suspending_stop(*args, **kwargs) -> AsyncMock:
        box = await create(*args, **kwargs)
        name = kwargs["phase"]

        async def stop() -> None:
            # A real provider call suspends; an unshielded stop in a cancelled scope would end here.
            await anyio.sleep(0)
            stopped.append(name)

        box.stop.side_effect = stop
        return box

    server._create_sandbox.side_effect = create_with_suspending_stop
    grading = asyncio.Event()

    async def hang(*_args, **_kwargs) -> VerifierResult:
        grading.set()
        await anyio.sleep_forever()

    server._run_verifier = AsyncMock(side_effect=hang)
    request = Request({"type": "http", "session": {}})
    await server.seed_session(request, seed(task))
    verify_request = DeepsweExternal1VerifyRequest.model_validate(verify_body(task))

    async with anyio.create_task_group() as group:
        scope = anyio.CancelScope()

        async def run() -> None:
            with scope:
                await server.verify(request, verify_request)

        group.start_soon(run)
        await grading.wait()
        # As a client disconnect cancels the request handler.
        scope.cancel()

    assert stopped == ["agent", "verifier"]
    assert not server._verifier_sandboxes and not server._session_id_to_sandbox

    server._run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    retried = await server.verify(request, verify_request)
    assert retried.reward == 1.0 and not retried.mask_sample
    # The agent sandbox is gone, but the saved capture still names it.
    assert retried.agent_sandbox_id == "sb-1" and retried.verifier_sandbox_id == "sb-3"
    assert server._collect_model_patch.await_count == 1, "the retry grades the saved capture"
    assert phases == ["agent", "verifier", "verifier"]


@pytest.mark.asyncio
async def test_cancelled_verify_after_capture_retry_rejects_a_verifier_that_reuses_the_agent_sandbox(
    task: InlineTask,
) -> None:
    server, _, _ = make_server()
    create = server._create_sandbox.side_effect
    grading = asyncio.Event()

    async def hang(*_args, **_kwargs) -> VerifierResult:
        grading.set()
        await anyio.sleep_forever()

    server._run_verifier = AsyncMock(side_effect=hang)
    request = Request({"type": "http", "session": {}})
    await server.seed_session(request, seed(task))
    verify_request = DeepsweExternal1VerifyRequest.model_validate(verify_body(task))
    async with anyio.create_task_group() as group:
        scope = anyio.CancelScope()

        async def run() -> None:
            with scope:
                await server.verify(request, verify_request)

        group.start_soon(run)
        await grading.wait()
        scope.cancel()

    async def reuse_agent_sandbox(*args, **kwargs) -> AsyncMock:
        box = await create(*args, **kwargs)
        box.serialize.return_value = {"sandbox_id": "sb-1"}
        return box

    server._create_sandbox.side_effect = reuse_agent_sandbox
    server._run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    retried = await server.verify(request, verify_request)
    assert retried.mask_sample and "reused the agent sandbox" in retried.verifier_error
    server._run_verifier.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "hang_in", "expected"),
    [("golden", "_execute_golden", ["agent"]), ("null", "_run_verifier", ["agent", "verifier"])],
)
async def test_cancelled_validation_verify_still_stops_its_sandboxes(
    task: InlineTask, mode: str, hang_in: str, expected: list[str]
) -> None:
    server, _, _ = make_server(mode=mode)
    stopped: list[str] = []
    create = server._create_sandbox.side_effect

    async def create_with_suspending_stop(*args, **kwargs) -> AsyncMock:
        box = await create(*args, **kwargs)
        name = kwargs["phase"]

        async def stop() -> None:
            await anyio.sleep(0)
            stopped.append(name)

        box.stop.side_effect = stop
        return box

    server._create_sandbox.side_effect = create_with_suspending_stop
    server._execute_golden = AsyncMock()
    reached = asyncio.Event()

    async def hang(*_args, **_kwargs) -> None:
        reached.set()
        await anyio.sleep_forever()

    setattr(server, hang_in, AsyncMock(side_effect=hang))
    verify_request = DeepsweExternal1VerifyRequest.model_validate(verify_body(task))
    async with anyio.create_task_group() as group:
        scope = anyio.CancelScope()

        async def run() -> None:
            with scope:
                await server.verify(Request({"type": "http", "session": {}}), verify_request)

        group.start_soon(run)
        await reached.wait()
        scope.cancel()

    assert stopped == expected


def test_seed_naming_a_resources_session_must_be_an_episode_seed(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    hybrid = task_row(task.data, "original instruction\n") | {"resources_session_id": "resources-session"}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=hybrid).status_code == 422
    assert not sandboxes


@pytest.mark.parametrize("mode", ["golden", "null"])
def test_validation_modes_refuse_episode_seeds(task: InlineTask, mode: str) -> None:
    server, sandboxes, _ = make_server(mode=mode)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        refused = client.post("/seed_session", json=seed(task).model_dump(mode="json"))
    assert refused.status_code == 422 and "golden/null-validation" in refused.text
    assert not sandboxes


def test_agent_run_seed_returns_the_handle_and_verify_consumes_it(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=task_row(task.data, "original instruction\n"))
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {"sandbox_handle": "sb-1", "sandbox_descriptor": {"sandbox_id": "sb-1"}}
        verified = client.post("/verify", json=verify_body(task) | {"sandbox_handle": "sb-1"}).json()
        assert verified["reward"] == 1.0 and not verified["mask_sample"]
        sandboxes[0].stop.assert_awaited_once()
        again = client.post("/verify", json=verify_body(task)).json()
        assert again["mask_sample"] and again["failure_kind"] == "session_lost"
    assert len(sandboxes) == 2


def test_session_conformance(task: InlineTask) -> None:
    server, _, _ = make_server()
    check_resources_session_contract(
        server.setup_webserver(),
        seed(task, "conformance-session"),
        keeps_state=True,
        holds_state=lambda session_id: session_id in server._session_id_to_sandbox,
        verify_body=verify_body(task),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["agent", "verifier"])
@pytest.mark.parametrize("failure", ["cancel", "error"])
async def test_setup_failure_keeps_real_sandbox_owned_until_cleanup(
    task: InlineTask, monkeypatch: pytest.MonkeyPatch, phase: str, failure: str
) -> None:
    """Exercise real startup ownership when setup or its first teardown cannot complete."""
    from types import SimpleNamespace

    import resources_servers.deepswe_external1.app as module
    from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
    from nemo_gym.sandbox.providers import SandboxHandle
    from nemo_gym.server_runtime import ResourcesCloseSessionRequest

    server, _, _ = make_server()
    request = Request({"type": "http", "session": {}})
    if phase == "verifier":
        await server.seed_session(request, seed(task))
    # Restore the production allocator: only the provider transport is a fake.
    server._create_sandbox = DeepsweExternal1ResourcesServer._create_sandbox.__get__(server)
    entered_setup = asyncio.Event()
    closed: list[str] = []
    fail_close = failure == "error"
    handle = SandboxHandle(sandbox_id="setup-box", provider_name="fake", raw=None)

    async def setup_exec(*_args, **_kwargs) -> SandboxExecResult:
        entered_setup.set()
        if failure == "cancel":
            await anyio.sleep_forever()
        raise RuntimeError("setup failed")

    async def close(box: SandboxHandle) -> None:
        await anyio.sleep(0)  # An unshielded cleanup cannot complete inside a cancelled scope.
        if fail_close:
            raise RuntimeError("stop failed")
        closed.append(box.sandbox_id)

    provider = SimpleNamespace(
        name="fake",
        create=AsyncMock(return_value=handle),
        exec=setup_exec,
        close=close,
        aclose=AsyncMock(),
        upload_file=AsyncMock(),
    )
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *_args: provider)
    monkeypatch.setattr(module, "resolve_provider_metadata", lambda *_args: {})

    async def run() -> None:
        if phase == "agent":
            await server.seed_session(request, seed(task))
        else:
            await server.verify(request, DeepsweExternal1VerifyRequest.model_validate(verify_body(task)))

    if failure == "cancel":
        async with anyio.create_task_group() as group:
            scope = anyio.CancelScope()

            async def cancel_request() -> None:
                with scope:
                    await run()

            group.start_soon(cancel_request)
            await entered_setup.wait()
            owned = server._session_id_to_sandbox if phase == "agent" else server._verifier_sandboxes
            assert isinstance(owned["resources-session"], AsyncSandbox)
            scope.cancel()
        assert closed == ["setup-box"]
        assert "resources-session" not in owned
    else:
        if phase == "agent":
            with pytest.raises(RuntimeError, match="stop failed"):
                await run()
        else:
            body = DeepsweExternal1VerifyRequest.model_validate(verify_body(task))
            response = await server.verify(request, body)
            assert response.mask_sample and not response.evaluation_completed
            assert response.failure_stage == "verifier_setup"
            assert response.cleanup_errors == ["verifier"]
            result = response.model_dump(mode="json")
            assert (await server.verify(request, body)).model_dump(mode="json") == result
            assert json.loads((Path(response.log_dir) / "result.json").read_text()) == result
        owned = server._session_id_to_sandbox if phase == "agent" else server._verifier_sandboxes
        assert isinstance(owned["resources-session"], AsyncSandbox)
        assert not closed
        fail_close = False
        await server.close_resources_session(request, ResourcesCloseSessionRequest.model_validate(CLOSE))
        assert closed == ["setup-box"]
        assert "resources-session" not in owned
    provider.create.assert_awaited_once()
