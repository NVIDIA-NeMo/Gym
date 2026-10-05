# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientOSError, ServerDisconnectedError
from fastapi import HTTPException
from fastapi.testclient import TestClient

import nemo_gym.server_utils as server_utils
import resources_servers.terminal_bench_2_1.app as terminal_bench_app
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21VerifyRequest,
)


@pytest.fixture
def setup(tmp_path: Path):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("exit 0\n")
    server = TerminalBench21ResourcesServer(
        config=TerminalBench21ResourcesServerConfig(
            sandbox_provider="sandbox", sandbox_config={}, host="", port=0, entrypoint="", name="tb21"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="task-sandbox"),
        exec=AsyncMock(return_value=SimpleNamespace(stdout="/work/task\n", stderr="", return_code=0)),
        serialize=AsyncMock(return_value={"sandbox_id": "task-sandbox"}),
        download=AsyncMock(side_effect=lambda remote, local: Path(local).write_text("1")),
        stop=AsyncMock(),
    )

    async def create(task, *, session_id=None):
        if session_id is not None:
            server._session_id_to_sandbox[session_id] = sandbox
        return sandbox

    server._create_sandbox = AsyncMock(side_effect=create)
    server._upload_folder = AsyncMock()
    seed = ResourcesSeedSessionRequest(
        resources_session_id="resources-session",
        episode_id={"rollout_id": "rollout", "attempt": 0},
        task_id={"taskset": "tb21", "task_id": "0"},
        task_data={"task_name": "terminal-bench/regex-log", "docker_image": "task-image", "task_folder": str(task)},
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "cookie-session"})
    return server, sandbox, seed, request


def close_body(seed: ResourcesSeedSessionRequest) -> ResourcesCloseSessionRequest:
    return ResourcesCloseSessionRequest(resources_session_id=seed.resources_session_id, episode_id=seed.episode_id)


def verify_body(seed: ResourcesSeedSessionRequest) -> TerminalBench21VerifyRequest:
    return TerminalBench21VerifyRequest.model_validate(
        seed.task_data
        | {
            "responses_create_params": {"input": "Solve the task"},
            "response": {
                "output": [],
                "id": "response",
                "created_at": 0,
                "model": "model",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )


@pytest.mark.parametrize("reward", [0, 1])
def test_resources_session_http_lifecycle_keeps_task_until_close(setup, reward: int) -> None:
    server, sandbox, seed, _ = setup
    sandbox.download.side_effect = lambda remote, local: Path(local).write_text(str(reward))
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=seed.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert response.json() == {
            "resources_session_id": seed.resources_session_id,
            "resources_tools": None,
            "sandbox_access": {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "sandbox",
                    "descriptor": {"sandbox_id": "task-sandbox"},
                },
                "workdir": "/work/task",
            },
        }
        assert client.cookies
        assert "task_folder" not in response.text
        server._upload_folder.assert_not_awaited()
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).json() == response.json()
        server._create_sandbox.assert_awaited_once()
        result = client.post("/verify", json=verify_body(seed).model_dump(mode="json"))
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == reward
        assert result.json()["evaluation_completed"] is True
        server._upload_folder.assert_awaited_once()
        assert server._upload_folder.await_args.args[2] == "/tests"
        sandbox.stop.assert_not_awaited()
        replay = client.post("/verify", json=verify_body(seed).model_dump(mode="json"))
        assert replay.status_code == 200
        assert replay.json() == result.json()
        server._upload_folder.assert_awaited_once()
        closed = client.post("/close_session", json=close_body(seed).model_dump(mode="json"))
        assert closed.status_code == 200
        assert closed.json() == {"resources_session_id": seed.resources_session_id}
        assert client.post("/close_session", json=close_body(seed).model_dump(mode="json")).json() == closed.json()
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409
        assert client.post("/verify", json=verify_body(seed).model_dump(mode="json")).status_code == 409
    sandbox.stop.assert_awaited_once()
    assert not server._session_id_to_sandbox
    assert not server._session_id_to_state


async def test_concurrent_seed_retries_create_only_one_sandbox(setup) -> None:
    server, sandbox, seed, request = setup
    results = await asyncio.gather(*(server.seed_session(request, seed) for _ in range(5)))
    assert all(result == results[0] for result in results)
    server._create_sandbox.assert_awaited_once()
    sandbox.serialize.assert_awaited_once()
    assert request.session[SESSION_ID_KEY] == seed.resources_session_id


@pytest.mark.parametrize("field", ["episode_id", "task_id", "task_data"])
async def test_seed_rejects_changed_request(setup, field: str) -> None:
    server, _, seed, request = setup
    await server.seed_session(request, seed)
    changed = seed.model_copy(deep=True)
    if field == "episode_id":
        changed.episode_id = changed.episode_id.model_copy(update={"rollout_id": "other"})
    elif field == "task_id":
        changed.task_id = changed.task_id.model_copy(update={"task_id": "other"})
    else:
        changed.task_data["docker_image"] = "other-image"
    with pytest.raises(HTTPException, match="different request"):
        await server.seed_session(request, changed)
    server._create_sandbox.assert_awaited_once()


async def test_close_fences_delayed_seed_and_checks_identity(setup) -> None:
    server, sandbox, seed, request = setup
    await server.close_resources_session(close_body(seed))
    with pytest.raises(HTTPException, match="already closed"):
        await server.seed_session(request, seed)
    wrong = close_body(seed)
    wrong.episode_id = wrong.episode_id.model_copy(update={"rollout_id": "other"})
    with pytest.raises(HTTPException, match="episode_id"):
        await server.close_resources_session(wrong)
    server._create_sandbox.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


async def test_failed_close_retains_owner_for_retry(setup) -> None:
    server, sandbox, seed, request = setup
    await server.seed_session(request, seed)
    sandbox.stop.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await server.close_resources_session(close_body(seed))
    assert server._session_id_to_sandbox[seed.resources_session_id] is sandbox
    assert seed.resources_session_id not in server._closed_sessions
    await server.close_resources_session(close_body(seed))
    await server.close_resources_session(close_body(seed))
    assert sandbox.stop.await_count == 2
    assert not server._session_id_to_sandbox


async def test_seed_preserves_primary_error_and_failed_cleanup_for_close(setup) -> None:
    server, sandbox, seed, request = setup
    sandbox.serialize.side_effect = RuntimeError("cannot serialize")
    sandbox.stop.side_effect = [RuntimeError("cannot stop"), None]
    with pytest.raises(RuntimeError, match="cannot serialize"):
        await server.seed_session(request, seed)
    assert server._session_id_to_sandbox[seed.resources_session_id] is sandbox
    with pytest.raises(HTTPException, match="no longer available"):
        await server.seed_session(request, seed)
    await server.close_resources_session(close_body(seed))
    assert not server._session_id_to_sandbox


@pytest.mark.parametrize("recovery", ["close", "shutdown"])
@pytest.mark.parametrize("stop_failure", ["error", "timeout"])
async def test_initial_setup_failure_retains_sandbox_for_cleanup_retry(
    setup, monkeypatch: pytest.MonkeyPatch, recovery: str, stop_failure: str
) -> None:
    server, _, seed, request = setup
    server.config.session_close_timeout_seconds = 0.01

    async def stop(handle):
        if provider.close.await_count == 1:
            if stop_failure == "timeout":
                await asyncio.Event().wait()
            raise RuntimeError("provider unavailable")

    provider = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(sandbox_id="task-sandbox")),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=1, stdout="", stderr="setup failed")),
        close=AsyncMock(side_effect=stop),
        aclose=AsyncMock(),
    )
    sandbox = AsyncSandbox(provider)
    monkeypatch.setattr(terminal_bench_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(terminal_bench_app, "resolve_provider_config", lambda *_: provider)
    monkeypatch.setattr(terminal_bench_app, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(terminal_bench_app, "AsyncSandbox", lambda _: sandbox)
    # Exercise real allocation/setup, not the fixture's successful-create shortcut.
    server._create_sandbox = TerminalBench21ResourcesServer._create_sandbox.__get__(server)
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        with pytest.raises(RuntimeError, match="Failed to prepare TerminalBench package sources"):
            await server.seed_session(request, seed)
        provider.create.assert_awaited_once()
        provider.close.assert_awaited_once()
        provider.aclose.assert_not_awaited()
        assert server._session_id_to_sandbox[seed.resources_session_id] is sandbox
        assert seed.resources_session_id not in server._closed_sessions
        if recovery == "close":
            await server.close_resources_session(close_body(seed))
            await server.close_resources_session(close_body(seed))
    assert provider.close.await_count == 2
    provider.aclose.assert_awaited_once()
    assert not server._session_id_to_sandbox


@pytest.mark.parametrize("disconnect", [ServerDisconnectedError, ClientOSError])
async def test_lost_verdict_response_replays_through_shared_http_retry(
    setup, monkeypatch: pytest.MonkeyPatch, disconnect: type[ServerDisconnectedError] | type[ClientOSError]
) -> None:
    server, sandbox, seed, _ = setup
    responses = []
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200

        async def send(**kwargs):
            response = client.post("/verify", json=kwargs["json"])
            responses.append(response)
            if len(responses) == 1:
                assert response.status_code == 200
                raise disconnect("lost completed verdict")
            return response

        monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: SimpleNamespace(request=send))
        response = await server_utils._request_with_retries(
            "POST",
            "http://testserver/verify",
            _internal=True,
            _max_connection_retries=2,
            json=verify_body(seed).model_dump(mode="json"),
        )
        assert [r.status_code for r in responses] == [200, 200]
        assert response.json() == responses[0].json()
        assert response.json()["reward"] == 1
        server._upload_folder.assert_awaited_once()
        sandbox.download.assert_awaited_once()
        sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("field", ["response", "responses_create_params", "docker_image"])
async def test_completed_verification_rejects_changed_request(setup, field: str) -> None:
    server, sandbox, seed, request = setup
    await server.seed_session(request, seed)
    body = verify_body(seed)
    original = body.model_copy(deep=True)
    result = await server.verify(request, body)
    if field == "response":
        body.response.id = "other-response"
    elif field == "responses_create_params":
        body.responses_create_params.input = "different input"
    else:
        body.docker_image = "different image"
    with pytest.raises(HTTPException, match="does not match") as error:
        await server.verify(request, body)
    assert error.value.status_code == 409
    # Neither caller mutation of the first request nor of its result changes the cache.
    result.reward = 0
    result.response.id = "mutated returned response"
    replay = await server.verify(request, original)
    assert replay.reward == 1
    assert replay.response.id == original.response.id
    server._upload_folder.assert_awaited_once()
    sandbox.download.assert_awaited_once()


async def test_concurrent_verification_replays_only_after_first_finishes(setup) -> None:
    server, sandbox, seed, request = setup
    await server.seed_session(request, seed)
    entered, release = asyncio.Event(), asyncio.Event()

    async def upload(*args):
        entered.set()
        await release.wait()

    server._upload_folder.side_effect = upload
    first = asyncio.create_task(server.verify(request, verify_body(seed)))
    await entered.wait()
    second = asyncio.create_task(server.verify(request, verify_body(seed)))
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    results = await asyncio.gather(first, second)
    assert results[0] == results[1]
    assert results[0].reward == 1
    server._upload_folder.assert_awaited_once()
    sandbox.download.assert_awaited_once()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_interrupted_verification_requires_retrying_episode(
    setup, failure: type[RuntimeError] | type[asyncio.CancelledError]
) -> None:
    server, _, seed, request = setup
    await server.seed_session(request, seed)
    server._verify = AsyncMock(side_effect=failure("verification interrupted"))
    with pytest.raises(failure):
        await server.verify(request, verify_body(seed))
    with pytest.raises(HTTPException, match="retry the episode") as error:
        await server.verify(request, verify_body(seed))
    assert error.value.status_code == 503
    server._verify.assert_awaited_once()
    await server.close_resources_session(close_body(seed))
    assert not server._session_id_to_state


@pytest.mark.parametrize("invalid", ["missing_tests", "golden_mode", "relative_workdir", "failed_pwd"])
async def test_resources_session_seed_rejects_invalid_setup(setup, invalid: str) -> None:
    server, sandbox, seed, request = setup
    if invalid == "missing_tests":
        (Path(seed.task_data["task_folder"]) / "tests/test.sh").unlink()
    elif invalid == "golden_mode":
        server.config.is_verifying_golden_patch = True
    elif invalid == "relative_workdir":
        sandbox.exec.return_value.stdout = "relative/path"
    else:
        sandbox.exec.return_value.return_code = 1
    with pytest.raises((HTTPException, RuntimeError)):
        await server.seed_session(request, seed)
    if invalid in {"missing_tests", "golden_mode"}:
        server._create_sandbox.assert_not_awaited()
    else:
        sandbox.stop.assert_awaited_once()


async def test_verify_rejects_different_task_without_consuming_session(setup) -> None:
    server, sandbox, seed, request = setup
    await server.seed_session(request, seed)
    changed = verify_body(seed).model_copy(update={"docker_image": "other-image"})
    with pytest.raises(HTTPException, match="does not match"):
        await server.verify(request, changed)
    server._upload_folder.assert_not_awaited()
    result = await server.verify(request, verify_body(seed))
    assert result.reward == 1
    sandbox.stop.assert_not_awaited()


async def test_close_waits_for_inflight_seed(setup) -> None:
    server, sandbox, seed, request = setup
    entered, release = asyncio.Event(), asyncio.Event()

    async def create(task, *, session_id):
        entered.set()
        await release.wait()
        server._session_id_to_sandbox[session_id] = sandbox
        return sandbox

    server._create_sandbox.side_effect = create
    seeding = asyncio.create_task(server.seed_session(request, seed))
    await entered.wait()
    closing = asyncio.create_task(server.close_resources_session(close_body(seed)))
    await asyncio.sleep(0)
    sandbox.stop.assert_not_awaited()
    release.set()
    await asyncio.gather(seeding, closing)
    sandbox.stop.assert_awaited_once()
    assert not server._session_id_to_sandbox


def test_shutdown_cleans_seeded_sandbox_and_legacy_seed_shape_stays_unchanged(setup) -> None:
    server, sandbox, seed, _ = setup
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/seed_session", json=seed.task_data)
        assert result.status_code == 200
        assert result.json() == {"sandbox_handle": "task-sandbox"}
        sandbox.serialize.assert_not_awaited()
    sandbox.stop.assert_awaited_once()


def test_process_local_sessions_reject_multiple_workers(setup) -> None:
    server, _, _, _ = setup
    with pytest.raises(ValueError, match="num_workers=1"):
        TerminalBench21ResourcesServer(
            config=server.config.model_copy(update={"num_workers": 2}), server_client=server.server_client
        )
