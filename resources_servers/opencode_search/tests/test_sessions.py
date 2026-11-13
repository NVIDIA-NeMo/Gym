# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions: the sandbox an episode seed lends and the one an agent's own /run seeds, run on the local sandbox
provider in a temporary directory, with the judge answered at the server's HTTP client."""

import asyncio
import json
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxResources, SandboxSpec, cpu_cap_env
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from nemo_gym.verification import JudgeError
from resources_servers.opencode_search.app import (
    SearchResourcesServer,
    SearchResourcesServerConfig,
    SearchSeedSessionRequest,
)


CORRECT = "extracted_final_answer: Paris\nreasoning: same city\ncorrect: yes"
WRONG = "extracted_final_answer: Lyon\nreasoning: another city\ncorrect: no"
NO_VERDICT = "I cannot decide whether the response is right."


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Start every sandbox on the local provider under tmp_path, and return the workdir of the lent ones."""
    named_configs = {
        "sandbox": {
            "local": {"workspace_root": str(tmp_path / "agent-runs")},
            "default_metadata": {"sandbox-api": "local"},
        }
    }
    monkeypatch.setattr("nemo_gym.server_runtime.sandbox_lending.get_global_config_dict", lambda: named_configs)
    monkeypatch.setattr("resources_servers.opencode_search.app.get_global_config_dict", lambda: named_configs)
    # The local provider cannot be reattached from another process, so it has no descriptor to hand out.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return tmp_path / "workspace"


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AsyncSandbox, SandboxSpec]]:
    started: list[tuple[AsyncSandbox, SandboxSpec]] = []
    real_start = AsyncSandbox.start

    async def start(self: AsyncSandbox, spec: Optional[SandboxSpec] = None, **kwargs: Any) -> AsyncSandbox:
        started.append((self, spec))
        return await real_start(self, spec, **kwargs)

    monkeypatch.setattr(AsyncSandbox, "start", start)
    return started


@pytest.fixture
def stopped(monkeypatch: pytest.MonkeyPatch) -> list[AsyncSandbox]:
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


def _sandbox_config(**fields: Any) -> dict[str, Any]:
    """configs/opencode_search.yaml's sandbox_config, with an env entry of its own."""
    return {
        "ttl_s": 23400,
        "ready_timeout_s": 1200,
        "derive_cpu_env": True,
        "resources": {"cpu": 2, "memory_mib": 16384, "disk_gib": 30},
        "provider_options": {"resource_requests": {"cpu": 1, "memory_mib": 512, "disk_gib": 30}},
        "metadata": {"benchmark": "search-browsecomp"},
        "env": {"EXTRA": "1"},
        **fields,
    }


def _server(
    sandbox_workdir: Optional[Path | str], sandbox_config: Optional[dict[str, Any]] = None
) -> SearchResourcesServer:
    config = SearchResourcesServerConfig(
        name="opencode_search",
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        sandbox_provider="sandbox",
        sandbox_config=_sandbox_config() if sandbox_config is None else sandbox_config,
        sandbox_image="python:3.12",
        sandbox_workdir=None if sandbox_workdir is None else str(sandbox_workdir),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_max_attempts=2,
        judge_retry_delay_seconds=0.0,
    )
    return SearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str) -> dict[str, Any]:
    return {
        "id": "resp",
        "created_at": 0.0,
        "model": "model",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _row() -> dict[str, Any]:
    """A dataset row, as an agent's own /run sends it to /seed_session."""
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "Which city?"}]},
        "question": "Which city?",
        "ground_truth": "Paris",
    }


def _verify_body(answer: str = "Exact Answer: Paris", **fields: Any) -> dict[str, Any]:
    return _row() | {"response": _response(answer)} | fields


def _seed(session_id: str = "session", task_data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="browsecomp_opencode:benchmark", task_id="t-1"),
        task_data={"question": "Which city?", "ground_truth": "Paris"} if task_data is None else task_data,
    ).model_dump(mode="json")


def _close(session_id: str = "session") -> dict[str, Any]:
    return ResourcesCloseSessionRequest(
        resources_session_id=session_id, episode_id=EpisodeId(rollout_id=f"rollout-{session_id}")
    ).model_dump(mode="json")


def _mock_judge(server: SearchResourcesServer, *texts: str) -> AsyncMock:
    """Answer the judge calls with the given texts in order, repeating the last."""
    replies = [json.dumps(_response(text)).encode() for text in texts]
    answered: list[bytes] = []

    async def post(**_: Any) -> MagicMock:
        answered.append(replies[min(len(answered), len(replies) - 1)])
        reply = MagicMock()
        reply.read = AsyncMock(return_value=answered[-1])
        return reply

    mock = AsyncMock(side_effect=post)
    server.server_client.post = mock
    return mock


def _graded(verified: Any) -> tuple[Any, ...]:
    body = verified.json()
    return body["reward"], body["mask_sample"], body["failure_kind"], body["evaluation_completed"]


# ---------------------------------------------------------------------------
# Episode seed: a sandbox lent to a session harness
# ---------------------------------------------------------------------------


def test_episode_sessions_follow_the_session_contract(workdir) -> None:
    server = _server(workdir)
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest.model_validate(_seed("contract")),
        keeps_state=True,
        holds_state=server.has_session_state,
        verify_body=_verify_body(),
    )


def test_an_episode_seed_lends_one_sandbox_and_its_close_stops_it(workdir, started, stopped) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed())

    assert seeded.status_code == 200, seeded.text
    assert seeded.json() == {
        "resources_session_id": "session",
        "resources_tools": None,
        "sandbox_access": {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "local"}},
            "workdir": str(workdir),
        },
    }
    # A repeated seed returns the same session and starts no second sandbox.
    again = client.post("/seed_session", json=_seed())
    assert again.status_code == 200 and again.json() == seeded.json()
    assert len(started) == 1

    assert client.post("/close_session", json=_close()).status_code == 200
    assert stopped == [started[0][0]]
    assert not server.has_session_state("session")
    # A repeated close succeeds and stops nothing more.
    assert client.post("/close_session", json=_close()).status_code == 200
    assert len(stopped) == 1


@pytest.mark.parametrize("derive_cpu_env", [True, False])
def test_a_lent_sandbox_is_the_agent_run_sandbox_in_the_workdir(workdir, started, derive_cpu_env) -> None:
    """Lending changes where the harness works and labels the task; image, limits, env and labels stay the same."""
    server = _server(workdir, sandbox_config=_sandbox_config(derive_cpu_env=derive_cpu_env))
    client = TestClient(server.setup_webserver())

    assert client.post("/seed_session", json=_row()).status_code == 200
    client.cookies.clear()
    assert client.post("/seed_session", json=_seed()).status_code == 200

    (_, agent_run), (_, lent) = started
    assert lent == replace(agent_run, workdir=str(workdir), metadata=agent_run.metadata | {"task_id": "t-1"})
    assert lent.env == (cpu_cap_env(2) if derive_cpu_env else {}) | {"EXTRA": "1"}


@pytest.mark.parametrize(
    "body",
    [
        _seed(task_data={"question": "Which city?"}),
        _seed(task_data={"ground_truth": "Paris"}),
        # A malformed episode seed: it names a resources session but no episode, task or task_data.
        {"resources_session_id": "session"},
    ],
    ids=["no_ground_truth", "no_question", "id_only"],
)
def test_an_episode_seed_with_bad_task_data_is_refused_and_starts_nothing(workdir, started, body) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())

    refused = client.post("/seed_session", json=body)

    assert refused.status_code == 422, refused.text
    assert started == []
    assert not server.has_session_state("session") and server.session_identity("session") is None
    # The refusal bound nothing, so the same ID seeds once its task_data is valid.
    assert client.post("/seed_session", json=_seed()).status_code == 200


def test_without_a_sandbox_workdir_an_episode_seed_is_refused(workdir, started) -> None:
    server = _server(None)
    client = TestClient(server.setup_webserver())

    refused = client.post("/seed_session", json=_seed())

    assert refused.status_code == 422 and "sandbox_workdir" in refused.text
    assert started == [] and not server.has_session_state("session")
    # An agent's own /run still gets its sandbox.
    client.cookies.clear()
    assert client.post("/seed_session", json=_row()).status_code == 200


def test_a_seed_for_another_episode_or_after_the_close_conflicts(workdir, started, stopped) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200

    other_episode = _seed() | {"episode_id": EpisodeId(rollout_id="another-rollout").model_dump(mode="json")}
    assert client.post("/seed_session", json=other_episode).status_code == 409
    assert client.post("/close_session", json=_close()).status_code == 200
    assert client.post("/seed_session", json=_seed()).status_code == 409
    assert len(started) == 1 and len(stopped) == 1


def test_an_agent_run_seed_cannot_replace_an_episode_session(workdir, started) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200

    # The episode seed's reply bound the cookie to the episode session.
    refused = client.post("/seed_session", json=_row())

    assert refused.status_code == 409, refused.text
    assert len(started) == 1 and server.has_session_state("session")


@pytest.mark.parametrize(
    "result",
    [
        SandboxExecResult(stdout="", stderr="mkdir: cannot create directory: Permission denied", return_code=1),
        SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout"),
    ],
    ids=["exit_code", "error_type"],
)
def test_an_episode_seed_that_cannot_create_its_workdir_stops_its_sandbox(
    workdir, started, stopped, monkeypatch, result
) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    commands: list[tuple[str, Optional[str]]] = []

    async def exec(self: AsyncSandbox, command: str, *, cwd: Optional[str] = None, **_: Any) -> SandboxExecResult:
        commands.append((command, cwd))
        return result

    monkeypatch.setattr(AsyncSandbox, "exec", exec)

    failed = client.post("/seed_session", json=_seed())

    assert failed.status_code == 500
    # From /, since a command's default directory is the workdir that does not exist yet.
    assert commands == [(f"mkdir -p -- {shlex.quote(str(workdir))}", "/")]
    assert stopped == [started[0][0]]
    assert not server.has_session_state("session") and server._lent_sandboxes.get("session") is None


def test_a_close_whose_stop_fails_keeps_the_sandbox_for_a_retried_close(workdir, started, monkeypatch) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    attempts: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        attempts.append(self)
        if len(attempts) == 1:
            raise RuntimeError("the provider refused the stop")
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)

    assert client.post("/close_session", json=_close()).status_code == 500
    assert server.has_session_state("session") and server._lent_sandboxes.get("session") is not None
    # The failed close fenced nothing, so its retry is still the owner's close and stops the sandbox.
    assert client.post("/close_session", json=_close()).status_code == 200
    assert attempts == [started[0][0]] * 2
    assert not server.has_session_state("session") and server._lent_sandboxes.get("session") is None


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["episode", "agent_run"])
def test_verify_stops_the_sandbox_before_the_judge_runs(workdir, started, stopped, seed) -> None:
    """Grading never reads the sandbox, so it is not kept running while the verify waits on the judge."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed() if seed == "episode" else _row()).status_code == 200
    judge = _mock_judge(server, CORRECT)
    stops_at_judge: list[int] = []
    answer = judge.side_effect

    async def post(**kwargs: Any) -> MagicMock:
        stops_at_judge.append(len(stopped))
        return await answer(**kwargs)

    judge.side_effect = post

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0, False, None, True)
    assert stops_at_judge == [1] and stopped == [started[0][0]]
    assert server._sandbox_access == {} and server._agent_run_sandboxes == {}


def test_a_verify_is_graded_once(workdir) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200
    first_judge = _mock_judge(server, CORRECT)
    first = client.post("/verify", json=_verify_body())
    assert first.status_code == 200 and _graded(first) == (1.0, False, None, True)

    # A judge asked again would now disagree, so only a replay returns the first verdict.
    second_judge = _mock_judge(server, WRONG)
    assert client.post("/verify", json=_verify_body()).json() == first.json()

    conflict = client.post("/verify", json=_verify_body("Exact Answer: Lyon"))
    assert conflict.status_code == 409, conflict.text
    # The refused body leaves the first verdict in place.
    assert client.post("/verify", json=_verify_body()).json() == first.json()
    assert first_judge.await_count == 1 and second_judge.await_count == 0


def test_a_verify_after_the_close_is_masked_as_lost(workdir) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200
    assert client.post("/close_session", json=_close()).status_code == 200
    judge = _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (0.0, True, "session_lost", False)
    assert judge.await_count == 0


def test_a_judge_without_a_verdict_is_masked_and_an_identical_retry_is_graded(workdir) -> None:
    """No verdict is not a measured 0: the row is masked, and the verify is not replayed but graded again."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200
    judge = _mock_judge(server, NO_VERDICT)

    inconclusive = client.post("/verify", json=_verify_body())

    assert inconclusive.status_code == 200, inconclusive.text
    assert _graded(inconclusive) == (0.0, True, "judge_unparseable", False)
    assert inconclusive.json()["judge_parsed"] is False
    assert judge.await_count == 2  # judge_max_attempts

    _mock_judge(server, CORRECT)
    retried = client.post("/verify", json=_verify_body())
    assert _graded(retried) == (1.0, False, None, True)


def test_a_judge_without_a_verdict_outranks_a_budget_ending(workdir) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    _mock_judge(server, NO_VERDICT)

    verified = client.post("/verify", json=_verify_body(opencode_terminal_reason="max_iteration"))

    assert _graded(verified) == (0.0, True, "judge_unparseable", False)


def test_a_judge_that_never_answers_sends_the_row_to_the_failures_sidecar(workdir) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200
    server.server_client.post = AsyncMock(side_effect=JudgeError("connection refused"))

    failed = client.post("/verify", json=_verify_body())

    assert failed.status_code == 200, failed.text
    assert _graded(failed) == (0.0, True, "judge_failed", False)
    assert failed.json()["_ng_failure_class"] == "judge_failed"
    # Nothing was graded, so an identical retry asks the judge again.
    _mock_judge(server, CORRECT)
    assert _graded(client.post("/verify", json=_verify_body())) == (1.0, False, None, True)


def test_a_verify_without_a_seed_is_graded(workdir) -> None:
    """gym eval reverify, or an agent run whose cookie was lost: the grade reads only the request."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    _mock_judge(server, WRONG)

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (0.0, False, None, True)


# ---------------------------------------------------------------------------
# Agent-run seed, kept for an agent's own /run
# ---------------------------------------------------------------------------


def test_an_agent_run_seed_replies_with_the_handle_of_the_sandbox_it_started(workdir, started) -> None:
    """The relay agent attaches to the sandbox the handle names; without it the agent starts one of its own."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_row())

    assert seeded.status_code == 200, seeded.text
    ((sandbox, spec),) = started
    assert seeded.json() == {"sandbox_handle": sandbox.sandbox_id}
    assert sandbox.sandbox_id
    # The spec the seed built before episode sessions, field for field.
    assert spec == SandboxSpec(
        image="python:3.12",
        ttl_s=23400,
        ready_timeout_s=1200,
        workdir=None,
        env=cpu_cap_env(2) | {"EXTRA": "1"},
        files={},
        metadata={"sandbox-api": "local", "benchmark": "search-browsecomp", "nemo_gym_agent": "opencode_search"},
        resources=SandboxResources(cpu=2, memory_mib=16384, disk_gib=30),
        entrypoint=None,
        provider_options={"resource_requests": {"cpu": 1, "memory_mib": 512, "disk_gib": 30}},
    )


@pytest.mark.parametrize(("reason", "kind"), [("max_iteration", "agent_limit"), ("", None)])
def test_an_agent_run_verify_grades_and_masks_a_budget_ending(workdir, started, stopped, reason, kind) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_row()).status_code == 200
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(opencode_terminal_reason=reason))

    assert verified.status_code == 200, verified.text
    # A budget ending keeps its judged reward; the mask and kind name the ending.
    assert _graded(verified) == (1.0, bool(kind), kind, True)
    assert stopped == [started[0][0]]


async def test_a_cancelled_agent_run_seed_stops_the_sandbox_it_registered(workdir, stopped, monkeypatch) -> None:
    server = _server(workdir)
    starting = asyncio.Event()

    async def start(self: AsyncSandbox, spec: Optional[SandboxSpec] = None, **_: Any) -> AsyncSandbox:
        starting.set()
        await asyncio.Event().wait()
        return self

    monkeypatch.setattr(AsyncSandbox, "start", start)
    seed = asyncio.create_task(
        server.seed_session(SimpleNamespace(session={SESSION_ID_KEY: "run"}), SearchSeedSessionRequest())
    )
    await asyncio.wait_for(starting.wait(), timeout=5)

    seed.cancel()
    (outcome,) = await asyncio.wait_for(asyncio.gather(seed, return_exceptions=True), timeout=5)

    assert isinstance(outcome, asyncio.CancelledError)
    assert len(stopped) == 1 and not server.has_session_state("run")


# ---------------------------------------------------------------------------
# Shutdown and startup
# ---------------------------------------------------------------------------


def test_shutdown_stops_every_sandbox_still_running(workdir, started, stopped) -> None:
    server = _server(workdir)
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
        client.cookies.clear()
        assert client.post("/seed_session", json=_row()).status_code == 200
        assert stopped == []

    assert len(started) == len(stopped) == 2
    assert {id(sandbox) for sandbox, _ in started} == {id(sandbox) for sandbox in stopped}
    assert server._sandbox_access == {} and server._agent_run_sandboxes == {}


@pytest.mark.parametrize("sandbox_workdir", ["workspace", "/", "/tmp", "/tmp/", "/workspace/../tmp"])
def test_a_workdir_a_harness_cannot_use_is_refused_at_startup(sandbox_workdir) -> None:
    with pytest.raises(ValueError, match="sandbox_workdir must be an absolute directory"):
        _server(sandbox_workdir)


def test_a_sandbox_config_a_lent_sandbox_cannot_take_is_refused_at_startup() -> None:
    with pytest.raises(ValueError, match="unknown_setting"):
        _server("/workspace", sandbox_config=_sandbox_config(unknown_setting=1))


def test_only_a_lending_server_needs_one_worker() -> None:
    assert _server(None).multiple_workers_unsupported_reason() is None
    assert _server("/workspace").multiple_workers_unsupported_reason() == "lent sandboxes live in one worker's memory"
