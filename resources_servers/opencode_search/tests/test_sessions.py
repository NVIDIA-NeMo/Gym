# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions: the sandbox an episode seed lends and the one an agent's own /run seeds, run on the local sandbox
provider in a temporary directory, with the judge answered at the server's HTTP client."""

import asyncio
import base64
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import PROVIDER_UNAVAILABLE, EpisodeId, FailureHeader, TaskId
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxExecResult,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    sandbox_failure,
)
from nemo_gym.server_runtime import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from nemo_gym.verification import JudgeError
from resources_servers.opencode_search.app import (
    _PLUGIN_LOG_ROW_CHARS,
    PLUGINS_NOT_LOADED,
    SearchResourcesServer,
    SearchResourcesServerConfig,
    SearchSeedSessionRequest,
    terminal_reason_from_plugin_log,
)


CORRECT = "extracted_final_answer: Paris\nreasoning: same city\ncorrect: yes"
WRONG = "extracted_final_answer: Lyon\nreasoning: another city\ncorrect: no"
NO_VERDICT = "I cannot decide whether the response is right."

REPOSITORY = Path(__file__).resolve().parents[3]
# The receipts plugin and the four plugins opencode_sandboxed_agent ships, by path in the repository, in load order.
PLUGINS = (
    "resources_servers/opencode_search/opencode_bundle/nemo_gym_receipts.js",
    "responses_api_agents/opencode_sandboxed_agent/plugins/nemo_gym_sentinel.js",
    "responses_api_agents/opencode_sandboxed_agent/plugins/nemo_gym_tool_surface.js",
    "responses_api_agents/opencode_sandboxed_agent/plugins/nemo_gym_turn_budget.js",
    "responses_api_agents/opencode_sandboxed_agent/plugins/nemo_gym_web_guard.js",
)
SENTINEL = "[nemo-gym-plugin] loaded sentinel version=1 plugins=\n"
TURN_CAP = "[nemo-gym-plugin] terminal=max_iteration session=s calls=401 cap=400\n"
# A web search as nemo_gym_web_guard.js reports it. The plugin writes the query as JSON, which keeps a character outside
# ASCII as it is, so a log that holds this line is longer in bytes than in characters.
MULTIBYTE_SEARCH = '[nemo-gym-plugin] [exa-mcp] tool=web_search_exa ms=5 bytes=1 query="café"\n'
# What an Environment Server sends with an episode's verify: the session's close follows it.
EPISODE_VERIFY = {CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
NODE = shutil.which("node")


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
    sandbox_workdir: Optional[Path | str], sandbox_config: Optional[dict[str, Any]] = None, **settings: Any
) -> SearchResourcesServer:
    config = SearchResourcesServerConfig(
        **{
            "name": "opencode_search",
            "host": "0.0.0.0",
            "port": 8080,
            "entrypoint": "app.py",
            "sandbox_provider": "sandbox",
            "sandbox_config": _sandbox_config() if sandbox_config is None else sandbox_config,
            "sandbox_image": "python:3.12",
            "sandbox_workdir": None if sandbox_workdir is None else str(sandbox_workdir),
            "judge_model_server": ModelServerRef(type="responses_api_models", name="judge"),
            "judge_max_attempts": 2,
            "judge_retry_delay_seconds": 0.0,
        }
        | settings
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


def _staged(stage: Path) -> dict[str, str]:
    """PLUGINS as episode_sandbox_files, named as opencode_sandboxed_agent names its staged plugins, under stage."""
    return {str(stage / f"nemo-gym-plugin-{Path(source).name}"): source for source in PLUGINS}


def _plugin_log_server(workdir: Path, log: Path, **settings: Any) -> tuple[SearchResourcesServer, TestClient]:
    """A server that reads the plugin log at log, and a client whose episode session it seeded."""
    server = _server(workdir, plugin_log_path=str(log), **settings)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    return server, client


def _record_execs(monkeypatch: pytest.MonkeyPatch, stopped: list[AsyncSandbox]) -> list[tuple[str, int]]:
    """Record every command run in a sandbox from now on, with the number of sandboxes stopped by then."""
    execs: list[tuple[str, int]] = []
    real_exec = AsyncSandbox.exec

    async def exec(self: AsyncSandbox, command: str, **kwargs: Any) -> SandboxExecResult:
        execs.append((command, len(stopped)))
        return await real_exec(self, command, **kwargs)

    monkeypatch.setattr(AsyncSandbox, "exec", exec)
    return execs


def _plugin_log_read(log_text: str, *, cut_after: Optional[int] = None) -> str:
    """What the plugin log read prints for a log holding log_text, line breaks aside: the log's size in bytes, then
    the log in base64. With cut_after, the base64 stops at the last whole group within that many bytes, as a provider
    that cut the output short could return it."""
    data = log_text.encode()
    encoded = base64.b64encode(data).decode()
    if cut_after is not None:
        encoded = encoded[: cut_after // 3 * 4]
    return f"{len(data)}:{encoded}"


def _failure_kinds(reply: Any) -> list[str]:
    """The failure kinds that a reply's failure report names."""
    report = FailureHeader.decode(reply.headers)
    return [] if report is None else [event.failure_kind for event in report.events.values()]


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


def test_an_episode_seed_stages_the_episode_files_and_env_in_its_sandbox(
    workdir, tmp_path, started, monkeypatch
) -> None:
    files = _staged(tmp_path / "stage")
    log = tmp_path / "stage" / "plugins.log"
    # The sources name files relative to the Gym root, wherever the server starts.
    monkeypatch.chdir(tmp_path)
    server = _server(
        workdir,
        episode_sandbox_files=files,
        episode_sandbox_env={"EXTRA": "episode", "NEMO_GYM_MAX_TURNS": "400", "NEMO_GYM_DISABLE_BASH": ""},
        plugin_log_path=str(log),
    )
    client = TestClient(server.setup_webserver())

    assert client.post("/seed_session", json=_row()).status_code == 200
    client.cookies.clear()
    assert client.post("/seed_session", json=_seed()).status_code == 200

    (_, agent_run), (_, lent) = started
    contents = {path: (REPOSITORY / source).read_bytes() for path, source in files.items()}
    assert lent.files == contents
    # Each file is where the harness is told to load it from, byte for byte.
    assert {path: Path(path).read_bytes() for path in files} == contents
    # Over sandbox_config.env. An empty value sets nothing, and the plugin log's path comes from plugin_log_path.
    assert lent.env == cpu_cap_env(2) | {
        "EXTRA": "episode",
        "NEMO_GYM_MAX_TURNS": "400",
        "NEMO_GYM_PLUGIN_LOG": str(log),
    }
    # An agent's own /run gets the sandbox it got before.
    assert agent_run.files == {} and agent_run.env == cpu_cap_env(2) | {"EXTRA": "1"}


@pytest.mark.parametrize(
    ("settings", "url"),
    [
        ({"host": "192.0.2.10"}, "http://192.0.2.10:8080/mcp"),
        # A URL the config names wins, and leaves this server's own address unchecked.
        (
            {"host": "127.0.0.1", "episode_sandbox_env": {"NEMO_GYM_EXA_BASE_URL": "http://198.51.100.7:9000/mcp"}},
            "http://198.51.100.7:9000/mcp",
        ),
    ],
    ids=["this_server", "configured"],
)
def test_with_the_search_cache_a_lent_sandbox_searches_through_its_proxy(
    workdir, tmp_path, started, settings, url
) -> None:
    server = _server(workdir, exa_cache_root=str(tmp_path / "search-cache"), **settings)
    client = TestClient(server.setup_webserver())

    assert client.post("/seed_session", json=_seed()).status_code == 200

    ((_, lent),) = started
    assert lent.env["NEMO_GYM_EXA_BASE_URL"] == url


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


def test_an_episode_seed_cannot_replace_an_agent_run_session(workdir, started) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_row()).status_code == 200
    (session_id,) = server._agent_run_sandboxes
    client.cookies.clear()

    refused = client.post("/seed_session", json=_seed(session_id))

    assert refused.status_code == 409, refused.text
    assert len(started) == 1 and server._lent_sandboxes.get(session_id) is None


def test_an_episode_seed_after_its_verify_conflicts(workdir, started) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200
    _mock_judge(server, CORRECT)
    assert client.post("/verify", json=_verify_body()).status_code == 200

    refused = client.post("/seed_session", json=_seed())

    assert refused.status_code == 409, refused.text
    assert len(started) == 1


@pytest.mark.parametrize(
    ("result", "status"),
    [
        # The image refuses the directory, as it would in every episode: 422, so the episode does not run again.
        (SandboxExecResult(stdout="", stderr="mkdir: cannot create directory: Permission denied", return_code=1), 422),
        # The command did not complete, which a fresh sandbox may clear: 500, which an Environment Server retries.
        (SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout"), 500),
    ],
    ids=["exit_code", "error_type"],
)
def test_an_episode_seed_that_cannot_create_its_workdir_stops_its_sandbox(
    workdir, started, stopped, monkeypatch, result, status
) -> None:
    server = _server(workdir)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    commands: list[tuple[str, Optional[str]]] = []

    async def exec(self: AsyncSandbox, command: str, *, cwd: Optional[str] = None, **_: Any) -> SandboxExecResult:
        commands.append((command, cwd))
        return result

    monkeypatch.setattr(AsyncSandbox, "exec", exec)

    failed = client.post("/seed_session", json=_seed())

    assert failed.status_code == status, failed.text
    # From /, since a command's default directory is the workdir that does not exist yet.
    assert commands == [(f"mkdir -p -- {shlex.quote(str(workdir))}", "/")]
    assert stopped == [started[0][0]]
    assert not server.has_session_state("session") and server._lent_sandboxes.get("session") is None


@pytest.mark.parametrize("failing", ["start", "serialize"])
def test_an_episode_seed_whose_sandbox_cannot_start_or_be_described_stops_it_for_a_retry(
    workdir, started, stopped, monkeypatch, failing
) -> None:
    """A provider failure may clear on a fresh sandbox, so the seed answers 500, which an Environment Server
    retries."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    if failing == "start":
        recorded_start = AsyncSandbox.start

        async def start(self: AsyncSandbox, spec: Optional[SandboxSpec] = None, **kwargs: Any) -> AsyncSandbox:
            await recorded_start(self, spec, **kwargs)
            raise RuntimeError("the provider lost the sandbox it started")

        monkeypatch.setattr(AsyncSandbox, "start", start)
    else:
        monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(side_effect=RuntimeError("cannot describe it")))

    failed = client.post("/seed_session", json=_seed())

    assert failed.status_code == 500, failed.text
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


@pytest.mark.parametrize(
    ("seed", "lending"),
    [("episode", True), ("agent_run", True), ("agent_run", False)],
    ids=["episode", "agent_run", "agent_run_without_lending"],
)
def test_verify_stops_the_sandbox_before_the_judge_runs(workdir, started, stopped, seed, lending) -> None:
    """Grading never reads the sandbox, so it is not kept running while the verify waits on the judge."""
    server = _server(workdir if lending else None)
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


def test_a_budget_ending_keeps_its_kind_when_the_judge_gives_no_verdict(workdir) -> None:
    """Recovery re-judges a judge_unparseable row from a payload that carries no ending, which would grade a capped
    run unmasked; agent_limit keeps it masked, as when the judge gave a verdict."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    _mock_judge(server, NO_VERDICT)

    verified = client.post("/verify", json=_verify_body(opencode_terminal_reason="max_iteration"))

    assert _graded(verified) == (0.0, True, "agent_limit", False)
    assert verified.json()["judge_parsed"] is False


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


def test_judge_replies_without_a_verdict_trip_the_dead_judge_breaker(workdir) -> None:
    """Such a row is masked, so it leaves the quality metrics as quietly as an unreachable judge's: the run stops after
    judge_max_consecutive_failures of them in a row, and a verdict starts the count again."""
    server = _server(workdir, judge_max_consecutive_failures=2)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)

    def verify(judge_reply: str) -> Any:
        _mock_judge(server, judge_reply)
        client.cookies.clear()
        return client.post("/verify", json=_verify_body())

    assert _graded(verify(NO_VERDICT)) == (0.0, True, "judge_unparseable", False)
    assert _graded(verify(CORRECT)) == (1.0, False, None, True)
    assert _graded(verify(NO_VERDICT)) == (0.0, True, "judge_unparseable", False)
    tripped = verify(NO_VERDICT)

    assert tripped.status_code == 500, tripped.text


def test_a_verify_without_a_seed_is_graded(workdir) -> None:
    """gym eval reverify, or an agent run whose cookie was lost: the grade reads only the request."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    _mock_judge(server, WRONG)

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (0.0, False, None, True)


# ---------------------------------------------------------------------------
# Plugin receipts: the episode's ending, and the plugins it ran with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        ([], ""),
        (["[nemo-gym-plugin] terminal=max_iteration session=s calls=401 cap=400"], "max_iteration"),
        (["[nemo-gym-plugin] terminal=max_iteration_abort session=s calls=406"], "max_iteration"),
        (["[nemo-gym-plugin] context_overflow session=s"], "context_window"),
        (
            ["[nemo-gym-plugin] context_overflow session=s", "[nemo-gym-plugin] context_overflow_recovered session=s"],
            "",
        ),
        # An overflow left unmatched outranks the turn budget.
        (
            [
                "[nemo-gym-plugin] terminal=max_iteration session=s calls=401 cap=400",
                "[nemo-gym-plugin] context_overflow session=s",
            ],
            "context_window",
        ),
        # A recovery matches one earlier overflow, never a later one.
        (
            [
                "[nemo-gym-plugin] context_overflow_recovered session=s",
                "[nemo-gym-plugin] context_overflow session=s",
            ],
            "context_window",
        ),
        # Only the plugins' lines count.
        (["terminal=max_iteration", "context_overflow"], ""),
    ],
    ids=[
        "no_lines",
        "max_iteration",
        "max_iteration_abort",
        "overflow",
        "recovered_overflow",
        "overflow_outranks_budget",
        "recovery_before_overflow",
        "unmarked_lines",
    ],
)
def test_the_ending_is_read_from_the_turn_budget_plugins_lines(lines, reason) -> None:
    assert terminal_reason_from_plugin_log("".join(f"{line}\n" for line in lines)) == reason


@pytest.mark.parametrize(
    ("lines", "reason", "kind"),
    [
        ("[nemo-gym-plugin] terminal=max_iteration session=s calls=401 cap=400\n", "max_iteration", "agent_limit"),
        ("[nemo-gym-plugin] context_overflow session=s\n", "context_window", "model_context_limit"),
        (
            "[nemo-gym-plugin] context_overflow session=s\n[nemo-gym-plugin] context_overflow_recovered session=s\n",
            "",
            None,
        ),
        # The read checks the bytes it returns against the file's size in bytes, so a log longer in bytes than in
        # characters is read whole, its ending with it.
        (MULTIBYTE_SEARCH + TURN_CAP, "max_iteration", "agent_limit"),
    ],
    ids=["max_iteration", "context_window", "recovered", "multibyte_search"],
)
def test_an_episode_takes_its_ending_from_the_plugin_log_before_its_sandbox_stops(
    workdir, tmp_path, started, stopped, monkeypatch, lines, reason, kind
) -> None:
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    log.write_text(SENTINEL + lines, encoding="utf-8")
    execs = _record_execs(monkeypatch, stopped)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    # As on the relay path: a budget ending keeps its judged reward, and the mask and kind name the ending.
    assert _graded(verified) == (1.0, bool(kind), kind, True)
    receipts = {key: verified.json()[key] for key in ("opencode_terminal_reason", "opencode_plugins_loaded")}
    assert receipts == {"opencode_terminal_reason": reason, "opencode_plugins_loaded": True}
    assert verified.json()["opencode_plugin_log"] == SENTINEL + lines
    # One read, while the sandbox still ran; then the sandbox stopped.
    assert [stops for _, stops in execs] == [0] and stopped == [started[0][0]]
    # An identical retry replays the reply, receipts included, without reading again.
    replayed = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)
    assert replayed.json() == verified.json() and len(execs) == 1


@pytest.mark.parametrize(
    ("log_text", "markers", "missing"),
    [
        (None, [], ["[nemo-gym-plugin] loaded sentinel"]),
        # An empty log is refused like a missing one; the detail tells them apart.
        ("", [], ["[nemo-gym-plugin] loaded sentinel"]),
        # The receipts plugin's own load line does not stand in for the sentinel.
        ("[nemo-gym-plugin] loaded receipts version=1\n", [], ["[nemo-gym-plugin] loaded sentinel"]),
        (
            SENTINEL + "[nemo-gym-plugin] tool_surface task=off bash=on\n",
            ["tool_surface task=off bash=off", "loaded sentinel"],
            ["tool_surface task=off bash=off"],
        ),
        # Longer in bytes than in characters: log_bytes counts bytes.
        (MULTIBYTE_SEARCH, [], ["[nemo-gym-plugin] loaded sentinel"]),
    ],
    ids=["no_log", "empty_log", "receipts_only", "required_marker", "multibyte_log"],
)
def test_an_episode_whose_plugins_did_not_load_is_refused_terminally(
    workdir, tmp_path, started, stopped, log_text, markers, missing
) -> None:
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log, required_plugin_markers=markers)
    if log_text is not None:
        log.write_text(log_text, encoding="utf-8")
    judge = _mock_judge(server, CORRECT)

    refused = client.post("/verify", json=_verify_body())

    # 422, which an Environment Server does not retry: the log was read, and it says the run lacked its plugins.
    assert refused.status_code == 422, refused.text
    detail = refused.json()["detail"]
    assert f"lacks {missing!r}" in detail
    # Whether the file existed, its size in bytes, and its two ends with line breaks escaped: here the whole log.
    text = log_text or ""
    assert f"exists={log_text is not None} log_bytes={len(text.encode())} head={text!r} tail={text!r}" in detail
    assert judge.await_count == 0 and stopped == [started[0][0]]
    # The log it read is kept, so a retry is refused the same way.
    assert client.post("/verify", json=_verify_body()).json()["detail"] == detail


def test_a_refusal_shows_the_two_ends_of_a_long_plugin_log_within_what_the_environment_server_keeps(
    workdir, tmp_path
) -> None:
    """An Environment Server keeps the first 2,000 characters of a refused reply, after its status line, as the row's
    failure_reason: the detail ends within them, with the end of the log."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log, required_plugin_markers=["tool_surface task=off bash=off"])
    calls = "".join(f"[nemo-gym-plugin] [exa-mcp] tool=web_search_exa ms={call} bytes=1\n" for call in range(1000))
    log.write_text(SENTINEL + calls + TURN_CAP)
    _mock_judge(server, CORRECT)

    refused = client.post("/verify", json=_verify_body())

    assert refused.status_code == 422, refused.text
    detail = refused.json()["detail"]
    # The head starts at the log's first line and the tail ends at its last, line breaks escaped.
    assert f"head={SENTINEL!r}"[:-1] in detail and detail.endswith(f"{TURN_CAP!r}"[1:])
    # With room for the status line before it.
    assert len(refused.text) <= 1800


# A read of SENTINEL + TURN_CAP cut short at the end of a base64 group, as a provider that caps a command's output
# could return it: it decodes cleanly, to a log that keeps the sentinel and loses the turn cap and its mask.
CUT_SHORT = SandboxExecResult(
    stdout=_plugin_log_read(SENTINEL + TURN_CAP, cut_after=len(SENTINEL)), stderr=None, return_code=0
)


@pytest.mark.parametrize(
    "failure",
    [
        SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout"),
        SandboxExecResult(stdout="", stderr="base64: permission denied", return_code=1),
        RuntimeError("the provider is unavailable"),
        CUT_SHORT,
    ],
    ids=["error_type", "exit_code", "raised", "cut_short"],
)
def test_a_plugin_log_that_cannot_be_read_fails_the_verify_for_a_retry(
    workdir, tmp_path, started, stopped, monkeypatch, failure
) -> None:
    """A read that fails gives no verdict: the 500 is retryable, so the episode runs again; nothing is graded."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    log.write_text(SENTINEL + TURN_CAP)
    read = AsyncMock(side_effect=failure) if isinstance(failure, Exception) else AsyncMock(return_value=failure)
    monkeypatch.setattr(AsyncSandbox, "exec", read)
    judge = _mock_judge(server, CORRECT)

    failed = client.post("/verify", json=_verify_body())

    assert failed.status_code == 500, failed.text
    assert read.await_count == 1 and judge.await_count == 0 and stopped == [started[0][0]]
    # Nothing was kept from the attempt, so the same verify again is masked rather than graded without the log.
    assert _graded(client.post("/verify", json=_verify_body())) == (0.0, True, "session_lost", False)


@pytest.mark.parametrize(
    ("read", "status", "kind"),
    [
        # No plugin wrote a log.
        (None, 422, PLUGINS_NOT_LOADED),
        (
            SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout"),
            500,
            "provider_io_timeout",
        ),
        (
            SandboxExecResult(stdout="", stderr="base64: permission denied", return_code=1),
            500,
            "provider_request_failed",
        ),
        (CUT_SHORT, 500, "provider_request_failed"),
        # A provider's own failure keeps its kind.
        (sandbox_failure(operation="exec", failure_kind=PROVIDER_UNAVAILABLE), 500, PROVIDER_UNAVAILABLE),
    ],
    ids=["plugins_not_loaded", "error_type", "exit_code", "cut_short", "provider_failure"],
)
def test_a_plugin_log_that_is_refused_or_cannot_be_read_names_its_failure_kind(
    workdir, tmp_path, monkeypatch, read, status, kind
) -> None:
    """An Environment Server records an episode whose verify did not grade by the kinds the reply's failure report
    names."""
    server = _server(workdir, plugin_log_path=str(tmp_path / "plugins.log"))
    app = server.setup_webserver()
    # As a running server has it: the middleware that reports each reply's failures in its failure header.
    server.setup_exception_middleware(app)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    if read is not None:
        mocked = AsyncMock(side_effect=read) if isinstance(read, Exception) else AsyncMock(return_value=read)
        monkeypatch.setattr(AsyncSandbox, "exec", mocked)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == status, verified.text
    assert _failure_kinds(verified) == [kind]


def test_a_plugin_log_returned_as_messages_joined_by_newlines_keeps_its_lines(workdir, tmp_path, monkeypatch) -> None:
    """A provider may return a command's output as messages joined by newlines. A message boundary inside a plugin line
    must not split it: the sentinel would read as missing, and the turn cap would go unmasked. Nor one inside the size
    the read prints, which would make the log read as cut short."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    log.write_text(SENTINEL + TURN_CAP)
    real_exec = AsyncSandbox.exec
    printed: list[str] = []

    async def exec(self: AsyncSandbox, command: str, **kwargs: Any) -> SandboxExecResult:
        result = await real_exec(self, command, **kwargs)
        printed.append("".join((result.stdout or "").split()))
        # Two characters a message, so that a boundary also falls inside the size.
        stdout = "\n".join(printed[-1][start : start + 2] for start in range(0, len(printed[-1]), 2))
        return replace(result, stdout=stdout)

    monkeypatch.setattr(AsyncSandbox, "exec", exec)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0, True, "agent_limit", True)
    assert verified.json()["opencode_plugin_log"] == SENTINEL + TURN_CAP
    # Line breaks aside, the read printed what _plugin_log_read gives where a test fakes the read.
    assert printed == [_plugin_log_read(SENTINEL + TURN_CAP)]


def test_a_plugin_log_too_long_for_the_row_is_read_whole_and_carried_by_its_two_ends(workdir, tmp_path) -> None:
    """The row carries at most _PLUGIN_LOG_ROW_CHARS characters of the plugin log, but the ending comes from all of it:
    here from a turn cap in the part of the log the row leaves out."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    calls = [f"[nemo-gym-plugin] [exa-mcp] tool=web_search_exa ms={call} bytes=1\n" for call in range(24000)]
    whole = SENTINEL + "".join(calls[:12000]) + TURN_CAP + "".join(calls[12000:])
    log.write_text(whole)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0, True, "agent_limit", True)
    carried = verified.json()["opencode_plugin_log"]
    assert len(carried) <= _PLUGIN_LOG_ROW_CHARS < len(whole)
    assert carried.startswith(SENTINEL) and carried.endswith(calls[-1]) and TURN_CAP not in carried


@pytest.mark.parametrize(
    ("headers", "graded", "judge_calls"),
    [
        # An Environment Server's verify: graded from its body, it would skip the plugin gate and the turn cap's mask.
        (EPISODE_VERIFY, (0.0, True, "session_lost", False), 0),
        # The relay agent's verify, or gym eval reverify: graded from its body, as without plugin_log_path.
        ({}, (1.0, False, None, True), 1),
    ],
    ids=["episode", "relay_or_reverify"],
)
def test_a_verify_for_a_sandbox_this_server_no_longer_holds(workdir, tmp_path, headers, graded, judge_calls) -> None:
    """As after a restart, which stops every sandbox: the verify of a run cut off by its turn cap reaches a server
    that holds nothing for its session."""
    log = tmp_path / "plugins.log"
    log.write_text(SENTINEL + TURN_CAP)
    server = _server(workdir, plugin_log_path=str(log))
    client = TestClient(server.setup_webserver())
    judge = _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=headers)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == graded
    assert judge.await_count == judge_calls and "opencode_plugin_log" not in verified.json()


@pytest.mark.parametrize(
    ("seed", "settings", "reason", "graded"),
    [
        ("episode", {}, "", (1.0, False, None, True)),
        # An agent's own /run sends its ending in the verify body, as the relay agent does.
        (
            "agent_run",
            {"plugin_log_path": "/nonexistent/plugins.log"},
            "max_iteration",
            (1.0, True, "agent_limit", True),
        ),
    ],
    ids=["episode_without_plugin_log_path", "agent_run"],
)
def test_without_a_plugin_log_to_read_nothing_is_read_or_refused(
    workdir, stopped, monkeypatch, seed, settings, reason, graded
) -> None:
    """A harness that brings no OpenCode plugins is graded as before."""
    server = _server(workdir, **settings)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed() if seed == "episode" else _row()).status_code == 200
    execs = _record_execs(monkeypatch, stopped)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(opencode_terminal_reason=reason))

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == graded
    assert execs == [] and "opencode_plugin_log" not in verified.json()


@pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the receipts plugin")
def test_the_receipts_plugin_copies_the_plugin_lines_to_the_log() -> None:
    ran = subprocess.run(
        [NODE, "--test", str(Path(__file__).with_name("nemo_gym_receipts.test.mjs"))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr


# What OpenCode does with the plugins it is configured with: it imports each, calls its factory, and runs the hooks
# the factories returned on every model call. Here, three calls.
_PLUGIN_HOST = """
import { pathToFileURL } from "node:url";
const hooks = [];
for (const path of process.argv.slice(1)) {
  const plugin = (await import(pathToFileURL(path).href)).default;
  hooks.push(await plugin({ client: { session: { abort: async () => {} } } }));
}
for (let call = 0; call < 3; call++) {
  for (const hook of hooks) await hook["chat.params"]?.({ sessionID: "session", message: {} });
}
"""


@pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the OpenCode plugins")
def test_the_staged_plugins_report_to_verify_through_the_plugin_log(workdir, tmp_path, started) -> None:
    files = _staged(tmp_path / "stage")
    server = _server(
        workdir,
        episode_sandbox_files=files,
        episode_sandbox_env={
            "NEMO_GYM_MAX_TURNS": "2",
            "NEMO_GYM_DISABLE_BASH": "1",
            "NEMO_GYM_DISABLE_SUBAGENTS": "1",
        },
        plugin_log_path=str(tmp_path / "stage" / "plugins.log"),
        required_plugin_markers=["tool_surface task=off bash=off"],
    )
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200
    ((_, lent),) = started
    # The staged plugins, run with the lent sandbox's environment as OpenCode would run them there.
    ran = subprocess.run(
        [NODE, "--input-type=module", "-e", _PLUGIN_HOST, *files],
        env=os.environ | lent.env,
        cwd=lent.workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 0, ran.stderr
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    # The third call ran past the turn budget of two.
    assert _graded(verified) == (1.0, True, "agent_limit", True)
    assert verified.json()["opencode_terminal_reason"] == "max_iteration"
    # The log holds what the plugins wrote to stderr, and nothing else that went there.
    plugin_lines = [line for line in ran.stderr.splitlines(keepends=True) if "[nemo-gym-plugin]" in line]
    assert verified.json()["opencode_plugin_log"] == "".join(plugin_lines)


# ---------------------------------------------------------------------------
# Export copy: the harness's session exports, copied to this host at verify
# ---------------------------------------------------------------------------

# What opencode_agent writes to its export_dir: the session inventory, each session's export and the run's event log.
EXPORTS = {
    "sessions.json": b'[{"id": "ses_root", "parent_id": null}]',
    "ses_root.json": b'{"info": {"id": "ses_root"}, "messages": []}',
    "ses_root.events.jsonl": b'{"type": "step_start"}\n',
}
EXPORT_BYTES = sum(len(data) for data in EXPORTS.values())


def _export_copy_server(
    workdir: Path, tmp_path: Path, **settings: Any
) -> tuple[SearchResourcesServer, TestClient, Path]:
    """A server that copies the exports in tmp_path/exports to tmp_path/copies, a client whose episode session it
    seeded, with EXPORTS in that directory, and where the copy of that episode's exports goes."""
    exports = tmp_path / "exports"
    server = _server(
        workdir, episode_export_dir=str(exports), episode_export_copy_dir=str(tmp_path / "copies"), **settings
    )
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    exports.mkdir()
    for name, data in EXPORTS.items():
        (exports / name).write_bytes(data)
    return server, client, tmp_path / "copies" / "rollout-session" / "t-1" / "attempt-0"


def _record_downloads(monkeypatch: pytest.MonkeyPatch, stopped: list[AsyncSandbox]) -> list[int]:
    """Record every download from a sandbox from now on, as the number of sandboxes stopped by then."""
    downloads: list[int] = []
    real_download = AsyncSandbox.download

    async def download(self: AsyncSandbox, remote_path: str, local_path: Path | str) -> None:
        downloads.append(len(stopped))
        await real_download(self, remote_path, local_path)

    monkeypatch.setattr(AsyncSandbox, "download", download)
    return downloads


def test_an_episode_verify_copies_the_session_exports_to_this_host_before_its_sandbox_stops(
    workdir, tmp_path, started, stopped, monkeypatch
) -> None:
    log = tmp_path / "plugins.log"
    log.write_text(SENTINEL)
    # Exports that hold exactly the limit are copied.
    server, client, copied_to = _export_copy_server(
        workdir, tmp_path, plugin_log_path=str(log), episode_export_max_bytes=EXPORT_BYTES
    )
    exports = tmp_path / "exports"
    # Not copied: a partial file opencode_agent renames into place, a link, a directory and a name that is not plain.
    (exports / ".ses_root.json.7.tmp").write_bytes(b"{")
    (exports / "ses_link.json").symlink_to(exports / "ses_root.json")
    (exports / "ses_child").mkdir()
    (exports / "ses root.json").write_bytes(b"{}")
    # An earlier copy of the same episode, as a retried episode would find it.
    copied_to.mkdir(parents=True)
    (copied_to / "ses_earlier.json").write_bytes(b"{}")
    execs = _record_execs(monkeypatch, stopped)
    downloads = _record_downloads(monkeypatch, stopped)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0, False, None, True)
    assert verified.json()["opencode_export_copy"] == {
        "status": "copied",
        "dir": str(copied_to),
        "files": {name: len(data) for name, data in EXPORTS.items()},
        "bytes": EXPORT_BYTES,
        "detail": "",
    }
    # Byte for byte, and nothing else: the earlier copy is gone.
    assert {path.name: path.read_bytes() for path in copied_to.iterdir()} == EXPORTS
    # The plugin log read, then the listing and every download, while the sandbox ran; then it stopped.
    assert [("base64" in command, stops) for command, stops in execs] == [(True, 0), (False, 0)]
    assert downloads == [0] * len(EXPORTS) and stopped == [started[0][0]]
    # An identical retry replays the reply, the copy's report included, without copying again.
    replayed = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)
    assert replayed.json() == verified.json()
    assert len(execs) == 2 and len(downloads) == len(EXPORTS)


@pytest.mark.parametrize(
    "case",
    [
        # One byte over the limit.
        "oversize",
        # The harness wrote no exports.
        "absent",
    ],
)
def test_exports_over_the_size_limit_or_absent_are_not_copied_and_change_no_grade(
    workdir, tmp_path, started, stopped, monkeypatch, case
) -> None:
    server, client, copied_to = _export_copy_server(workdir, tmp_path, episode_export_max_bytes=EXPORT_BYTES - 1)
    if case == "absent":
        shutil.rmtree(tmp_path / "exports")
    copied_to.mkdir(parents=True)
    (copied_to / "ses_earlier.json").write_bytes(b"{}")
    downloads = _record_downloads(monkeypatch, stopped)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0, False, None, True)
    copy = verified.json()["opencode_export_copy"]
    listed = {name: len(data) for name, data in EXPORTS.items()} if case == "oversize" else {}
    expected = {"status": case, "dir": str(copied_to), "files": listed, "bytes": sum(listed.values())}
    assert {key: copy[key] for key in expected} == expected
    detail = f"exceed episode_export_max_bytes ({EXPORT_BYTES - 1})" if case == "oversize" else "has no"
    assert detail in copy["detail"]
    # Nothing was downloaded, and the earlier copy is gone too.
    assert downloads == [] and not copied_to.exists()
    assert stopped == [started[0][0]]


@pytest.mark.parametrize(
    ("failure", "detail"),
    [
        ("listing_exit_code", "exit 1"),
        ("listing_error_type", "error timeout"),
        # As a provider that caps a command's output could return it, without the listing's end.
        ("listing_cut_short", "cut short"),
        ("download_raises", "RuntimeError: the provider lost the file"),
        ("download_cut_short", "size differs from the listing"),
        ("download_hangs", "ran past"),
    ],
)
def test_an_export_copy_that_fails_keeps_no_file_and_changes_no_grade(
    workdir, tmp_path, started, stopped, monkeypatch, caplog, failure, detail
) -> None:
    server, client, copied_to = _export_copy_server(workdir, tmp_path)
    copied_to.mkdir(parents=True)
    (copied_to / "ses_earlier.json").write_bytes(b"{}")
    listings = {
        "listing_exit_code": SandboxExecResult(stdout="", stderr="cd: permission denied", return_code=1),
        "listing_error_type": SandboxExecResult(
            stdout=None, stderr="timed out", return_code=125, error_type="timeout"
        ),
        "listing_cut_short": SandboxExecResult(stdout="13:sessions.json/", stderr=None, return_code=0),
    }
    if failure in listings:
        monkeypatch.setattr(AsyncSandbox, "exec", AsyncMock(return_value=listings[failure]))
    else:
        monkeypatch.setattr("resources_servers.opencode_search.app._EXPORT_COPY_TIMEOUT_S", 0.5)
        real_download = AsyncSandbox.download
        downloads: list[str] = []

        async def download(self: AsyncSandbox, remote_path: str, local_path: Path | str) -> None:
            downloads.append(remote_path)
            if failure == "download_hangs":
                # Far past the copy's time limit, which the test lowers.
                await asyncio.sleep(30)
            elif failure == "download_cut_short":
                Path(local_path).write_bytes(b"{")
            # The first file is copied, so a partial copy is left behind when the second fails.
            elif len(downloads) > 1:
                raise RuntimeError("the provider lost the file")
            else:
                await real_download(self, remote_path, local_path)

        monkeypatch.setattr(AsyncSandbox, "download", download)
    _mock_judge(server, CORRECT)

    with caplog.at_level("WARNING"):
        verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0, False, None, True)
    copy = verified.json()["opencode_export_copy"]
    assert (copy["status"], copy["dir"], copy["files"], copy["bytes"]) == ("failed", str(copied_to), {}, 0)
    assert detail in copy["detail"]
    assert "were not copied" in caplog.text
    # Nothing is kept, neither a part of this copy nor the earlier one, and the sandbox stopped all the same.
    assert not copied_to.exists()
    assert stopped == [started[0][0]]


@pytest.mark.parametrize(
    ("episode", "task_id", "directory"),
    [
        (EpisodeId(rollout_id="rollout-session"), "t-1", "rollout-session/t-1/attempt-0"),
        # Another attempt of the rollout: its capture key and its attempt directory are its own.
        (EpisodeId(rollout_id="rollout-session", attempt=2), "t-1", "rollout-session-a2/t-1/attempt-2"),
        # Task IDs that are not plain directory names.
        (EpisodeId(rollout_id="rollout-session"), "../set/7", "rollout-session/.._set_7/attempt-0"),
        (EpisodeId(rollout_id="rollout-session"), "..", "rollout-session/__/attempt-0"),
    ],
    ids=["first_attempt", "later_attempt", "task_id_with_slashes", "task_id_of_dots"],
)
def test_each_attempt_of_an_episode_copies_its_exports_to_a_directory_of_its_own(
    workdir, tmp_path, episode, task_id, directory
) -> None:
    exports, copies = tmp_path / "exports", tmp_path / "copies"
    exports.mkdir()
    (exports / "sessions.json").write_bytes(EXPORTS["sessions.json"])
    server = _server(workdir, episode_export_dir=str(exports), episode_export_copy_dir=str(copies))
    client = TestClient(server.setup_webserver())
    seed = ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id=episode,
        task_id=TaskId(taskset="browsecomp_opencode:benchmark", task_id=task_id),
        task_data={"question": "Which city?", "ground_truth": "Paris"},
    )
    assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert verified.json()["opencode_export_copy"]["dir"] == str(copies / directory)
    assert (copies / directory / "sessions.json").read_bytes() == EXPORTS["sessions.json"]


@pytest.mark.parametrize(
    ("seed", "copy_settings"),
    [
        ("episode", ()),
        # Where the harness writes its exports, but nowhere on this host to copy them to.
        ("episode", ("episode_export_dir",)),
        # An agent's own /run borrows no sandbox from this server, so the relay arm copies nothing.
        ("agent_run", ("episode_export_dir", "episode_export_copy_dir")),
    ],
    ids=["default", "without_a_copy_dir", "agent_run"],
)
def test_without_an_export_copy_dir_or_a_lent_sandbox_nothing_is_copied(
    workdir, tmp_path, stopped, monkeypatch, seed, copy_settings
) -> None:
    paths = {"episode_export_dir": str(tmp_path / "exports"), "episode_export_copy_dir": str(tmp_path / "copies")}
    server = _server(workdir, **{name: paths[name] for name in copy_settings})
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed() if seed == "episode" else _row()).status_code == 200
    (tmp_path / "exports").mkdir()
    (tmp_path / "exports" / "sessions.json").write_bytes(EXPORTS["sessions.json"])
    execs = _record_execs(monkeypatch, stopped)
    downloads = _record_downloads(monkeypatch, stopped)
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0, False, None, True)
    assert execs == [] and downloads == [] and "opencode_export_copy" not in verified.json()
    assert not (tmp_path / "copies").exists()


# ---------------------------------------------------------------------------
# Agent-run seed, kept for an agent's own /run
# ---------------------------------------------------------------------------

# The relay preset runs this server with lending off (sandbox_workdir unset); turning it on must not change this arm.
LENDING = pytest.mark.parametrize("lending", [False, True], ids=["without_lending", "with_lending"])


@LENDING
def test_an_agent_run_seed_replies_with_the_handle_of_the_sandbox_it_started(workdir, started, lending) -> None:
    """The relay agent attaches to the sandbox the handle names; without it the agent starts one of its own."""
    server = _server(workdir if lending else None)
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


@LENDING
@pytest.mark.parametrize(("reason", "kind"), [("max_iteration", "agent_limit"), ("", None)])
def test_an_agent_run_verify_grades_and_masks_a_budget_ending(
    workdir, started, stopped, lending, reason, kind
) -> None:
    server = _server(workdir if lending else None)
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


@pytest.mark.parametrize(
    "sandbox_workdir",
    # The last two are refused by opencode_agent: a path with a NUL, and one in its own storage.
    ["workspace", "/", "/tmp", "/tmp/", "/workspace/../tmp", "/work\x00space", "/tmp/nemo-gym-opencode-sessions/x"],
)
def test_a_workdir_a_harness_cannot_use_is_refused_at_startup(sandbox_workdir) -> None:
    with pytest.raises(ValueError, match="sandbox_workdir must be an absolute directory"):
        _server(sandbox_workdir)


def test_a_sandbox_config_a_lent_sandbox_cannot_take_is_refused_at_startup() -> None:
    with pytest.raises(ValueError, match="unknown_setting"):
        _server("/workspace", sandbox_config=_sandbox_config(unknown_setting=1))


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        (
            {"plugin_log_path": "/tmp/plugins.log", "episode_sandbox_env": {"NEMO_GYM_PLUGIN_LOG": "/tmp/other.log"}},
            "must equal plugin_log_path",
        ),
        ({"episode_sandbox_env": {"NEMO_GYM_PLUGIN_LOG": "/tmp/plugins.log"}}, "must equal plugin_log_path"),
        ({"required_plugin_markers": ["tool_surface task=off bash=off"]}, "set plugin_log_path"),
        ({"plugin_log_path": "plugins.log"}, "plugin_log_path must be an absolute sandbox path"),
        # The model's tools could change what lies in its working directory.
        ({"plugin_log_path": "/workspace/plugins.log"}, "plugin_log_path must lie outside sandbox_workdir"),
        # opencode_agent's rules for its export_dir.
        ({"episode_export_dir": "exports"}, "episode_export_dir must be an absolute sandbox directory"),
        ({"episode_export_dir": "/"}, "episode_export_dir must be an absolute sandbox directory"),
        (
            {"episode_export_dir": "/tmp/nemo-gym-opencode-sessions/exports"},
            "episode_export_dir must be an absolute sandbox directory",
        ),
        ({"episode_export_dir": "/workspace"}, "must be neither sandbox_workdir, inside it nor above it"),
        ({"episode_export_dir": "/workspace/exports"}, "must be neither sandbox_workdir, inside it nor above it"),
        (
            {"episode_export_dir": "/tmp/exports", "episode_export_copy_dir": "copies"},
            "episode_export_copy_dir must be an absolute directory on this host",
        ),
        ({"episode_export_copy_dir": "/copies"}, "episode_export_copy_dir needs episode_export_dir"),
        ({"episode_export_max_bytes": 0}, "greater than 0"),
        ({"episode_sandbox_files": {"nemo-gym-plugin.js": PLUGINS[0]}}, "must map absolute sandbox paths"),
        ({"episode_sandbox_files": {"/workspace/p.js": PLUGINS[0]}}, "must lie outside sandbox_workdir"),
        (
            {"episode_sandbox_files": {"/tmp/nemo-gym-opencode-runtime-1.17.11/p.js": PLUGINS[0]}},
            "where opencode_agent keeps its runtime",
        ),
        ({"episode_sandbox_files": {"/tmp/p.js": str(REPOSITORY / PLUGINS[0])}}, "relative to the Gym root"),
        ({"episode_sandbox_files": {"/tmp/p.js": "../p.js"}}, "relative to the Gym root"),
        # The plugins read one key, and the lent sandbox's environment cannot pick one from a pool for each row.
        ({"episode_sandbox_env": {"EXA_API_KEYS": "k0:k1"}}, "set one key as EXA_API_KEY"),
        ({"sandbox_config": _sandbox_config(env={"EXA_API_KEYS": "k0:k1"})}, "set one key as EXA_API_KEY"),
        ({"exa_cache_root": "/search-cache", "host": "127.0.0.1"}, "cannot reach the search cache proxy"),
        ({"exa_cache_root": "/search-cache", "host": "0.0.0.0"}, "cannot reach the search cache proxy"),
    ],
    ids=[
        "plugin_log_paths_differ",
        "plugin_log_path_unset",
        "markers_without_plugin_log",
        "relative_plugin_log",
        "plugin_log_inside_workdir",
        "relative_export_dir",
        "root_export_dir",
        "export_dir_in_opencode_storage",
        "export_dir_is_workdir",
        "export_dir_inside_workdir",
        "relative_export_copy_dir",
        "export_copy_dir_without_export_dir",
        "no_export_size",
        "relative_episode_file",
        "episode_file_inside_workdir",
        "episode_file_in_opencode_storage",
        "absolute_episode_file_source",
        "parent_episode_file_source",
        "search_key_pool",
        "search_key_pool_in_sandbox_config",
        "search_cache_on_loopback",
        "search_cache_on_any_address",
    ],
)
def test_episode_settings_that_cannot_work_are_refused_at_startup(settings, error) -> None:
    with pytest.raises(ValueError, match=error):
        _server("/workspace", **settings)


def test_an_export_dir_above_the_workdir_is_refused_at_startup() -> None:
    """opencode_agent refuses an export_dir that holds its workdir, as it refuses one inside it."""
    with pytest.raises(ValueError, match="must be neither sandbox_workdir, inside it nor above it"):
        _server("/workspace/task", episode_export_dir="/workspace")


@pytest.mark.parametrize("named", ["/tmp/plugins.log", ""])
def test_plugin_log_settings_that_agree_are_accepted(named) -> None:
    """NEMO_GYM_PLUGIN_LOG may repeat plugin_log_path, or be left empty for plugin_log_path to set."""
    server = _server(
        "/workspace", plugin_log_path="/tmp/plugins.log", episode_sandbox_env={"NEMO_GYM_PLUGIN_LOG": named}
    )

    assert server._lent_sandbox_config().env["NEMO_GYM_PLUGIN_LOG"] == "/tmp/plugins.log"


def test_without_lending_the_search_cache_needs_no_sandbox_reachable_address() -> None:
    """Only a lent sandbox is given the proxy's address: an agent's own /run works it out itself, so a server that
    lends nothing starts on loopback as before."""
    server = _server(None, exa_cache_root="/search-cache", host="127.0.0.1")

    assert server._lent_sandboxes is None


@pytest.mark.parametrize(
    "source",
    [
        "resources_servers/opencode_search/opencode_bundle/nemo_gym_missing.js",
        "resources_servers/opencode_search/opencode_bundle",
    ],
    ids=["missing", "directory"],
)
def test_an_episode_file_that_cannot_be_read_is_refused_at_startup(source) -> None:
    with pytest.raises(FileNotFoundError, match=f"source '{source}' is not a file"):
        _server("/workspace", episode_sandbox_files={"/tmp/nemo-gym-plugin-missing.js": source})


def test_only_a_lending_server_needs_one_worker() -> None:
    assert _server(None).multiple_workers_unsupported_reason() is None
    assert _server("/workspace").multiple_workers_unsupported_reason() == "lent sandboxes live in one worker's memory"


# ---------------------------------------------------------------------------
# Grading: empty answers, failed runs, judge failures and the verify row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seeded", [False, True], ids=["without_a_seed", "episode"])
@pytest.mark.parametrize("answer", ["", " \n\t"], ids=["empty", "whitespace"])
@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("", None),
        ("max_iteration", "agent_limit"),
        ("context_window", "model_context_limit"),
        ("agent_timed_out", "agent_timeout"),
    ],
    ids=["no_ending", "max_iteration", "context_window", "agent_timed_out"],
)
def test_an_empty_answer_is_a_graded_zero_without_a_judge_call(workdir, seeded, answer, reason, kind) -> None:
    """No answer is a measured 0: the judge is not asked, the row counts as graded (evaluation_completed), and a budget
    ending still masks it with its kind."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    if seeded:
        assert client.post("/seed_session", json=_seed()).status_code == 200
    judge = _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(answer, opencode_terminal_reason=reason))

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (0.0, kind is not None, kind, True)
    row = verified.json()
    assert (row["final_text"], row["extracted_final_answer"], row["judge_parsed"]) == ("", "", False)
    assert (row["judge_attempts"], row["judge_time_taken"]) == (0, 0.0)
    assert bool(row["failure_reason"]) == (kind is not None)
    assert judge.await_count == 0


def test_an_episode_with_an_empty_answer_keeps_its_plugin_log_mask(workdir, tmp_path) -> None:
    """The ending read from the plugin log masks an empty answer as it masks a judged one."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    log.write_text(SENTINEL + TURN_CAP)
    judge = _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body(""), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (0.0, True, "agent_limit", True)
    assert judge.await_count == 0


def test_a_think_block_is_stripped_before_the_judge(workdir) -> None:
    """The judge grades what follows a reasoning model's <think> block, never the block itself, which usually spans
    several lines."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    judge = _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=_verify_body("<think>Marseille,\nperhaps</think>Exact Answer: Paris"))

    assert _graded(verified) == (1.0, False, None, True)
    judge_request = judge.await_args.kwargs["json"].model_dump_json()
    assert "Marseille" not in judge_request and "Exact Answer: Paris" in judge_request


def test_an_empty_plugin_log_is_refused_as_one_without_the_sentinel(workdir, tmp_path, started, stopped) -> None:
    """A log that exists but holds nothing reads as "", which lacks the sentinel's line: the run is refused, terminally,
    as when no log was written."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    log.write_text("")
    judge = _mock_judge(server, CORRECT)

    refused = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY)

    assert refused.status_code == 422, refused.text
    assert "[nemo-gym-plugin] loaded sentinel" in refused.json()["detail"]
    assert judge.await_count == 0 and stopped == [started[0][0]]
    # The empty log it read is kept, so a retry is refused the same way.
    assert client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY).status_code == 422


# The line nemo_gym_turn_budget.js reports for an overflow of the context window that compaction did not recover from.
OVERFLOW = "[nemo-gym-plugin] context_overflow session=s\n"
# The outcomes of a failed run that the default agent outcome policy sends to /verify, and model_context_limit, which the
# session preset adds; each with the plugin log line of the ending that masks the row, and that ending's kind.
POLICY_VERIFIED_FAILURES = [
    ({"failure_kind": "agent_limit"}, TURN_CAP, "agent_limit"),
    ({"failure_kind": "agent_invalid_output"}, TURN_CAP, "agent_limit"),
    ({"failure_kind": "provider_oom_killed"}, TURN_CAP, "agent_limit"),
    ({"failure_kind": "model_request_error", "status_code": 400}, TURN_CAP, "agent_limit"),
    ({"failure_kind": "model_request_error", "status_code": 413}, TURN_CAP, "agent_limit"),
    ({"failure_kind": "model_request_error", "status_code": 422}, TURN_CAP, "agent_limit"),
    ({"failure_kind": "model_context_limit"}, OVERFLOW, "model_context_limit"),
]


def _failed_response(answer: str, outcome: dict[str, Any]) -> dict[str, Any]:
    """A failed run's response as a harness reports it: its outcome in the metadata, and the answer the run reached, or
    no output at all."""
    response = _response(answer) | {"status": "failed", "metadata": {"ng_agent_outcome": json.dumps(outcome)}}
    return response if answer else response | {"output": []}


@pytest.mark.parametrize("answer", ["Exact Answer: Paris", ""], ids=["answer", "no_output"])
@pytest.mark.parametrize(
    ("outcome", "ending", "kind"),
    POLICY_VERIFIED_FAILURES,
    ids=[
        "agent_limit",
        "agent_invalid_output",
        "provider_oom_killed",
        "model_request_error_400",
        "model_request_error_413",
        "model_request_error_422",
        "model_context_limit",
    ],
)
def test_a_failed_response_the_outcome_policy_verifies_is_graded_with_its_receipts(
    workdir, tmp_path, outcome, ending, kind, answer
) -> None:
    """verify reads no response status: a failed run's last answer is graded, or scores 0 without a judge call when
    there is none, the plugin log's ending masks the row, and the failure report single_agent_turn forwards comes back.
    The grade replaces request fields of the same names."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    log.write_text(SENTINEL + ending)
    judge = _mock_judge(server, CORRECT)
    failure_report = {"failure_id": None, "events": {}, "omitted": True}
    body = _verify_body(
        response=_failed_response(answer, outcome),
        failure_report=failure_report,
        # Request fields named like the grade's.
        reward=0.5,
        mask_sample=False,
        failure_kind="session_lost",
        failure_reason="a reason from the request",
        evaluation_completed=False,
    )

    verified = client.post("/verify", json=body, headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == (1.0 if answer else 0.0, True, kind, True)
    assert verified.json()["failure_reason"] != "a reason from the request"
    assert verified.json()["failure_report"] == failure_report
    assert judge.await_count == (1 if answer else 0)


@pytest.mark.parametrize(
    ("ending", "graded"),
    [(OVERFLOW, (1.0, True, "model_context_limit", True)), ("", (1.0, False, None, True))],
    ids=["overflow", "no_ending"],
)
def test_an_incomplete_response_is_masked_only_by_the_plugin_log(workdir, tmp_path, ending, graded) -> None:
    """opencode_agent ends a run that overflowed the context window as incomplete, as it ends one that timed out, and
    puts agent_timeout_sec on every response. verify reads neither: the plugin log's ending alone masks the row, never
    as agent_timeout."""
    log = tmp_path / "plugins.log"
    server, client = _plugin_log_server(workdir, log)
    log.write_text(SENTINEL + ending)
    _mock_judge(server, CORRECT)
    response = _response("Exact Answer: Paris") | {
        "status": "incomplete",
        "metadata": {"harness_execution": "sandbox", "agent_timeout_sec": "21600"},
    }

    verified = client.post("/verify", json=_verify_body(response=response), headers=EPISODE_VERIFY)

    assert verified.status_code == 200, verified.text
    assert _graded(verified) == graded


def test_a_row_masked_without_a_grade_names_its_failure_and_gives_a_reason(workdir) -> None:
    """A lost session, a judge without a verdict and a judge that never answered each leave a masked row, not counted as
    graded, that says why."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed()).status_code == 200
    assert client.post("/close_session", json=_close()).status_code == 200
    replies = [client.post("/verify", json=_verify_body())]
    _mock_judge(server, NO_VERDICT)
    client.cookies.clear()
    replies.append(client.post("/verify", json=_verify_body()))
    server.server_client.post = AsyncMock(side_effect=JudgeError("connection refused"))
    client.cookies.clear()
    replies.append(client.post("/verify", json=_verify_body()))

    assert [_graded(reply) for reply in replies] == [
        (0.0, True, "session_lost", False),
        (0.0, True, "judge_unparseable", False),
        (0.0, True, "judge_failed", False),
    ]
    reasons = [reply.json()["failure_reason"] for reply in replies]
    assert all(isinstance(reason, str) and reason.strip() for reason in reasons), reasons


def test_a_judge_that_never_answers_trips_the_dead_judge_breaker(workdir) -> None:
    """Rollouts in a row whose judge could not be reached trip the breaker, and verify answers 500; a verdict starts the
    count again."""
    server = _server(workdir, judge_max_consecutive_failures=2)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)

    def verify() -> Any:
        client.cookies.clear()
        return client.post("/verify", json=_verify_body())

    server.server_client.post = AsyncMock(side_effect=JudgeError("connection refused"))
    first = verify()
    assert first.status_code == 200, first.text
    assert _graded(first) == (0.0, True, "judge_failed", False)
    tripped = verify()
    assert tripped.status_code == 500, tripped.text

    _mock_judge(server, CORRECT)
    assert _graded(verify()) == (1.0, False, None, True)
    # The count started again, so one more unreachable judge sends its row to the failures sidecar and stops nothing.
    server.server_client.post = AsyncMock(side_effect=JudgeError("connection refused"))
    assert _graded(verify()) == (0.0, True, "judge_failed", False)


def test_a_zero_limit_turns_the_dead_judge_breaker_off(workdir) -> None:
    """With judge_max_consecutive_failures 0, rollouts without a verdict are masked one by one, and verify never answers
    500."""
    server = _server(workdir, judge_max_consecutive_failures=0)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    _mock_judge(server, NO_VERDICT)

    replies = []
    for _ in range(5):
        client.cookies.clear()
        replies.append(client.post("/verify", json=_verify_body()))

    assert [reply.status_code for reply in replies] == [200] * 5
    assert {_graded(reply) for reply in replies} == {(0.0, True, "judge_unparseable", False)}


# single_agent_turn's verify body for an episode: the task_data keys flat at the top level, then the request, the agent's
# response and the failure report it forwards.
SESSION_VERIFY_BODY = {
    "question": "Which city?",
    "ground_truth": "Paris",
    "responses_create_params": {"input": [{"role": "user", "content": "Which city?"}]},
    "response": _response("Exact Answer: Paris"),
    "failure_report": None,
}
# The grading fields of a verify row, with their types.
SEARCH_FIELD_TYPES = {
    "extracted_final_answer": str,
    "final_text": str,
    "judge_parsed": bool,
    "judge_attempts": int,
    "judge_time_taken": float,
    "mask_sample": bool,
    "evaluation_completed": bool,
}


@pytest.mark.parametrize(
    ("seed", "body", "headers"),
    [(_row(), _verify_body(opencode_terminal_reason=""), {}), (_seed(), SESSION_VERIFY_BODY, EPISODE_VERIFY)],
    ids=["relay", "session"],
)
def test_a_graded_row_keeps_the_search_fields_and_echoes_its_request(workdir, seed, body, headers) -> None:
    """The relay agent's verify body and single_agent_turn's get the same row: the grading fields under their names and
    types, and the request at the top level as it was sent."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=seed).status_code == 200
    _mock_judge(server, CORRECT)

    verified = client.post("/verify", json=body, headers=headers)

    assert verified.status_code == 200, verified.text
    row = verified.json()
    assert {field: type(row[field]) for field in SEARCH_FIELD_TYPES} == SEARCH_FIELD_TYPES
    assert _graded(verified) == (1.0, False, None, True)
    assert (row["extracted_final_answer"], row["final_text"]) == ("Paris", "Exact Answer: Paris")
    assert (row["judge_parsed"], row["judge_attempts"], row["failure_reason"]) == (True, 1, None)
    # The request comes back at the top level as it was sent, so the session's task_data keys stay flat.
    sent = {key: value for key, value in body.items() if key not in ("responses_create_params", "response")}
    assert {key: row[key] for key in sent} == sent and "task_data" not in row
    assert row["responses_create_params"]["input"][0]["content"] == "Which city?"
    assert row["response"]["output"][0]["content"][0]["text"] == "Exact Answer: Paris"


# ---------------------------------------------------------------------------
# Lifecycle: re-seeds, stops that fail, hang or are cancelled, and sandboxes the relay agent stopped
# ---------------------------------------------------------------------------

from collections.abc import Awaitable

import anyio

from nemo_gym.core import FailureHeader
from nemo_gym.sandbox import LocalProvider
from resources_servers.opencode_search.app import SearchVerifyRequest


async def _seed_agent_run(server: SearchResourcesServer) -> AsyncSandbox:
    """Seed an agent's own /run under the cookie session "run", and return the sandbox it started."""
    await asyncio.wait_for(
        server.seed_session(SimpleNamespace(session={SESSION_ID_KEY: "run"}), SearchSeedSessionRequest()), timeout=30
    )
    return server._agent_run_sandboxes["run"]


async def _seed_episode(server: SearchResourcesServer) -> None:
    """Seed the episode session "session" as an Environment Server does."""
    seed = ResourcesSeedSessionRequest.model_validate(_seed())
    await asyncio.wait_for(server.seed_session(SimpleNamespace(session={}), seed), timeout=30)


def _record_finished_stops(monkeypatch: pytest.MonkeyPatch) -> list[AsyncSandbox]:
    """Make every stop await once before it stops its sandbox, as a provider's request does, and record each sandbox
    whose stop finished. A cancellation delivered at that await cuts short a stop that is not shielded."""
    finished: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        await anyio.sleep(0)
        await real_stop(self)
        finished.append(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return finished


async def _cancel_once_reached(reached: asyncio.Event, call: Awaitable[Any]) -> None:
    """Await call, and once reached is set, cancel it as the server cancels a request whose client disconnected: at that
    await and at every later one that is not shielded."""
    scope = anyio.CancelScope()

    async def run() -> None:
        with scope:
            await call

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(run)
            await reached.wait()
            scope.cancel()


@LENDING
def test_a_reused_agent_run_cookie_stops_the_first_sandbox_before_it_starts_another(
    workdir, started, stopped, monkeypatch, lending
) -> None:
    """A second seed under the same cookie replaces the run: its sandbox stops before another starts, rather than stay
    running with nothing left holding it."""
    server = _server(workdir if lending else None)
    client = TestClient(server.setup_webserver())
    first = client.post("/seed_session", json=_row())
    stops_at_start: list[int] = []
    recorded_start = AsyncSandbox.start

    async def start(self: AsyncSandbox, spec: Optional[SandboxSpec] = None, **kwargs: Any) -> AsyncSandbox:
        stops_at_start.append(len(stopped))
        return await recorded_start(self, spec, **kwargs)

    monkeypatch.setattr(AsyncSandbox, "start", start)

    second = client.post("/seed_session", json=_row())

    assert first.status_code == second.status_code == 200, second.text
    # The second sandbox started once the first had been stopped.
    assert stops_at_start == [1] and stopped == [started[0][0]] and len(started) == 2
    assert second.json() == {"sandbox_handle": started[1][0].sandbox_id} != first.json()
    assert list(server._agent_run_sandboxes.values()) == [started[1][0]]


@LENDING
def test_an_agent_run_reseed_whose_stop_fails_is_refused_and_keeps_the_first_sandbox(
    workdir, started, monkeypatch, lending
) -> None:
    """A stop that fails raises and keeps its sandbox, so a later stop can try again: the re-seed answers 500 and starts
    nothing in its place."""
    server = _server(workdir if lending else None)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_row()).status_code == 200
    (session_id,) = server._agent_run_sandboxes
    stop = AsyncMock(side_effect=RuntimeError("the provider refused the stop"))
    monkeypatch.setattr(AsyncSandbox, "stop", stop)

    refused = client.post("/seed_session", json=_row())

    assert refused.status_code == 500, refused.text
    assert stop.await_count == 1 and len(started) == 1
    assert server._agent_run_sandboxes == {session_id: started[0][0]}


def test_an_episode_seed_cannot_take_over_an_agent_run_session(workdir, started, stopped) -> None:
    """The converse of an agent run's seed refused over an episode session: the run keeps its sandbox."""
    server = _server(workdir)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_row()).status_code == 200
    (cookie_session,) = server._agent_run_sandboxes
    client.cookies.clear()

    refused = client.post("/seed_session", json=_seed(cookie_session))

    assert refused.status_code == 409, refused.text
    assert len(started) == 1 and stopped == [] and server.has_session_state(cookie_session)


async def test_an_agent_run_stop_finishes_when_its_release_is_cancelled(workdir, monkeypatch) -> None:
    """A client that disconnects cancels every later await of its request. The stop of an agent run's sandbox is
    shielded from that: it finishes, and the server lets the sandbox go."""
    server = _server(None)
    sandbox = await _seed_agent_run(server)
    finished: list[AsyncSandbox] = []

    async def stop(self: AsyncSandbox) -> None:
        # An await, as a provider's request has: a cancellation delivered here cuts short a stop that is not shielded.
        await anyio.sleep(0)
        finished.append(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)

    with anyio.CancelScope() as scope:
        scope.cancel()
        await server.release_session_state("run")

    assert finished == [sandbox] and server._agent_run_sandboxes == {}


async def test_an_agent_run_stop_that_hangs_times_out_and_keeps_its_sandbox(workdir, monkeypatch) -> None:
    """A provider stop that never returns would hold a re-seed, a verify or the shutdown forever. It gives up after
    STOP_TIMEOUT_S, the bound LentSandboxes applies, and the server keeps the sandbox for a later stop."""
    server = _server(None)
    sandbox = await _seed_agent_run(server)
    monkeypatch.setattr("resources_servers.opencode_search.app.STOP_TIMEOUT_S", 0.05)

    async def stop(self: AsyncSandbox) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(AsyncSandbox, "stop", stop)

    release = asyncio.create_task(server.release_session_state("run"))
    done, _ = await asyncio.wait({release}, timeout=5)

    assert done == {release}
    with pytest.raises(TimeoutError):
        release.result()
    assert server._agent_run_sandboxes == {"run": sandbox}


@pytest.mark.parametrize("seed", ["episode", "agent_run"])
def test_a_verify_whose_stop_fails_keeps_its_grade_and_the_stop_is_retried(
    workdir, started, monkeypatch, caplog, seed
) -> None:
    """A stop before the judge that fails is logged and leaves the grade as it is. verifying() tries again on exit and
    reports session_release_failed; the server keeps the sandbox, and the episode's close, or the shutdown after an
    agent run, stops it."""
    server = _server(workdir)
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    attempts: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        attempts.append(self)
        if len(attempts) <= 2:
            raise RuntimeError("the provider refused the stop")
        await real_stop(self)

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed() if seed == "episode" else _row()).status_code == 200
        session_id = "session" if seed == "episode" else next(iter(server._agent_run_sandboxes))
        monkeypatch.setattr(AsyncSandbox, "stop", stop)
        judge = _mock_judge(server, CORRECT)

        with caplog.at_level("WARNING"):
            verified = client.post("/verify", json=_verify_body(), headers=EPISODE_VERIFY if seed == "episode" else {})

        assert verified.status_code == 200, verified.text
        assert _graded(verified) == (1.0, False, None, True) and judge.await_count == 1
        # The stop before the judge and verifying()'s on exit both failed, and the server still holds the sandbox.
        assert attempts == [started[0][0]] * 2 and server.has_session_state(session_id)
        assert "before grading" in caplog.text
        report = FailureHeader.decode(verified.headers)
        assert report is not None and "session_release_failed" in {e.failure_kind for e in report.events.values()}
        if seed == "episode":
            assert client.post("/close_session", json=_close()).status_code == 200
            assert attempts == [started[0][0]] * 3 and not server.has_session_state(session_id)
    # No close follows an agent run: the shutdown stops its sandbox.
    assert attempts == [started[0][0]] * 3 and not server.has_session_state(session_id)


async def test_a_verify_cancelled_while_it_stops_its_sandbox_still_stops_it(workdir, monkeypatch) -> None:
    """A client that disconnects cancels its verify at every await from then on. The stop before the judge is shielded:
    it finishes, and the server lets the sandbox go instead of holding one whose stop was cut short."""
    server = _server(None)
    sandbox = await _seed_agent_run(server)
    stopping, cancelled = asyncio.Event(), asyncio.Event()
    finished: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopping.set()
        await cancelled.wait()
        await real_stop(self)
        finished.append(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    _mock_judge(server, CORRECT)
    request = SimpleNamespace(session={SESSION_ID_KEY: "run"}, headers={})
    scope = anyio.CancelScope()

    async def verify() -> None:
        with scope:
            await server.verify(request, SearchVerifyRequest.model_validate(_verify_body()))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(verify)
            await stopping.wait()
            scope.cancel()
            cancelled.set()

    assert finished == [sandbox] and server._agent_run_sandboxes == {}


async def test_a_verify_cancelled_while_it_reads_the_plugin_log_still_stops_its_sandbox(
    workdir, tmp_path, started, monkeypatch
) -> None:
    """A verify cancelled before it stopped the sandbox, here while it reads the plugin log, stops it as it ends."""
    server = _server(workdir, plugin_log_path=str(tmp_path / "plugins.log"))
    await _seed_episode(server)
    finished = _record_finished_stops(monkeypatch)
    reading = asyncio.Event()

    async def exec(self: AsyncSandbox, command: str, **_: Any) -> SandboxExecResult:
        reading.set()
        await anyio.sleep_forever()

    monkeypatch.setattr(AsyncSandbox, "exec", exec)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"}, headers=EPISODE_VERIFY)

    await _cancel_once_reached(reading, server.verify(request, SearchVerifyRequest.model_validate(_verify_body())))

    assert finished == [started[0][0]]
    assert not server.has_session_state("session") and server._lent_sandboxes.get("session") is None


@pytest.mark.parametrize(
    "cancelled_in", ["start", "exec"], ids=["while_its_sandbox_starts", "while_it_makes_the_workdir"]
)
async def test_an_episode_seed_cancelled_at_an_await_stops_its_sandbox(workdir, monkeypatch, cancelled_in) -> None:
    """A seed cancelled as its client disconnects leaves no sandbox running. LentSandboxes stops one whose start was
    cancelled; release_session_state stops one whose workdir the seed was creating, though the seed never stored its
    access."""
    server = _server(workdir)
    finished = _record_finished_stops(monkeypatch)
    reached: list[AsyncSandbox] = []
    entered = asyncio.Event()

    async def hang(self: AsyncSandbox, *_: Any, **__: Any) -> None:
        reached.append(self)
        entered.set()
        await anyio.sleep_forever()

    monkeypatch.setattr(AsyncSandbox, cancelled_in, hang)

    await _cancel_once_reached(entered, _seed_episode(server))

    assert len(reached) == 1 and finished == reached
    assert not server.has_session_state("session") and server._lent_sandboxes.get("session") is None
    assert server._sandbox_access == {}


def test_a_sandbox_the_relay_agent_stopped_first_does_not_break_a_reseed_or_the_shutdown(
    workdir, tmp_path, started, stopped
) -> None:
    """The relay agent stops its sandbox itself, also when it refuses a run and never verifies, which leaves this server
    holding a sandbox that is gone. A re-seed of the cookie stops it again, which the provider takes as stopped, before
    it starts another; the shutdown stops the second."""
    server = _server(None)
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_row()).status_code == 200
        ((first, _),) = started
        workspace = tmp_path / "agent-runs" / first.sandbox_id
        assert workspace.is_dir()
        # The agent stops it through a provider client of its own: this server's client still takes it to be running.
        asyncio.run(LocalProvider().close(first._handle))
        assert not workspace.exists()

        reseeded = client.post("/seed_session", json=_row())

        assert reseeded.status_code == 200, reseeded.text
        assert len(started) == 2 and stopped == [first]
        assert reseeded.json() == {"sandbox_handle": started[1][0].sandbox_id}
    assert stopped == [first, started[1][0]] and server._agent_run_sandboxes == {}


def test_a_stop_that_fails_at_shutdown_is_logged_and_the_other_sandboxes_still_stop(
    workdir, started, monkeypatch, caplog
) -> None:
    """The shutdown stops every agent run's sandbox. A stop that fails there, as with a provider that refuses to stop a
    sandbox the relay agent already stopped, is logged; the other sandboxes still stop, and the server still exits."""
    server = _server(None)
    attempts: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        attempts.append(self)
        if self is started[0][0]:
            raise RuntimeError("the provider refused the stop: the sandbox is gone")
        await real_stop(self)

    with caplog.at_level("ERROR"), TestClient(server.setup_webserver()) as client:
        for _ in range(2):
            client.cookies.clear()
            assert client.post("/seed_session", json=_row()).status_code == 200
        monkeypatch.setattr(AsyncSandbox, "stop", stop)

    assert len(attempts) == 2 and {id(sandbox) for sandbox in attempts} == {id(sandbox) for sandbox, _ in started}
    assert started[1][0].teardown_confirmed is True and server._agent_run_sandboxes == {}
    assert "Failed to stop the sandbox of agent run" in caplog.text
