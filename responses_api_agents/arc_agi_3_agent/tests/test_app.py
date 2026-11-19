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
import datetime as dt
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from responses_api_agents.arc_agi_3_agent import app as agent_app
from responses_api_agents.arc_agi_3_agent.app import (
    ArcAgi3Agent,
    ArcAgi3AgentConfig,
    ArcAgi3AgentRunRequest,
    redact,
    repo_path,
)
from responses_api_agents.arc_agi_3_agent.call_log import GameCallLog
from responses_api_agents.arc_agi_3_agent.harness_runtime import AttemptResult, GameStartError


class FakeHttpResponse:
    def __init__(self, payload: dict, status: int = 200, cookies=None):
        self._payload = payload
        self.status = status
        self.ok = status < 400
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class FakeRuntime:
    def __init__(self, attempt=None, raises=None):
        self.attempt, self.raises = attempt, raises
        self.run_dir = Path("/tmp/arc3-run")
        self.run_config_sha256 = "c0ffee"
        self.start_time = None
        self.start_time_utc = None
        self.api_key = "key-1"
        self.calls = []
        self.env = {
            "LOCAL_ANALYZER_TEMPERATURE": "0.7",
            "LOCAL_ANALYZER_TOP_K": "20",
            "LOCAL_ANALYZER_MAX_OUTPUT": "12288",
        }

    async def play_game(self, meta, index, transport, game_key=None):
        self.calls.append((meta["full_game_id"], index))
        if self.raises:
            raise self.raises
        return self.attempt


def make_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides) -> ArcAgi3Agent:
    monkeypatch.setattr(agent_app, "ensure_flask", lambda: None)
    monkeypatch.setattr(agent_app, "ensure_harness", lambda source, dest: tmp_path / "harness")
    config = ArcAgi3AgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="arc_agi_3_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="arc_agi_3_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        environments_dir=str(tmp_path / "games"),
        **overrides,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.get = AsyncMock(return_value=FakeHttpResponse({"url": "http://game-server.invalid", "games": 25}))
    return ArcAgi3Agent(config=config, server_client=server_client)


def run_request() -> ArcAgi3AgentRunRequest:
    return ArcAgi3AgentRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "Play ls20"}]},
        verifier_metadata={"game_id": "ls20", "full_game_id": "ls20-9607627b", "dispatch_index": 3},
    )


def test_config_defaults_and_hyperparameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path, temperature=1.0, request_seed=5, context_keep_tokens=12288)
    assert agent.config.play_seconds == 7200 and agent.config.concurrency == 25
    assert agent.config.chat_template_kwargs == {}
    hyper = agent.hyperparameters()
    assert hyper["temperature"] == 1.0 and hyper["request_seed"] == 5 and hyper["top_p"] is None
    assert "seed" not in hyper
    assert agent.run_dir().parent == PARENT_DIR / "results" / "arc_agi_3"
    assert make_agent(monkeypatch, tmp_path, run_name="x").run_dir().name == "x"


def test_one_worker_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_workers must be 1"):
        make_agent(monkeypatch, tmp_path, num_workers=2)


def test_repo_path() -> None:
    assert repo_path("/abs/dir") == Path("/abs/dir")
    assert repo_path("rel/dir") == PARENT_DIR / "rel/dir"


def test_run_sends_record_to_verifier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path, reasoning_effort="high")
    record = {"state": "gave_up", "levels_completed": 1, "history": [{}, {}]}
    runtime = FakeRuntime(
        attempt=AttemptResult(record=record, persistence_error="OSError: full", scorecard_id="card-1", guid="guid-1")
    )
    runtime.start_time = dt.datetime(2026, 11, 4, 12, 0, 0)
    runtime.start_time_utc = "2026-11-04T12:00:00+00:00"
    agent._runtime = runtime
    sent = {}

    async def post(server_name, url_path, json, cookies=None):
        sent.update(server_name=server_name, url_path=url_path, body=json)
        return FakeHttpResponse({**json, "reward": 0.5, "rhae": 50.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    result = asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    assert runtime.calls == [("ls20-9607627b", 3)]
    assert sent["server_name"] == "arc_agi_3_resources_server" and sent["url_path"] == "/verify"
    assert sent["body"]["game_record"] == record and sent["body"]["agent_error"] is None
    # F1: a save failure travels as persistence_error next to the intact record; F7: the run clock origin goes along.
    assert sent["body"]["persistence_error"] == "OSError: full"
    assert sent["body"]["benchmark_start_time"] == "2026-11-04T12:00:00+00:00"
    # The authority: the attempt's scorecard, its game run and the key that can close the card.
    assert sent["body"]["scorecard_id"] == "card-1" and sent["body"]["game_guid"] == "guid-1"
    assert sent["body"]["scorecard_api_key"] == "key-1"
    summary = json.loads(sent["body"]["response"]["output"][0]["content"][0]["text"])
    assert summary == {
        "game_id": "ls20",
        "state": "gave_up",
        "levels_completed": 1,
        "actions": 2,
        "run_dir": "/tmp/arc3-run",
    }
    assert result.reward == 0.5 and result.run_dir == "/tmp/arc3-run" and result.run_config_sha256 == "c0ffee"
    assert result.harness == "arc3-kaggle-kit"
    (invocation,) = result.ng_agent_observations["records"]
    assert invocation["kind"] == "agent_invocation" and invocation["stop_reason"] == "gave_up"
    assert invocation["model_calls"] == [] and result.agent_call_log == []
    # The harness's revision and the effective sampling travel with every rollout.
    assert result.harness_revision == {
        "name": "arc3-kaggle-kit",
        "kaggle_harness_dataset": "eladsarafian/arc3-c237-j30vfh-20261102-harness version 1",
        "kaggle_config_dataset": "eladsarafian/arc3-c237-j30vfh-20261102-config version 1",
        "source_commit": "30d2cc8bb582c4126e1020d084a94468a057dea7",  # pragma: allowlist secret
    }
    assert result.sampling["temperature"] == 0.7 and result.sampling["top_k"] == 20
    assert result.sampling["max_output_tokens"] == 12288 and result.sampling["top_p"] is None
    assert result.sampling["reasoning_effort"] == "high"
    assert result.sampling["reasoning_effort_kwarg"] == "reasoning_effort"


def test_run_routes_calls_on_the_rollout_path_and_reports_drains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Training mode: the game's calls go to its run's rollout path; the result lists the gameplay calls and the drain;
    with the per-game clock the verifier measures RHAE@T from the game's own start."""
    agent = make_agent(monkeypatch, tmp_path, run_clock="per_game")
    monkeypatch.setattr(
        type(agent), "url_path_for_run", lambda self, path, body: f"/ng-rollout/7-0/training-token-capture{path}"
    )
    record = {"state": "not_won", "levels_completed": 1, "history": [{}]}
    seen = {}

    class Transport:
        def __init__(self):
            self.log = GameCallLog()

        def take_call_log(self, key):
            return self.log

    transport = Transport()

    class Runtime(FakeRuntime):
        async def play_game(self, meta, index, transport_, game_key=None):
            seen["path"] = agent._game_url_paths[game_key]
            system = {"role": "system", "content": "play"}
            turns = [{"role": "user", "content": f"s{i}"} for i in range(4)]
            transport.log.add(
                role="gameplay",
                history_id="main",
                messages=[system, *turns],
                status=200,
                response={"id": "r0"},
                step=4,
                level=1,
            )
            transport.log.add(
                role="helper", history_id="drain-summary", messages=[system, *turns], status=200, response={"id": "s0"}
            )
            transport.log.add_log(
                logged_at=1.0, severity="WARNING", logger="h", message="drain summary unusable", step=4, level=1
            )
            transport.log.add(
                role="gameplay",
                history_id="main",
                status=200,
                response={"id": "r1"},
                messages=[system, turns[-1], {"role": "user", "content": "[Summary of earlier history]"}],
            )
            return AttemptResult(record=record, started_at_utc="2026-11-17T08:00:00+00:00", game_key=game_key)

    agent._runtime, agent._transport = Runtime(), transport
    sent = {}

    async def post(server_name, url_path, json, cookies=None):
        sent.update(body=json)
        return FakeHttpResponse({**json, "reward": 0.1})

    agent.server_client.post = AsyncMock(side_effect=post)
    result = asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    assert seen["path"] == "/ng-rollout/7-0/training-token-capture/v1/chat/completions"
    assert agent._game_url_paths == {}
    assert sent["body"]["benchmark_start_time"] == "2026-11-17T08:00:00+00:00"
    invocation, compaction = result.ng_agent_observations["records"]
    assert [c["response_id"] for c in invocation["model_calls"]] == ["r0", "r1"]
    assert compaction["kind"] == "context_compaction"
    assert [c["response_id"] for c in compaction["model_calls"]] == ["s0"]
    assert compaction["before_model_call"]["response_id"] == "r0"
    log = result.agent_call_log
    assert [(c["index"], c["role"], c["history_id"], c["response_id"]) for c in log] == [
        (0, "gameplay", "main", "r0"),
        (1, "helper", "drain-summary", "s0"),
        (2, "gameplay", "main", "r1"),
    ]
    assert (log[0]["step"], log[0]["level"]) == (4, 1)
    assert [(e["level"], e["message"]) for e in result.harness_log["lines"]] == [(1, "drain summary unusable")]


def test_run_reports_a_model_that_never_answered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A dead model endpoint (every call an HTTP error): the game still ends with a record, and the verifier gets
    model_error, so the result is invalid and the rollout masked, never a valid 0-action score."""
    agent = make_agent(monkeypatch, tmp_path)
    log = GameCallLog()

    class Transport:
        def take_call_log(self, key):
            return log

    class Runtime(FakeRuntime):
        async def play_game(self, meta, index, transport_, game_key=None):
            for status in (404, 404, 502):
                log.add(role="gameplay", history_id="main", messages=[], status=status, response=None)
            return AttemptResult(record={"state": "gave_up", "levels_completed": 0, "history": []}, game_key=game_key)

    agent._runtime, agent._transport = Runtime(), Transport()
    sent = {}

    async def post(server_name, url_path, json, cookies=None):
        sent.update(body=json)
        return FakeHttpResponse({**json, "reward": 0.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    result = asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    assert sent["body"]["model_error"] == "every model call of the game failed (3 calls; last: HTTP 502)"
    assert sent["body"]["agent_error"] is None
    assert result.ng_agent_observations["records"][0]["status"] == "failed"


def test_run_scores_a_game_that_cannot_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    agent._runtime = FakeRuntime(raises=GameStartError("pins differ"))
    sent = {}

    async def post(server_name, url_path, json, cookies=None):
        sent.update(body=json)
        return FakeHttpResponse({**json, "reward": 0.0})

    agent.server_client.post = AsyncMock(side_effect=post)
    result = asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    assert sent["body"]["game_record"] is None
    assert sent["body"]["agent_error"] == "GameStartError: pins differ"
    assert result.reward == 0.0


def test_post_model_and_aggregate_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)

    class Morsel:
        value = "abc"

    calls = []

    async def post(server_name, url_path, json, cookies=None):
        calls.append((server_name, url_path, cookies))
        if url_path == "/aggregate_metrics":
            return FakeHttpResponse({"agent_metrics": {"rhae": 12.5}, "key_metrics": {"rhae": 12.5}})
        return FakeHttpResponse({"choices": []}, cookies={"session": Morsel()})

    agent.server_client.post = AsyncMock(side_effect=post)
    status, text, cookies = asyncio.run(agent._post_model({"messages": []}, {"session": "old"}))
    assert status == 200 and json.loads(text) == {"choices": []} and cookies == {"session": "abc"}
    assert calls[0] == ("policy_model", "/v1/chat/completions", {"session": "old"})
    metrics = asyncio.run(agent.aggregate_metrics(MagicMock()))
    assert calls[1][:2] == ("arc_agi_3_resources_server", "/aggregate_metrics")
    assert metrics.key_metrics == {"rhae": 12.5}


def test_other_errors_are_not_scored_as_not_started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F1: only GameStartError means "never started"; anything else fails the rollout loudly.
    agent = make_agent(monkeypatch, tmp_path)
    agent._runtime = FakeRuntime(raises=OSError("disk"))
    agent.server_client.post = AsyncMock()
    with pytest.raises(OSError):
        asyncio.run(agent.run(MagicMock(cookies={}), run_request()))
    agent.server_client.post.assert_not_called()


def test_requires_context_error_passthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F3: without the model server's context-error passthrough the agent refuses to start.
    agent = make_agent(monkeypatch, tmp_path)
    agent.server_client.global_config_dict = {
        "policy_model": {"responses_api_models": {"vllm_model": {"base_url": ["http://x/v1"]}}}
    }
    with pytest.raises(RuntimeError, match="propagate_context_overflow_errors"):
        asyncio.run(agent._ensure_runtime())


def test_redact() -> None:
    config = {
        "api_key": "sk-123",  # pragma: allowlist secret
        "base_url": ["http://user:pw@host:8000/v1?token=abc&x=1"],  # pragma: allowlist secret
        "nested": {"Authorization": "Bearer t", "hf_token": "", "ok": 3, "max_tokens": 9, "thinking_token_budget": 8},
        "other": object,
    }
    out = redact(config)
    assert out["api_key"] == "REDACTED" and out["nested"]["Authorization"] == "REDACTED"
    assert out["base_url"] == ["http://REDACTED@host:8000/v1?token=REDACTED&x=1"]
    assert out["nested"]["hf_token"] == "" and out["nested"]["ok"] == 3 and isinstance(out["other"], str)
    assert out["nested"]["max_tokens"] == 9 and out["nested"]["thinking_token_budget"] == 8
    assert redact({"HF_TOKEN": "x", "access-token": "y"}) == {"HF_TOKEN": "REDACTED", "access-token": "REDACTED"}


def test_run_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F8: the effective configuration, redacted, with the engine's own report (or why it could not be read).
    agent = make_agent(monkeypatch, tmp_path, seed=3, extra_body={"thinking_token_budget": 8192})
    monkeypatch.setattr(agent_app, "load_manifest", lambda: {"provenance": {"tool_agent_py_sha256": "22a2ae7b"}})
    (tmp_path / "games" / "ls20" / "v1").mkdir(parents=True)
    (tmp_path / "games" / "ls20" / "v1" / "metadata.json").write_text("{}")

    async def unreachable(method, url, **kwargs):
        raise ConnectionError("engine down")

    monkeypatch.setattr(agent_app, "request", unreachable)
    model_config = {"base_url": ["http://h:8000/v1"], "api_key": "secret", "propagate_context_overflow_errors": True}
    config = asyncio.run(agent.run_config({"LOCAL_ANALYZER_TEMPERATURE": "0.7"}, model_config))
    assert config["model_server"]["config"]["api_key"] == "REDACTED"
    assert config["engine_observed"] == {"error": "ConnectionError: engine down"}
    assert config["transport"]["extra_body"] == {"thinking_token_budget": 8192} and config["agent"]["seed"] == 3
    assert config["environments"]["games"] == {"ls20-v1": {"version": "v1", "files_sha256": {}}}
    assert config["harness"]["provenance"]["tool_agent_py_sha256"] == "22a2ae7b"


def test_lifespan_shutdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F6: the server's shutdown stops and finalizes the runtime.
    from fastapi.testclient import TestClient

    agent = make_agent(monkeypatch, tmp_path)
    calls = []

    class Runtime:
        async def shutdown(self, transport):
            calls.append(transport)

    agent._runtime, agent._transport = Runtime(), "transport"
    with TestClient(agent.setup_webserver()):
        pass
    assert calls == ["transport"]


class StartupRuntime:
    """A HarnessRuntime stand-in whose start blocks until released (or raises)."""

    created: list = []

    def __init__(self, **kwargs):
        self.run_dir = Path("/tmp/arc3-run")
        self.started, self.release = threading.Event(), threading.Event()
        self.fail = kwargs.pop("_fail", False)
        self.shutdowns, self.aborted = [], False
        StartupRuntime.created.append(self)

    def start(self, proxy, run_config):
        self.started.set()
        assert self.release.wait(5), "the test never released the start"
        if self.fail:
            raise OSError("harness import failed")

    def abort_start(self):
        self.aborted = True

    async def shutdown(self, transport):
        self.shutdowns.append(transport)


def patch_runtime_start(monkeypatch: pytest.MonkeyPatch, fail_first: bool = False) -> list:
    StartupRuntime.created = []
    created = StartupRuntime.created

    def factory(**kwargs):
        return StartupRuntime(_fail=fail_first and not created, **kwargs)

    async def run_config(self, env, model_config):
        return {}

    monkeypatch.setattr(ArcAgi3Agent, "model_server_config", lambda self: {"propagate_context_overflow_errors": True})
    monkeypatch.setattr(ArcAgi3Agent, "resolve_model_base_url", lambda self, name: "http://policy/v1")
    monkeypatch.setattr(ArcAgi3Agent, "run_config", run_config)
    monkeypatch.setattr(agent_app, "load_harness_env_toml", lambda root: {})
    monkeypatch.setattr(agent_app, "compose_harness_env", lambda base, **kwargs: {})
    monkeypatch.setattr(agent_app, "HarnessRuntime", factory)
    return created


async def wait_for(predicate, seconds: float = 5.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def test_cancel_during_runtime_start_keeps_one_owned_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # F6 residual: the first request is cancelled while the runtime starts. The start keeps running under the agent;
    # the next request waits for the same runtime (no second one), and shutdown waits for the start, then closes it.
    agent = make_agent(monkeypatch, tmp_path)
    created = patch_runtime_start(monkeypatch)

    async def main():
        first = asyncio.ensure_future(agent._ensure_runtime())
        await wait_for(lambda: created and created[0].started.is_set())
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.ensure_future(agent._ensure_runtime())
        await asyncio.sleep(0.05)
        assert not second.done() and len(created) == 1
        shutdown = asyncio.ensure_future(agent.shutdown())
        await asyncio.sleep(0.05)
        assert not shutdown.done()  # waits for the start (a thread cannot be interrupted)
        created[0].release.set()
        runtime = await second
        await shutdown
        assert runtime is created[0] and agent._runtime is runtime and len(created) == 1
        assert runtime.shutdowns == [agent._transport] and agent._transport is not None
        with pytest.raises(RuntimeError, match="shutting down"):
            await agent._ensure_runtime()

    asyncio.run(main())


def test_failed_runtime_start_is_torn_down_and_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A start that raises tears down what it built; the request sees the error; the next request starts afresh.
    agent = make_agent(monkeypatch, tmp_path)
    created = patch_runtime_start(monkeypatch, fail_first=True)

    async def main():
        first = asyncio.ensure_future(agent._ensure_runtime())
        await wait_for(lambda: created and created[0].started.is_set())
        created[0].release.set()
        with pytest.raises(OSError, match="harness import failed"):
            await first
        assert created[0].aborted and agent._runtime is None and agent._transport is None
        second = asyncio.ensure_future(agent._ensure_runtime())
        await wait_for(lambda: len(created) == 2 and created[1].started.is_set())
        created[1].release.set()
        assert await second is created[1] and not created[1].aborted

    asyncio.run(main())


def test_make_exact_version() -> None:
    # arc_agi's ONLINE make asks for the metadata of the name only; the patch asks for the pinned full id.
    from responses_api_agents.arc_agi_3_agent import harness_runtime
    from responses_api_agents.arc_agi_3_official_agent import worker

    for module in (harness_runtime, worker):
        calls = []

        class FakeArcade:
            def _create_remote_wrapper(self, game_id, version, scorecard_id):
                calls.append((game_id, version, scorecard_id))

        module.make_exact_version(FakeArcade)
        module.make_exact_version(FakeArcade)  # idempotent
        FakeArcade()._create_remote_wrapper("sg0005", "aee9edbb", "card")
        FakeArcade()._create_remote_wrapper("ls20", None, "card")
        assert calls == [("sg0005-aee9edbb", None, "card"), ("ls20", None, "card")]
    assert_same_body(worker.make_exact_version, harness_runtime.make_exact_version)


def test_set_reset_timeout() -> None:
    # Only RESET waits longer (the engine imports the game on its first RESET); other calls keep their own timeout.
    from responses_api_agents.arc_agi_3_agent import harness_runtime
    from responses_api_agents.arc_agi_3_official_agent import worker

    for module in (harness_runtime, worker):
        posts = []

        class Session:
            def post(self, url, json=None, headers=None, timeout=None):
                posts.append((url, timeout))

        class FakeWrapper:  # arc_agi.remote_wrapper.RemoteEnvironmentWrapper's shape
            def __init__(self):
                self._session = Session()

            def reset(self):
                self._session.post("/api/cmd/RESET", json={}, headers={}, timeout=10)
                return "frame"

            def step(self):
                self._session.post("/api/cmd/ACTION1", json={}, headers={}, timeout=10)

        module.set_reset_timeout(FakeWrapper, 60.0)
        module.set_reset_timeout(FakeWrapper, 120.0)  # idempotent: the last value wins
        wrapper = FakeWrapper()
        assert wrapper.reset() == "frame"
        wrapper.step()
        assert posts == [("/api/cmd/RESET", 120.0), ("/api/cmd/ACTION1", 10)]
        assert "post" not in vars(wrapper._session)
    assert_same_body(worker.set_reset_timeout, harness_runtime.set_reset_timeout)


def assert_same_body(copy, original) -> None:
    """The official worker (run in the harness's own venv, without Gym) carries copies of these functions."""
    import inspect

    assert inspect.getsource(copy).split('"""')[2] == inspect.getsource(original).split('"""')[2]
