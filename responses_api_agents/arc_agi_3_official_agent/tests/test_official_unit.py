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
"""Unit tests of the official-harness agent: config validation, model-server checks, actions, route, end states."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from nemo_gym import PARENT_DIR
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from responses_api_agents.arc_agi_3_official_agent import app as official_app
from responses_api_agents.arc_agi_3_official_agent.app import (
    API_KEY_ENV,
    ArcAgi3OfficialAgent,
    ArcAgi3OfficialAgentConfig,
    Attempt,
    WorkerOutcome,
    check_model_server,
    validate_benchmark_config,
)
from responses_api_agents.arc_agi_3_official_agent.setup_official_harness import load_manifest, verify_tree


ENV_YAML = PARENT_DIR / "resources_servers/arc_agi_3/configs/arc_agi_3_official.yaml"


def shipped_agent_config() -> dict:
    config = yaml.safe_load(ENV_YAML.read_text())
    return config["arc_agi_3_official_agent"]["responses_api_agents"]["arc_agi_3_official_agent"]


def shipped_benchmark_config(harness_type: str = "provider") -> dict:
    return shipped_agent_config()["benchmark_configs"][harness_type]


class FakeHttpResponse:
    def __init__(self, payload, status: int = 200):
        self._text = payload if isinstance(payload, str) else json.dumps(payload)
        self.status = status
        self.ok = status < 400
        self.cookies = {}

    async def read(self) -> bytes:
        return self._text.encode()


def make_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides) -> ArcAgi3OfficialAgent:
    monkeypatch.setattr(official_app, "ensure_flask", lambda: None)
    monkeypatch.setattr(official_app, "ensure_official_harness", lambda source, dest: tmp_path / "official")
    config = ArcAgi3OfficialAgentConfig(
        host="127.0.0.1",
        port=8123,
        entrypoint="",
        name="arc_agi_3_official_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="arc_agi_3_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        environments_dir=overrides.pop("environments_dir", str(tmp_path / "games")),
        benchmark_configs=overrides.pop("benchmark_configs", shipped_agent_config()["benchmark_configs"]),
        **overrides,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.get = AsyncMock(return_value=FakeHttpResponse({"url": "http://game-server.invalid", "games": 25}))
    return ArcAgi3OfficialAgent(config=config, server_client=server_client)


# ---- config ----------------------------------------------------------------------------------------------------------


def test_shipped_provider_config_is_the_continuous_conversation_with_reasoning_replay() -> None:
    assert shipped_agent_config()["harness_type"] == "provider"
    entry = validate_benchmark_config(shipped_benchmark_config("provider"), "policy_model", "provider")
    runtime = entry["runtime"]
    assert runtime["state"] == "continuous_conversation"
    assert runtime["adapter_id"] == "open_source.chat_completions.v1"
    assert runtime["sdk"] == "openai-python" and runtime["api"] == "chat_completions"
    assert runtime["reasoning_replay"] == "reasoning_content"
    assert runtime["compaction"] == {
        "strategy": "harness_summary",
        "trigger_tokens": 192000,
        "summary_input_headroom_tokens": 8192,
    }
    assert entry["request"]["stream"] is False and entry["request"]["model"] == "policy_model"
    assert entry["request"]["max_tokens"] == 12288
    assert entry["request"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert entry["client"] == {"api_key_env": API_KEY_ENV}
    assert entry["agent"] == {"MAX_ACTIONS_BASELINE_MULTIPLIER": 5, "MAX_CONTEXT_LENGTH": 262144}


def test_shipped_standard_config_is_manual_rolling_without_reasoning_replay() -> None:
    entry = validate_benchmark_config(shipped_benchmark_config("standard"), "policy_model", "standard")
    runtime = entry["runtime"]
    assert runtime == {"state": "manual_rolling", "sdk": "openai-python", "api": "chat_completions"}
    assert entry["agent"] == {"MAX_ACTIONS_BASELINE_MULTIPLIER": 5, "MAX_CONTEXT_LENGTH": 110592}
    assert entry["request"]["max_tokens"] == 12288 and entry["request"]["stream"] is False
    assert entry["client"] == {"api_key_env": API_KEY_ENV}


def test_the_agent_uses_the_selected_types_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for harness_type, state in (("provider", "continuous_conversation"), ("standard", "manual_rolling")):
        agent = make_agent(monkeypatch, tmp_path, harness_type=harness_type)
        assert (
            agent._benchmark_config["id"] == f"gym-{'provider-adapter' if harness_type == 'provider' else 'standard'}"
        )
        assert agent._benchmark_config["runtime"]["state"] == state
    with pytest.raises(ValueError, match="harness_type"):
        make_agent(monkeypatch, tmp_path, harness_type="other")


def test_minimal_config_gets_the_fixed_fields() -> None:
    entry = validate_benchmark_config({"request": {"max_tokens": 64}}, "m")
    assert entry["id"] == "gym-policy"
    assert entry["runtime"]["state"] == "continuous_conversation"
    assert entry["runtime"]["reasoning_replay"] == "reasoning_content"
    assert entry["pricing"] == {}


@pytest.mark.parametrize(
    "change, match",
    [
        ({"runtime": {"state": "manual_rolling"}}, "runtime.state"),
        ({"runtime": {"state": "server"}}, "runtime.state"),
        ({"runtime": {"adapter_id": "inception.chat_completions.v1"}}, "runtime.adapter_id"),
        ({"runtime": {"sdk": "anthropic-python"}}, "runtime.sdk"),
        ({"runtime": {"api": "responses"}}, "runtime.api"),
        ({"runtime": {"reasoning_replay": "none"}}, "reasoning_replay"),
        ({"runtime": {"compaction": {"strategy": "native"}}}, "compaction"),
        ({"agent": {"analysis_mode": True}}, "analysis_mode"),
        ({"request": {"stream": True}}, "stream"),
        ({"client": {"base_url": "https://api.example.com/v1"}}, "base_url"),
        ({"client": {"api_key_env": "OPENAI_API_KEY"}}, "api_key_env"),
    ],
)
def test_configs_that_could_drop_reasoning_or_bypass_gym_are_rejected(change: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_benchmark_config(change, "m")


@pytest.mark.parametrize(
    "change, match",
    [
        ({"runtime": {"state": "continuous_conversation"}}, "runtime.state"),
        ({"runtime": {"reasoning_replay": "reasoning_content"}}, "reasoning_replay"),
        ({"runtime": {"compaction": {"strategy": "harness_summary"}}}, "compaction"),
        ({"request": {"stream": True}}, "stream"),
        ({"client": {"base_url": "https://api.example.com/v1"}}, "base_url"),
    ],
)
def test_standard_configs_must_be_manual_rolling_through_gym(change: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_benchmark_config(change, "m", "standard")


@pytest.mark.parametrize("mode", ["reasoning_content", "reasoning", "reasoning_aliases"])
def test_every_reasoning_replay_mode_is_accepted(mode: str) -> None:
    assert (
        validate_benchmark_config({"runtime": {"reasoning_replay": mode}}, "m")["runtime"]["reasoning_replay"] == mode
    )


def test_model_server_must_keep_history_reasoning() -> None:
    good = {
        "uses_reasoning_parser": True,
        "uses_interleaved_reasoning": True,
        "propagate_context_overflow_errors": True,
    }
    check_model_server(good, "p", True)
    check_model_server({**good, "propagate_context_overflow_errors": False}, "p", False)
    with pytest.raises(RuntimeError, match="uses_reasoning_parser"):
        check_model_server({**good, "uses_reasoning_parser": False}, "p", True)
    with pytest.raises(RuntimeError, match="uses_interleaved_reasoning"):
        check_model_server({**good, "uses_interleaved_reasoning": False}, "p", True)
    with pytest.raises(RuntimeError, match="propagate_context_overflow_errors"):
        check_model_server({**good, "propagate_context_overflow_errors": False}, "p", True)
    # The standard harness resends no reasoning: only the context-length passthrough is required.
    check_model_server({"propagate_context_overflow_errors": True}, "p", True, "standard")
    with pytest.raises(RuntimeError, match="propagate_context_overflow_errors"):
        check_model_server({}, "p", True, "standard")


def test_agent_refuses_several_server_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_workers"):
        make_agent(monkeypatch, tmp_path, num_workers=2)


def test_agent_rejects_a_reasoning_dropping_config_at_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime.state"):
        make_agent(monkeypatch, tmp_path, benchmark_configs={"provider": {"runtime": {"state": "manual_rolling"}}})


def test_route_base_url_is_per_game(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    assert agent.route_base_url("3-ls20-9607627b") == "http://127.0.0.1:8123/official/3-ls20-9607627b/v1"


# ---- the model route -----------------------------------------------------------------------------------------------


def test_route_maps_reasoning_both_ways(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    sent = []

    async def post(server_name, url_path, json, cookies=None):
        sent.append((server_name, url_path, json))
        content = "<think>new reasoning</think>[turn 2] ACTION2"
        return FakeHttpResponse({"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]})

    agent.server_client.post = post
    agent._attempts["1-ls20"] = Attempt("1-ls20")
    payload = {
        "model": "policy_model",
        "messages": [
            {"role": "system", "content": "You are playing a game."},
            {"role": "user", "content": "State: NOT_FINISHED"},
            {"role": "assistant", "content": "[turn 1] ACTION1", "reasoning_content": "old reasoning"},
            {"role": "user", "content": "State: NOT_FINISHED"},
        ],
        "max_tokens": 64,
        "stream": False,
        "top_k": 20,
        "chat_template_kwargs": {"preserve_thinking": True},
    }

    class FakeRequest:
        async def json(self):
            return payload

    response = asyncio.run(agent.route_chat_completions("1-ls20", FakeRequest()))
    server_name, url_path, body = sent[0]
    assert server_name == "policy_model" and url_path == "/v1/chat/completions"
    assert body["messages"][2] == {"role": "assistant", "content": "<think>old reasoning</think>[turn 1] ACTION1"}
    assert json.loads(body["metadata"]["extra_body"]) == {"top_k": 20}
    assert json.loads(body["metadata"]["chat_template_kwargs"]) == {"preserve_thinking": True}
    assert "stream" not in body
    reply = json.loads(response.body)["choices"][0]["message"]
    assert reply["content"] == "[turn 2] ACTION2"
    assert reply["reasoning_content"] == "new reasoning" and reply["reasoning"] == "new reasoning"


def test_route_passes_a_context_length_error_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    detail = json.dumps({"detail": "This model's maximum context length is 1000 tokens."})

    async def post(server_name, url_path, json, cookies=None):
        return FakeHttpResponse(detail, status=400)

    agent.server_client.post = post
    agent._attempts["k"] = Attempt("k")

    class FakeRequest:
        async def json(self):
            return {"messages": [{"role": "user", "content": "x"}], "max_tokens": 5}

    response = asyncio.run(agent.route_chat_completions("k", FakeRequest()))
    assert response.status_code == 400 and b"maximum context length" in response.body


def test_route_refuses_a_game_not_in_play(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    agent._attempts["done"] = Attempt("done", playing=False)
    for key in ("done", "unknown"):
        response = asyncio.run(agent.route_chat_completions(key, None))
        assert response.status_code == 410


def test_route_refuses_streaming(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = make_agent(monkeypatch, tmp_path)
    agent._attempts["k"] = Attempt("k")

    class FakeRequest:
        async def json(self):
            return {"messages": [], "stream": True}

    assert asyncio.run(agent.route_chat_completions("k", FakeRequest())).status_code == 400


def test_route_logs_every_call_with_its_tags_on_the_rollout_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from responses_api_agents.arc_agi_3_agent.call_log import call_log_entries
    from responses_api_agents.arc_agi_3_official_agent.worker import CALL_HEADERS

    agent = make_agent(monkeypatch, tmp_path)
    answers = [
        FakeHttpResponse({"id": "r0", "choices": [{"index": 0, "message": {"role": "assistant", "content": "A"}}]}),
        FakeHttpResponse({"detail": "busy"}, status=503),
        OSError("refused"),
    ]
    paths = []

    async def post(server_name, url_path, json, cookies=None):
        paths.append(url_path)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    agent.server_client.post = post
    attempt = Attempt("k", url_path="/ng-rollout/5-0/v1/chat/completions")
    agent._attempts["k"] = attempt

    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers

        async def json(self):
            return {"messages": [{"role": "user", "content": "State: NOT_FINISHED\nLevels completed: 1"}]}

    tags = {CALL_HEADERS["step"]: "7", CALL_HEADERS["level"]: "2"}
    asyncio.run(agent.route_chat_completions("k", FakeRequest(tags)))
    summary = {**tags, CALL_HEADERS["role"]: "helper", CALL_HEADERS["history_id"]: "harness-summary"}
    assert asyncio.run(agent.route_chat_completions("k", FakeRequest(summary))).status_code == 503
    with pytest.raises(OSError):
        asyncio.run(agent.route_chat_completions("k", FakeRequest({})))
    assert paths == ["/ng-rollout/5-0/v1/chat/completions"] * 3
    entries = call_log_entries(attempt.calls)
    assert [(e["role"], e["history_id"], e["status"], e["response_id"], e["step"], e["level"]) for e in entries] == [
        ("gameplay", "main", 200, "r0", 7, 2),
        ("helper", "harness-summary", 503, None, 7, 2),
        ("gameplay", "main", 0, None, None, None),
    ]
    assert entries[2]["error"] == "OSError: refused"


def test_run_returns_the_call_log_revision_and_sampling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from responses_api_agents.arc_agi_3_agent.call_log import GameCallLog
    from responses_api_agents.arc_agi_3_official_agent.app import ArcAgi3OfficialRunRequest

    agent = make_agent(monkeypatch, tmp_path, reasoning_effort="max")
    monkeypatch.setattr(ArcAgi3OfficialAgent, "url_path_for_run", lambda self, path, body: f"/ng-rollout/9-1{path}")
    seen = {}

    async def play_game(self, meta, index, url_path="/v1/chat/completions"):
        seen["url_path"] = url_path
        log = GameCallLog()
        log.add(role="gameplay", history_id="main", messages=[], status=200, response={"id": "r0"}, step=1, level=1)
        log.add(role="helper", history_id="harness-summary", messages=[], status=200, response={"id": "s0"})
        record = {"state": "gave_up", "levels_completed": 0, "actions": 1, "stop_reason": "level_action_budget"}
        return {
            "record": record,
            "scorecard_id": "c",
            "guid": "g",
            "solver_note": None,
            "key": "0-ls20-v1",
            "call_log": log,
            "clock_origin": "2026-11-19T10:00:00+00:00",
        }

    monkeypatch.setattr(ArcAgi3OfficialAgent, "play_game", play_game)

    async def post(server_name, url_path, json, cookies=None):
        return FakeHttpResponse({**json, "reward": 0.0})

    agent.server_client.post = post
    body = ArcAgi3OfficialRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "Play ls20"}]},
        verifier_metadata={"game_id": "ls20", "full_game_id": "ls20-v1", "dispatch_index": 0},
    )
    result = asyncio.run(agent.run(MagicMock(cookies={}), body))
    assert seen["url_path"] == "/ng-rollout/9-1/v1/chat/completions"
    assert result.stop_reason == "level_action_budget"  # sent to /verify, which keeps it in the rollout
    assert result.model_error is None  # a call was answered
    assert result.benchmark_start_time == "2026-11-19T10:00:00+00:00"  # the game's clock origin, RHAE@T's
    (invocation,) = result.ng_agent_observations["records"]
    assert invocation["invocation_id"] == "0-ls20-v1" and [c["response_id"] for c in invocation["model_calls"]] == [
        "r0"
    ]
    assert result.ng_agent_observations["source"] == "arc_agi_3_official_agent"
    assert [(c["role"], c["history_id"]) for c in result.agent_call_log] == [
        ("gameplay", "main"),
        ("helper", "harness-summary"),
    ]
    assert result.harness_revision == {
        "name": "arc-agi-3-benchmarking",
        "repository": "https://github.com/arcprize/arc-agi-3-benchmarking",
        "commit": "27f3d03de3f2f6fa1b8c1e01cd7a9d7d77e830f0",  # pragma: allowlist secret
    }
    shipped = shipped_benchmark_config("provider")["request"]
    assert result.sampling["reasoning_effort"] == "max" and "model" not in result.sampling
    assert {k: result.sampling[k] for k in shipped if k not in ("model", "stream")} == {
        k: v for k, v in shipped.items() if k not in ("model", "stream")
    }


def test_worker_tags_model_calls_and_summaries() -> None:
    from responses_api_agents.arc_agi_3_official_agent.worker import CALL_HEADERS, tag_model_calls

    seen = []

    class Client:
        _custom_headers = {}

    class Compactor:
        def compact(self, **kwargs):
            seen.append(dict(Client._custom_headers))
            return "summary"

    agent = MagicMock(_client=Client(), _summary_compactor=Compactor())
    report = {"actions": 4, "levels_completed": 1}
    tag = tag_model_calls(agent, report)
    assert Client._custom_headers == {
        CALL_HEADERS["step"]: "5",
        CALL_HEADERS["level"]: "2",
        CALL_HEADERS["role"]: "gameplay",
        CALL_HEADERS["history_id"]: "main",
    }
    assert agent._summary_compactor.compact(state=None) == "summary"
    assert seen[0][CALL_HEADERS["role"]] == "helper" and seen[0][CALL_HEADERS["history_id"]] == "harness-summary"
    assert Client._custom_headers[CALL_HEADERS["role"]] == "gameplay"
    report.update(actions=9, levels_completed=2)
    tag()
    assert (Client._custom_headers[CALL_HEADERS["step"]], Client._custom_headers[CALL_HEADERS["level"]]) == ("10", "3")
    with pytest.raises(RuntimeError, match="cannot be tagged"):
        tag_model_calls(MagicMock(_client=object()), report)


# ---- end states ----------------------------------------------------------------------------------------------


def test_end_states_from_the_worker_report() -> None:
    end = ArcAgi3OfficialAgent._end_state
    live, cancelled = Attempt("k"), Attempt("k", cancelled=True)
    assert end(live, WorkerOutcome(report={"state": "WIN"})) == ("won", None, "won")
    assert end(cancelled, WorkerOutcome(report={"state": "WIN"}))[0] == "won"  # a won game stays won
    assert end(cancelled, WorkerOutcome(report={"state": "NOT_FINISHED"})) == (
        "cancelled",
        "the agent server shut down",
        "cancelled",
    )
    assert end(live, WorkerOutcome(report={"state": "NOT_FINISHED", "error": "API_ERROR: x"})) == (
        "crashed",
        "API_ERROR: x",
        "crashed",
    )
    stopped = end(live, WorkerOutcome(report={"state": "NOT_FINISHED"}, window_stopped=True))
    assert stopped[0] == "gave_up" and stopped[2] == "play_window"
    # Upstream's action budgets and the play window share the state gave_up; the stop reason tells them apart, and
    # which budget fired: the level's own allowance, or the whole game's cap (also when the report does not say).
    level_budget = {"state": "GAME_OVER", "exit_reason": "ACTION_BUDGET", "action_budget": "level"}
    assert end(live, WorkerOutcome(report=level_budget)) == (
        "gave_up",
        "harness exit: ACTION_BUDGET",
        "level_action_budget",
    )
    total_budget = dict(level_budget, action_budget="total")
    assert end(live, WorkerOutcome(report=total_budget))[2] == "total_action_budget"
    unknown = {"state": "NOT_FINISHED", "exit_reason": "ACTION_BUDGET"}
    assert end(live, WorkerOutcome(report=unknown))[2] == "total_action_budget"
    assert end(live, WorkerOutcome(report={"state": "NOT_FINISHED", "exit_reason": "TIME_BUDGET"})) == (
        "gave_up",
        "harness exit: TIME_BUDGET",
        "play_window",
    )
    assert end(live, WorkerOutcome(report={"state": "NOT_FINISHED", "exit_reason": "UNKNOWN"}))[2] == "harness_end"


def test_worker_tells_which_action_budget_fired() -> None:
    """Upstream's is_done ends a game with ACTION_BUDGET on its total hard cap (checked first) or on the current level's
    allowance; its main loop also stops at the game's total MAX_ACTIONS."""
    from types import SimpleNamespace

    from responses_api_agents.arc_agi_3_official_agent.worker import action_budget_fired

    def agent(actions: int, level: int, level_actions: int, budgets=(4, 6), cap=None):
        frame = SimpleNamespace(levels_completed=level)
        return SimpleNamespace(
            action_counter=actions,
            frames=[SimpleNamespace(levels_completed=0), frame],
            _level_action_budgets=list(budgets),
            _level_action_counter=level_actions,
            _max_actions_hard_cap=cap,
        )

    # the total cap reached with one action on the final level: not that level's failure
    assert action_budget_fired(agent(actions=5, level=1, level_actions=1, cap=5)) == "total"
    # the cap and the level's allowance on the same action: upstream checks the cap first
    assert action_budget_fired(agent(actions=10, level=1, level_actions=6, cap=10)) == "total"
    assert action_budget_fired(agent(actions=7, level=1, level_actions=6)) == "level"
    assert action_budget_fired(agent(actions=4, level=0, level_actions=4)) == "level"
    # no baselines (no per-level budgets): upstream's fallback MAX_ACTIONS is the game's total
    assert action_budget_fired(agent(actions=11, level=0, level_actions=11, budgets=())) == "total"


# ---- install -------------------------------------------------------------------------------------------------------


def test_manifest_pins_the_commit_and_license() -> None:
    manifest = load_manifest()
    provenance = manifest["provenance"]
    assert provenance["commit"] == "27f3d03de3f2f6fa1b8c1e01cd7a9d7d77e830f0"  # pragma: allowlist secret
    assert provenance["license"] == "MIT" and provenance["repository"].endswith("arcprize/arc-agi-3-benchmarking")
    assert "benchmarking/agent.py" in manifest["files"] and "LICENSE" in manifest["files"]


def test_verify_tree_reports_changed_missing_and_extra_files(tmp_path: Path) -> None:
    manifest = {"provenance": {"commit": "c"}, "files": {"a.py": "0" * 64, "LICENSE": None}}
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "extra.py").write_text("y")
    with pytest.raises(ValueError) as info:
        verify_tree(tmp_path, manifest)
    message = str(info.value)
    assert "sha256 differs: a.py" in message and "missing LICENSE" in message and "unpinned file extra.py" in message


def test_run_reports_a_model_that_never_answered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from responses_api_agents.arc_agi_3_agent.call_log import GameCallLog
    from responses_api_agents.arc_agi_3_official_agent.app import ArcAgi3OfficialRunRequest

    agent = make_agent(monkeypatch, tmp_path)

    async def play_game(self, meta, index, url_path="/v1/chat/completions"):
        log = GameCallLog()
        log.add(role="gameplay", history_id="main", messages=[], status=404, response=None)
        log.add(role="gameplay", history_id="main", messages=[], status=0, response=None, error="TimeoutError: ")
        record = {"state": "gave_up", "levels_completed": 0, "actions": 0, "stop_reason": "harness_end"}
        return {
            "record": record,
            "scorecard_id": "c",
            "guid": "g",
            "solver_note": None,
            "key": "0-ls20-v1",
            "call_log": log,
            "clock_origin": None,
        }

    monkeypatch.setattr(ArcAgi3OfficialAgent, "play_game", play_game)
    sent = {}

    async def post(server_name, url_path, json, cookies=None):
        sent.update(json)
        return FakeHttpResponse({**json, "reward": 0.0})

    agent.server_client.post = post
    body = ArcAgi3OfficialRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "Play ls20"}]},
        verifier_metadata={"game_id": "ls20", "full_game_id": "ls20-v1", "dispatch_index": 0},
    )
    asyncio.run(agent.run(MagicMock(cookies={}), body))
    assert sent["model_error"] == "every model call of the game failed (2 calls; last: TimeoutError: )"
