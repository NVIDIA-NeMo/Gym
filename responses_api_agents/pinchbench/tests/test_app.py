# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the PinchBench Gym agent.

These cover the pure launcher/parser logic (env construction, sandbox spec, result +
transcript parsing) without launching a sandbox or invoking the model — so they run
fast and offline.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import AgentInvocation, ToolCallObservation
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pinchbench.app import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
    PinchBenchAgent,
    PinchBenchAgentConfig,
    PinchBenchRunRequest,
    SandboxKilledError,
    _classify_task_failure,
)


def make_config(**over) -> PinchBenchAgentConfig:
    base = dict(
        name="pinchbench",
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        model_base_url="http://endpoint/v1",
        model_api_key="sk-policy",
        model_name="vendor/model",
        judge_model="judge/model",
        judge_base_url="http://endpoint/v1",
        judge_api_key="sk-judge",
        brave_api_key="brave-key",
    )
    base.update(over)
    return PinchBenchAgentConfig(**base)


def make_agent(**over) -> PinchBenchAgent:
    return PinchBenchAgent(config=make_config(**over), server_client=MagicMock(spec=ServerClient))


def _records(bundle, record_type):
    return [record for record in bundle.records if isinstance(record, record_type)]


def test_sanity_construct():
    agent = make_agent()
    assert agent.config.task_timeout_s == 1800


@pytest.mark.asyncio
async def test_responses_not_implemented():
    agent = make_agent()
    with pytest.raises(NotImplementedError):
        await agent.responses(MagicMock(), MagicMock())


def test_task_env_gateway_mode():
    env = make_agent()._task_env("task_x")
    assert env["TASK_ID"] == "task_x"
    assert env["OPENCLAW_GATEWAY_TOKEN"] == "pinchbench-local"
    assert env["MODEL_NAME"] == "vendor/model"
    assert env["JUDGE_BASE_URL"] == "http://endpoint/v1"
    assert env["BRAVE_API_KEY"] == "brave-key"
    assert "NEMO_GYM_OBSERVABILITY_ENABLED" not in env
    assert make_agent()._task_env("task_x", "1-2")["NEMO_GYM_OBSERVABILITY_ENABLED"] == "1"


def test_task_env_prefixes_configured_gym_model_servers():
    agent = make_agent(
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
    )
    with patch.object(
        PinchBenchAgent,
        "resolve_model_base_url",
        side_effect=lambda _self, name, rollout_id: f"http://{name}/ng-rollout/{rollout_id}/v1",
        autospec=True,
    ):
        env = agent._task_env("task_x", "7-2")

    assert env["MODEL_BASE_URL"] == "http://policy/ng-rollout/7-2/v1"
    assert env["JUDGE_BASE_URL"] == "http://judge/ng-rollout/7-2/v1"


def test_direct_exec_wrapper_sets_provider_and_agent_timeout_ceiling(tmp_path):
    agent = make_agent(openclaw_provider_timeout_seconds=14400)
    wrapper = agent._write_direct_exec_wrapper(tmp_path)
    wrapper_text = wrapper.read_text()

    assert 'custom_provider["timeoutSeconds"] = provider_timeout_s' in wrapper_text
    assert 'defaults["timeoutSeconds"] = provider_timeout_s' in wrapper_text


def test_build_spec_from_config(tmp_path):
    image = tmp_path / "pinchbench.sif"
    image.touch()
    agent = make_agent(
        sandbox_spec={
            "image": str(image),
            "ready_timeout_s": 600,
            "resources": {"cpu": 4, "memory_mib": 8192},
        }
    )
    spec = agent._build_spec("task_x")
    assert spec.image == str(image)
    assert spec.ready_timeout_s == 600
    assert spec.resources.cpu == 4 and spec.resources.memory_mib == 8192
    assert spec.metadata == {"task_id": "task_x"}
    # the per-task env (incl the in-sandbox gateway token) is injected into the spec
    assert spec.env["TASK_ID"] == "task_x"
    assert spec.env["OPENCLAW_GATEWAY_TOKEN"]


def _write_result(out_dir, task_id, mean, gtype, breakdown, notes):
    payload = {
        "tasks": [
            {
                "task_id": task_id,
                "grading": {
                    "runs": [
                        {
                            "task_id": task_id,
                            "score": mean,
                            "max_score": 1.0,
                            "grading_type": gtype,
                            "breakdown": breakdown,
                            "notes": notes,
                        }
                    ],
                    "mean": mean,
                },
            }
        ]
    }
    (out_dir / "0001_model.json").write_text(json.dumps(payload))


def test_parse_result_hybrid(tmp_path):
    _write_result(tmp_path, "task_x", 0.82, "hybrid", {"automated.a": 1.0, "llm_judge.quality": 0.9}, "looks good")
    r = make_agent()._parse_result("task_x", tmp_path)
    assert r["reward"] == pytest.approx(0.82)
    assert r["grading_type"] == "hybrid"
    assert r["breakdown"]["llm_judge.quality"] == 0.9
    assert r["notes"] == "looks good"
    assert r["status"] == "success"


def test_parse_result_missing_task(tmp_path):
    _write_result(tmp_path, "other_task", 1.0, "automated", {}, "")
    r = make_agent()._parse_result("task_x", tmp_path)
    assert r["reward"] == 0.0
    assert r["status"] == "missing_task"


def test_parse_result_no_output(tmp_path):
    r = make_agent()._parse_result("task_x", tmp_path)
    assert r["reward"] == 0.0
    assert r["status"] == "error"


def test_response_from_transcript(tmp_path):
    tdir = tmp_path / "0001_transcripts"
    tdir.mkdir()
    events = [
        {"type": "message", "message": {"role": "user", "content": "do X"}},
        {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}},
    ]
    (tdir / "task_x.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    resp = make_agent()._response_from_transcript("task_x", tmp_path)
    assert resp.output[0].content[0].text == "Done."
    assert make_agent()._content_text(resp.output[0].content) == "Done."


def test_response_from_transcript_common_output_items_and_usage(tmp_path):
    tdir = tmp_path / "0001_transcripts"
    tdir.mkdir()
    events = [
        {
            "id": "assistant-1",
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "Need to search first."},
                    {
                        "type": "toolCall",
                        "id": "call_1",
                        "name": "web_search",
                        "arguments": {"query": "AAPL"},
                        "partialArgs": '{"query": "AAPL"}',
                    },
                ],
                "usage": {"input": 11, "output": 5, "cacheRead": 2},
            },
        },
        {
            "id": "tool-1",
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_1",
                "toolName": "web_search",
                "content": [{"type": "text", "text": '{"provider": "tavily"}'}],
            },
        },
        {
            "id": "assistant-2",
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        },
    ]
    (tdir / "task_x.jsonl").write_text("\n".join(json.dumps(e) for e in events))

    resp = make_agent()._response_from_transcript("task_x", tmp_path)

    assert [item.type for item in resp.output] == ["reasoning", "function_call", "function_call_output", "message"]
    assert resp.output[0].summary[0].text == "Need to search first."
    assert resp.output[1].name == "web_search"
    assert resp.output[1].arguments == '{"query": "AAPL"}'
    assert resp.output[2].call_id == "call_1"
    assert resp.output[2].output == '{"provider": "tavily"}'
    assert resp.output[3].content[0].text == "Done."
    assert resp.usage.input_tokens == 18
    assert resp.usage.output_tokens == 8
    assert resp.usage.input_tokens_details.cached_tokens == 2
    assert resp.usage.total_tokens == 26


def test_collect_transcript_archives(tmp_path):
    out = tmp_path / "out"
    (out / "0001_transcripts").mkdir(parents=True)
    (out / "0001_transcripts" / "task_x.jsonl").write_text(
        json.dumps({"type": "message", "message": {"role": "assistant", "content": "hi"}})
    )
    agent = make_agent(transcripts_dir=str(tmp_path / "archive"))
    events, archive = agent._collect_transcript("task_x", out, "runid123")
    assert len(events) == 1
    assert archive and (tmp_path / "archive" / "task_x_runid123").exists()


@pytest.mark.asyncio
async def test_run_returns_zero_on_failure_never_raises(tmp_path, monkeypatch):
    """A container/parse failure must yield reward 0 + status=error, NOT a 500 —
    otherwise ng_collect_rollouts (fail-fast) aborts the whole collection."""
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def boom(task_id, out_dir, rollout_id=None):
        raise RuntimeError("sandbox exploded")

    monkeypatch.setattr(agent, "_run_in_sandbox", boom)
    body = MagicMock()
    body.model_dump.return_value = {
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
        "verifier_metadata": {"task_id": "task_x"},
    }

    resp = await agent.run(body=body)  # must not raise
    assert resp.reward == 0.0
    assert resp.status == "error"
    assert resp.task_id == "task_x"
    assert "sandbox exploded" in resp.grading_notes


def _run_body(task_id="task_x"):
    body = MagicMock()
    body.model_dump.return_value = {
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
        "verifier_metadata": {"task_id": task_id},
    }
    return body


def _observed_run_body(task_id="task_x"):
    return PinchBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": "solve"},
            "verifier_metadata": {"task_id": task_id},
            "_ng_task_index": 1,
            "_ng_rollout_index": 2,
        }
    )


def test_hierarchy_discovery_reports_specific_root_gap(tmp_path):
    sessions_dir = tmp_path / "openclaw_sessions" / "agents" / "main" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "sessions.json").write_text(json.dumps({"other-root": {"sessionId": "other-session"}}))

    observations = make_agent()._build_observations(
        _observed_run_body(),
        make_agent()._empty_response("task_x"),
        [],
        tmp_path,
        "expected-session",
    )

    hierarchy_gaps = [gap for gap in observations.gaps if gap.code == "subagent_hierarchy_unavailable"]
    assert [gap.detail for gap in hierarchy_gaps] == ["root_session_not_found"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc,expected_class,no_persist,terminal",
    [
        (RuntimeError("sandbox exploded"), "legitimate", False, False),
        (SandboxKilledError("direct apptainer exec killed (rc=-15)"), "kill_shaped", True, False),
        (TimeoutError("direct apptainer exec timed out"), "timeout_exceeded", False, True),
    ],
)
async def test_failure_routing_sentinels(exc, expected_class, no_persist, terminal, tmp_path, monkeypatch):
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def fail(task_id, out_dir, rollout_id=None):
        raise exc

    monkeypatch.setattr(agent, "_run_in_sandbox", fail)
    dumped = (await agent.run(body=_run_body())).model_dump()
    assert dumped.get(NG_FAILURE_CLASS_KEY) == expected_class
    assert bool(dumped.get(NG_NO_PERSIST_KEY)) is no_persist
    assert bool(dumped.get(NG_TERMINAL_KEY)) is terminal


@pytest.mark.asyncio
async def test_successful_task_carries_no_routing_sentinels(tmp_path, monkeypatch):
    """Scored rollouts must keep landing in the main jsonl (no sentinel keys)."""
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def ok(task_id, out_dir, rollout_id=None):
        return None

    monkeypatch.setattr(agent, "_run_in_sandbox", ok)
    monkeypatch.setattr(
        agent,
        "_parse_result",
        lambda task_id, out_dir: {
            "reward": 1.0,
            "grading_type": "automated",
            "breakdown": {},
            "notes": "ok",
            "status": "success",
        },
    )
    monkeypatch.setattr(agent, "_collect_transcript", lambda task_id, out_dir, run_id: ([], ""))
    resp = await agent.run(body=_run_body())
    dumped = resp.model_dump()
    assert dumped["reward"] == 1.0
    assert "ng_agent_observations" not in dumped
    for key in (NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY):
        assert key not in dumped


@pytest.mark.asyncio
async def test_run_returns_correlated_openclaw_observations(tmp_path, monkeypatch):
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "archive"))
    agent.server_client.global_config_dict = {"observability_enabled": True}
    rollout_ids = []

    async def run_in_sandbox(task_id, out_dir, rollout_id=None):
        rollout_ids.append(rollout_id)
        transcript_dir = out_dir / "0001_transcripts"
        transcript_dir.mkdir(parents=True)
        events = [
            {"type": "session", "id": "session-1"},
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "need to search"},
                        {"type": "toolCall", "id": "call-1", "name": "search"},
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "content": [{"type": "text", "text": "result"}],
                    "timestamp": 1_750_000_002_000,
                    "details": {"durationMs": 500, "status": "completed"},
                },
            },
        ]
        (transcript_dir / f"{task_id}.jsonl").write_text("\n".join(json.dumps(event) for event in events))

    monkeypatch.setattr(agent, "_run_in_sandbox", run_in_sandbox)
    monkeypatch.setattr(
        agent,
        "_parse_result",
        lambda *_: {
            "reward": 1.0,
            "grading_type": "automated",
            "breakdown": {},
            "notes": "ok",
            "status": "success",
        },
    )

    result = await agent.run(body=_observed_run_body())

    assert rollout_ids == ["1-2"]
    assert result.ng_agent_observations.source == "openclaw"
    assert _records(result.ng_agent_observations, AgentInvocation)[0].invocation_id == "session-1"
    assert _records(result.ng_agent_observations, ToolCallObservation)[0].duration_ms == 500


@pytest.mark.asyncio
async def test_observation_failure_does_not_change_result(tmp_path, monkeypatch):
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "archive"))
    agent.server_client.global_config_dict = {"observability_enabled": True}

    async def run_in_sandbox(task_id, out_dir, rollout_id=None):
        return None

    monkeypatch.setattr(agent, "_run_in_sandbox", run_in_sandbox)
    monkeypatch.setattr(
        agent,
        "_parse_result",
        lambda *_: {
            "reward": 1.0,
            "grading_type": "automated",
            "breakdown": {},
            "notes": "ok",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        "responses_api_agents.pinchbench.app.build_openclaw_observations",
        MagicMock(side_effect=RuntimeError("observer failed")),
    )

    result = await agent.run(body=_observed_run_body())

    assert result.reward == 1.0
    assert [gap.code for gap in result.ng_agent_observations.gaps] == ["observation_capture_failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,exception_type",
    [("timeout", TimeoutError), ("sandbox", RuntimeError)],
)
async def test_managed_sandbox_exec_errors_skip_download(tmp_path, monkeypatch, error_type, exception_type):
    agent = make_agent(sandbox_provider={"opensandbox": {}})

    class FailedSandbox:
        download_called = False

        async def start(self, spec):
            return None

        async def exec(self, command, timeout_s):
            return SandboxExecResult(stdout=None, stderr="provider failed", return_code=125, error_type=error_type)

        async def download(self, source, target):
            self.download_called = True

        async def stop(self):
            return None

    sandbox = FailedSandbox()
    monkeypatch.setattr("responses_api_agents.pinchbench.app.AsyncSandbox", lambda _: sandbox)

    with pytest.raises(exception_type):
        await agent._run_in_sandbox("task_x", tmp_path)
    assert sandbox.download_called is False


@pytest.mark.parametrize(
    "exc,expected",
    [
        (SandboxKilledError("rc=-15"), "kill_shaped"),
        (TimeoutError("timed out"), "timeout_exceeded"),
        (RuntimeError("exec failed"), "legitimate"),
        (FileNotFoundError("apptainer"), "legitimate"),
    ],
)
def test_classify_task_failure(exc, expected):
    assert _classify_task_failure(exc) == expected


# --- _task_env optional injections ---


def test_task_env_injects_tavily_key_when_set():
    env = make_agent(web_search_provider="tavily", tavily_api_key="test-tavily-key", brave_api_key=None)._task_env(
        "t"
    )  # pragma: allowlist secret
    assert env["TAVILY_API_KEY"] == "test-tavily-key"  # pragma: allowlist secret
    assert "BRAVE_API_KEY" not in env


def test_task_env_omits_brave_key_when_not_set():
    env = make_agent(web_search_provider="tavily", tavily_api_key="test-tavily-key", brave_api_key=None)._task_env(
        "t"
    )  # pragma: allowlist secret
    assert "BRAVE_API_KEY" not in env


@pytest.mark.parametrize("seconds,expected", [(300, "300"), (14400, "14400")])
def test_task_env_injects_provider_timeout_when_set(seconds, expected):
    env = make_agent(openclaw_provider_timeout_seconds=seconds)._task_env("t")
    assert env["PINCHBENCH_PROVIDER_TIMEOUT_SECONDS"] == expected


def test_task_env_omits_provider_timeout_when_not_set():
    assert "PINCHBENCH_PROVIDER_TIMEOUT_SECONDS" not in make_agent()._task_env("t")


# --- run() edge cases ---


@pytest.mark.asyncio
async def test_run_raises_on_missing_task_id(tmp_path):
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))
    body = MagicMock()
    body.model_dump.return_value = {"verifier_metadata": {}}
    with pytest.raises(ValueError, match="task_id"):
        await agent.run(body=body)


@pytest.mark.asyncio
async def test_non_clean_exit_rc_present_in_raw_rollout(tmp_path, monkeypatch):
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def non_clean(task_id, out_dir, rollout_id=None):
        return 1

    monkeypatch.setattr(agent, "_run_in_sandbox", non_clean)
    monkeypatch.setattr(
        agent,
        "_parse_result",
        lambda *_: {"reward": 0.0, "grading_type": "unknown", "breakdown": {}, "notes": "", "status": "success"},
    )
    monkeypatch.setattr(agent, "_response_from_transcript", lambda *_: agent._empty_response("task_x"))
    monkeypatch.setattr(agent, "_collect_transcript", lambda *_: ([], ""))
    resp = await agent.run(body=_run_body())
    assert resp.raw_rollout["non_clean_exit_rc"] == 1


# --- signal-kill detection in _run_in_apptainer_direct ---


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode", [-15, 137, 143])
async def test_signal_killed_apptainer_raises_sandbox_killed_error(tmp_path, returncode):
    agent = make_agent(
        sandbox_spec={"image": "docker://test"},
        sandbox_provider={"apptainer": {"direct_exec": True}},
    )
    proc = MagicMock()
    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=None)
    proc.kill = MagicMock()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(SandboxKilledError):
            await agent._run_in_apptainer_direct("task_x", tmp_path, {"direct_exec": True})


# --- transcript parsing edge cases ---


def test_response_from_transcript_deduplicates_events_by_id(tmp_path):
    tdir = tmp_path / "0001_transcripts"
    tdir.mkdir()
    event = {
        "id": "e1",
        "type": "message",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    }
    (tdir / "task_x.jsonl").write_text(json.dumps(event) + "\n" + json.dumps(event))
    resp = make_agent()._response_from_transcript("task_x", tmp_path)
    assert sum(1 for item in resp.output if item.type == "message" and item.content[0].text == "Hi") == 1


def test_response_from_transcript_uses_details_when_content_empty(tmp_path):
    tdir = tmp_path / "0001_transcripts"
    tdir.mkdir()
    events = [
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_1",
                "content": [],
                "details": {"status": "ok", "count": 3},
            },
        }
    ]
    (tdir / "task_x.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    resp = make_agent()._response_from_transcript("task_x", tmp_path)
    result = next(item for item in resp.output if item.type == "function_call_output")
    assert json.loads(result.output) == {"status": "ok", "count": 3}


def test_read_transcript_events_tolerates_malformed_json(tmp_path):
    tdir = tmp_path / "0001_transcripts"
    tdir.mkdir()
    (tdir / "task_x.jsonl").write_text('{"valid": true}\nNOT JSON\nnull\n[]\n{"also": "valid"}')
    events = make_agent()._read_transcript_events("task_x", tmp_path)
    assert len(events) == 5
    assert all("raw" in events[index] for index in (1, 2, 3))


def test_transcript_parsing_ignores_non_object_message_and_usage():
    events = [
        {"type": "message", "message": "invalid"},
        {"type": "message", "message": {"role": "assistant", "content": "Done.", "usage": "invalid"}},
    ]

    response = make_agent()._response_from_transcript_events("task_x", events)

    assert response.output[0].content[0].text == "Done."
    assert response.usage.total_tokens == 0


def test_tool_call_arguments_with_dict_is_json_serialized():
    block = {"name": "search", "arguments": {"q": "AAPL"}, "partialArgs": '{"q":"stale"}'}
    assert json.loads(make_agent()._tool_call_arguments(block)) == {"q": "AAPL"}


def test_tool_call_arguments_absent_returns_empty_object():
    assert make_agent()._tool_call_arguments({"name": "search"}) == "{}"


def test_collect_transcript_returns_empty_when_no_transcript_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    events, archive = make_agent(transcripts_dir=str(tmp_path / "arch"))._collect_transcript("task_x", out, "run1")
    assert events == []
    assert archive == ""


def test_parse_result_empty_runs_defaults_grading_type(tmp_path):
    payload = {"tasks": [{"task_id": "task_x", "grading": {"runs": [], "mean": 0.5}}]}
    (tmp_path / "result.json").write_text(json.dumps(payload))
    r = make_agent()._parse_result("task_x", tmp_path)
    assert r["reward"] == pytest.approx(0.5)
    assert r["grading_type"] == "unknown"
    assert r["status"] == "success"
