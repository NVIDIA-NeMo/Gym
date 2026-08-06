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
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ContextCompactionObservation,
    ToolCallObservation,
)
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
    assert agent.config.openclaw_mode == "gateway"  # per-task gateway: proven working mode
    assert agent.config.task_timeout_s == 1800


@pytest.mark.asyncio
async def test_responses_not_implemented():
    agent = make_agent()
    with pytest.raises(NotImplementedError):
        await agent.responses(MagicMock(), MagicMock())


def test_task_env_gateway_mode():
    env = make_agent(openclaw_mode="gateway")._task_env("task_x")
    assert env["TASK_ID"] == "task_x"
    assert env["OPENCLAW_GATEWAY_TOKEN"]  # gateway daemon mode
    assert "PINCHBENCH_FORCE_LOCAL" not in env
    assert env["MODEL_NAME"] == "vendor/model"
    assert env["JUDGE_BASE_URL"] == "http://endpoint/v1"
    assert env["BRAVE_API_KEY"] == "brave-key"
    assert "NEMO_GYM_OBSERVABILITY_ENABLED" not in env
    assert make_agent(openclaw_mode="gateway")._task_env("task_x", "1-2")["NEMO_GYM_OBSERVABILITY_ENABLED"] == "1"


def test_task_env_prefixes_configured_gym_model_servers() -> None:
    model_ref = ModelServerRef(type="responses_api_models", name="policy")
    judge_ref = ModelServerRef(type="responses_api_models", name="judge")
    agent = make_agent(model_server=model_ref, judge_model_server=judge_ref)
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
    assert 'agent["timeoutSeconds"] = provider_timeout_s' in wrapper_text


def test_build_spec_from_config():
    agent = make_agent(
        sandbox_spec={
            "image": "/sif/pinchbench.sif",
            "ready_timeout_s": 600,
            "resources": {"cpu": 4, "memory_mib": 8192},
        }
    )
    spec = agent._build_spec("task_x")
    assert spec.image == "/sif/pinchbench.sif"
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
    assert r["reward"] == 0.0 and r["status"] == "missing_task"


def test_parse_result_no_output(tmp_path):
    r = make_agent()._parse_result("task_x", tmp_path)  # empty dir
    assert r["reward"] == 0.0 and r["status"] == "error"


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


def test_transcript_reader_preserves_non_object_records_as_raw_evidence(tmp_path):
    tdir = tmp_path / "0001_transcripts"
    tdir.mkdir()
    (tdir / "task_x.jsonl").write_text('null\n[]\n{"type":"message","message":{"role":"assistant"}}\n')

    events = make_agent()._read_transcript_events("task_x", tmp_path)

    assert events[:2] == [{"raw": "null"}, {"raw": "[]"}]


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


def test_transcript_parsing_ignores_non_object_message_and_usage():
    events = [
        {"type": "message", "message": "invalid"},
        {
            "type": "message",
            "message": {"role": "assistant", "content": "Done.", "usage": "invalid"},
        },
    ]

    resp = make_agent()._response_from_transcript_events("task_x", events)

    assert resp.output[0].content[0].text == "Done."
    assert resp.usage.total_tokens == 0


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


# Failure routing: rows without `_ng_failure_class` land in the main jsonl, where
# resume counts them as done forever — the pre-fix behavior these tests pin.


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


@pytest.mark.asyncio
async def test_generic_failure_routes_to_sidecar_not_main(tmp_path, monkeypatch):
    """A failed task must carry a failure class so it never lands in the main jsonl."""
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def boom(task_id, out_dir, rollout_id=None):
        raise RuntimeError("sandbox exploded")

    monkeypatch.setattr(agent, "_run_in_sandbox", boom)
    resp = await agent.run(body=_run_body())
    dumped = resp.model_dump()
    assert dumped.get(NG_FAILURE_CLASS_KEY) == "legitimate"  # sidecar, bounded retry
    assert not dumped.get(NG_NO_PERSIST_KEY)
    assert not dumped.get(NG_TERMINAL_KEY)


@pytest.mark.asyncio
async def test_signal_killed_sandbox_is_kill_shaped_and_unpersisted(tmp_path, monkeypatch):
    """Walltime SIGTERM shape: no row anywhere; resume's set-difference re-dispatches."""
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def killed(task_id, out_dir, rollout_id=None):
        raise SandboxKilledError("direct apptainer exec killed (rc=-15) for task task_x")

    monkeypatch.setattr(agent, "_run_in_sandbox", killed)
    resp = await agent.run(body=_run_body())
    dumped = resp.model_dump()
    assert dumped.get(NG_FAILURE_CLASS_KEY) == "kill_shaped"
    assert dumped.get(NG_NO_PERSIST_KEY) is True


@pytest.mark.asyncio
async def test_task_timeout_is_terminal_sidecar(tmp_path, monkeypatch):
    """Per-task timeout consumed its budget: one sidecar row, never retried."""
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def slow(task_id, out_dir, rollout_id=None):
        raise TimeoutError("direct apptainer exec timed out for task task_x")

    monkeypatch.setattr(agent, "_run_in_sandbox", slow)
    resp = await agent.run(body=_run_body())
    dumped = resp.model_dump()
    assert dumped.get(NG_FAILURE_CLASS_KEY) == "timeout_exceeded"
    assert dumped.get(NG_TERMINAL_KEY) is True
    assert not dumped.get(NG_NO_PERSIST_KEY)


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
async def test_run_returns_openclaw_observations_when_enabled(tmp_path, monkeypatch):
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "archive"))
    agent.server_client.global_config_dict = {"observability_enabled": True}

    async def run_in_sandbox(task_id, out_dir, rollout_id=None):
        transcript_dir = out_dir / "0001_transcripts"
        transcript_dir.mkdir(parents=True)
        root_transcript = transcript_dir / f"{task_id}.jsonl"
        root_transcript.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session", "id": "session-1"}),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {"role": "user", "content": "retained pinch input"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {"type": "reasoning", "text": "need to search"},
                                    {"type": "toolCall", "id": "call-1", "name": "search"},
                                ],
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "toolResult",
                                "toolCallId": "call-1",
                                "content": [{"type": "text", "text": "result"}],
                                "timestamp": 1_750_000_002_000,
                                "details": {"durationMs": 500, "status": "completed"},
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "compaction",
                            "summary": "condensed context",
                            "firstKeptEntryId": "entry-7",
                            "tokensBefore": 120_000,
                        }
                    ),
                ]
            )
        )
        sessions_dir = out_dir / "openclaw_sessions" / "agents" / "main" / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "session-1.jsonl").write_text(root_transcript.read_text())
        child_transcript = sessions_dir / "child-session.jsonl"
        child_transcript.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session", "id": "child-session"}),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "user",
                                "content": [{"type": "text", "text": "research this"}],
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "child done"}],
                            },
                        }
                    ),
                ]
            )
        )
        (sessions_dir / "sessions.json").write_text(
            json.dumps(
                {
                    "agent:main:main": {"sessionId": "session-1"},
                    "agent:main:subagent:child": {
                        "sessionId": "child-session",
                        "spawnedBy": "agent:main:main",
                    },
                }
            )
        )

    monkeypatch.setattr(agent, "_run_in_sandbox", run_in_sandbox)
    monkeypatch.setattr(
        agent,
        "_parse_result",
        lambda *args: {
            "reward": 1.0,
            "grading_type": "automated",
            "breakdown": {},
            "notes": "ok",
            "status": "success",
        },
    )
    body = PinchBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": "solve"},
            "verifier_metadata": {"task_id": "task_x"},
            "_ng_task_index": 1,
            "_ng_rollout_index": 2,
        }
    )

    result = await agent.run(body=body)

    observations = result.ng_agent_observations
    assert observations is not None
    assert observations.source == "openclaw"
    assert result.model_dump()["ng_agent_observations"]["source"] == "openclaw"
    invocations = _records(observations, AgentInvocation)
    tool_calls = _records(observations, ToolCallObservation)
    compactions = _records(observations, ContextCompactionObservation)
    assert invocations[0].invocation_id == "agent:main:main"
    assert [item.type for item in invocations[0].conversation] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert invocations[0].conversation[0].content == "retained pinch input"
    assert invocations[0].conversation[1].summary[0].text == "need to search"
    assert invocations[1].parent_invocation_id == "agent:main:main"
    assert "research this" in str(invocations[1].conversation)
    assert tool_calls[0].duration_ms == 500
    assert compactions[0].summary == "condensed context"


@pytest.mark.asyncio
async def test_managed_sandbox_timeout_is_terminal(tmp_path, monkeypatch):
    agent = make_agent(
        work_root=str(tmp_path / "work"),
        transcripts_dir=str(tmp_path / "archive"),
        sandbox_provider={"opensandbox": {}},
    )
    agent.server_client.global_config_dict = {"observability_enabled": True}

    class TimedOutSandbox:
        sandbox_id = "sandbox-1"
        download_called = False

        async def start(self, spec):
            return None

        async def exec(self, command, timeout_s):
            return SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout")

        async def download(self, source, target):
            self.download_called = True

        async def stop(self):
            return None

    sandbox = TimedOutSandbox()
    monkeypatch.setattr("responses_api_agents.pinchbench.app.AsyncSandbox", lambda config: sandbox)

    result = await agent.run(body=_observed_run_body())

    dumped = result.model_dump()
    assert dumped[NG_FAILURE_CLASS_KEY] == "timeout_exceeded"
    assert dumped[NG_TERMINAL_KEY] is True
    assert result.ng_agent_observations.source == "openclaw"
    assert sandbox.download_called is False


@pytest.mark.asyncio
async def test_managed_sandbox_error_skips_download(tmp_path, monkeypatch):
    agent = make_agent(
        work_root=str(tmp_path / "work"),
        transcripts_dir=str(tmp_path / "archive"),
        sandbox_provider={"opensandbox": {}},
    )

    class FailedSandbox:
        download_called = False

        async def start(self, spec):
            return None

        async def exec(self, command, timeout_s):
            return SandboxExecResult(stdout=None, stderr="provider failed", return_code=125, error_type="sandbox")

        async def download(self, source, target):
            self.download_called = True

        async def stop(self):
            return None

    sandbox = FailedSandbox()
    monkeypatch.setattr("responses_api_agents.pinchbench.app.AsyncSandbox", lambda config: sandbox)

    with pytest.raises(RuntimeError, match=r"sandbox.*provider failed"):
        await agent._run_in_sandbox("task_x", tmp_path)
    assert sandbox.download_called is False


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
        lambda *args: {
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
    body = PinchBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": "solve"},
            "verifier_metadata": {"task_id": "task_x"},
            "_ng_task_index": 1,
            "_ng_rollout_index": 2,
        }
    )

    result = await agent.run(body=body)

    assert result.reward == 1.0
    assert result.status == "success"
    assert result.response.output[0].content[0].text == ""
    observations = result.ng_agent_observations
    assert observations is not None
    assert observations.source == "openclaw"
    assert [gap.code for gap in observations.gaps] == ["observation_capture_failed"]


def test_classify_task_failure_mapping():
    assert _classify_task_failure(SandboxKilledError("rc=-15")) == "kill_shaped"
    assert _classify_task_failure(TimeoutError("timed out")) == "timeout_exceeded"
    assert _classify_task_failure(RuntimeError("exec failed")) == "legitimate"
    assert _classify_task_failure(FileNotFoundError("apptainer")) == "legitimate"
