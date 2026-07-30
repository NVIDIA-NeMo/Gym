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
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
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
    assert env["MODEL_BASE_URL"] == "http://endpoint/v1"
    assert env["JUDGE_BASE_URL"] == "http://endpoint/v1"
    assert env["BRAVE_API_KEY"] == "brave-key"


def test_task_env_gateway_hardening_knobs():
    env = make_agent(
        gateway_ready_timeout_seconds=123,
        gateway_start_jitter_s=17,
        gateway_port_low=23000,
        gateway_port_high=24000,
    )._task_env("task_x")

    assert env["PINCHBENCH_GATEWAY_READY_TIMEOUT_SECONDS"] == "123"
    assert env["PINCHBENCH_GATEWAY_START_JITTER_S"] == "17"
    assert env["PINCHBENCH_GATEWAY_PORT_LOW"] == "23000"
    assert env["PINCHBENCH_GATEWAY_PORT_HIGH"] == "24000"


def _rollout_body() -> PinchBenchRunRequest:
    return PinchBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"task_id": "task_x"},
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
        }
    )


def test_task_env_observability_prefixes_configured_model_server_before_v1():
    agent = make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client.global_config_dict = {
        "observability_enabled": True,
        "policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 9001}}},
    }
    agent.server_client._build_server_base_url.side_effect = lambda cfg: f"http://{cfg['host']}:{cfg['port']}"

    env = agent._task_env("task_x", _rollout_body())

    assert env["MODEL_BASE_URL"] == "http://127.0.0.1:9001/ng-rollout/12-3/v1"
    assert env["JUDGE_BASE_URL"] == "http://endpoint/v1"


def test_task_env_observability_does_not_prefix_direct_endpoint():
    agent = make_agent()
    agent.server_client.global_config_dict = {"observability_enabled": True}
    body = PinchBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"task_id": "task_x"},
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
        }
    )

    env = agent._task_env("task_x", body)

    assert env["MODEL_BASE_URL"] == "http://endpoint/v1"
    assert env["JUDGE_BASE_URL"] == "http://endpoint/v1"


def test_direct_exec_wrapper_stages_checkout_run_task(tmp_path):
    agent = make_agent()
    wrapper = agent._write_direct_exec_wrapper(tmp_path)
    staged_run_task = tmp_path / "run_task_current.sh"

    assert wrapper.exists()
    assert staged_run_task.exists()
    assert staged_run_task.stat().st_mode & 0o100

    wrapper_text = wrapper.read_text()
    staged_text = staged_run_task.read_text()

    assert "run_task_current.sh" in wrapper_text
    assert "/opt/run_task.sh" not in wrapper_text
    assert "run_task_patched.sh" not in wrapper_text
    assert "PINCHBENCH_GATEWAY_READY_TIMEOUT_SECONDS" in staged_text
    assert "select_gateway_port" in staged_text


def test_direct_exec_wrapper_sets_provider_and_agent_timeout_ceiling(tmp_path):
    agent = make_agent(openclaw_provider_timeout_seconds=14400)
    wrapper = agent._write_direct_exec_wrapper(tmp_path)
    wrapper_text = wrapper.read_text()

    assert 'custom_provider["timeoutSeconds"] = provider_timeout_s' in wrapper_text
    assert 'defaults["timeoutSeconds"] = provider_timeout_s' in wrapper_text
    assert 'agent["timeoutSeconds"] = provider_timeout_s' in wrapper_text
    assert 'diagnostics["stuckSessionAbortMs"] = provider_timeout_s * 1000' in wrapper_text


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

    async def boom(task_id, out_dir, body=None):
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


@pytest.mark.asyncio
async def test_debug_artifacts_preserved_on_failure(tmp_path, monkeypatch):
    agent = make_agent(
        work_root=str(tmp_path / "work"),
        transcripts_dir=str(tmp_path / "arch"),
        preserve_debug_artifacts=True,
    )

    async def boom(task_id, out_dir, body=None):
        sandbox_out = out_dir / "sandbox" / "out"
        sandbox_out.mkdir(parents=True)
        (out_dir / "sandbox" / "apptainer.stdout.log").write_text("apptainer stdout")
        (out_dir / "sandbox" / "apptainer.stderr.log").write_text("apptainer stderr")
        (sandbox_out / "run.log").write_text("inner run log")
        (sandbox_out / "gateway.log").write_text("inner gateway log")
        raise RuntimeError("sandbox exploded")

    monkeypatch.setattr(agent, "_run_in_sandbox", boom)

    resp = await agent.run(body=_run_body())
    debug_path = resp.raw_rollout["debug_artifacts_to"]
    debug_dir = tmp_path / "arch" / "_debug_artifacts"
    assert str(debug_dir) in debug_path
    assert (Path(debug_path) / "sandbox" / "out" / "run.log").read_text() == "inner run log"
    meta = json.loads((Path(debug_path) / "debug_meta.json").read_text())
    assert meta["reason"] == "exception"
    assert meta["failure"]["type"] == "RuntimeError"
    assert "sandbox/out/gateway.log" in meta["copied"]
    assert not any((tmp_path / "work").iterdir())


@pytest.mark.asyncio
async def test_debug_artifacts_preserved_for_success_without_transcript(tmp_path, monkeypatch):
    agent = make_agent(
        work_root=str(tmp_path / "work"),
        transcripts_dir=str(tmp_path / "arch"),
        preserve_debug_artifacts=True,
    )

    async def ok_without_transcript(task_id, out_dir, body=None):
        _write_result(out_dir, task_id, 0.5, "automated", {"ok": True}, "graded without transcript")

    monkeypatch.setattr(agent, "_run_in_sandbox", ok_without_transcript)

    resp = await agent.run(body=_run_body())
    debug_path = resp.raw_rollout["debug_artifacts_to"]
    meta = json.loads((Path(debug_path) / "debug_meta.json").read_text())
    assert resp.reward == pytest.approx(0.5)
    assert meta["reason"] == "no_transcript_events"
    assert meta["result"]["status"] == "success"
    assert meta["result_files"][0]["matching_tasks"][0]["grading_notes"] == "graded without transcript"


# Failure routing: rows without `_ng_failure_class` land in the main jsonl, where
# resume counts them as done forever — the pre-fix behavior these tests pin.


def _run_body(task_id="task_x"):
    body = MagicMock()
    body.model_dump.return_value = {
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
        "verifier_metadata": {"task_id": task_id},
    }
    return body


@pytest.mark.asyncio
async def test_generic_failure_routes_to_sidecar_not_main(tmp_path, monkeypatch):
    """A failed task must carry a failure class so it never lands in the main jsonl."""
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def boom(task_id, out_dir):
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

    async def killed(task_id, out_dir, body=None):
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

    async def slow(task_id, out_dir, body=None):
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

    async def ok(task_id, out_dir, body=None):
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
    monkeypatch.setattr(agent, "_response_from_transcript", lambda task_id, out_dir: agent._empty_response(task_id))
    monkeypatch.setattr(agent, "_collect_transcript", lambda task_id, out_dir, run_id: ([], ""))
    resp = await agent.run(body=_run_body())
    dumped = resp.model_dump()
    assert dumped["reward"] == 1.0
    for key in (NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY):
        assert key not in dumped


def test_classify_task_failure_mapping():
    assert _classify_task_failure(SandboxKilledError("rc=-15")) == "kill_shaped"
    assert _classify_task_failure(TimeoutError("timed out")) == "timeout_exceeded"
    assert _classify_task_failure(RuntimeError("exec failed")) == "legitimate"
    assert _classify_task_failure(FileNotFoundError("apptainer")) == "legitimate"
