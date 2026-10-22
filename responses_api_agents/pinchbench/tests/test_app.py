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
from nemo_gym.rollout_collection import is_terminal_failure
from nemo_gym.rollout_observability import AgentInvocation, ToolCallObservation
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandbox.providers.apptainer.provider import ApptainerCreateVerificationError
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pinchbench.app import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
    PinchBenchAgent,
    PinchBenchAgentConfig,
    PinchBenchRunRequest,
    SandboxExecError,
    SandboxKilledError,
    SandboxNoOutputError,
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
        # The host has no apptainer binary: the sandbox could not run at all.
        (FileNotFoundError("apptainer"), "provider_unavailable"),
        # A real failure: every task of a run failed at once with a bare host-side error and no model call.
        (FileNotFoundError(2, "No such file or directory"), "provider_unavailable"),
        # Real failures: the container image was gone, and the container stalled in setup (stderr kept).
        (
            SandboxExecError(
                "direct apptainer exec failed for task task_csv_finance_report: FATAL:   While checking container "
                "encryption: could not open image <path>: failed to retrieve path for <path>: lstat <path>: no such "
                "file or directory"
            ),
            "provider_unavailable",
        ),
        (
            SandboxExecError(
                "direct apptainer exec failed for task task_csv_finance_report: INFO:    Terminating squashfuse_ll "
                "after timeout\nINFO:    Cleanup error: while stopping driver for <path>: squashfuse_ll exited: "
                "fuse: failed to unmount <path>: Invalid argument"
            ),
            "provider_unavailable",
        ),
        # A real failure: the sandbox failed its readiness probe.
        (
            ApptainerCreateVerificationError(
                "sandbox 'nemo-gym-<hex>' failed readiness probe: return_code=1, "
                "stderr='ERROR  : Failed to enter in user namespace: Invalid argument'"
            ),
            "provider_unavailable",
        ),
        # A real failure: the exec exited cleanly but wrote no archive.
        (
            SandboxNoOutputError("direct apptainer exec did not produce <path> for task task_todo_list_cleanup"),
            "provider_unavailable",
        ),
    ],
)
@pytest.mark.parametrize("eval_mode", [False, True], ids=["training", "evaluation"])
def test_classify_task_failure(exc, expected, eval_mode, monkeypatch):
    monkeypatch.setattr("responses_api_agents.pinchbench.app.is_eval_mode", lambda: eval_mode)
    # Outside evaluation every one of these keeps the class it always had: legitimate.
    if not eval_mode and expected == "provider_unavailable":
        expected = "legitimate"
    # Evaluation caps a kill before the run started like every retry: kill_shaped would be re-dispatched forever.
    if eval_mode and expected == "kill_shaped":
        expected = "provider_unavailable"
    assert _classify_task_failure(exc) == expected
    row = {"_ng_failure_class": _classify_task_failure(exc), "grading_notes": f"run failed: {exc}"}
    if expected == "provider_unavailable":
        assert not is_terminal_failure(row)


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


@pytest.mark.asyncio
async def test_a_failed_apptainer_exec_with_no_archive_raises_sandbox_exec_error(tmp_path):
    agent = make_agent(
        sandbox_spec={"image": "docker://test"},
        sandbox_provider={"apptainer": {"direct_exec": True}},
    )
    proc = MagicMock()
    proc.returncode = 255
    proc.wait = AsyncMock(return_value=None)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(SandboxExecError, match="direct apptainer exec failed"):
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


# --- the sandbox rule: once the agent's run started, a dead or empty sandbox is the model's ---


@pytest.mark.parametrize(
    "exc",
    [
        SandboxKilledError("direct apptainer exec killed (rc=137)", run_started=True),
        SandboxExecError("direct apptainer exec failed", run_started=True),
        SandboxNoOutputError("direct apptainer exec did not produce out.tgz", run_started=True),
    ],
    ids=["killed", "exec-failed", "no-output"],
)
def test_a_sandbox_lost_after_the_run_started_is_the_models_final_zero(exc, monkeypatch):
    # A memory bomb (137), kill -9 -1, or a deleted /sandbox/out: the model's commands can cause each one.
    monkeypatch.setattr("responses_api_agents.pinchbench.app.is_eval_mode", lambda: True)
    assert _classify_task_failure(exc) == "agent_run_error"


def test_a_kill_before_the_run_started_is_retried_with_a_cap(monkeypatch):
    # Never kill_shaped in evaluation: kill_shaped writes no sidecar row, so it is re-dispatched with no attempt cap.
    monkeypatch.setattr("responses_api_agents.pinchbench.app.is_eval_mode", lambda: True)
    assert _classify_task_failure(SandboxKilledError("rc=-15")) == "provider_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "failure_class", "cause", "terminal", "no_persist"),
    [
        (SandboxKilledError("rc=137", run_started=True), "agent_run_error", "model", True, False),
        (SandboxKilledError("rc=-15"), "provider_unavailable", "infra", False, False),
        (SandboxExecError("create failed"), "provider_unavailable", "infra", False, False),
    ],
    ids=["killed-after-start", "killed-before-start", "exec-failed-before-start"],
)
async def test_evaluation_routes_sandbox_losses_by_whether_the_run_started(
    exc, failure_class, cause, terminal, no_persist, tmp_path, monkeypatch
):
    monkeypatch.setattr("responses_api_agents.pinchbench.app.is_eval_mode", lambda: True)
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def fail(task_id, out_dir, rollout_id=None):
        raise exc

    monkeypatch.setattr(agent, "_run_in_sandbox", fail)
    dumped = (await agent.run(body=_run_body())).model_dump()
    assert dumped.get(NG_FAILURE_CLASS_KEY) == failure_class
    assert dumped.get("_ng_failure_cause") == cause
    assert bool(dumped.get(NG_TERMINAL_KEY)) is terminal
    assert bool(dumped.get(NG_NO_PERSIST_KEY)) is no_persist
    assert is_terminal_failure(dumped) is terminal


@pytest.mark.asyncio
async def test_the_direct_exec_marks_whether_the_agents_run_started(tmp_path):
    agent = make_agent(
        sandbox_spec={"image": "docker://test"},
        sandbox_provider={"apptainer": {"direct_exec": True}},
    )
    proc = MagicMock()
    proc.returncode = 137
    proc.wait = AsyncMock(return_value=None)
    proc.kill = MagicMock()

    async def started_then_killed(*args, **kwargs):
        (tmp_path / "sandbox" / ".agent_started").touch()
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=started_then_killed):
        with pytest.raises(SandboxKilledError) as caught:
            await agent._run_in_apptainer_direct("task_x", tmp_path, {"direct_exec": True})
    assert caught.value.run_started is True
    (tmp_path / "w").mkdir()
    assert ".agent_started" in agent._write_direct_exec_wrapper(tmp_path / "w").read_text()


class _CollectionResponse:
    def __init__(self, payload: dict) -> None:
        self.status, self.ok, self.payload = 200, True, payload

    def release(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run_started", "expected_dispatches"),
    [(True, 1), (False, 3)],
    ids=["killed-after-the-run-started", "killed-before-the-run-started"],
)
async def test_a_killed_sandbox_is_not_re_dispatched_on_every_resume(
    run_started, expected_dispatches, tmp_path, monkeypatch
):
    """Before this change a kill was kill_shaped with no sidecar row, so every --resume ran it again with no attempt cap.

    A kill after the agent's run started is the model's final 0 (one dispatch). A kill before it is retried, and each
    retry writes a sidecar row, so NEMO_GYM_MAX_ROLLOUT_ATTEMPTS (3) caps it.
    """
    import json

    from omegaconf import OmegaConf

    import nemo_gym.rollout_collection as rc
    from nemo_gym.reward_profile import compute_aggregate_metrics

    monkeypatch.setattr("responses_api_agents.pinchbench.app.is_eval_mode", lambda: True)
    monkeypatch.setattr(rc, "is_eval_mode", lambda: True)
    monkeypatch.delenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", raising=False)
    monkeypatch.setattr(rc, "get_global_config_dict", MagicMock(return_value={}))
    agent = make_agent(work_root=str(tmp_path / "work"), transcripts_dir=str(tmp_path / "arch"))

    async def killed(task_id, out_dir, rollout_id=None):
        raise SandboxKilledError("direct apptainer exec killed (rc=137) for task task_x", run_started=run_started)

    monkeypatch.setattr(agent, "_run_in_sandbox", killed)
    dispatches = 0

    async def post(server_name, url_path, json, **kwargs):
        nonlocal dispatches
        if url_path == "/run":
            dispatches += 1
            return _CollectionResponse((await agent.run(body=_run_body())).model_dump(mode="json", by_alias=True))
        return _CollectionResponse(compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

    server_client = MagicMock()
    server_client.post = AsyncMock(side_effect=post)
    server_client.global_config_dict = OmegaConf.create(
        {
            "my_agent": {"responses_api_agents": {"impl": {}}},
            "my_environment_server": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}},
        }
    )
    monkeypatch.setattr(rc, "setup_server_client_utils", lambda *args, **kwargs: server_client)

    async def raise_for_status(response):
        return None

    async def get_response_json(response):
        return response.payload

    monkeypatch.setattr(rc, "raise_for_status", raise_for_status)
    monkeypatch.setattr(rc, "get_response_json", get_response_json)

    input_fpath = tmp_path / "input.jsonl"
    input_fpath.write_text(
        json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}}) + "\n"
    )
    config = rc.RolloutCollectionConfig(
        input_jsonl_fpath=str(input_fpath),
        output_jsonl_fpath=str(tmp_path / "output.jsonl"),
        resume_from_cache=True,
        disable_health_check=True,
    )
    # The first collection and five --resume chain hops: before this change each one dispatched the task again.
    for _ in range(6):
        try:
            await rc.RolloutCollectionHelper().run_from_config(config)
        except RuntimeError as error:
            # A hop whose only rollout failed again has no score to report; the next --resume still runs.
            assert "None of the 1 dispatched rollouts produced a result" in str(error)

    assert dispatches == expected_dispatches
