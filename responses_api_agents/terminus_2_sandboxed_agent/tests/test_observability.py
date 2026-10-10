# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_gym.openai_utils as openai_utils
from nemo_gym.base_responses_api_model import CaptureStore, merge_model_call_capture_into_record
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    Terminus2Agent,
    Terminus2AgentConfig,
    Terminus2AgentRunRequest,
)


@pytest.fixture
def execution(monkeypatch):
    config = Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_1_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="swebench_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=2,
        enable_summarize=True,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        dump_trajectory=False,
        debug=False,
        model_context_limit=32_000,
        model_output_limit=4_000,
        interleaved_thinking=False,
        llm_request_timeout=60,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": True}
    server = Terminus2Agent(config=config, server_client=client)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")), stop=AsyncMock()
    )
    monkeypatch.setattr(Terminus2Agent, "_connect_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    calls, agents, commands = [], [], []
    mode = SimpleNamespace(value="success")

    async def transport(**kwargs):
        # Exercise the real client header merge and HTTP-status retry loop.
        await asyncio.sleep(0)
        owner = kwargs["headers"].get("x-session-id")
        index = sum(call["client_session_id"] == owner for call in calls)
        request = deepcopy(kwargs["json"])
        timeout = mode.value == "all_fail" or (mode.value == "retry" and index == 0)
        http_retry = mode.value in {"http_retry", "rate_limit"} and index == 0
        retry_status = 429 if mode.value == "rate_limit" else 500
        reject = (
            (
                mode.value in {"compaction", "compaction_missing_id", "cancel_summary", "short", "synthetic"}
                and index == 1
            )
            or (mode.value in {"short", "synthetic"} and index == 2)
            or (mode.value == "synthetic" and index == 3)
        )
        synthetic = mode.value == "synthetic" and index == 4
        cancel = (mode.value == "cancel_model" and index == 1) or (mode.value == "cancel_summary" and index == 2)
        answer = json.dumps(
            {
                "analysis": "inspect",
                "plan": "run commands",
                "commands": [{"keystrokes": "pwd\n", "duration": 0.1}, {"keystrokes": "ls\n", "duration": 0.1}],
                "task_complete": True,
            }
        )
        if mode.value == "parse_error" and index == 0:
            answer = "not valid JSON"
        response = {
            "id": ""
            if mode.value == "missing_id" or (mode.value == "compaction_missing_id" and index == 2)
            else ("reused" if mode.value == "reused_id" else f"resp_{len(calls)}"),
            "created_at": 0,
            "model": "policy_model",
            "object": "response",
            "output": [
                {
                    "type": "reasoning",
                    "id": f"reasoning_{len(calls)}",
                    "summary": [{"type": "summary_text", "text": "reasoning"}],
                },
                {
                    "id": f"msg_{len(calls)}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": answer, "annotations": []}],
                },
            ],
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": True,
            "usage": {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        }
        if reject:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        error = timeout or http_retry or synthetic or cancel
        calls.append(
            {
                "model_call_id": f"call_{len(calls)}",
                "client_session_id": owner,
                "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                "request": request,
                "response": None if error else deepcopy(response),
                "status_code": retry_status if http_retry else (None if error else 200),
                "error_category": "timeout" if timeout else ("server_error" if error else None),
            }
        )
        assert kwargs["url"].endswith("/v1/responses")
        if timeout:
            raise TimeoutError
        if cancel:
            raise asyncio.CancelledError
        if synthetic:
            raise ValueError("fallback request failed")
        payload = json.dumps(response).encode()
        return SimpleNamespace(
            status=retry_status if http_retry else 200,
            ok=not http_retry,
            read=AsyncMock(return_value=payload),
            content=SimpleNamespace(read=AsyncMock(return_value=b"retry")),
        )

    monkeypatch.setattr(openai_utils, "request", transport)

    async def setup(agent, environment):
        agents.append(agent)
        if mode.value == "shell_recovery":
            agent._max_episodes = 4

        async def send_keys(keystrokes, **kwargs):
            if mode.value == "shell_recovery" and len(calls) == 2:
                raise app_module.ShellExitedError("the interactive shell exited")
            if mode.value == "failure":
                raise ValueError("terminal command failed")
            if mode.value == "cancel_tool":
                raise asyncio.CancelledError
            if mode.value == "batch_timeout":
                raise TimeoutError
            commands.append(keystrokes)

        agent._session = SimpleNamespace(
            is_session_alive=AsyncMock(return_value=True),
            send_keys=send_keys,
            get_incremental_output=AsyncMock(return_value="terminal output"),
            capture_pane=AsyncMock(return_value="current screen"),
            recover_shell=AsyncMock(return_value="The interactive shell exited. Shell state reset; commands skipped."),
        )

    # Keep Harbor's real run loop, parser, query/retry/fallback and summarization hooks.
    monkeypatch.setattr(app_module.NeMoGymTerminus2, "setup", setup)
    monkeypatch.setattr(app_module.Terminus2, "_build_skills_section", AsyncMock(return_value=None))
    monkeypatch.setattr(app_module.Terminus2, "_count_total_tokens", lambda self, chat: 0)

    async def run(rollout_id="test-rollout", indexed=True, **identity):
        body = Terminus2AgentRunRequest(
            responses_create_params={"input": "solve this"},
            _ng_rollout_id=rollout_id,
            **({"_ng_task_index": 0, "_ng_rollout_index": 0} if indexed else {}),
            **identity,
        )
        request = SimpleNamespace(
            json=AsyncMock(return_value=body.model_dump(by_alias=True) | {"_ng_rollout_id": rollout_id}),
            session={app_module.SESSION_ID_KEY: rollout_id},
            cookies={},
        )

        async def post(**kwargs):
            if kwargs["url_path"] == "/seed_session":
                return SimpleNamespace(
                    status=200, ok=True, cookies={}, json=AsyncMock(return_value={"sandbox_handle": "sandbox"})
                )
            result = kwargs["json"] | {"reward": 1.0}
            return SimpleNamespace(status=200, ok=True, read=AsyncMock(return_value=json.dumps(result).encode()))

        client.post = post
        return json.loads((await server.run(request, body)).model_dump_json(by_alias=True))

    return SimpleNamespace(
        run=run,
        calls=calls,
        mode=mode,
        client=client,
        agents=agents,
        commands=commands,
        server=server,
        sandbox=sandbox,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception,status", [(ValueError("setup failed"), "failed"), (asyncio.CancelledError(), "incomplete")]
)
async def test_setup_failure_is_a_rollout_result_and_releases_sandbox(execution, monkeypatch, exception, status):
    monkeypatch.setattr(app_module.NeMoGymTerminus2, "setup", AsyncMock(side_effect=exception))
    result = await execution.run()
    assert result["terminus2_completed"] is False
    assert type(exception).__name__ in result["error"]
    assert result["ng_agent_observations"]["records"][0]["status"] == status
    assert not execution.calls
    execution.sandbox.stop.assert_awaited_once()
    assert not execution.server._session_sandboxes


@pytest.mark.asyncio
async def test_setup_uses_the_rollout_timeout(execution, monkeypatch):
    execution.server.config.sandbox_timeout = 0.01
    started = asyncio.Event()

    async def setup(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app_module.NeMoGymTerminus2, "setup", setup)
    result = await asyncio.wait_for(execution.run(), timeout=1)
    assert started.is_set()
    assert result["terminus2_completed"] is False
    assert "TimeoutError" in result["error"]
    assert result["ng_agent_observations"]["records"][0]["status"] == "incomplete"
    assert not execution.calls
    execution.sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("upload_fails", [False, True])
async def test_local_tmux_asset_bypasses_sandbox_mount(execution, tmp_path, upload_fails):
    asset = tmp_path / "tmux binary"
    asset.write_bytes(b"configured binary bytes")
    execution.server.config.local_tmux_binary_path = str(asset)
    execution.server.config.remote_tmux_binary_path = "/unreadable/mounted/tmux"
    uploaded = []

    async def upload(source, destination):
        uploaded.append((Path(source).read_bytes(), destination))
        if upload_fails:
            raise OSError("sandbox upload unavailable")

    execution.sandbox.upload = AsyncMock(side_effect=upload)
    result = await execution.run()
    assert uploaded == [(asset.read_bytes(), "/usr/local/bin/tmux")]
    assert all("cp " not in call.args[0] for call in execution.sandbox.exec.await_args_list)
    assert result["terminus2_completed"] is not upload_fails
    assert len(execution.calls) == (0 if upload_fails else 2)
    if upload_fails:
        assert "sandbox upload unavailable" in result["error"]
    else:
        assert any("chmod +x /usr/local/bin/tmux" in call.args[0] for call in execution.sandbox.exec.await_args_list)
    execution.sandbox.stop.assert_awaited_once()


def test_missing_local_tmux_asset_fails_before_serving_rollouts(execution, tmp_path):
    config = execution.server.config.model_copy(update={"local_tmux_binary_path": str(tmp_path / "missing-tmux")})
    with pytest.raises(ValueError, match="Local tmux binary is not a file"):
        Terminus2Agent(config=config, server_client=execution.client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failures,message", [(2, "Input/output error"), (3, "Input/output error"), (1, "Permission denied")]
)
async def test_tmux_install_retries_io_errors_without_aborting_collection(execution, monkeypatch, failures, message):
    execution.server.config.remote_tmux_binary_path = "/mounted binaries/tmux"
    attempts = []
    sleep = AsyncMock()
    monkeypatch.setattr(app_module.asyncio, "sleep", sleep)

    async def sandbox_exec(command, **kwargs):
        if "&& cp " in command:
            attempts.append((command, kwargs))
            return SimpleNamespace(return_code=int(len(attempts) <= failures), stdout=message, stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    execution.sandbox.exec.side_effect = sandbox_exec
    result = await execution.run()
    recovered = failures == 2
    assert result["terminus2_completed"] is recovered
    assert len(attempts) == (3 if message == "Input/output error" else 1)
    assert all("cp '/mounted binaries/tmux' /usr/local/bin/tmux" in command for command, _ in attempts)
    assert all(kwargs["timeout_s"] == 30 for _, kwargs in attempts)
    assert [call.args[0] for call in sleep.await_args_list if call.args[0]] == (
        [1, 2] if message == "Input/output error" else []
    )
    assert len(execution.calls) == (2 if recovered else 0)
    if not recovered:
        assert message in result["error"]
        assert result["ng_agent_observations"]["records"][0]["status"] == "failed"
    execution.sandbox.stop.assert_awaited_once()
    assert not execution.server._session_sandboxes


@pytest.mark.asyncio
async def test_shell_exit_requires_a_confirmation_after_the_model_sees_the_state_reset(execution):
    execution.mode.value = "shell_recovery"
    result = await execution.run()
    assert result["terminus2_completed"] is True
    assert len(execution.calls) == 3
    assert execution.commands == ["pwd\n", "ls\n", "pwd\n", "ls\n"]
    assert "Shell state reset" in str(execution.calls[2]["request"]["input"])


def save_and_check(execution, result, tmp_path):
    store = CaptureStore(tmp_path / "capture")
    for call in execution.calls:
        store.record("test-rollout", call)
    result["_ng_rollout_id"] = "test-rollout"
    merge_model_call_capture_into_record(result, [store.root], include_payloads=True)
    _attach_trajectory_record(result, result)
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(result) + "\n")
    [health] = run_health_checks(path, workers=1).rollouts
    return result["ng_trajectory"], health


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,call_count,turn_count,status",
    [
        ("success", 2, 2, "completed"),
        ("retry", 3, 2, "completed"),
        ("http_retry", 3, 2, "completed"),
        ("rate_limit", 3, 2, "completed"),
        ("all_fail", 30, 0, "incomplete"),
        ("compaction", 6, 2, "completed"),
        ("compaction_missing_id", 6, 2, "completed"),
        ("cancel_summary", 3, 1, "incomplete"),
        ("batch_timeout", 2, 2, "completed"),
        ("short", 5, 2, "completed"),
        ("synthetic", 5, 1, "completed"),
        ("failure", 1, 1, "failed"),
        ("cancel_tool", 1, 1, "incomplete"),
        ("cancel_model", 2, 1, "incomplete"),
        ("missing_id", 2, 0, "completed"),
        ("parse_error", 2, 2, "completed"),
    ],
)
async def test_real_harbor_decisions_survive_saved_projection(
    execution, tmp_path, mode, call_count, turn_count, status
):
    execution.mode.value = mode
    result = await execution.run(task_id="task-identity")
    trajectory, health = save_and_check(execution, result, tmp_path)
    [invocation] = trajectory["invocations"]
    turns = trajectory["turns"]
    assert invocation["status"] == status
    assert len(execution.calls) == call_count
    assert len(turns) == turn_count
    assert {call["client_session_id"] for call in execution.calls} == {invocation["invocation_id"]}
    assert {ref["model_call_id"] for ref in invocation["model_calls"]} == {
        call["model_call_id"] for call in execution.calls
    }
    for turn in turns:
        [ref] = turn["model_calls"]
        [call] = [
            call for call in execution.calls if call["response"] and call["response"]["id"] == ref["response_id"]
        ]
        assert turn["question"] == call["request"]["input"]
        assert turn["answer"] == [
            item.model_dump(mode="json")
            for item in app_module.NeMoGymResponse.model_validate(call["response"]).output
            if item.type != "reasoning"
        ]
        assert turn["reasoning_content"] == "reasoning"
        assert turn["task_id"] == trajectory["task_id"] == "task-identity"
        assert turn["rollout_id"] == trajectory["rollout_id"] == "test-rollout"
        assert turn["resolved"] is None
        assert turn["timestamp"] > 0
    assert {
        "rollout_ended_on_failed_model_call",
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "model_call_runaway_generation",
        "trajectory_capture_mismatch",
    }.isdisjoint(health.unobserved)
    assert ("agent_turn_hollow" in health.unobserved) == (turn_count == 0)
    assert "rollout_token_count_mismatch" in health.unobserved
    assert not health.policy_calls_observed
    assert "rollout_token_count_mismatch" not in {finding.check for finding in health.findings}
    assert ("turns_unavailable" in {gap["code"] for gap in trajectory["gaps"]}) == (turn_count == 0)
    assert not execution.agents[0]._nemo_gym_llm._is_compacting
    assert not execution.agents[0]._is_check_proactive_summarization
    if mode == "success":
        assert len(execution.commands) == 4
        assert [turn["turn_no"] for turn in turns] == [1, 2]
        assert [turn["step_count"] for turn in turns] == [0, 1]
        assert not health.findings
    if mode in {"compaction", "compaction_missing_id", "short"}:
        [compaction] = [r for r in result["ng_agent_observations"]["records"] if r["kind"] == "context_compaction"]
        assert compaction["outcome"] == ("failed" if mode == "short" else "completed")
        assert len(compaction["model_calls"]) == ({"compaction": 3, "compaction_missing_id": 2, "short": 1}[mode])
        assert turns[1]["question"][-1]["content"] != "terminal output"
        assert turns[1]["model_calls"][0]["response_id"] == execution.calls[-1]["response"]["id"]
        assert {ref["response_id"] for ref in compaction["model_calls"]}.isdisjoint(
            ref["response_id"] for turn in turns for ref in turn["model_calls"]
        )
    if mode == "batch_timeout":
        assert [turn["step_count"] for turn in turns] == [0, 1]
    if mode == "cancel_summary":
        [compaction] = [r for r in result["ng_agent_observations"]["records"] if r["kind"] == "context_compaction"]
        assert compaction["outcome"] == "aborted"
    if mode == "parse_error":
        assert turns[0]["answer"][0]["content"][0]["text"] == "not valid JSON"
        assert [turn["step_count"] for turn in turns] == [0, 0]


@pytest.mark.asyncio
async def test_reused_response_id_is_not_arbitrarily_selected(execution, tmp_path):
    execution.mode.value = "reused_id"
    result = await execution.run()
    trajectory, health = save_and_check(execution, result, tmp_path)
    assert len(trajectory["turns"]) == 1
    assert "model_response_id_reused" in {gap["code"] for gap in trajectory["gaps"]}
    assert "rollout_token_count_mismatch" in health.unobserved
    assert "trajectory_capture_mismatch" in {finding.check for finding in health.findings}


@pytest.mark.asyncio
async def test_concurrent_executions_have_distinct_turns_and_ownership(execution):
    results = await asyncio.gather(execution.run("first"), execution.run("second"))
    ids = [result["ng_agent_observations"]["records"][0]["invocation_id"] for result in results]
    assert len(set(ids)) == 2
    for result, owner in zip(results, ids):
        assert [turn["turn_no"] for turn in result["ng_trajectory"]["turns"]] == [1, 2]
        for turn in result["ng_trajectory"]["turns"]:
            assert turn["invocation_id"] == owner
            [ref] = turn["model_calls"]
            [call] = [call for call in execution.calls if call["response"]["id"] == ref["response_id"]]
            assert call["client_session_id"] == owner


@pytest.mark.asyncio
async def test_disabled_observability_preserves_output_and_headers(execution):
    execution.client.global_config_dict = {"observability_enabled": False}
    result = await execution.run()
    assert "ng_agent_observations" not in result
    assert "ng_trajectory" not in result
    assert result["terminus2_completed"] is True
    assert result["response"]["output"]
    assert all(call["client_session_id"] is None for call in execution.calls)


@pytest.mark.asyncio
async def test_missing_rollout_identity_does_not_emit_observations(execution):
    result = await execution.run(None, indexed=False)
    assert "ng_agent_observations" not in result
    assert "ng_trajectory" not in result


@pytest.mark.asyncio
async def test_parser_exception_keeps_observed_answer(execution, monkeypatch, tmp_path):
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser

    def parse(self, response):
        raise ValueError("parser failed")

    monkeypatch.setattr(TerminusJSONPlainParser, "parse_response", parse)
    result = await execution.run()
    trajectory, health = save_and_check(execution, result, tmp_path)
    assert result["error"] and result["terminus2_completed"] is False
    assert len(trajectory["turns"]) == 1
    assert trajectory["turns"][0]["model_calls"][0]["response_id"] == execution.calls[0]["response"]["id"]
    assert "agent_turn_hollow" not in health.unobserved


@pytest.mark.asyncio
async def test_proactive_summarization_flag_resets_on_cancellation(execution, monkeypatch):
    monkeypatch.setattr(
        app_module.Terminus2, "_check_proactive_summarization", AsyncMock(side_effect=asyncio.CancelledError)
    )
    result = await execution.run()
    assert result["ng_agent_observations"]["records"][0]["status"] == "incomplete"
    assert not result["ng_trajectory"]["turns"]
    assert not execution.agents[0]._is_check_proactive_summarization
