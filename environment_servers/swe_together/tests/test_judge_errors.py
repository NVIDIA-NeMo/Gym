# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A judge seed or execution failure stays the reported cause when the session close also fails."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectionError, ClientResponseError

from environment_servers.swe_together import evaluation
from environment_servers.swe_together.judge_evidence import collect_judge_evidence
from environment_servers.swe_together.schemas import SWETRequest
from environment_servers.swe_together.task import TaskData
from nemo_gym.trajectory import TrajectoryRecord


SEED, ACTIVATE, CLOSE = (
    "/v1/agent_sessions",
    "/ng-rollout/one/ng-agent-instance/judge/v1/responses",
    "/v1/agent_sessions/close",
)
CONFIRMED = {"agent_session_id": "judge-test", "cleanup_confirmed": True}
UNCONFIRMED = {"agent_session_id": "judge-test"}


def http_error(status):
    return ClientResponseError(SimpleNamespace(real_url="http://judge"), (), status=status, message="error")


async def run_judge(tmp_path, monkeypatch, *, failures, close, verdict=None):
    """Run the judge against a fake agent; ``failures`` maps a path to the exception it raises."""
    monkeypatch.setattr(evaluation, "uuid4", lambda: SimpleNamespace(hex="test"))
    monkeypatch.setattr(evaluation, "stage_judge", AsyncMock(return_value="judge"))
    monkeypatch.setattr(evaluation, "raise_for_status", AsyncMock())
    monkeypatch.setattr(evaluation, "read_test_reward", AsyncMock(return_value={}))
    reader = AsyncMock(return_value=verdict) if verdict else AsyncMock(side_effect=FileNotFoundError("verdict.json"))
    monkeypatch.setattr(evaluation, "read_text", reader)
    (tmp_path / "canonical_goals.json").write_text(json.dumps({"completeness_goals": [{"id": "goal", "weight": 1}]}))
    calls = []

    async def post(name, path, *, json, cookies, _control=False):
        calls.append(path)
        if path in failures:
            raise failures[path]
        payload = close if path == CLOSE else {"status": "completed", "output": []} if path == ACTIVATE else {}
        return SimpleNamespace(cookies={}, payload=payload)

    async def payload(response):
        return response.payload

    monkeypatch.setattr(evaluation, "get_response_json", payload)
    task = TaskData(task_id="task", image="image", image_digest="sha256:" + "1" * 64)
    req = SWETRequest(
        episode_id={"rollout_id": "one"},
        task={"task_id": {"taskset": "swet", "task_id": "task"}, "task_input": {"task_data": task}},
    )
    evidence = {}
    run = evaluation.run_judge(
        client=SimpleNamespace(post=post),
        agent=SimpleNamespace(name="judge"),
        sandbox=SimpleNamespace(serialize=AsyncMock(return_value={})),
        provider="sandbox",
        task_dir=tmp_path,
        patch="patch",
        workdir="/workspace",
        episode=req.episode_id,
        activation_path=ACTIVATE,
        task_id=req.task.task_id,
        timeout=100,
        cookies={},
        artifact_dir=tmp_path,
        evidence=evidence,
    )
    return run, calls, evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_path", [SEED, ACTIVATE])
@pytest.mark.parametrize("close_failure", ["unconfirmed", "close_request_error"])
async def test_primary_error_survives_failed_close(tmp_path, monkeypatch, failed_path, close_failure):
    primary = http_error(500)
    failures = {failed_path: primary}
    if close_failure == "close_request_error":
        failures[CLOSE] = ClientConnectionError("close lost")
    run, calls, evidence = await run_judge(tmp_path, monkeypatch, failures=failures, close=UNCONFIRMED)
    with pytest.raises(ClientResponseError) as raised:
        await run
    assert raised.value is primary
    # A failed seed is not proof that no session state exists, so close is still attempted.
    assert calls[-1] == CLOSE
    expected = "Judge session close was not confirmed" if close_failure == "unconfirmed" else "close lost"
    assert evidence["judge_close_error"] == expected
    assert json.loads((tmp_path / "judge-close-error.json").read_text()) == {"error": expected}
    assert evidence["judge_transport_error"].startswith("ClientResponseError: 500")


@pytest.mark.asyncio
async def test_primary_error_replaces_unreadable_verdict_after_confirmed_close(tmp_path, monkeypatch):
    primary = http_error(500)
    run, calls, evidence = await run_judge(tmp_path, monkeypatch, failures={SEED: primary}, close=CONFIRMED)
    with pytest.raises(ClientResponseError) as raised:
        await run
    assert raised.value is primary
    assert calls == [SEED, CLOSE]
    assert "judge_close_error" not in evidence


@pytest.mark.asyncio
async def test_cleanup_failure_surfaces_without_primary_error(tmp_path, monkeypatch):
    run, calls, evidence = await run_judge(tmp_path, monkeypatch, failures={}, close=UNCONFIRMED)
    with pytest.raises(RuntimeError, match="^Judge cleanup was not confirmed$"):
        await run
    assert calls == [SEED, ACTIVATE, CLOSE]
    assert evidence["judge_close_error"] == "Judge session close was not confirmed"


@pytest.mark.asyncio
async def test_cancellation_propagates_past_failed_close(tmp_path, monkeypatch):
    run, calls, evidence = await run_judge(
        tmp_path, monkeypatch, failures={SEED: asyncio.CancelledError()}, close=UNCONFIRMED
    )
    with pytest.raises(asyncio.CancelledError):
        await run
    assert calls == [SEED, CLOSE]
    assert evidence["judge_close_error"] == "Judge session close was not confirmed"


@pytest.mark.asyncio
async def test_successful_judge_is_unchanged(tmp_path, monkeypatch):
    raw = json.dumps({"goal_results": [{"id": "goal", "met": True}]})
    run, calls, evidence = await run_judge(tmp_path, monkeypatch, failures={}, close=CONFIRMED, verdict=raw)
    verdict = await run
    assert calls == [SEED, ACTIVATE, CLOSE]
    assert verdict["judge_score"] == 1.0
    assert verdict["judge_close"]["cleanup_confirmed"] is True
    assert "judge_close_error" not in evidence and "judge_transport_error" not in evidence


@pytest.mark.parametrize("attempt", [0, 2])
async def test_the_judge_session_collects_only_its_own_calls_from_the_episode_capture(tmp_path, monkeypatch, attempt):
    monkeypatch.setattr(evaluation, "uuid4", lambda: SimpleNamespace(hex="first"))
    monkeypatch.setattr(evaluation, "stage_judge", AsyncMock(return_value="judge"))
    monkeypatch.setattr(evaluation, "raise_for_status", AsyncMock())
    monkeypatch.setattr(evaluation, "read_test_reward", AsyncMock(return_value={}))
    monkeypatch.setattr(
        evaluation, "read_text", AsyncMock(return_value=json.dumps({"goal_results": [{"id": "goal", "met": True}]}))
    )
    monkeypatch.setattr(evaluation, "get_response_json", AsyncMock(side_effect=lambda response: response.payload))
    (tmp_path / "canonical_goals.json").write_text(json.dumps({"completeness_goals": [{"id": "goal", "weight": 1}]}))
    req = SWETRequest(
        episode_id={"rollout_id": "parent", "attempt": attempt},
        task={
            "task_id": {"taskset": "swet", "task_id": "task"},
            "task_input": {"task_data": TaskData(task_id="task", image="image", image_digest="sha256:" + "1" * 64)},
        },
    )
    key = req.episode_id.capture_key
    model = {"type": "responses_api_models", "name": "judge"}
    # A judge capture under the old ``<key>-judge`` episode is left alone.
    old_path = tmp_path / f"{key}-judge.capture.jsonl"
    old_bytes = json.dumps({"model_call_id": "old-session", "request": {}, "response": {}})
    old_path.write_text(old_bytes)
    capture = tmp_path / f"{key}.capture.jsonl"
    # The assistant's calls share the episode's capture key and are not judge evidence.
    capture.write_text(
        json.dumps(
            {
                "model_call_id": "assistant-call",
                "agent_instance": "assistant",
                "agent_kind": "assistant",
                "model_ref": {"type": "responses_api_models", "name": "policy"},
                "request": {"messages": []},
                "response": {"id": "assistant-call", "content": []},
                "status_code": 200,
            }
        )
        + "\n"
    )
    activation_path = f"/ng-rollout/{key}/ng-agent-instance/judge/v1/responses"
    seen = []
    session = None
    encode = json.dumps

    async def post(name, path, *, json, cookies, _control=False):
        nonlocal session
        if path == SEED:
            session = json.agent_session_id
            seen.append((json.episode_id.capture_key, json.agent_instance))
            payload = {}
        elif path == CLOSE:
            seen.append((json.episode_id.capture_key, None))
            payload = {
                "agent_session_id": session,
                "cleanup_confirmed": True,
                "agent_observations": {
                    "source": "claude_code",
                    "records": [
                        {
                            "kind": "agent_invocation",
                            "invocation_id": session,
                            "status": "completed",
                            "model_calls": [{"model_ref": model, "response_id": session}],
                        }
                    ],
                },
            }
        else:
            assert path == activation_path
            exchange = {
                "model_call_id": session,
                "agent_instance": "judge",
                "agent_kind": "judge",
                "model_ref": model,
                "request": {"messages": []},
                "response": {"id": session, "content": []},
                "status_code": 200,
            }
            with capture.open("a") as handle:
                handle.write(encode(exchange) + "\n")
            payload = {"status": "completed", "output": []}
        return SimpleNamespace(cookies={}, payload=payload)

    evidence = {}
    verdict = await evaluation.run_judge(
        client=SimpleNamespace(post=post),
        agent=SimpleNamespace(name="judge"),
        sandbox=SimpleNamespace(serialize=AsyncMock(return_value={})),
        provider="sandbox",
        task_dir=tmp_path,
        patch="patch",
        workdir="/workspace",
        episode=req.episode_id,
        activation_path=activation_path,
        task_id=req.task.task_id,
        timeout=100,
        cookies={},
        artifact_dir=tmp_path,
        evidence=evidence,
    )

    assert seen == [(key, "judge"), (key, None)]
    assert (evidence["judge_capture_episode"], evidence["judge_agent_instance"]) == (key, "judge")
    assert verdict["judge_score"] == 1 and verdict["judge_close"]["cleanup_confirmed"] is True
    parent = TrajectoryRecord(task_id="task", rollout_id=key)
    result = await collect_judge_evidence(
        parent, evidence, {"observability_enabled": True, "model_call_capture_dir": str(tmp_path)}, tmp_path
    )
    assert [call.model_call_id for call in result.model_calls] == ["judge-first"]
    assert not result.gaps
    raw = (tmp_path / "judge-model-calls.capture.jsonl").read_text().splitlines()
    assert [json.loads(line)["model_call_id"] for line in raw] == ["judge-first"]
    assert old_path.read_text() == old_bytes
