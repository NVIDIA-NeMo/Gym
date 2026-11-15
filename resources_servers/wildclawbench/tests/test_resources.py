# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource cleanup retains ownership until the native container is removed."""

import hashlib
import sys

import pytest

from responses_api_agents.wildclawbench_agent.execution import NativeProcess
from responses_api_agents.wildclawbench_agent.tests.helpers import make_pipeline


async def test_failed_cleanup_keeps_owner_and_unknown_close_is_idempotent(tmp_path, monkeypatch):
    _, _, resources, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    process = NativeProcess(resources.config.execution, tmp_path, "sha256:" + "a" * 64)
    resources._owned["session"] = process

    async def failed(self, stage, timeout):
        raise RuntimeError("container runtime unavailable")

    monkeypatch.setattr(NativeProcess, "invoke", failed)
    with pytest.raises(RuntimeError):
        await resources.release_session_state("session")
    assert resources.has_session_state("session")
    calls = []

    async def succeeded(self, stage, timeout):
        calls.append(stage)
        return {}

    monkeypatch.setattr(NativeProcess, "invoke", succeeded)
    await resources.release_session_state("session")
    await resources.release_session_state("session")
    await resources.release_session_state("unknown")
    assert calls == ["cleanup"]
    assert not resources.has_session_state("session")


def test_typed_session_ownership_fences_and_lost_seed_reply(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nemo_gym.core import EpisodeId, TaskId
    from nemo_gym.server_runtime import ResourcesSeedSessionRequest

    _, _, resources, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    cleanup = []

    async def invoke(self, stage, timeout):
        cleanup.append(self.output.name)
        return {}

    monkeypatch.setattr(NativeProcess, "invoke", invoke)
    seed = ResourcesSeedSessionRequest(
        resources_session_id="one",
        episode_id=EpisodeId(rollout_id="3-2"),
        task_id=TaskId(taskset="wildclaw", task_id="task"),
        task_data={"verifier_metadata": {"task_id": "task"}},
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
    )
    client = TestClient(resources.setup_webserver(), raise_server_exceptions=False)
    for _ in range(2):
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
    assert resources.has_session_state("one")
    client.cookies.clear()
    foreign = seed.model_copy(update={"resources_session_id": "foreign"})
    assert client.post("/seed_session", json=foreign.model_dump(mode="json")).status_code == 200
    assert resources._owned["foreign"].output != resources._owned["one"].output
    assert not cleanup and resources.has_session_state("one")
    assert (
        client.post(
            "/close_session",
            json={"resources_session_id": "foreign", "episode_id": seed.episode_id.model_dump(mode="json")},
        ).status_code
        == 200
    )
    cleanup.clear()
    close = {"resources_session_id": "one", "episode_id": seed.episode_id.model_dump(mode="json")}
    wrong = dict(close, episode_id={"rollout_id": "other"})
    assert client.post("/close_session", json=wrong).status_code == 409
    assert resources.has_session_state("one")
    for _ in range(2):
        assert client.post("/close_session", json=close).status_code == 200
    assert cleanup == [hashlib.sha256(b"one").hexdigest()] and not resources.has_session_state("one")
    client.cookies.clear()
    assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409
    # An unrelated new execution can be closed without the cookie from its lost seed reply.
    fresh = seed.model_copy(update={"resources_session_id": "new"})
    assert client.post("/seed_session", json=fresh.model_dump(mode="json")).status_code == 200
    client.cookies.clear()
    assert client.post("/close_session", json=dict(close, resources_session_id="new")).status_code == 200
    assert cleanup == [hashlib.sha256(b"one").hexdigest(), hashlib.sha256(b"new").hexdigest()]
    assert not resources._owned


@pytest.mark.parametrize("session_id", ["invalid", "../escape", "a" * 63])
def test_resources_owns_output_path_and_replaces_capture_pointer(tmp_path, monkeypatch, session_id):
    from fastapi.testclient import TestClient

    from responses_api_agents.wildclawbench_agent.execution import execution_key_for_episode

    _, _, resources, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    body = dict(
        resources_session_id=session_id,
        episode_id={"rollout_id": "3-2", "attempt": 1},
        task_id={"taskset": "wildclaw", "task_id": "task"},
        task_data={"verifier_metadata": {}},
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
    )
    client = TestClient(resources.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=body).status_code == 200
    key = execution_key_for_episode(resources.config.execution, "3-2-a1")
    assert key == hashlib.sha256(session_id.encode()).hexdigest()
    output = resources._owned[session_id].output
    assert output.name == key and output.parent == tmp_path / "episodes"
    # A later run can reuse the capture route, with a fresh private folder.
    client.cookies.clear()
    body["resources_session_id"] = session_id + "-new-run"
    assert client.post("/seed_session", json=body).status_code == 200
    current = execution_key_for_episode(resources.config.execution, "3-2-a1")
    assert current != key
    assert (output / "lease.json").is_file()
    close = dict(resources_session_id=body["resources_session_id"], episode_id=body["episode_id"])
    assert client.post("/close_session", json=close).status_code == 200
    assert (resources._owned[session_id].output / "lease.json").is_file()
    with pytest.raises(ValueError, match="closed"):
        execution_key_for_episode(resources.config.execution, "3-2-a1")


def test_grading_rejects_a_changed_answer_or_model():
    from nemo_gym.core import NeMoGymResponse
    from nemo_gym.server_runtime import BaseVerifyRequest, ResourcesSessionConflictError
    from resources_servers.wildclawbench.app import verify_response_binding
    from responses_api_agents.openclaw_agent.app import openclaw_session_conversation

    events = [
        {
            "type": "message",
            "id": "answer",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "original answer"}]},
        }
    ]
    response = NeMoGymResponse(
        id="r",
        created_at=0,
        model="policy",
        object="response",
        output=openclaw_session_conversation(events),
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    body = BaseVerifyRequest(responses_create_params={"input": "task"}, response=response)
    verify_response_binding(body, {"events": events}, "policy")
    for changed in [response.model_copy(update={"model": "other"}), response.model_copy(update={"output": []})]:
        with pytest.raises(ResourcesSessionConflictError):
            verify_response_binding(body.model_copy(update={"response": changed}), {"events": events}, "policy")


def test_verification_retry_keeps_original_grade_and_rejects_changed_response(tmp_path, monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from nemo_gym.core import NeMoGymResponse
    from responses_api_agents.openclaw_agent.app import openclaw_session_conversation

    _, _, resources, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    client = TestClient(resources.setup_webserver(), raise_server_exceptions=False)
    seed = dict(
        resources_session_id="scored",
        episode_id={"rollout_id": "3-2"},
        task_id={"taskset": "wildclaw", "task_id": "task"},
        task_data={"verifier_metadata": {"task_id": "task"}},
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
    )
    assert client.post("/seed_session", json=seed).status_code == 200
    output = resources._owned["scored"].output
    events = [
        {
            "type": "message",
            "id": "original",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "original answer"}]},
        }
    ]
    (output / "agent_result.json").write_text(json.dumps({"events": events}))
    (output / "worker.json").write_text(
        json.dumps({"metadata": {"task_id": "task"}, "evaluation_protocol": {"gym_protocol": "agent_sessions_v1"}})
    )
    (output / "agent_closed.json").write_text('{"cleanup_confirmed":true}')
    calls = []

    async def invoke(self, stage, timeout):
        calls.append(stage)
        assert (output / "agent_closed.json").is_file()
        return dict(reward=0.25, native_reward=0.25, mask_sample=False, errors=[], task_id="task")

    monkeypatch.setattr(NativeProcess, "invoke", invoke)
    response = NeMoGymResponse(
        id="r",
        created_at=0,
        model="policy",
        object="response",
        output=openclaw_session_conversation(events),
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    body = {"responses_create_params": seed["responses_create_params"], "response": response.model_dump(mode="json")}
    first = client.post("/verify", json=body)
    assert first.status_code == 200 and first.json()["reward"] == 0.25
    assert client.post("/verify", json=body).json()["reward"] == 0.25
    assert calls == ["verify", "cleanup"]
    body["response"]["id"] = "different-answer"
    assert client.post("/verify", json=body).status_code == 409
    assert calls == ["verify", "cleanup"]


def test_prepared_rows_have_source_bound_metadata_and_reject_missing_digests():
    import json
    from pathlib import Path

    from pydantic import ValidationError

    from nemo_gym.data import load_task_data_schema, normalize_task_fields
    from resources_servers.wildclawbench.task_data import TaskData

    directory = Path(__file__).parents[1]
    schema = load_task_data_schema(directory)
    for line in (directory / "data/example.jsonl").read_text().splitlines():
        row = json.loads(line)
        metadata, conflicts = normalize_task_fields(row)
        assert not conflicts
        validated = schema.validate_python(metadata)
        assert validated.task_id == row["verifier_metadata"]["task_id"]
        incomplete = dict(metadata)
        del incomplete["workspace_sha256"]
        with pytest.raises(ValidationError):
            TaskData.model_validate(incomplete)


async def test_aggregation_retains_valid_zero_and_blocks_incomplete_or_failed_headline(tmp_path, monkeypatch):
    from nemo_gym.config import AggregateMetricsRequest
    from responses_api_agents.wildclawbench_agent import metrics

    environment, _, resources, calls = make_pipeline(tmp_path, sys.executable, monkeypatch)
    pin = dict(revision="r", tasks={"a": "a.md", "b": "b.md"}, files={"a.md": "a" * 64, "b.md": "b" * 64})
    monkeypatch.setattr(metrics, "source_pin", lambda: pin)
    rows = [
        dict(
            task_id=k,
            task_sha256=k * 64,
            source_revision="r",
            evaluation_protocol={"gym_protocol": "agent_sessions_v1"},
            reward=float(i),
            native_reward=float(i),
            mask_sample=False,
            ng_perf={"completion_tokens": 10 * (i + 1), "num_turns": 2},
            _ng_task_index=i,
            _ng_rollout_index=0,
        )
        for i, k in enumerate(["a", "b"])
    ]

    async def aggregate(values):
        request = AggregateMetricsRequest(verify_responses=values)
        direct = await resources.aggregate_metrics(request)
        forwarded = await environment.aggregate_metrics(request)
        assert forwarded.key_metrics == direct.key_metrics
        assert all(forwarded.agent_metrics[key] == value for key, value in direct.agent_metrics.items())
        assert calls[-1] == "/aggregate_metrics"
        assert forwarded.perf_summary["total_completion_tokens"] == sum(
            row.get("ng_perf", {}).get("completion_tokens", 0) for row in values
        )
        return forwarded

    complete = await aggregate(rows)
    assert complete.key_metrics == {"wildclawbench/overall_score": 0.5}
    masked = await aggregate([rows[0], dict(rows[1], reward=0.0, native_reward=None, mask_sample=True)])
    assert masked.agent_metrics["wildclawbench/num_scored"] == 1
    assert masked.agent_metrics["wildclawbench/partial_mean_score"] == 0
    assert not masked.key_metrics
    failed = await aggregate(rows + [{"_ng_failure_class": "environment_server_failed"}])
    assert failed.agent_metrics["wildclawbench/num_failed"] == 1
    assert not failed.key_metrics
