# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource cleanup retains ownership until the native container is removed."""

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
        task_data={"wildclaw_execution_key": "a" * 64, "verifier_metadata": {"task_id": "task"}},
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
    )
    client = TestClient(resources.setup_webserver(), raise_server_exceptions=False)
    for _ in range(2):
        assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 200
    assert resources.has_session_state("one")
    client.cookies.clear()
    foreign = seed.model_copy(update={"resources_session_id": "foreign"})
    assert client.post("/seed_session", json=foreign.model_dump(mode="json")).status_code == 409
    assert not cleanup and resources.has_session_state("one")
    close = {"resources_session_id": "one", "episode_id": seed.episode_id.model_dump(mode="json")}
    wrong = dict(close, episode_id={"rollout_id": "other"})
    assert client.post("/close_session", json=wrong).status_code == 409
    assert resources.has_session_state("one")
    for _ in range(2):
        assert client.post("/close_session", json=close).status_code == 200
    assert cleanup == ["a" * 64] and not resources.has_session_state("one")
    client.cookies.clear()
    assert client.post("/seed_session", json=seed.model_dump(mode="json")).status_code == 409
    # An unrelated new execution can be closed without the cookie from its lost seed reply.
    fresh = seed.model_copy(
        update={"resources_session_id": "new", "task_data": dict(seed.task_data, wildclaw_execution_key="b" * 64)}
    )
    assert client.post("/seed_session", json=fresh.model_dump(mode="json")).status_code == 200
    client.cookies.clear()
    assert client.post("/close_session", json=dict(close, resources_session_id="new")).status_code == 200
    assert cleanup == ["a" * 64, "b" * 64]
    assert not resources._owned


@pytest.mark.parametrize("key", ["invalid", "../escape", "a" * 63])
def test_invalid_execution_key_is_terminal_before_allocation(tmp_path, monkeypatch, key):
    from fastapi.testclient import TestClient

    _, _, resources, _ = make_pipeline(tmp_path, sys.executable, monkeypatch)
    body = dict(
        resources_session_id="bad",
        episode_id={"rollout_id": "3-2"},
        task_id={"taskset": "wildclaw", "task_id": "task"},
        task_data={"verifier_metadata": {}, "wildclaw_execution_key": key},
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
    )
    response = TestClient(resources.setup_webserver(), raise_server_exceptions=False).post("/seed_session", json=body)
    assert response.status_code == 422
    assert not resources._owned
    assert not (tmp_path / "episodes").exists()


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
        task_data={"verifier_metadata": {"task_id": "task"}, "wildclaw_execution_key": "c" * 64},
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
    )
    assert client.post("/seed_session", json=seed).status_code == 200
    output = tmp_path / "episodes" / ("c" * 64)
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
