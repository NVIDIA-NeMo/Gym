# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA model rejections reach the shared environment and bounded retry policy."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pydantic import ValidationError

from environment_servers.single_agent_turn.app import AgentOutcomePolicyConfig
from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _episode_record,
    failures_path_for,
)
from nemo_gym.rollout_observability import AgentInvocation
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.provenance import provenance_metadata
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_app import agent, close, open_session, result
from responses_api_agents.nooa_agent.tests.test_config import agent_config, invocation_config
from responses_api_agents.nooa_agent.tests.test_environment import _environment, _paths, _request
from responses_api_agents.nooa_agent.tests.test_runner import model_http_error
from responses_api_agents.nooa_agent.tests.test_sandbox_entrypoint import payload


def rejection_agent(status: int) -> NOOAAgent:
    client = MagicMock(spec=ServerClient)
    failure = model_http_error(status)
    failure.message = "Provider rejected this request; diagnostic mentions HTTP 500"
    client.post = AsyncMock(side_effect=failure)
    invocation = invocation_config(
        agent_class="responses_api_agents.nooa_agent.tests.test_runner:PolicyAgent",
        invocation_adapter="responses_api_agents.nooa_agent.tests.test_runner:invoke_policy",
    )
    return NOOAAgent(config=agent_config(nooa=invocation.model_dump()), server_client=client)


@pytest.mark.parametrize(
    ("status", "action", "expected_action"),
    [
        (400, "default", "verify"),
        (400, "fail", "fail"),
        (400, "retry", "retry"),
        (413, "default", "verify"),
        (413, "fail", "fail"),
        (413, "retry", "retry"),
        (422, "default", "verify"),
        (422, "fail", "fail"),
        (422, "retry", "retry"),
        (401, "default", "fail"),
        (401, "verify", "verify"),
        (401, "retry", "retry"),
        (403, "default", "fail"),
        (404, "default", "fail"),
        (503, "default", "retry"),
        (503, "fail", "fail"),
        (503, "verify", "verify"),
    ],
)
async def test_model_rejection_obeys_environment_policy_and_closes(
    status: int, action: str, expected_action: str
) -> None:
    instance = rejection_agent(status)
    environment, client = _environment(instance)
    if action != "default":
        environment.config.agent_outcome_policy = AgentOutcomePolicyConfig(by_status_code={status: action})

    episode = await environment.run_request(_request())

    if expected_action == "verify":
        assert episode.failure is None
        assert episode.result.reward == 1
        response = episode.result.response
        observations = episode.result.ng_agent_observations
        assert _paths(client)[-3:] == ["/v1/agent_sessions/close", "/verify", "/close_session"]
    else:
        assert episode.result is None
        assert episode.failure.stage == "agent"
        assert episode.failure.terminal is (expected_action == "fail")
        response = episode.failure.partial_response
        observations = episode.failure.agent_observations
        assert "/verify" not in _paths(client)
        assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
    assert response.status == "failed"
    root = next(
        record
        for record in observations.records
        if isinstance(record, AgentInvocation) and record.parent_invocation_id is None
    )
    assert root.status == "failed"
    assert root.error_type == "GenerationError"
    assert orjson.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {
        "failure_kind": "model_request_error",
        "status_code": status,
    }
    assert "diagnostic mentions HTTP 500" in response.error.message
    assert NOOATerminationReason.MODEL_REJECTED in {gap.code for gap in observations.gaps}
    receipt = next(iter(instance._closed_session_records.values())).close_response
    assert receipt.agent_observations == observations
    assert receipt.resources_cookies == {"resource": "seed"}
    assert client.post.call_args_list[-1].kwargs["cookies"] == receipt.resources_cookies
    # CodeAct exhausts its three attempts before the environment applies its policy.
    assert instance.server_client.post.await_count == 3
    for call in instance.server_client.post.await_args_list:
        assert call.kwargs["url_path"] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
        assert call.kwargs["cookies"] == {}


@pytest.mark.parametrize("action", ["fail", "retry"])
@pytest.mark.parametrize("prior_attempts", [1, 2])
async def test_rejection_policy_respects_collector_attempt_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str, prior_attempts: int
) -> None:
    environment, _ = _environment(rejection_agent(413))
    environment.config.agent_outcome_policy = AgentOutcomePolicyConfig(by_status_code={413: action})
    episode = await environment.run_request(_request())
    assert episode.failure is not None
    record = _episode_record(episode.model_dump(mode="json"))
    assert "reward" not in record
    assert record["_ng_failure_terminal"] is (action == "fail")
    key = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(tmp_path / "input.jsonl"),
        output_jsonl_fpath=str(tmp_path / "output.jsonl"),
        resume_from_cache=True,
    )
    config.materialized_jsonl_fpath.write_bytes(orjson.dumps(key) + b"\n")
    Path(config.output_jsonl_fpath).write_bytes(b"")
    failures_path_for(Path(config.output_jsonl_fpath)).write_bytes(
        (orjson.dumps({**record, **key}) + b"\n") * prior_attempts
    )
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "2")

    pending, *_ = RolloutCollectionHelper()._load_from_cache(config)

    assert pending == ([{**key, ATTEMPT_INDEX_KEY_NAME: 1}] if action == "retry" and prior_attempts == 1 else [])


@pytest.mark.parametrize("status", [400, 401, 413, 422, 503])
async def test_worker_result_and_checkpoint_preserve_model_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    instance = rejection_agent(status)
    p = payload()
    p.invocation = instance.config.nooa
    monkeypatch.setattr(entrypoint, "_ModelClient", MagicMock(return_value=instance.server_client))

    artifact = await entrypoint.execute(p)
    path = tmp_path / "result.json"
    entrypoint._write_json(path, artifact)
    restored = entrypoint.SandboxResult.model_validate_json(path.read_text()).run_result()

    assert artifact.error is None
    assert artifact.response.status == "failed"
    assert artifact.model_status_code == restored.model_status_code == status
    assert restored.termination_reason is NOOATerminationReason.MODEL_REJECTED
    assert restored.termination_error == artifact.response.error.message
    assert restored.model_cookies == p.request.model_cookies
    assert restored.resource_cookies == p.request.resource_cookies
    assert restored.episode.observations == artifact.observations
    assert restored.trajectory == artifact.trajectory
    root = next(item for item in restored.trajectory.invocations if item.parent_invocation_id is None)
    assert root.status == "failed"
    assert root.error_type == "GenerationError"
    assert instance.server_client.post.await_count == 3
    checkpoint_path = tmp_path / "checkpoint.json"
    entrypoint.write_checkpoint(checkpoint_path, restored)
    checkpoint = entrypoint.SandboxCheckpoint.model_validate_json(checkpoint_path.read_text())
    assert checkpoint.model_status_code == checkpoint.run_result().model_status_code == status
    assert checkpoint.partial is True
    assert checkpoint.response.status == "incomplete"
    assert checkpoint.termination_reason is None
    assert "partial_checkpoint" in {gap.code for gap in checkpoint.observations.gaps}

    # The controller must classify the restored structured status, not parse diagnostic text.
    instance, run = agent(monkeypatch)
    run.return_value = restored
    request = await open_session(instance)
    response = await instance.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert orjson.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {
        "failure_kind": "model_request_error",
        "status_code": status,
    }
    await close(instance)


@pytest.mark.parametrize("invalid", [True, "400", 399, 600])
def test_worker_rejects_invalid_structured_model_status(invalid: object) -> None:
    with pytest.raises(ValidationError, match="model_status_code"):
        entrypoint.SandboxResult(
            observations={"source": "nooa"}, model_cookies={}, resource_cookies={}, model_status_code=invalid
        )


async def test_reported_rejection_preserves_other_response_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, run = agent(monkeypatch)
    rejected = result(reason=NOOATerminationReason.MODEL_REJECTED)
    rejected.model_status_code = 422
    rejected.episode.response.metadata = {"existing": "retained"}
    run.return_value = rejected
    request = await open_session(instance)

    response = await instance.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))

    assert response.metadata == {
        "existing": "retained",
        AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"model_request_error","status_code":422}',
        **provenance_metadata(),
    }
    assert "nooa_commit" in response.metadata
    assert rejected.episode.response.metadata == {"existing": "retained"}
    await close(instance)
