# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_recovery import IncompleteEvaluationError
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client


def response(output=None):
    return NeMoGymResponse(
        id="resp_saved",
        object="response",
        created_at=0,
        model="controlled",
        output=[] if output is None else output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(mode="json")


@pytest.mark.parametrize("slot", ["response", "_ng_failure_response", "_ng_failure_partial_response"])
def test_retained_generation_is_structural_not_placeholder_presence(slot):
    from nemo_gym.rollout_recovery import has_saved_generation

    assert has_saved_generation({slot: response()})
    for placeholder in (
        {},
        {"error": "failed"},
        {"output": []},
        {"status": "failed", "output": []},
        response() | {"status": "failed", "error": {"code": "server_error", "message": "diagnostic only"}},
    ):
        assert not has_saved_generation({slot: placeholder})


@pytest.mark.parametrize("empty_output", [False, True])
async def test_native_cleanup_partial_response_survives_collection(tmp_path, monkeypatch, empty_output):
    import httpx
    from omegaconf import OmegaConf

    from tests.single_agent_turn_helpers import _environment_server, _request, _Response

    environment, dependencies = _environment_server()
    if empty_output:
        dependencies.responses[2] = _Response(response())
    dependencies.responses = dependencies.responses[:3] + [
        TimeoutError("controlled lost agent cleanup reply"),
        _Response({"resources_session_id": "resource"}),
    ]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
    output = artifacts / "rollouts.jsonl"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        environment_server_routes={"test:source": "environment"},
        route_failures_to_sidecar=True,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment.setup_webserver()), base_url="http://environment"
    ) as http:

        async def post(**kwargs):
            result = await http.post("/run", json=kwargs["json"])
            assert result.status_code == 200
            return FakeResponse(result.status_code, result.json())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(
            {"environment": {"environment_servers": {"single_agent_turn": {}}}}
        )
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
        with pytest.raises(IncompleteEvaluationError):
            await RolloutCollectionHelper().run_from_config(config)
        store = RolloutStore.read(output)
        assert store.pending() == []
        failure = store.failures()[0]
        assert "_ng_failure_partial_response" in failure
        assert has_actual_generation(failure)
        try:
            await RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": True}))
        except IncompleteEvaluationError:
            pass
        assert client.post.await_count == 1
        assert len([call for call in dependencies.calls if call[1].endswith("/v1/responses")]) == 1


def has_actual_generation(row):
    from nemo_gym.rollout_recovery import has_saved_generation

    return has_saved_generation(row)


@pytest.mark.parametrize("action", ["retry", "fail"])
@pytest.mark.parametrize(
    "output",
    [
        [],
        [
            {
                "type": "message",
                "id": "partial",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "partial diagnostic", "annotations": []}],
            }
        ],
    ],
)
async def test_failed_agent_partial_output_keeps_the_selected_policy(tmp_path, monkeypatch, action, output):
    import httpx
    from omegaconf import OmegaConf

    from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome, AgentOutcomePolicyConfig
    from tests.single_agent_turn_helpers import _environment_server, _request, _Response

    environment, dependencies = _environment_server()
    environment.config.agent_outcome_policy = AgentOutcomePolicyConfig(by_status_code={503: action})
    failed = response(output) | {
        "status": "failed",
        "error": {"code": "server_error", "message": "controlled model failure"},
        "metadata": {
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(
                failure_kind="model_request_error", status_code=503
            ).model_dump_json()
        },
    }
    replies = dependencies.responses.copy()
    replies[2] = _Response(failed)
    del replies[4]
    dependencies.responses = replies + replies
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
    output_path = artifacts / "rollouts.jsonl"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output_path),
        environment_server_routes={"test:source": "environment"},
        route_failures_to_sidecar=True,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment.setup_webserver()), base_url="http://environment"
    ) as http:

        async def post(**kwargs):
            result = await http.post("/run", json=kwargs["json"])
            assert result.status_code == 200
            return FakeResponse(result.status_code, result.json())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(
            {"environment": {"environment_servers": {"single_agent_turn": {}}}}
        )
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
        for resume in [False, True]:
            try:
                await RolloutCollectionHelper().run_from_config(
                    config.model_copy(update={"resume_from_cache": resume})
                )
            except IncompleteEvaluationError:
                pass
            store = RolloutStore.read(output_path)
            failure = store.failures()[0]
            assert failure["_ng_failure_terminal"] is (action == "fail")
            assert failure["_ng_failure_partial_response"] == failed
            assert "reward" not in failure
            assert bool(store.pending()) is (action == "retry")
        assert client.post.await_count == (2 if action == "retry" else 1)
        assert not any(path == "/verify" for _, path, _ in dependencies.calls)


@pytest.mark.parametrize("route", [False, True])
@pytest.mark.parametrize("status,blocked", [(200, True), (201, True), (299, True), (503, False), (None, False)])
async def test_selected_model_body_delivery_blocks_collection_resume(tmp_path, monkeypatch, route, status, blocked):
    import json

    from nemo_gym.typed_exceptions import NgFailureEvent

    failure = NgFailureEvent(
        boundary="model",
        failure_kind="model_request_error",
        failure_reason="Controlled model request failure",
        operation="model_request",
        operation_id="operation",
        attempt_id="operation-1",
        http_status=status,
        request_upload_complete=False if status is None else None,
    )
    from aiohttp import ClientResponseError

    from tests.unit_tests.test_rollout_collection import http_error

    error = http_error(500)
    error.__cause__ = failure
    post = AsyncMock(side_effect=error)
    client = install_fake_server_client(monkeypatch, post)
    monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps({"agent_ref": {"name": "my_agent"}, "responses_create_params": {"input": []}}) + "\n")
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        route_failures_to_sidecar=route,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    for resume in (False, True):
        try:
            await RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": resume}))
        except (IncompleteEvaluationError, ClientResponseError):
            pass
    assert post.await_count == (1 if blocked or not route else 2)
    store = RolloutStore.read(output)
    # Default-mode diagnostic failures are independently uncertain-delivery guarded.
    assert bool(store.pending()) is (not blocked and route)
    if route:
        rows = store.failures()
        assert all(failure.event_id in row["_ng_failure_events"] for row in rows)
        assert all("reward" not in row for row in rows)


def test_unselected_body_failure_does_not_override_selected_preanswer_infrastructure():
    from nemo_gym.rollout_recovery import has_unclassified_execution_failure

    events = {
        "old": {
            "boundary": "model",
            "failure_kind": "model_request_error",
            "operation": "model_request",
            "http_status": 200,
        },
        "current": {
            "boundary": "model",
            "failure_kind": "model_request_error",
            "operation": "model_request",
            "http_status": 503,
        },
        "wrapper": {"boundary": "agent", "failure_kind": "agent_route_failed", "cause_id": "current"},
    }
    row = {"_ng_failure_class": "agent_run_error", "_ng_failure_events": events, "_ng_failure_id": "wrapper"}
    assert not has_unclassified_execution_failure(row)
    events["wrapper"]["cause_id"] = "old"
    assert has_unclassified_execution_failure(row)
