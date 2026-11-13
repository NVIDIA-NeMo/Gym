# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Gym's real response schemas, verifier routes, cohort and mask contract."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.video_mcqa.tests.helpers import grade_failed_run_and_replay
from resources_servers.videomme_v2.app import (
    CohortManifest,
    VideoMMEV2ResourcesServer,
    VideoMMEV2ResourcesServerConfig,
    VideoMMEV2VerifyRequest,
    final_assistant_text,
)
from resources_servers.videomme_v2.scoring import EXECUTION_FAILURE_METADATA_KEY
from resources_servers.videomme_v2.task import VideoQuestion, prepare_agentic, prepare_single_turn


REVISION = "a" * 40


def question(identity: str = "shape-1") -> VideoQuestion:
    return VideoQuestion(
        task_id=identity,
        dataset_source="https://example.invalid/invented-shapes",
        dataset_revision=REVISION,
        video_id="shapes",
        video_source="https://example.invalid/shapes.mp4",
        video_sha256="b" * 64,
        question="Which shape moves?",
        options=("Circle", "Square", "Triangle"),
    )


def response(text: str | None = r"\boxed{B}", **updates: object) -> NeMoGymResponse:
    output = (
        []
        if text is None
        else [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    )
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "object": "response",
            "model": "fixture",
            "output": output,
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            **updates,
        }
    )


def body(text: str | None = r"\boxed{B}", *, identity: str = "shape-1", **updates: object) -> VideoMMEV2VerifyRequest:
    row = prepare_single_turn(question(identity), expected_answer="B", video_url="https://example.invalid/shapes.mp4")
    return VideoMMEV2VerifyRequest.model_validate({**row, "response": response(text).model_dump(), **updates})


@pytest.fixture
def server(tmp_path) -> VideoMMEV2ResourcesServer:
    manifest = tmp_path / "cohort.json"
    manifest.write_text(json.dumps({"dataset_revision": REVISION, "task_ids": ["shape-1", "shape-2"]}))
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return VideoMMEV2ResourcesServer(
        config=VideoMMEV2ResourcesServerConfig(
            name="videomme_v2", host="127.0.0.1", port=8000, entrypoint="app.py", cohort_manifest=manifest
        ),
        server_client=client,
    )


@pytest.mark.parametrize(
    ("text", "outcome", "predicted", "reward"),
    [
        (r"\boxed{B}", "correct", "B", 1.0),
        (r"\boxed{A}", "incorrect", "A", 0.0),
        (None, "no_answer", None, 0.0),
        ("B", "format_error", None, 0.0),
        (r"\boxed{D}", "format_error", None, 0.0),
        (r"\boxed{A} or \boxed{B}", "ambiguous_answer", None, 0.0),
    ],
)
async def test_verify(server, text, outcome, predicted, reward) -> None:
    result = await server.verify(body(text))
    assert (result.outcome, result.predicted_answer, result.reward) == (outcome, predicted, reward)
    assert result.evaluation_completed and not result.mask_sample


async def test_failed_model_is_not_an_incorrect_answer(server) -> None:
    failed = response(r"\boxed{B}", status="failed", error={"code": "server_error", "message": "fixture"})
    result = await server.verify(body(response=failed.model_dump()))
    assert result.outcome == "infrastructure_failure"
    assert result.reward == 0.0 and result.mask_sample and not result.evaluation_completed
    assert result.predicted_answer is None


async def test_tool_call_invalidates_single_turn_answer(server) -> None:
    generated = response()
    raw = generated.model_dump()
    raw["output"].append(
        {"id": "call", "type": "function_call", "call_id": "call", "name": "unavailable_tool", "arguments": "{}"}
    )
    result = await server.verify(body(response=raw))
    assert result.outcome == "execution_failure" and result.reward == 0.0 and not result.mask_sample


@pytest.fixture
def agentic_server(tmp_path, server) -> VideoMMEV2ResourcesServer:
    manifest = tmp_path / "agentic-cohort.json"
    manifest.write_text(
        json.dumps({"dataset_revision": REVISION, "setting": "agentic", "task_ids": ["shape-1", "shape-2"]})
    )
    return VideoMMEV2ResourcesServer(
        config=server.config.model_copy(update={"cohort_manifest": manifest}),
        server_client=server.server_client,
    )


def agentic_body() -> VideoMMEV2VerifyRequest:
    row = prepare_agentic(question(), expected_answer="B", video_path="video/shapes.mp4")
    generated = response().model_dump()
    generated["output"].insert(
        0, {"id": "call", "type": "function_call", "call_id": "call", "name": "watch_video", "arguments": "{}"}
    )
    return VideoMMEV2VerifyRequest.model_validate({**row, "response": generated})


async def test_agentic_tool_calls_do_not_invalidate_a_final_answer(agentic_server) -> None:
    result = await agentic_server.verify(agentic_body())
    assert result.outcome == "correct" and result.reward == 1.0 and result.evaluation_completed
    assert result.setting == "agentic" and result.predicted_answer == "B"


@pytest.mark.parametrize(
    "updates",
    [{}, {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}],
    ids=["failed_run", "length_limited"],
)
async def test_failed_agent_run_stays_zero_when_its_saved_answer_is_replayed(agentic_server, updates) -> None:
    row = prepare_agentic(question(), expected_answer="B", video_path="video/shapes.mp4")
    failed = response(metadata={EXECUTION_FAILURE_METADATA_KEY: "true"}, **updates).model_dump()
    saved, replayed = await grade_failed_run_and_replay(agentic_server, VideoMMEV2VerifyRequest, row, failed)
    assert final_assistant_text(NeMoGymResponse.model_validate(saved["response"])) == r"\boxed{B}"
    for result in (saved, replayed.model_dump()):
        assert (result["outcome"], result["predicted_answer"], result["reward"]) == ("execution_failure", None, 0)
        assert result["evaluation_completed"] and not result["mask_sample"]
    plain = response(**updates).model_dump()
    assert (
        await agentic_server.verify(VideoMMEV2VerifyRequest.model_validate({**row, "response": plain}))
    ).outcome == ("correct")


async def test_setting_cannot_be_mixed_with_another_cohort(server, agentic_server) -> None:
    with pytest.raises(ValueError, match="setting"):
        await server.verify(agentic_body())
    with pytest.raises(ValueError, match="setting"):
        await agentic_server.verify(body())


async def test_agentic_uses_a_separate_metric_namespace(agentic_server) -> None:
    measured = (await agentic_server.verify(agentic_body())).model_dump()
    measured.update({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0})
    metrics = await agentic_server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[measured]))
    assert metrics.key_metrics["mean/Video_MME_v2_agentic"] == 0.5
    assert metrics.key_metrics["Video_MME_v2_agentic/coverage"] == 0.5
    assert metrics.key_metrics["Video_MME_v2_agentic/is_complete"] is False
    assert "mean/Video_MME_v2" not in metrics.key_metrics


def test_reasoning_and_previous_messages_are_not_final_evidence() -> None:
    reasoning = {"id": "thinking", "type": "reasoning", "summary": [{"type": "summary_text", "text": r"\boxed{B}"}]}
    value = response(None, output=[reasoning])
    assert final_assistant_text(value) == ""
    early = response(r"\boxed{B}").model_dump()["output"]
    late = response(r"\boxed{A}").model_dump()["output"]
    assert final_assistant_text(response(output=[reasoning, *early, *late])) == r"\boxed{A}"


async def test_reverify_is_idempotent(server) -> None:
    first = await server.verify(body())
    second = await server.verify(VideoMMEV2VerifyRequest.model_validate(first.model_dump()))
    assert first.model_dump() == second.model_dump()


async def test_incomplete_generation_can_only_score_its_actual_final_text(server) -> None:
    result = await server.verify(
        body(
            None,
            response=response(
                None, status="incomplete", incomplete_details={"reason": "max_output_tokens"}
            ).model_dump(),
        )
    )
    assert result.outcome == "no_answer" and result.reward == 0


async def test_wrong_cohort_is_rejected(server) -> None:
    with pytest.raises(ValueError, match="frozen"):
        await server.verify(body(identity="foreign"))
    row = body().model_dump()
    row["verifier_metadata"]["dataset_revision"] = "c" * 40
    with pytest.raises(ValueError, match="frozen"):
        await server.verify(VideoMMEV2VerifyRequest.model_validate(row))
    with pytest.raises(ValidationError, match="IDs differ"):
        body(task_id="other")


async def test_fixed_cohort_missing_and_masked_do_not_inflate_score(server) -> None:
    correct = (await server.verify(body())).model_dump()
    correct.update({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0})
    partial = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[correct]))
    assert partial.key_metrics["mean/Video_MME_v2"] == 0.5
    assert partial.key_metrics["Video_MME_v2/coverage"] == 0.5
    assert partial.key_metrics["Video_MME_v2/is_complete"] is False
    failed = (
        await server.verify(body(identity="shape-2", response=response(None, status="failed").model_dump()))
    ).model_dump()
    failed.update({TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0})
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[correct, failed]))
    assert result.agent_metrics["Video_MME_v2/returned"] == 2
    assert result.key_metrics["mean/Video_MME_v2"] == 0.5
    assert result.agent_metrics["Video_MME_v2/outcomes/infrastructure_failure"] == 1


async def test_empty_aggregate_is_provisional_not_a_finished_zero(server) -> None:
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    assert result.key_metrics["Video_MME_v2/coverage"] == 0
    assert result.agent_metrics["Video_MME_v2/missing"] == 2
    assert result.key_metrics["Video_MME_v2/is_complete"] is False


async def test_completed_cohort_and_repeats(server) -> None:
    server._cohort = server._cohort.model_copy(update={"num_repeats": 2})
    results = []
    for repeat in (0, 1):
        for index, identity in enumerate(("shape-1", "shape-2")):
            item = (
                await server.verify(body(r"\boxed{B}" if repeat == 0 else r"\boxed{A}", identity=identity))
            ).model_dump()
            results.append({**item, TASK_INDEX_KEY_NAME: index, ROLLOUT_INDEX_KEY_NAME: repeat})
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=results))
    assert result.agent_metrics["Video_MME_v2/expected"] == 4
    assert result.key_metrics["mean/Video_MME_v2"] == 0.5
    assert result.key_metrics["Video_MME_v2/is_complete"] is True


async def test_bad_aggregation_inputs_are_not_silently_merged(server) -> None:
    item = (await server.verify(body())).model_dump()
    for rows in (
        [item, item],
        [{**item, ROLLOUT_INDEX_KEY_NAME: 1}],
        [{**item, ROLLOUT_INDEX_KEY_NAME: True}],
        [{**item, "task_id": "wrong"}],
        [{**item, "setting": "agentic"}],
        [{**item, "answer_policy": "boxed_or_letter_v1"}],
        [{**item, "verifier_metadata": {**item["verifier_metadata"], "dataset_revision": "c" * 40}}],
    ):
        with pytest.raises(ValueError):
            await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))


@pytest.mark.parametrize("ids", [[], ["same", "same"], [" "], [""]])
def test_invalid_frozen_ids(ids) -> None:
    with pytest.raises(ValidationError):
        CohortManifest(dataset_revision=REVISION, task_ids=ids)


def test_real_http_routes_and_private_metadata(server) -> None:
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/verify", json=body().model_dump())
        assert result.status_code == 200 and result.json()["reward"] == 1
        assert client.get("/reverify_mode").json() == "stateless"
        invalid = body().model_dump()
        invalid["task_id"] = "different"
        assert client.post("/verify", json=invalid).status_code == 422
        params = result.json()["responses_create_params"]
        assert "expected_answer" not in json.dumps(params)
        assert "dataset_revision" not in json.dumps(params)


def test_stateless_session_protocol(server) -> None:
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest(
            resources_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout", attempt=0),
            task_id=TaskId(taskset="invented", task_id="shape-1"),
            task_data={},
        ),
        keeps_state=False,
    )
