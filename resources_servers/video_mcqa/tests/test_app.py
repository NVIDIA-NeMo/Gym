# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.lvbench.app import LVBenchResourcesServer
from resources_servers.video_mcqa.app import (
    AccuracyCohort,
    VideoMCQAResourcesServerConfig,
    VideoMCQAVerifyRequest,
    final_assistant_text,
)
from resources_servers.video_mcqa.prepare import prepare_dataset
from resources_servers.video_mcqa.scoring import EXECUTION_FAILURE_METADATA_KEY
from resources_servers.video_mcqa.tests.helpers import (
    REVISION,
    grade_failed_run_and_replay,
    lv_record,
    media,
    response,
    v1_record,
)
from resources_servers.videomme_v1.app import VideoMMEV1ResourcesServer


def make_server(
    tmp_path, *, benchmark="videomme_v1", setting="single_turn", num_repeats=1, answer_policy="boxed_only_v1"
):
    records = (
        [v1_record(), v1_record("two", duration="long", domain="Other shapes")]
        if benchmark == "videomme_v1"
        else [lv_record(), lv_record("two", question_type=["reasoning"])]
    )
    rows, cohort = prepare_dataset(
        records,
        benchmark=benchmark,
        dataset_revision=REVISION,
        media=media(source="/data/source.mp4", sandbox_path="video/shapes.mp4"),
        setting=setting,
        num_repeats=num_repeats,
        answer_policy=answer_policy,
    )
    path = tmp_path / f"{benchmark}-{setting}-{num_repeats}-{answer_policy}.json"
    path.write_text(cohort.model_dump_json())
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    cls = VideoMMEV1ResourcesServer if benchmark == "videomme_v1" else LVBenchResourcesServer
    server = cls(
        config=VideoMCQAResourcesServerConfig(
            name=benchmark, host="127.0.0.1", port=8000, entrypoint="app.py", cohort_manifest=path
        ),
        server_client=client,
    )
    return server, rows, cohort


def request(row, text=r"\boxed{B}", **updates):
    return VideoMCQAVerifyRequest.model_validate({**copy.deepcopy(row), "response": response(text), **updates})


async def measured(server, row, text=r"\boxed{B}", *, index=0, repeat=0, **updates):
    result = (await server.verify(request(row, text, **updates))).model_dump()
    result[TASK_INDEX_KEY_NAME] = index
    result[ROLLOUT_INDEX_KEY_NAME] = repeat
    return result


@pytest.mark.parametrize("benchmark", ["videomme_v1", "lvbench"])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
@pytest.mark.parametrize(
    "text,outcome,predicted,reward",
    [
        (r"\boxed{B}", "correct", "B", 1),
        (r"\boxed{C}", "incorrect", "C", 0),
        (None, "no_answer", None, 0),
        ("B", "format_error", None, 0),
        (r"\boxed{E}", "format_error", None, 0),
        (r"\boxed{A} or \boxed{B}", "ambiguous_answer", None, 0),
    ],
)
async def test_verify_and_reverify(tmp_path, benchmark, setting, text, outcome, predicted, reward) -> None:
    server, rows, _ = make_server(tmp_path, benchmark=benchmark, setting=setting)
    result = await server.verify(request(rows[0], text))
    assert (result.outcome, result.predicted_answer, result.reward) == (outcome, predicted, reward)
    assert result.evaluation_completed and not result.mask_sample
    assert await server.verify(VideoMCQAVerifyRequest.model_validate(result.model_dump())) == result


@pytest.mark.parametrize("benchmark", ["videomme_v1", "lvbench"])
async def test_model_failure_is_masked_and_cannot_be_called_correct(tmp_path, benchmark) -> None:
    server, rows, _ = make_server(tmp_path, benchmark=benchmark)
    failed = response(status="failed", error={"code": "server_error", "message": "fixture"})
    result = await measured(server, rows[0], response=failed)
    assert result["outcome"] == "infrastructure_failure" and result["mask_sample"]
    assert result["failure_kind"] == f"{benchmark}:infrastructure_failure"
    assert result["predicted_answer"] is None and result["reward"] == 0
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[result]))
    prefix = "Video_MME" if benchmark == "videomme_v1" else "LVBench"
    assert metrics.key_metrics[f"{prefix}/coverage"] == 0
    assert metrics.key_metrics[f"{prefix}/raw_acc"] == 0
    assert metrics.agent_metrics[f"{prefix}/expected"] == 2
    assert metrics.agent_metrics[f"{prefix}/missing"] == 1


@pytest.mark.parametrize("benchmark", ["videomme_v1", "lvbench"])
@pytest.mark.parametrize(
    "updates",
    [{}, {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}],
    ids=["failed_run", "length_limited"],
)
async def test_failed_agent_run_stays_zero_when_its_saved_answer_is_replayed(tmp_path, benchmark, updates) -> None:
    server, rows, _ = make_server(tmp_path, benchmark=benchmark, setting="agentic")
    failed = response(metadata={EXECUTION_FAILURE_METADATA_KEY: "true"}, **updates)
    saved, replayed = await grade_failed_run_and_replay(server, VideoMCQAVerifyRequest, rows[0], failed)
    # The saved row keeps the correct-looking answer for inspection.
    assert final_assistant_text(NeMoGymResponse.model_validate(saved["response"])) == r"\boxed{B}"
    for result in (saved, replayed.model_dump()):
        assert (result["outcome"], result["predicted_answer"], result["reward"]) == ("execution_failure", None, 0)
        assert result["evaluation_completed"] and not result["mask_sample"]
    # Without the agent's marker the same answer is correct; a failed model call stays inconclusive.
    assert (await server.verify(request(rows[0], response=response(**updates)))).outcome == "correct"
    infrastructure = await server.verify(
        request(rows[0], response={**failed, "status": "failed", "error": {"code": "server_error", "message": "x"}})
    )
    assert infrastructure.outcome == "infrastructure_failure" and infrastructure.mask_sample


@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
async def test_tool_calls_are_invalid_only_for_single_turn(tmp_path, setting) -> None:
    server, rows, _ = make_server(tmp_path, setting=setting)
    generated = response()
    generated["output"].insert(
        0, {"id": "call", "type": "function_call", "call_id": "call", "name": "watch_video", "arguments": "{}"}
    )
    result = await server.verify(request(rows[0], response=generated))
    assert result.outcome == ("execution_failure" if setting == "single_turn" else "correct")
    assert result.reward == (0 if setting == "single_turn" else 1) and not result.mask_sample


@pytest.mark.parametrize("benchmark", ["videomme_v1", "lvbench"])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
async def test_metrics_namespaces_and_frozen_denominators(tmp_path, benchmark, setting) -> None:
    server, rows, _ = make_server(tmp_path, benchmark=benchmark, setting=setting)
    result = await measured(server, rows[0])
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[result]))
    prefix = ("Video_MME" if benchmark == "videomme_v1" else "LVBench") + ("_agentic" if setting == "agentic" else "")
    assert metrics.key_metrics[f"{prefix}/raw_acc"] == 50
    assert metrics.key_metrics[f"mean/{prefix}"] == 0.5
    assert metrics.key_metrics[f"{prefix}/coverage"] == 0.5
    assert metrics.key_metrics[f"{prefix}/is_complete"] is False
    assert metrics.agent_metrics[f"{prefix}/missing"] == 1
    assert not any("nonlinear" in key or "Video_MME_v2" in key for key in metrics.key_metrics)
    if benchmark == "videomme_v1":
        assert metrics.key_metrics[f"{prefix}/duration/short/raw_acc"] == 100
        assert metrics.key_metrics[f"{prefix}/duration/long/raw_acc"] == 0
        assert f"{prefix}/duration/medium/raw_acc" not in metrics.key_metrics
        assert metrics.key_metrics[f"{prefix}/domain/Invented%20shapes/raw_acc"] == 100
    else:
        assert metrics.key_metrics[f"{prefix}/task_type/entity%20recognition/raw_acc"] == 100
        assert metrics.key_metrics[f"{prefix}/task_type/reasoning/raw_acc"] == 50
    if setting == "agentic":
        assert not any(key.startswith(f"{prefix.removesuffix('_agentic')}/") for key in metrics.key_metrics)


async def test_lv_overlap_is_not_a_macro_average(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, benchmark="lvbench")
    results = [await measured(server, rows[0]), await measured(server, rows[1], r"\boxed{A}", index=1)]
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=results))
    assert metrics.key_metrics["LVBench/raw_acc"] == 50
    assert metrics.key_metrics["LVBench/task_type/entity%20recognition/raw_acc"] == 100
    assert metrics.key_metrics["LVBench/task_type/reasoning/raw_acc"] == 50
    assert metrics.agent_metrics["LVBench/expected"] == 2
    assert metrics.agent_metrics["LVBench/task_type/reasoning/expected"] == 2
    assert metrics.key_metrics["LVBench/is_complete"] is True


async def test_repeats_are_frozen_and_not_deduplicated(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, num_repeats=2)
    results = [await measured(server, rows[0]), await measured(server, rows[0], r"\boxed{A}", repeat=1)]
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=results))
    assert metrics.agent_metrics["Video_MME/expected"] == 4
    assert metrics.key_metrics["Video_MME/raw_acc"] == 25
    assert metrics.agent_metrics["Video_MME/duration/short/expected"] == 2
    assert metrics.key_metrics["Video_MME/duration/short/raw_acc"] == 50


async def test_no_returned_results_is_provisional_not_perfect(tmp_path) -> None:
    server, _, _ = make_server(tmp_path)
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    assert metrics.key_metrics["Video_MME/raw_acc"] == 0
    assert metrics.key_metrics["Video_MME/coverage"] == 0
    assert metrics.agent_metrics["Video_MME/missing"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_revision", "c" * 40),
        ("dataset_source", "https://example.invalid/other"),
        ("task_id", "videomme_v1:unknown"),
        ("level", 1),
        ("group_id", "group"),
        ("option_count", 3),
    ],
)
async def test_foreign_or_v2_metadata_is_rejected(tmp_path, field, value) -> None:
    server, rows, _ = make_server(tmp_path)
    row = copy.deepcopy(rows[0])
    row["verifier_metadata"][field] = value
    if field == "task_id":
        row["task_id"] = value
    with pytest.raises(ValueError):
        await server.verify(request(row))


async def test_wrong_setting_and_mismatched_top_level_id(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path)
    with pytest.raises(ValueError, match="setting"):
        await server.verify(request(rows[0], setting="agentic"))
    with pytest.raises(ValidationError, match="task IDs"):
        request(rows[0], task_id="other")


@pytest.mark.parametrize(
    "field,value",
    [
        (ROLLOUT_INDEX_KEY_NAME, True),
        (ROLLOUT_INDEX_KEY_NAME, -1),
        (ROLLOUT_INDEX_KEY_NAME, 1),
        ("task_id", "other"),
        ("setting", "agentic"),
        ("answer_policy", "boxed_or_letter_v1"),
        ("reward", 0.5),
        ("evaluation_completed", False),
    ],
)
async def test_aggregation_rejects_corrupt_or_foreign_results(tmp_path, field, value) -> None:
    server, rows, _ = make_server(tmp_path)
    result = await measured(server, rows[0])
    result[field] = value
    with pytest.raises(ValueError):
        await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[result]))


async def test_duplicate_results_fail_before_subset_filtering(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path)
    result = await measured(server, rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[result, result]))


@pytest.mark.parametrize(
    "updates",
    [
        {"dataset_source": "other"},
        {"task_ids": ["other:one"]},
        {"task_ids": ["videomme_v1:"]},
        {"task_ids": ["videomme_v1:one", "videomme_v1:one"]},
        {"task_subsets": {}},
        {"num_repeats": True},
        {"num_repeats": 0},
        {"benchmark": "longvideobench"},
        {"setting": "other"},
        {"schema_version": 2},
    ],
)
def test_invalid_frozen_cohort(tmp_path, updates) -> None:
    _, _, cohort = make_server(tmp_path)
    with pytest.raises(ValidationError):
        AccuracyCohort.model_validate({**cohort.model_dump(), **updates})


def test_benchmark_server_cannot_load_another_benchmarks_cohort(tmp_path) -> None:
    server, _, _ = make_server(tmp_path)
    with pytest.raises(ValueError, match="different benchmark"):
        LVBenchResourcesServer(config=server.config, server_client=server.server_client)


def test_real_http_routes_and_model_visible_projection(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, benchmark="lvbench")
    client = TestClient(server.setup_webserver())
    result = client.post("/verify", json=request(rows[0]).model_dump())
    assert result.status_code == 200 and result.json()["reward"] == 1
    assert "expected_answer" not in str(result.json()["responses_create_params"])
    measured_result = {**result.json(), TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
    aggregate = client.post("/aggregate_metrics", json={"verify_responses": [measured_result]})
    assert aggregate.status_code == 200 and aggregate.json()["key_metrics"]["LVBench/raw_acc"] == 50


async def test_opt_in_bare_letter_policy(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, answer_policy="boxed_or_letter_v1")
    result = await server.verify(request(rows[0], "B"))
    assert result.reward == 1 and result.answer_policy == "boxed_or_letter_v1"


def test_only_last_final_assistant_message_is_graded() -> None:
    early = response()["output"]
    late = response(r"\boxed{A}")["output"]
    reasoning = {"id": "thinking", "type": "reasoning", "summary": [{"type": "summary_text", "text": r"\boxed{B}"}]}
    assert final_assistant_text(NeMoGymResponse.model_validate(response(output=[reasoning]))) == ""
    combined = NeMoGymResponse.model_validate(response(output=[*early, reasoning, *late]))
    assert final_assistant_text(combined) == r"\boxed{A}"
