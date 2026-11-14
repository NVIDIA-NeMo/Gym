# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped metrics through the actual Gym verifier and aggregation HTTP route."""

import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.server_utils import ServerClient
from resources_servers.videomme_v2.app import (
    CohortManifest,
    VideoMMEV2ResourcesServer,
    VideoMMEV2ResourcesServerConfig,
    VideoMMEV2VerifyRequest,
)
from resources_servers.videomme_v2.task import prepare_agentic, prepare_single_turn
from resources_servers.videomme_v2.tests.test_app import REVISION, question, response
from resources_servers.videomme_v2.tests.test_metrics import group


def make_server(tmp_path, *, setting="single_turn", **updates) -> VideoMMEV2ResourcesServer:
    groups = [
        group("one", 1),
        group("two", 2, group_type="logic", group_structure="[1,[2,3],4]"),
        group("three", 3, group_type="logic", group_structure="[[1,2],3,4]"),
    ]
    values = {
        "schema_version": 2,
        "dataset_revision": REVISION,
        "setting": setting,
        "task_ids": [task_id for value in groups for task_id in value.task_ids],
        "groups": [value.model_dump(mode="json") for value in groups],
        **updates,
    }
    manifest = tmp_path / f"cohort-{setting}.json"
    manifest.write_text(json.dumps(values))
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return VideoMMEV2ResourcesServer(
        config=VideoMMEV2ResourcesServerConfig(
            name="videomme_v2", host="127.0.0.1", port=8000, entrypoint="app.py", cohort_manifest=manifest
        ),
        server_client=client,
    )


def request(server, value, index, correct=True, **response_updates) -> VideoMMEV2VerifyRequest:
    item = replace(question(value.task_ids[index]), level=value.level, group_id=value.group_id)
    if server._cohort.setting == "single_turn":
        row = prepare_single_turn(item, expected_answer="B", video_url="https://example.invalid/shapes.mp4")
    else:
        row = prepare_agentic(item, expected_answer="B", video_path="video/shapes.mp4")
    return VideoMMEV2VerifyRequest.model_validate(
        {**row, "response": response(r"\boxed{B}" if correct else r"\boxed{A}", **response_updates)}
    )


async def rows_for(server, patterns, repeat=0):
    rows = []
    for value, pattern in zip(server._cohort.groups, patterns):
        for index, correct in enumerate(pattern):
            row = (await server.verify(request(server, value, index, correct))).model_dump()
            rows.append(
                {
                    **row,
                    TASK_INDEX_KEY_NAME: server._cohort.task_ids.index(row["task_id"]),
                    ROLLOUT_INDEX_KEY_NAME: repeat,
                }
            )
    return rows


@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
async def test_primary_metrics_and_level_breakdowns(tmp_path, setting) -> None:
    server = make_server(tmp_path, setting=setting)
    patterns = [(True, True, True, False), (True, False, True, True), (False, True, True, True)]
    rows = await rows_for(server, patterns)
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=list(reversed(rows))))
    prefix = "Video_MME_v2" if setting == "single_turn" else "Video_MME_v2_agentic"
    assert result.key_metrics[f"{prefix}/raw_acc"] == 75
    assert result.key_metrics[f"{prefix}/nonlinear_score"] == pytest.approx((56.25 + 100 / 3 + 10) / 3)
    assert result.key_metrics[f"mean/{prefix}"] == 0.75
    for level, nonlinear in ((1, 56.25), (2, 100 / 3), (3, 10)):
        assert result.key_metrics[f"{prefix}/L{level}/raw_acc"] == 75
        assert result.key_metrics[f"{prefix}/L{level}/nonlinear_score"] == pytest.approx(nonlinear)
        assert result.key_metrics[f"{prefix}/L{level}/group_coverage"] == 1
    assert result.key_metrics[f"{prefix}/nonlinear_available"] is True
    assert result.key_metrics[f"{prefix}/is_complete"] is True
    assert result.agent_metrics[f"{prefix}/expected_groups"] == 3
    other = "Video_MME_v2_agentic" if setting == "single_turn" else "Video_MME_v2"
    assert f"{other}/raw_acc" not in result.key_metrics
    assert "groups" not in rows[0]["responses_create_params"]
    assert "expected_answer" not in json.dumps(rows[0]["responses_create_params"])


async def test_repeats_keep_independent_group_scores(tmp_path) -> None:
    server = make_server(tmp_path, num_repeats=2)
    first = await rows_for(server, [(True, True, False, False)] * 3)
    second = await rows_for(server, [(False, False, True, True)] * 3, repeat=1)
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=first + second))
    assert result.key_metrics["Video_MME_v2/raw_acc"] == 50
    assert result.key_metrics["Video_MME_v2/L1/nonlinear_score"] == 25
    assert result.key_metrics["Video_MME_v2/L2/nonlinear_score"] == pytest.approx(100 / 6)
    assert result.key_metrics["Video_MME_v2/L3/nonlinear_score"] == 10
    assert result.agent_metrics["Video_MME_v2/expected_groups"] == 6
    assert result.agent_metrics["Video_MME_v2/expected"] == 24


async def test_missing_results_do_not_shrink_nonlinear_denominator(tmp_path) -> None:
    server = make_server(tmp_path)
    rows = await rows_for(server, [(True,) * 4] * 3)
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows[:1]))
    assert result.key_metrics["Video_MME_v2/raw_acc"] == pytest.approx(100 / 12)
    assert result.key_metrics["Video_MME_v2/nonlinear_score"] == pytest.approx(6.25 / 3)
    assert result.key_metrics["Video_MME_v2/group_coverage"] == 0
    assert result.key_metrics["Video_MME_v2/is_complete"] is False
    assert result.key_metrics["Video_MME_v2/L1/raw_acc"] == 25
    assert result.key_metrics["Video_MME_v2/L2/raw_acc"] == 0


async def test_masked_failure_is_inconclusive_and_cannot_complete_a_group(tmp_path) -> None:
    server = make_server(tmp_path)
    rows = await rows_for(server, [(True,) * 4] * 3)
    rows[0] = (await server.verify(request(server, server._cohort.groups[0], 0, status="failed"))).model_dump()
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert result.key_metrics["Video_MME_v2/raw_acc"] == pytest.approx(100 * 11 / 12)
    assert result.key_metrics["Video_MME_v2/group_coverage"] == pytest.approx(2 / 3)
    assert result.agent_metrics["Video_MME_v2/nonlinear_complete"] is False


async def test_result_metadata_must_match_frozen_group_and_level(tmp_path) -> None:
    server = make_server(tmp_path)
    value = request(server, server._cohort.groups[0], 0)
    for updates, message in (({"level": 2}, "level"), ({"group_id": "other"}, "group")):
        payload = value.model_dump()
        payload["verifier_metadata"].update(updates)
        with pytest.raises(ValueError, match=message):
            await server.verify(VideoMMEV2VerifyRequest.model_validate(payload))
        row = (await server.verify(value)).model_dump()
        row["verifier_metadata"].update(updates)
        with pytest.raises(ValueError, match=message):
            await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[row]))


async def test_actual_http_aggregation_exposes_both_metrics(tmp_path) -> None:
    server = make_server(tmp_path)
    rows = await rows_for(server, [(True,) * 4] * 3)
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/aggregate_metrics", json={"verify_responses": rows})
    assert result.status_code == 200
    assert result.json()["key_metrics"]["Video_MME_v2/raw_acc"] == 100
    assert result.json()["key_metrics"]["Video_MME_v2/nonlinear_score"] == 100


def test_v2_requires_group_partition_but_legacy_v1_still_loads() -> None:
    legacy = {"dataset_revision": REVISION, "task_ids": ["one"]}
    assert CohortManifest(**legacy).schema_version == 1
    with pytest.raises(ValidationError, match="complete question groups"):
        CohortManifest(**legacy, schema_version=2)
    value = group()
    valid = {**legacy, "task_ids": value.task_ids, "schema_version": 2, "groups": [value]}
    assert CohortManifest(**valid).groups[0] == value
    with pytest.raises(ValidationError, match="partition"):
        CohortManifest(**{**valid, "task_ids": [*value.task_ids, "missing"]})


@pytest.mark.parametrize("levels", [{"one": 1.0}, {"one": True}, {"one": 4}, {"other": 1}])
def test_cohort_rejects_invalid_level_maps(levels) -> None:
    with pytest.raises(ValidationError):
        CohortManifest(dataset_revision=REVISION, task_ids=["one"], task_levels=levels)


async def test_legacy_level_map_gives_raw_breakdowns_without_fake_group_score(tmp_path) -> None:
    server = make_server(
        tmp_path,
        schema_version=1,
        groups=[],
        task_ids=["a", "b", "c"],
        task_levels={"a": 1, "b": 2, "c": 3},
    )
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    for prefix in ("Video_MME_v2", "Video_MME_v2/L1", "Video_MME_v2/L2", "Video_MME_v2/L3"):
        assert result.key_metrics[f"{prefix}/raw_acc"] == 0
        assert result.key_metrics[f"{prefix}/nonlinear_available"] is False
        assert f"{prefix}/nonlinear_score" not in result.key_metrics
