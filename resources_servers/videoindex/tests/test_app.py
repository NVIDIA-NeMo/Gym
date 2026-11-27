# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-level verify and fixed-cohort 4-capability-group aggregation tests for VideoIndexResourcesServer."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest, ModelServerRef
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig, VideoMCQAVerifyRequest
from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.tests.helpers import response
from resources_servers.videoindex.app import VERIFIER_FIXTURE, VideoIndexResourcesServer
from resources_servers.videoindex.dataset import CAPABILITY_GROUPS, RELEASES
from resources_servers.videoindex.prepare import prepare_videoindex_dataset
from resources_servers.videoindex.scoring import LETTERS, VIDEOINDEX_RULE_POLICY


JUDGE = ModelServerRef(type="responses_api_models", name="video_mcqa_answer_judge")


def synthetic_records() -> list[dict[str, object]]:
    """Four synthetic items — one per capability group — covering 2, 4, 8, and 10 options."""
    option_counts = [2, 4, 8, 10]
    answer_indices = [1, 2, 7, 9]
    records: list[dict[str, object]] = []
    for index, (group, count, ans_idx) in enumerate(zip(CAPABILITY_GROUPS, option_counts, answer_indices)):
        options = [f"Synthetic choice {i + 1}" for i in range(count)]
        records.append(
            {
                "item_id": f"synth_{index}",
                "benchmark": "SyntheticBench",
                "capability_group": group,
                "question": f"Synthetic question {index}?",
                "options": options,
                "answer": LETTERS[ans_idx],
                "answer_idx": ans_idx,
                "video": f"videos/clip_{index}.mp4",
                "video_id": f"clip_{index}",
            }
        )
    return records


def synthetic_media(records: list[dict[str, object]], *, setting: str) -> dict[str, VideoSource]:
    media: dict[str, VideoSource] = {}
    for index, record in enumerate(records):
        rel = str(record["video"])
        digest = f"{index:064x}"
        if setting == "single_turn":
            media[rel] = VideoSource(source_url=rel, sha256=digest, video_url=f"file:///data/{rel}")
        else:
            media[rel] = VideoSource(source_url=rel, sha256=digest, source=f"/data/{rel}", sandbox_path=rel)
    return media


def scripted_judge(*contents: str | None) -> tuple[MagicMock, list[dict]]:
    calls: list[dict] = []
    replies = iter(contents)

    async def post(*, server_name: str, url_path: str, json: object, **kwargs: object) -> SimpleNamespace:
        calls.append({"server_name": server_name, "url_path": url_path, "json": json})
        content = next(replies)
        payload = {
            "id": "judge-reply",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=json_module.dumps(payload).encode()))

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = post
    return client, calls


json_module = json


def make_server(
    tmp_path: Path,
    *,
    benchmark: str = "videoindex_v1_0",
    setting: str = "single_turn",
    answer_policy: str = VIDEOINDEX_RULE_POLICY,
    num_repeats: int = 1,
    judge_replies: tuple[str | None, ...] = (),
    judge: ModelServerRef | None = None,
) -> tuple[VideoIndexResourcesServer, list[dict[str, object]], list[dict]]:
    records = synthetic_records()
    release = RELEASES[benchmark]
    rows, cohort = prepare_videoindex_dataset(
        records,
        benchmark=benchmark,
        dataset_revision=release.dataset_revision,
        media=synthetic_media(records, setting=setting),
        setting=setting,
        answer_policy=answer_policy,
        num_repeats=num_repeats,
    )
    manifest = tmp_path / f"{benchmark}-{setting}-{answer_policy}.json"
    manifest.write_text(cohort.model_dump_json(), encoding="utf-8")
    client, calls = scripted_judge(*judge_replies)
    config = VideoMCQAResourcesServerConfig(
        name="videoindex",
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        cohort_manifest=manifest,
        judge_model_server=judge,
        judge_chat_completion_create_params={"messages": []},
    )
    return VideoIndexResourcesServer(config=config, server_client=client), rows, calls


def make_request(row: dict[str, object], text: str, **updates: object) -> VideoMCQAVerifyRequest:
    return VideoMCQAVerifyRequest.model_validate({**copy.deepcopy(row), "response": response(text, **updates)})


@pytest.mark.parametrize("benchmark", ["videoindex_v1_0", "videoindex_v1_1"])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
async def test_rule_verify_http_and_four_capability_group_metrics(
    tmp_path: Path, benchmark: str, setting: str
) -> None:
    server, rows, calls = make_server(tmp_path, benchmark=benchmark, setting=setting)
    # Verify over HTTP TestClient to ensure FastAPI response model validates videoindex_rule_v1 cleanly.
    http = TestClient(server.setup_webserver())
    replies = [
        "B",  # perception (2 options, gold B) -> correct
        "The answer is (A)",  # temporal (4 options, gold C) -> incorrect
        "<think>Check options</think>Answer: H",  # spatial_physical (8 options, gold H) -> correct
        "J.",  # reasoning_knowledge (10 options, gold J) -> correct
    ]
    verify_responses = []
    for index, (row, text) in enumerate(zip(rows, replies)):
        resp = http.post("/verify", json=make_request(row, text).model_dump())
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["answer_policy"] == VIDEOINDEX_RULE_POLICY
        verify_responses.append({**data, TASK_INDEX_KEY_NAME: index, ROLLOUT_INDEX_KEY_NAME: 0})

    assert calls == []
    assert [item["outcome"] for item in verify_responses] == ["correct", "incorrect", "correct", "correct"]
    assert [item["predicted_answer"] for item in verify_responses] == ["B", "A", "H", "J"]

    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=verify_responses))
    base = RELEASES[benchmark].metric_name + ("_agentic" if setting == "agentic" else "")
    assert metrics.key_metrics[f"mean/{base}"] == pytest.approx(0.75)
    assert metrics.key_metrics[f"{base}/raw_acc"] == pytest.approx(75.0)
    assert metrics.key_metrics[f"{base}/coverage"] == pytest.approx(1.0)
    assert metrics.key_metrics[f"{base}/is_complete"] is True
    assert metrics.key_metrics[f"{base}/capability_group/perception/raw_acc"] == pytest.approx(100.0)
    assert metrics.key_metrics[f"{base}/capability_group/temporal/raw_acc"] == pytest.approx(0.0)
    assert metrics.key_metrics[f"{base}/capability_group/spatial_physical/raw_acc"] == pytest.approx(100.0)
    assert metrics.key_metrics[f"{base}/capability_group/reasoning_knowledge/raw_acc"] == pytest.approx(100.0)


async def test_missing_and_infra_failed_rows_preserve_frozen_denominator(tmp_path: Path) -> None:
    server, rows, _ = make_server(tmp_path, benchmark="videoindex_v1_1", setting="single_turn")
    r0 = (await server.verify(make_request(rows[0], "B"))).model_dump()
    r1 = (await server.verify(make_request(rows[1], "C", status="failed"))).model_dump()
    assert r1["outcome"] == "infrastructure_failure" and r1["mask_sample"] is True
    assert r1["failure_kind"] == "videoindex_v1_1:infrastructure_failure"
    verify_responses = [
        {**r0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
        {**r1, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
    ]
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=verify_responses))
    assert metrics.key_metrics["Video_Index_v1_1/raw_acc"] == pytest.approx(25.0)
    assert metrics.key_metrics["Video_Index_v1_1/coverage"] == pytest.approx(0.25)
    assert metrics.key_metrics["Video_Index_v1_1/is_complete"] is False
    assert metrics.agent_metrics["Video_Index_v1_1/missing"] == 2
    assert metrics.agent_metrics["Video_Index_v1_1/outcomes/infrastructure_failure"] == 1


async def test_llm_extraction_scores_ten_option_choice_and_requires_judge(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs judge_model_server"):
        make_server(tmp_path, answer_policy="llm_extraction_v1", judge=None)

    server, rows, calls = make_server(
        tmp_path,
        benchmark="videoindex_v1_0",
        setting="agentic",
        answer_policy="llm_extraction_v1",
        judge=JUDGE,
        judge_replies=('{"choice": "J"}', '{"choice": "none"}'),
    )
    # Row 3 has 10 options (A-J) with gold J.
    res = await server.verify(make_request(rows[3], "The tenth option matches best."))
    assert (res.outcome, res.predicted_answer, res.reward, res.answer_policy) == (
        "correct",
        "J",
        1.0,
        "llm_extraction_v1",
    )
    assert "J. Synthetic choice 10" in calls[0]["json"].messages[0]["content"]

    no_choice = await server.verify(make_request(rows[0], "Cannot tell."))
    assert (no_choice.outcome, no_choice.predicted_answer, no_choice.reward) == ("no_choice", None, 0.0)


async def test_server_rejects_mismatched_policy_and_eleven_choices(tmp_path: Path) -> None:
    server, rows, _ = make_server(tmp_path, benchmark="videoindex_v1_0", setting="single_turn")
    bad_row = copy.deepcopy(rows[0])
    bad_row["verifier_metadata"]["option_count"] = 11
    bad_row["verifier_metadata"]["options"] = [f"Option {i}" for i in range(11)]
    with pytest.raises(ValueError, match="between 2 and 10"):
        await server.verify(make_request(bad_row, "B"))

    r0 = (await server.verify(make_request(rows[0], "B"))).model_dump()
    r0["answer_policy"] = "boxed_only_v1"
    with pytest.raises(ValueError, match="answer policy differs"):
        await server.aggregate_metrics(
            AggregateMetricsRequest(verify_responses=[{**r0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}])
        )


async def test_verifier_fixture_passes_contract() -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE,
        reward_range=(0.0, 1.0),
        higher_is_better=True,
        determinism="stochastic",
    )
    assert len(results) >= 3


async def test_num_repeats_two_aggregation_and_startup_mismatch_guard(tmp_path: Path) -> None:
    server, rows, _ = make_server(tmp_path, benchmark="videoindex_v1_0", setting="single_turn", num_repeats=2)
    # Repeat 0: all 4 tasks answered (3 correct, 1 wrong). Repeat 1: 2 tasks answered (both correct), 2 missing.
    r0_replies = ["B", "A", "H", "J"]
    r1_replies = ["B", "C"]
    verify_responses = []
    for idx, reply in enumerate(r0_replies):
        data = (await server.verify(make_request(rows[idx], reply))).model_dump()
        verify_responses.append({**data, TASK_INDEX_KEY_NAME: idx, ROLLOUT_INDEX_KEY_NAME: 0})
    for idx, reply in enumerate(r1_replies):
        data = (await server.verify(make_request(rows[idx], reply))).model_dump()
        verify_responses.append({**data, TASK_INDEX_KEY_NAME: idx, ROLLOUT_INDEX_KEY_NAME: 1})

    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=verify_responses))
    assert metrics.agent_metrics["Video_Index_v1_0/expected"] == 8
    assert metrics.agent_metrics["Video_Index_v1_0/missing"] == 2
    assert metrics.key_metrics["Video_Index_v1_0/raw_acc"] == pytest.approx(5 / 8 * 100)
    assert metrics.key_metrics["Video_Index_v1_0/coverage"] == pytest.approx(6 / 8)
    assert metrics.key_metrics["Video_Index_v1_0/is_complete"] is False

    # Within-repeat duplicate must still be rejected.
    dup = [verify_responses[0], verify_responses[0]]
    with pytest.raises(ValueError, match="duplicate"):
        await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=dup))

    # Server startup must fail fast before model dispatch when global_config num_repeats disagrees with cohort.
    manifest = tmp_path / "videoindex_v1_0-single_turn-videoindex_rule_v1.json"
    bad_client = MagicMock(spec=ServerClient)
    bad_client.global_config_dict = {"num_repeats": 1}
    with pytest.raises(ValueError, match="disagrees with frozen cohort num_repeats"):
        VideoIndexResourcesServer(
            config=VideoMCQAResourcesServerConfig(
                name="videoindex",
                host="127.0.0.1",
                port=8000,
                entrypoint="app.py",
                cohort_manifest=manifest,
            ),
            server_client=bad_client,
        )

    ok_client = MagicMock(spec=ServerClient)
    ok_client.global_config_dict = {"num_repeats": 2}
    ok_server = VideoIndexResourcesServer(
        config=VideoMCQAResourcesServerConfig(
            name="videoindex",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            cohort_manifest=manifest,
        ),
        server_client=ok_client,
    )
    assert ok_server._cohort.num_repeats == 2
