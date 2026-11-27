# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest, ModelServerRef
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.cgbench.app import VERIFIER_FIXTURE, CGBenchResourcesServer
from resources_servers.cgbench.dataset import DATASET_REVISION, SLICES, CGBenchBenchmark
from resources_servers.cgbench.prepare import CGBenchPromptStyle, prepare_cgbench_dataset
from resources_servers.cgbench.scoring import CGBENCH_RULE_POLICY, CGBenchAnswerPolicy
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig, VideoMCQAVerifyRequest
from resources_servers.video_mcqa.dataset import VideoSource
from resources_servers.video_mcqa.tests.helpers import response


JUDGE = ModelServerRef(type="responses_api_models", name="cgbench_answer_judge")


def _sample_records() -> list[dict[str, object]]:
    return [
        {
            "qid": 101,
            "video_uid": "vid_alpha",
            "question": "Which synthetic label appears at the start?",
            "choices": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
            "answer": "Epsilon",
            "right_answer": "E",
            "clue_intervals": [[12, 20]],
            "domain": "Electonic/Social Gaming",
            "sub_category": "2D Spatial Perception",
            "duration": 599,  # 0-10m
        },
        {
            "qid": 102,
            "video_uid": "vid_beta",
            "question": "Why did the narrator pause?",
            "choices": ["One", "Two", "Three", "Four", "Five", "Six", "Seven"],
            "answer": "Seven",
            "right_answer": "G",
            "clue_intervals": [[100, 120]],
            "domain": "Music & TV show",
            "sub_category": "Event Cognition",
            "duration": 1800,  # 20-30m
        },
        {
            "qid": 103,
            "video_uid": "vid_gamma",
            "question": "Which nonexistent prop is mentioned?",
            "choices": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
            "answer": "P8",
            "right_answer": "H",
            "clue_intervals": [],
            "domain": "News",
            "sub_category": "Hallucination",
            "duration": 3660,  # 60m+
        },
    ]


def _sample_media(setting: str) -> dict[str, VideoSource]:
    videos = ("vid_alpha", "vid_beta", "vid_gamma")
    media: dict[str, VideoSource] = {}
    for index, uid in enumerate(videos):
        sha = f"{index + 1:064x}"
        if setting == "single_turn":
            media[uid] = VideoSource(
                source_url=f"cg_videos_720p/{uid}.mp4",
                sha256=sha,
                video_url=f"https://example.invalid/cg_videos_720p/{uid}.mp4",
            )
        else:
            media[uid] = VideoSource(
                source_url=f"cg_videos_720p/{uid}.mp4",
                sha256=sha,
                source=f"/data/cg_videos_720p/{uid}.mp4",
                sandbox_path=f"videos/{uid}.mp4",
            )
    return media


def _scripted_judge(*contents: str | None) -> tuple[MagicMock, list[dict]]:
    calls: list[dict] = []
    replies = iter(contents)

    async def post(*, server_name: str, url_path: str, json: object, **kwargs: object) -> SimpleNamespace:
        calls.append({"server_name": server_name, "url_path": url_path, "json": json})
        payload = {
            "id": "judge-reply",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": next(replies)}}
            ],
        }
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=str(json_dumps(payload)).encode()))

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = post
    return client, calls


def json_dumps(obj: object) -> str:
    return json.dumps(obj)


def _make_server(
    tmp_path,
    *,
    benchmark: CGBenchBenchmark = "cgbench",
    setting: str = "single_turn",
    answer_policy: CGBenchAnswerPolicy = CGBENCH_RULE_POLICY,
    prompt: CGBenchPromptStyle = "official",
    judge_replies: tuple[str | None, ...] = (),
):
    rows, cohort = prepare_cgbench_dataset(
        _sample_records(),
        benchmark=benchmark,
        dataset_revision=DATASET_REVISION,
        media=_sample_media(setting),
        setting=setting,  # type: ignore[arg-type]
        answer_policy=answer_policy,
        prompt=prompt,
    )
    manifest = tmp_path / f"{benchmark}-{setting}-{answer_policy}.json"
    manifest.write_text(cohort.model_dump_json(), encoding="utf-8")
    if answer_policy == "llm_extraction_v1":
        client, calls = _scripted_judge(*judge_replies)
        judge_ref = JUDGE
    else:
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {}
        calls = []
        judge_ref = None
    server = CGBenchResourcesServer(
        config=VideoMCQAResourcesServerConfig(
            name="cgbench",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            cohort_manifest=manifest,
            judge_model_server=judge_ref,
        ),
        server_client=client,
    )
    return server, rows, cohort, calls


def _verify_request(row: dict[str, object], text: str, **updates: object) -> VideoMCQAVerifyRequest:
    return VideoMCQAVerifyRequest.model_validate({**copy.deepcopy(row), "response": response(text), **updates})


@pytest.mark.parametrize("benchmark", ["cgbench", "cgbench_mini", "cgbench_hard"])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
async def test_verify_and_aggregate_metrics_across_all_slices_and_dimensions(
    tmp_path, benchmark: CGBenchBenchmark, setting: str
) -> None:
    server, rows, _, _ = _make_server(tmp_path, benchmark=benchmark, setting=setting)
    with TestClient(server.setup_webserver()) as client:
        r0 = client.post("/verify", json=_verify_request(rows[0], '```json\n{"result": "E"}\n```').model_dump())
        r1 = client.post("/verify", json=_verify_request(rows[1], '```json\n{"result": "A"}\n```').model_dump())
        r2 = client.post("/verify", json=_verify_request(rows[2], '```json\n{"result": "H"}\n```').model_dump())
        assert r0.status_code == 200 and r0.json()["outcome"] == "correct" and r0.json()["predicted_answer"] == "E"
        assert r1.status_code == 200 and r1.json()["outcome"] == "incorrect" and r1.json()["predicted_answer"] == "A"
        assert r2.status_code == 200 and r2.json()["outcome"] == "correct" and r2.json()["predicted_answer"] == "H"
        assert server.checkpoint_verify == "replay"

        responses = []
        for idx, resp in enumerate((r0.json(), r1.json(), r2.json())):
            resp[TASK_INDEX_KEY_NAME] = idx
            resp[ROLLOUT_INDEX_KEY_NAME] = 0
            responses.append(resp)

        agg = client.post("/aggregate_metrics", json=AggregateMetricsRequest(verify_responses=responses).model_dump())
        assert agg.status_code == 200
        body = agg.json()

    base = SLICES[benchmark].metric_name + ("_agentic" if setting == "agentic" else "")
    agent_metrics = body["agent_metrics"]
    key_metrics = body["key_metrics"]

    assert agent_metrics[f"{base}/expected"] == 3
    assert agent_metrics[f"{base}/correct"] == 2
    assert agent_metrics[f"{base}/missing"] == 0
    assert agent_metrics[f"{base}/is_complete"] is True
    assert agent_metrics[f"{base}/raw_acc"] == pytest.approx(200.0 / 3.0)
    assert key_metrics[f"mean/{base}"] == pytest.approx(2.0 / 3.0)

    # Category dimension (3 official macro-categories)
    assert agent_metrics[f"{base}/category/Perception/raw_acc"] == pytest.approx(100.0)
    assert agent_metrics[f"{base}/category/Cognition/raw_acc"] == pytest.approx(0.0)
    assert agent_metrics[f"{base}/category/Hallucination/raw_acc"] == pytest.approx(100.0)

    # Sub-category dimension (URL-encoded)
    assert agent_metrics[f"{base}/sub_category/{quote('2D Spatial Perception', safe='')}/raw_acc"] == pytest.approx(
        100.0
    )
    assert agent_metrics[f"{base}/sub_category/{quote('Event Cognition', safe='')}/raw_acc"] == pytest.approx(0.0)
    assert agent_metrics[f"{base}/sub_category/Hallucination/raw_acc"] == pytest.approx(100.0)

    # Domain dimension (URL-encoded, preserving upstream spelling)
    assert agent_metrics[f"{base}/domain/{quote('Electonic/Social Gaming', safe='')}/raw_acc"] == pytest.approx(100.0)
    assert agent_metrics[f"{base}/domain/{quote('Music & TV show', safe='')}/raw_acc"] == pytest.approx(0.0)
    assert agent_metrics[f"{base}/domain/News/raw_acc"] == pytest.approx(100.0)

    # Duration dimension (official 10-minute right-closed buckets)
    assert agent_metrics[f"{base}/duration/0-10m/raw_acc"] == pytest.approx(100.0)
    assert agent_metrics[f"{base}/duration/20-30m/raw_acc"] == pytest.approx(0.0)
    assert agent_metrics[f"{base}/duration/{quote('60m+', safe='')}/raw_acc"] == pytest.approx(100.0)


async def test_frozen_denominator_with_missing_and_infra_failed_rows(tmp_path) -> None:
    server, rows, _, _ = _make_server(tmp_path, benchmark="cgbench_mini", setting="single_turn")
    r0 = (await server.verify(_verify_request(rows[0], '```json\n{"result": "E"}\n```'))).model_dump()
    r0[TASK_INDEX_KEY_NAME] = 0
    r0[ROLLOUT_INDEX_KEY_NAME] = 0

    # Mark row 1 as infrastructure failure via mask_sample=True; omit row 2 entirely.
    r1 = (await server.verify(_verify_request(rows[1], '```json\n{"result": "G"}\n```'))).model_dump()
    r1[TASK_INDEX_KEY_NAME] = 1
    r1[ROLLOUT_INDEX_KEY_NAME] = 0
    r1["mask_sample"] = True

    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[r0, r1]))
    am = metrics.agent_metrics
    assert am["CG_Bench_mini/expected"] == 3
    assert am["CG_Bench_mini/returned"] == 2
    assert am["CG_Bench_mini/missing"] == 1
    assert am["CG_Bench_mini/correct"] == 1
    assert am["CG_Bench_mini/raw_acc"] == pytest.approx(100.0 / 3.0)
    assert am["CG_Bench_mini/coverage"] == pytest.approx(1.0 / 3.0)
    assert am["CG_Bench_mini/is_complete"] is False
    assert am["CG_Bench_mini/outcomes/infrastructure_failure"] == 1


async def test_boxed_only_and_llm_extraction_policies(tmp_path) -> None:
    boxed_server, boxed_rows, _, _ = _make_server(
        tmp_path, benchmark="cgbench_hard", setting="single_turn", answer_policy="boxed_only_v1", prompt="boxed"
    )
    res_ok = await boxed_server.verify(_verify_request(boxed_rows[0], r"\boxed{E}"))
    assert res_ok.outcome == "correct" and res_ok.answer_policy == "boxed_only_v1"
    res_fmt = await boxed_server.verify(_verify_request(boxed_rows[0], '{"result": "E"}'))
    assert res_fmt.outcome == "format_error"

    judge_server, judge_rows, _, calls = _make_server(
        tmp_path,
        benchmark="cgbench_hard",
        setting="agentic",
        answer_policy="llm_extraction_v1",
        judge_replies=('{"choice": "E"}', '{"choice": "none"}'),
    )
    j0 = await judge_server.verify(_verify_request(judge_rows[0], "Clearly Epsilon is the right one."))
    j1 = await judge_server.verify(_verify_request(judge_rows[1], "I cannot tell from the video."))
    assert j0.outcome == "correct" and j0.predicted_answer == "E" and j0.answer_policy == "llm_extraction_v1"
    assert j1.outcome == "no_choice" and j1.predicted_answer is None
    assert len(calls) == 2


async def test_rejects_more_than_eight_options_and_wrong_slice(tmp_path) -> None:
    server, rows, _, _ = _make_server(tmp_path, benchmark="cgbench", setting="single_turn")
    bad_options_row = copy.deepcopy(rows[0])
    bad_options_row["verifier_metadata"]["option_count"] = 9
    bad_options_row["verifier_metadata"]["options"] = [f"O{i}" for i in range(9)]
    with pytest.raises(ValueError, match="between 2 and 8 choices"):
        await server.verify(_verify_request(bad_options_row, '{"result": "E"}'))

    wrong_slice_row = copy.deepcopy(rows[0])
    wrong_slice_row["task_id"] = "cgbench_mini:101"
    wrong_slice_row["verifier_metadata"]["task_id"] = "cgbench_mini:101"
    with pytest.raises(ValueError, match="differs from the frozen cohort"):
        await server.verify(_verify_request(wrong_slice_row, '{"result": "E"}'))


async def test_verifier_fixture_passes_contract() -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE,
        reward_range=(0.0, 1.0),
        higher_is_better=True,
        determinism="stochastic",
    )
    assert len(results) >= 6


async def test_verify_fenced_placeholder_then_inline_answer_and_conflict(tmp_path) -> None:
    server, rows, _, _ = _make_server(tmp_path, benchmark="cgbench", setting="single_turn")
    ok = await server.verify(
        _verify_request(
            rows[0],
            'Use format:\n```json\n{"result": "option"}\n```\nFinal answer: {"result": "E"}',
        )
    )
    assert ok.outcome == "correct"
    assert ok.predicted_answer == "E"
    assert ok.reward == 1.0

    ambiguous = await server.verify(
        _verify_request(
            rows[0],
            '```json\n{"result": "A"}\n```\nFinal answer: {"result": "E"}',
        )
    )
    assert ambiguous.outcome == "ambiguous_answer"
    assert ambiguous.predicted_answer is None
    assert ambiguous.reward == 0.0


def test_num_repeats_mismatch_guard(tmp_path) -> None:
    _, cohort = prepare_cgbench_dataset(
        _sample_records(),
        benchmark="cgbench",
        dataset_revision=DATASET_REVISION,
        media=_sample_media("single_turn"),
        setting="single_turn",
        num_repeats=2,
    )
    manifest = tmp_path / "cohort-r2.json"
    manifest.write_text(cohort.model_dump_json(), encoding="utf-8")

    bad_client = MagicMock(spec=ServerClient)
    bad_client.global_config_dict = {"num_repeats": 1}
    with pytest.raises(ValueError, match="num_repeats=1 disagrees with frozen cohort num_repeats=2"):
        CGBenchResourcesServer(
            config=VideoMCQAResourcesServerConfig(
                name="cgbench", host="127.0.0.1", port=8000, entrypoint="app.py", cohort_manifest=manifest
            ),
            server_client=bad_client,
        )

    ok_client = MagicMock(spec=ServerClient)
    ok_client.global_config_dict = {"num_repeats": 2}
    server = CGBenchResourcesServer(
        config=VideoMCQAResourcesServerConfig(
            name="cgbench", host="127.0.0.1", port=8000, entrypoint="app.py", cohort_manifest=manifest
        ),
        server_client=ok_client,
    )
    assert server._cohort.num_repeats == 2
