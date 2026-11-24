# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Question sets, a video sent as labelled frames, and empty_answer_reward.

Invented questions and a scripted judge only; no API key or network.
"""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest
from nemo_gym.data import TaskDataValidator, load_task_data_schema
from resources_servers.lvbench.app import LVBenchResourcesServer
from resources_servers.video_mcqa.app import (
    VideoMCQAResourcesServer,
    VideoMCQAResourcesServerConfig,
    VideoMCQAVerifyRequest,
)
from resources_servers.video_mcqa.question_set import QuestionSetCohort
from resources_servers.video_mcqa.scoring import llm_extraction_prompt
from resources_servers.video_mcqa.task import (
    VerifierMetadata,
    VideoQuestion,
    prepare_agentic,
    prepare_single_turn,
    video_parts,
)
from resources_servers.video_mcqa.tests.helpers import REVISION, grade_failed_run_and_replay, response
from resources_servers.video_mcqa.tests.test_llm_extraction import JUDGE, scripted_judge
from resources_servers.videomme_v1.app import VideoMMEV1ResourcesServer


SOURCE = "https://example.invalid/invented-question-set"
SERVERS = Path(__file__).resolve().parents[2]
FRAMES = [
    ("This is a video:\nFrame 1 sampled at 0.25 seconds: ", "file:///data/frames/shapes/00000.jpg"),
    ("\nFrame 2 sampled at 0.75 seconds: ", "file:///data/frames/shapes/00001.jpg"),
    ("\nFrame 3 sampled at 1.25 seconds: ", "file:///data/frames/shapes/00002.jpg"),
]
SIX_OPTIONS = ("Circle", "Square", "Triangle", "Star", "Moon", "Sun")
CALL = {"id": "call", "type": "function_call", "call_id": "call", "name": "watch_video", "arguments": "{}"}
THINKING = {"id": "thinking", "type": "reasoning", "summary": [{"type": "summary_text", "text": r"\boxed{B}"}]}


def question(identity: str = "shapes:one") -> VideoQuestion:
    return VideoQuestion(
        task_id=identity,
        dataset_source=SOURCE,
        dataset_revision=REVISION,
        video_id="shapes",
        video_source="https://example.invalid/shapes.mp4",
        video_sha256="b" * 64,
        question="Which invented shape moves first?",
        options=SIX_OPTIONS,
    )


def frames_row(identity: str = "shapes:one", frames=FRAMES) -> dict:
    return json.loads(json.dumps(prepare_single_turn(question(identity), expected_answer="B", frames=frames)))


def agentic_row(identity: str = "shapes:one") -> dict:
    row = prepare_agentic(question(identity), expected_answer="B", video_path="video/shapes.mp4")
    return {
        **row,
        "video_binding": {"source": "/data/shapes.mp4", "sandbox_path": "video/shapes.mp4", "sha256": "b" * 64},
    }


def make_server(tmp_path, *replies, answer_policy="llm_extraction_v1", setting="single_turn", **options):
    cohort = QuestionSetCohort(
        question_set="invented_shapes",
        dataset_source=SOURCE,
        dataset_revision=REVISION,
        setting=setting,
        answer_policy=answer_policy,
        task_ids=("shapes:one", "shapes:two"),
    )
    path = tmp_path / f"question-set-{setting}-{answer_policy}.json"
    path.write_text(cohort.model_dump_json())
    client, calls = scripted_judge(*replies)
    config = VideoMCQAResourcesServerConfig(
        name="video_mcqa",
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        cohort_manifest=path,
        judge_model_server=JUDGE if answer_policy == "llm_extraction_v1" else None,
        judge_chat_completion_create_params={"messages": [], "max_completion_tokens": 512},
        **options,
    )
    return VideoMCQAResourcesServer(config=config, server_client=client), calls


def request(row: dict, text: str | None = r"\boxed{B}", **updates) -> VideoMCQAVerifyRequest:
    return VideoMCQAVerifyRequest.model_validate({**copy.deepcopy(row), "response": response(text, **updates)})


def request_with(row: dict, *items: dict) -> VideoMCQAVerifyRequest:
    return request(row, None, output=list(items))


def message(text: str) -> dict:
    return response(text)["output"][0]


def test_frames_are_label_and_image_parts_then_the_question() -> None:
    (user,) = frames_row()["responses_create_params"]["input"]
    content = user["content"]
    assert [part["type"] for part in content] == ["input_text", "input_image"] * 3 + ["input_text"]
    assert [part["text"] for part in content[:-1:2]] == [label for label, _ in FRAMES]
    assert [part["image_url"] for part in content[1:-1:2]] == [url for _, url in FRAMES]
    assert {part["detail"] for part in content[1:-1:2]} == {"auto"}
    assert content[-1]["text"].endswith("E. Moon\nF. Sun") and "Frame" not in content[-1]["text"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"video_url": "https://example.invalid/shapes.mp4", "frames": FRAMES},
        {"frames": []},
        {"frames": "file:///data/frames/shapes/00000.jpg"},
        {"frames": [(" ", "file:///data/frames/shapes/00000.jpg")]},
        {"frames": [("Frame 1: ", "")]},
        {"frames": [("Frame 1: ", "file:///a\x00b.jpg")]},
    ],
    ids=["no_video", "both", "no_frames", "a_string", "blank_label", "blank_url", "nul_byte"],
)
def test_a_video_is_one_video_part_or_nonempty_labelled_frames(kwargs) -> None:
    with pytest.raises(ValueError):
        video_parts(**kwargs)


def test_a_frames_row_keeps_its_question_and_options_for_the_verifier() -> None:
    stored = frames_row()["verifier_metadata"]
    assert (stored["question"], stored["options"], stored["option_count"]) == (
        "Which invented shape moves first?",
        list(SIX_OPTIONS),
        6,
    )
    assert frames_row(frames=FRAMES[:1])["verifier_metadata"] == stored


async def test_a_frames_row_is_graded_from_the_judges_reading_of_its_final_answer(tmp_path) -> None:
    server, calls = make_server(tmp_path, '{"choice": "B"}', '{"choice": "F"}')
    row = frames_row()
    result = await server.verify(request(row, "<think>The circle?</think>The square moves first, so B."))
    assert (result.outcome, result.predicted_answer, result.reward, result.mask_sample) == ("correct", "B", 1.0, False)
    (call,) = calls
    stored = VerifierMetadata(**row["verifier_metadata"])
    assert call["json"].messages == [
        {"role": "user", "content": llm_extraction_prompt(stored, "The square moves first, so B.")}
    ]
    content = call["json"].messages[0]["content"]
    assert "Which invented shape moves first?\nA. Circle\nB. Square" in content and "F. Sun\n</question>" in content
    assert "Frame" not in content and "file://" not in content
    # Any of the question's six options can be read, and is scored against the gold label.
    result = await server.verify(request(row, "The sun, F."))
    assert (result.outcome, result.predicted_answer, result.reward) == ("incorrect", "F", 0.0)
    assert result.failure_kind is None and result.answer_policy == "llm_extraction_v1"


async def test_rule_policies_read_only_the_final_answer_of_a_frames_row(tmp_path) -> None:
    server, calls = make_server(tmp_path, answer_policy="boxed_only_v1")
    row = frames_row()
    assert (await server.verify(request(row, r"\boxed{B}"))).outcome == "correct"
    assert (await server.verify(request(row, r"\boxed{F}"))).predicted_answer == "F"
    assert (await server.verify(request(row, "B"))).outcome == "format_error"
    assert calls == []


@pytest.mark.parametrize(
    "updates",
    [
        {"task_id": "shapes:three"},
        {"dataset_revision": "c" * 40},
        {"dataset_source": "https://example.invalid/other-set"},
        {"level": 2},
        {"group_id": "group-1"},
    ],
)
async def test_a_question_set_verifies_only_its_own_tasks(tmp_path, updates) -> None:
    server, _ = make_server(tmp_path, answer_policy="boxed_only_v1")
    row = frames_row()
    row["verifier_metadata"].update(updates)
    row["task_id"] = row["verifier_metadata"]["task_id"]
    with pytest.raises(ValueError):
        await server.verify(request(row))


async def test_a_question_set_verifies_its_setting_only(tmp_path) -> None:
    server, _ = make_server(tmp_path, answer_policy="boxed_only_v1")
    with pytest.raises(ValueError, match="setting"):
        await server.verify(request(frames_row() | {"setting": "agentic"}))


@pytest.mark.parametrize(
    "updates",
    [
        {"task_ids": ["shapes:one", "shapes:one"]},
        {"task_ids": [" "]},
        {"task_ids": ["videomme_v1:one"]},
        {"task_ids": ["lvbench:one"]},
        {"question_set": "invented shapes"},
        {"dataset_revision": "latest"},
        {"dataset_source": " "},
        {"task_subsets": {}},
    ],
)
def test_question_set_cohorts_are_frozen_and_never_use_a_benchmark_scope(updates) -> None:
    cohort = {
        "question_set": "invented_shapes",
        "dataset_source": SOURCE,
        "dataset_revision": REVISION,
        "setting": "single_turn",
        "task_ids": ["shapes:one"],
    }
    assert QuestionSetCohort.model_validate(cohort).answer_policy == "boxed_only_v1"
    with pytest.raises(ValidationError):
        QuestionSetCohort.model_validate({**cohort, **updates})


@pytest.mark.parametrize("server_type", [VideoMMEV1ResourcesServer, LVBenchResourcesServer])
def test_benchmark_servers_refuse_a_question_set(tmp_path, server_type) -> None:
    server, _ = make_server(tmp_path, answer_policy="boxed_only_v1")
    with pytest.raises(ValueError, match="different benchmark"):
        server_type(config=server.config, server_client=server.server_client)


async def test_question_set_accuracy_is_reported_under_its_name(tmp_path) -> None:
    server, _ = make_server(tmp_path, answer_policy="boxed_only_v1")
    results = [
        (await server.verify(request(frames_row(), r"\boxed{B}"))).model_dump(),
        (await server.verify(request(frames_row("shapes:two"), ""))).model_dump(),
    ]
    for index, result in enumerate(results):
        result |= {TASK_INDEX_KEY_NAME: index, ROLLOUT_INDEX_KEY_NAME: 0}
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=results))
    agent = metrics.agent_metrics
    assert (agent["invented_shapes/raw_acc"], agent["mean/invented_shapes"]) == (50, 0.5)
    assert (agent["invented_shapes/outcomes/correct"], agent["invented_shapes/outcomes/no_answer"]) == (1, 1)
    assert agent["invented_shapes/is_complete"] is True
    assert not [
        key for key in agent if key.startswith("invented_shapes/") and key.count("/") > 1 and "outcomes" not in key
    ]
    assert metrics.key_metrics["mean/invented_shapes"] == 0.5


@pytest.mark.parametrize("text", ["", "   ", "<think>still thinking"])
async def test_without_empty_answer_reward_an_attempt_without_an_answer_scores_zero(tmp_path, text) -> None:
    server, calls = make_server(tmp_path)
    result = await server.verify(request(frames_row(), text))
    assert (result.outcome, result.reward, result.mask_sample, calls) == ("no_answer", 0.0, False, [])


@pytest.mark.parametrize("text", ["", "   ", "<think>still thinking"])
async def test_empty_answer_reward_scores_an_attempt_without_an_answer(tmp_path, text) -> None:
    server, calls = make_server(tmp_path, empty_answer_reward=-0.5)
    result = await server.verify(request(frames_row(), text))
    assert (result.outcome, result.reward, result.mask_sample, result.evaluation_completed, calls) == (
        "no_answer",
        -0.5,
        False,
        True,
        [],
    )


async def test_empty_answer_reward_leaves_other_outcomes_alone(tmp_path) -> None:
    server, calls = make_server(tmp_path, '{"choice": "none"}', '{"choice": "C"}', empty_answer_reward=-0.5)
    row = frames_row()
    # An answer that names no option, or a wrong one, is a measured 0, not the empty-answer reward.
    assert ((await server.verify(request(row, "Either A or C."))).outcome, len(calls)) == ("no_choice", 1)
    assert (await server.verify(request(row, "C"))).reward == 0.0
    # A response without any output stays 0, as Gym's agents require for a failed run.
    empty = await server.verify(request_with(row))
    assert (empty.outcome, empty.reward, empty.mask_sample) == ("no_answer", 0.0, False)
    tool_call = await server.verify(request_with(row, message(r"\boxed{B}"), CALL))
    assert (tool_call.outcome, tool_call.reward) == ("execution_failure", 0.0)
    failed = await server.verify(request(row, r"\boxed{B}", status="failed"))
    assert (failed.outcome, failed.reward, failed.mask_sample) == ("infrastructure_failure", 0.0, True)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("items", "outcome", "reward"),
    [
        ((THINKING,), "no_answer", -0.5),
        ((THINKING, message("   ")), "no_answer", -0.5),
        ((THINKING, message(r"\boxed{B}")), "correct", 1.0),
    ],
)
async def test_empty_answer_reward_reads_the_output_items(tmp_path, items, outcome, reward) -> None:
    server, _ = make_server(tmp_path, answer_policy="boxed_only_v1", empty_answer_reward=-0.5)
    result = await server.verify(request_with(frames_row(), *items))
    assert (result.outcome, result.reward, result.mask_sample) == (outcome, reward, False)


@pytest.mark.parametrize("penalty", [None, -0.5])
async def test_an_agents_empty_response_check_still_gets_zero(tmp_path, penalty) -> None:
    # Gym's agents verify a failed run (execution_failure_reward_zero) as a response without output and require an
    # unmasked 0; a replay of the saved row then grades the saved answer.
    server, _ = make_server(tmp_path, answer_policy="boxed_only_v1", setting="agentic", empty_answer_reward=penalty)
    saved, replayed = await grade_failed_run_and_replay(
        server, VideoMCQAVerifyRequest, agentic_row(), response(r"\boxed{B}")
    )
    assert (saved["outcome"], saved["reward"], saved["mask_sample"]) == ("no_answer", 0.0, False)
    assert (replayed.outcome, replayed.reward) == ("correct", 1.0)


async def test_accuracy_reads_the_outcome_not_the_empty_answer_reward(tmp_path) -> None:
    server, _ = make_server(tmp_path, answer_policy="boxed_only_v1", empty_answer_reward=-0.5)
    unanswered = (await server.verify(request(frames_row(), "<think>still thinking"))).model_dump()
    unanswered |= {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
    assert unanswered["reward"] == -0.5
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[unanswered]))
    assert metrics.agent_metrics["invented_shapes/raw_acc"] == 0
    assert metrics.agent_metrics["invented_shapes/outcomes/no_answer"] == 1
    # Any other reward on an unanswered attempt is a corrupt row.
    with pytest.raises(ValueError, match="disagree"):
        await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[unanswered | {"reward": -0.25}]))


def test_question_set_rows_are_valid_task_data_and_benchmark_rows_keep_four_choices() -> None:
    validator = TaskDataValidator("video_mcqa", load_task_data_schema(SERVERS / "video_mcqa"), "rows.jsonl")
    grouped_row = frames_row("shapes:two")
    grouped_row["verifier_metadata"]["group_id"] = "group-1"
    unprepared_row = frames_row("shapes:three")
    del unprepared_row["verifier_metadata"]["options"]
    for index, row in enumerate([frames_row(), agentic_row(), grouped_row, unprepared_row]):
        validator.validate_row(index, row)
    assert (validator.report.rows, validator.report.error_rows) == (4, 2)
    assert "no v2 group or level" in validator.report.errors[0] and "options" in validator.report.errors[1]
    # VideoMME-v1 and LVBench check their own four choices.
    for benchmark in ("videomme_v1", "lvbench"):
        validator = TaskDataValidator(benchmark, load_task_data_schema(SERVERS / benchmark), "rows.jsonl")
        validator.validate_row(0, frames_row())
        assert validator.report.error_rows == 1 and "four choices" in validator.report.errors[0]
