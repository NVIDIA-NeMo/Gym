# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""answer_policy llm_extraction_v1 with a scripted judge: no API key, no network."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest, ModelServerRef
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding
from nemo_gym.verification import JudgeError
from resources_servers.lvbench.app import LVBenchResourcesServer
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig, VideoMCQAVerifyRequest
from resources_servers.video_mcqa.prepare import prepare_dataset
from resources_servers.video_mcqa.scoring import (
    LLM_EXTRACTION_PROMPT,
    NO_CHOICE,
    aggregate_scores,
    grade_answer,
    llm_extraction_prompt,
    parse_llm_choice,
    prompt_text,
)
from resources_servers.video_mcqa.task import VerifierMetadata, VideoQuestion, prepare_single_turn
from resources_servers.video_mcqa.tests.helpers import REVISION, lv_record, media, response, v1_record
from resources_servers.videomme_v1.app import VideoMMEV1ResourcesServer
from resources_servers.videomme_v2.app import (
    VideoMMEV2ResourcesServer,
    VideoMMEV2ResourcesServerConfig,
    VideoMMEV2VerifyRequest,
)


JUDGE = ModelServerRef(type="responses_api_models", name="answer_judge")


def judge_reply(content: str | None) -> bytes:
    return json.dumps(
        {
            "id": "judge-reply",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }
    ).encode()


def scripted_judge(*contents: str | None | Exception) -> tuple[MagicMock, list[dict]]:
    """A server client whose judge model answers with ``contents`` in turn (or raises) and records each request."""
    calls: list[dict] = []
    replies = iter(contents)

    async def post(*, server_name: str, url_path: str, json: object, **kwargs: object) -> SimpleNamespace:
        calls.append({"server_name": server_name, "url_path": url_path, "json": json})
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=judge_reply(reply)))

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = post
    return client, calls


def make_server(tmp_path, *contents, benchmark="videomme_v1", judge=JUDGE, answer_policy="llm_extraction_v1"):
    records = [v1_record(), v1_record("two")] if benchmark == "videomme_v1" else [lv_record(), lv_record("two")]
    rows, cohort = prepare_dataset(
        records,
        benchmark=benchmark,
        dataset_revision=REVISION,
        media=media(),
        setting="single_turn",
        answer_policy=answer_policy,
        prompt=benchmark,
    )
    path = tmp_path / f"{benchmark}-{answer_policy}.json"
    path.write_text(cohort.model_dump_json())
    client, calls = scripted_judge(*contents)
    server_type = VideoMMEV1ResourcesServer if benchmark == "videomme_v1" else LVBenchResourcesServer
    config = VideoMCQAResourcesServerConfig(
        name=benchmark,
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        cohort_manifest=path,
        judge_model_server=judge,
        judge_chat_completion_create_params={"messages": [], "max_completion_tokens": 512},
    )
    return server_type(config=config, server_client=client), rows, calls


def request(row, text, **updates):
    return VideoMCQAVerifyRequest.model_validate({**copy.deepcopy(row), "response": response(text, **updates)})


def metadata(task_id: str = "videomme_v1:one") -> VerifierMetadata:
    return VerifierMetadata(
        task_id=task_id,
        expected_answer="B",
        option_count=4,
        dataset_source="https://example.invalid/source",
        dataset_revision=REVISION,
        video_id="shapes",
        video_source="https://example.invalid/shapes.mp4",
        video_sha256="b" * 64,
    )


@pytest.mark.parametrize(
    "reply,choice",
    [
        ('{"choice": "B"}', "B"),
        ('{"choice": "b"}', "B"),
        ('```json\n{"choice": "D"}\n```', "D"),
        ('The answer gives one option.\n{"choice": "A"}', "A"),
        ('<think>{"choice": "C"} is tempting</think>{"choice": "A"}', "A"),
        ('{"choice": "none"}', NO_CHOICE),
        ('{"choice": "None"}', NO_CHOICE),
    ],
)
def test_judge_reply_parsing(reply, choice) -> None:
    assert parse_llm_choice(reply, option_count=4) == choice


@pytest.mark.parametrize(
    "reply",
    [None, "", "B", '{"choice": "E"}', '{"choice": ""}', '{"letter": "B"}', '{"choice": "B"', '["B"]', "<think>B"],
)
def test_a_reply_that_names_no_option_is_unusable(reply) -> None:
    with pytest.raises(ValueError):
        parse_llm_choice(reply, option_count=4)


def test_judge_sees_prompt_text_and_answer_but_no_media_or_gold() -> None:
    row = prepare_single_turn(
        VideoQuestion(
            task_id="videomme_v1:one",
            dataset_source="https://example.invalid/source",
            dataset_revision=REVISION,
            video_id="shapes",
            video_source="https://example.invalid/shapes.mp4",
            video_sha256="b" * 64,
            question="Which invented shape moves?",
            options=("Circle", "Square", "Triangle", "Star"),
        ),
        expected_answer="B",
        video_url="data:video/mp4;base64,SECRET_VIDEO",
        prompt="videomme_v1",
    )
    text = prompt_text(row["responses_create_params"]["input"])
    assert text.endswith("D. Star\nThe best answer is:") and "SECRET_VIDEO" not in text
    request_text = llm_extraction_prompt(text, "Answer: B")
    assert request_text == LLM_EXTRACTION_PROMPT.format(prompt=text, answer="Answer: B")
    assert (
        "<prompt>\nSelect the best answer" in request_text
        and "<final_answer>\nAnswer: B\n</final_answer>" in request_text
    )
    assert prompt_text("plain text") == "plain text"
    system = {"role": "system", "content": "Be brief."}
    assert prompt_text([system, *row["responses_create_params"]["input"]]).startswith("Be brief.\nSelect the best")


@pytest.mark.parametrize(
    "choice,outcome,predicted,reward",
    [("B", "correct", "B", 1.0), ("C", "incorrect", "C", 0.0), (NO_CHOICE, "no_choice", None, 0.0)],
)
def test_grading_a_judged_choice(choice, outcome, predicted, reward) -> None:
    score = grade_answer(
        "The answer is (B)", metadata(), setting="single_turn", answer_policy="llm_extraction_v1", judged_choice=choice
    )
    assert (score.outcome, score.predicted_answer, score.reward, score.evaluation_completed) == (
        outcome,
        predicted,
        reward,
        True,
    )


def test_judged_choice_belongs_to_llm_extraction_only() -> None:
    with pytest.raises(ValueError, match="needs the judge's reading"):
        grade_answer("B", metadata(), setting="single_turn", answer_policy="llm_extraction_v1")
    with pytest.raises(ValueError, match="needs the judge's reading"):
        grade_answer("B", metadata(), setting="single_turn", answer_policy="llm_extraction_v1", judged_choice="E")
    with pytest.raises(ValueError, match="Only llm_extraction_v1"):
        grade_answer(r"\boxed{B}", metadata(), setting="single_turn", judged_choice="B")
    empty = grade_answer("<think>B", metadata(), setting="single_turn", answer_policy="llm_extraction_v1")
    assert (empty.outcome, empty.reward) == ("no_answer", 0.0)


def test_no_choice_is_a_measured_zero_in_the_denominator() -> None:
    scores = [
        grade_answer("B", metadata(), setting="single_turn", answer_policy="llm_extraction_v1", judged_choice="B"),
        grade_answer(
            "Maybe A or C",
            metadata("videomme_v1:two"),
            setting="single_turn",
            answer_policy="llm_extraction_v1",
            judged_choice=NO_CHOICE,
        ),
    ]
    summary = aggregate_scores(
        scores,
        expected_task_ids=["videomme_v1:one", "videomme_v1:two", "videomme_v1:three"],
        dataset_revision=REVISION,
        setting="single_turn",
        answer_policy="llm_extraction_v1",
    )
    assert (summary.correct, summary.completed, summary.missing, summary.is_complete) == (1, 2, 1, False)
    assert summary.outcome_counts["no_choice"] == 1 and summary.avg_accuracy == pytest.approx(1 / 3)


@pytest.mark.parametrize("benchmark", ["videomme_v1", "lvbench"])
async def test_the_judges_choice_is_scored(tmp_path, benchmark) -> None:
    server, rows, calls = make_server(tmp_path, '{"choice": "B"}', '{"choice": "C"}', benchmark=benchmark)
    result = await server.verify(request(rows[0], "<think>Probably C.</think>Answer: B"))
    assert (result.outcome, result.predicted_answer, result.reward) == ("correct", "B", 1.0)
    assert result.answer_policy == "llm_extraction_v1" and not result.mask_sample and result.evaluation_completed
    (call,) = calls
    assert (call["server_name"], call["url_path"]) == ("answer_judge", "/v1/chat/completions")
    # Only the configured parameters and the message reach the judge, not the schema's defaults.
    assert set(call["json"].model_dump(exclude_unset=True)) == {"messages", "max_completion_tokens"}
    assert call["json"].max_completion_tokens == 512
    (message,) = call["json"].messages
    question = rows[0]["responses_create_params"]["input"][0]["content"][1]["text"]
    assert message == {"role": "user", "content": llm_extraction_prompt(question, "Answer: B")}
    assert "Probably" not in message["content"] and "example.invalid" not in message["content"]
    # A replay judges the saved answer again; a different reading is scored as read.
    replayed = await server.verify(VideoMCQAVerifyRequest.model_validate(result.model_dump()))
    assert (replayed.outcome, replayed.predicted_answer, len(calls)) == ("incorrect", "C", 2)


async def observed_verify(server, body) -> tuple[object, list | None]:
    invocation = VerifierInvocation(invocation_id="verify-1")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = await server.verify(body)
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation.grading_operations


async def test_a_judged_grade_is_bound_to_the_judges_reply(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, '{"choice": "C"}', "no verdict")
    result, operations = await observed_verify(server, request(rows[0], "Answer: C"))
    (operation,) = operations
    assert (operation.kind, operation.outcome, result.outcome) == ("llm_judge", "completed", "incorrect")
    assert [(ref.model_ref.name, ref.response_id) for ref in operation.dependencies] == [
        ("answer_judge", "judge-reply")
    ]
    assert operation.result_binding_sha256 == grading_result_binding(result)
    with pytest.raises(JudgeError):
        await observed_verify(server, request(rows[0], "Answer: C"))
    # Without a judge call there is no judge grading to record.
    result, operations = await observed_verify(server, request(rows[0], ""))
    assert (result.outcome, operations) == ("no_answer", None)


async def test_no_judge_call_without_an_answer_to_read(tmp_path) -> None:
    server, rows, calls = make_server(tmp_path)
    empty = await server.verify(request(rows[0], "<think>still thinking"))
    assert (empty.outcome, empty.reward, empty.mask_sample) == ("no_answer", 0.0, False)
    failed = await server.verify(request(rows[0], "Answer: B", status="failed"))
    assert (failed.outcome, failed.mask_sample, failed.evaluation_completed) == ("infrastructure_failure", True, False)
    assert calls == []


async def test_a_judge_that_reads_no_option_gives_no_choice(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, '{"choice": "none"}')
    result = await server.verify(request(rows[0], "Either A or C."))
    assert (result.outcome, result.predicted_answer, result.reward) == ("no_choice", None, 0.0)
    assert result.evaluation_completed and not result.mask_sample


async def test_an_unusable_reply_raises_a_judge_failure(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, "I cannot tell.")
    with pytest.raises(JudgeError, match="Unusable answer-judge reply"):
        await server.verify(request(rows[0], "Answer: B"))


# A 401 is not retried, so the failed call ends at once.
REJECTED = ClientResponseError(MagicMock(), (), status=401, message="Unauthorized")


@pytest.mark.parametrize("reply", ["no verdict", REJECTED], ids=["unusable_reply", "failed_call"])
def test_judge_failures_are_masked_rows_for_rejudging_not_wrong_answers(tmp_path, reply) -> None:
    server, rows, calls = make_server(tmp_path, reply)
    client = TestClient(server.setup_webserver())
    result = client.post("/verify", json=request(rows[0], "Answer: B").model_dump())
    assert result.status_code == 200 and len(calls) == 1
    row = result.json()
    assert (row["reward"], row["mask_sample"], row["failure_kind"]) == (0.0, True, "judge_failed")
    assert row["_ng_failure_class"] == "judge_failed"
    # The policy's answer is kept, so recovery judges it again without sampling the policy.
    assert row["response"]["output"][0]["content"][0]["text"] == "Answer: B"


async def test_a_judge_failed_task_stays_missing_in_the_frozen_cohort(tmp_path) -> None:
    server, rows, _ = make_server(tmp_path, '{"choice": "B"}')
    result = (await server.verify(request(rows[0], "B"))).model_dump()
    result |= {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
    # Rollout collection keeps a judge-failed row in the failures file, out of aggregation: its task is missing.
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[result]))
    assert metrics.key_metrics["Video_MME/raw_acc"] == 50 and metrics.key_metrics["Video_MME/is_complete"] is False
    assert metrics.agent_metrics["Video_MME/missing"] == 1


def test_a_judge_policy_cohort_needs_a_judge(tmp_path) -> None:
    with pytest.raises(ValueError, match="needs judge_model_server"):
        make_server(tmp_path, judge=None)


async def test_rule_policies_never_call_a_configured_judge(tmp_path) -> None:
    server, rows, calls = make_server(tmp_path, answer_policy="boxed_only_v1")
    result = await server.verify(request(rows[0], r"\boxed{B}"))
    assert (result.outcome, calls) == ("correct", [])


def test_letter_prompts_are_prepared_for_their_own_benchmark_only() -> None:
    def prepare(benchmark, **kwargs):
        record = v1_record() if benchmark == "videomme_v1" else lv_record()
        return prepare_dataset(
            [record], benchmark=benchmark, dataset_revision=REVISION, media=media(), setting="single_turn", **kwargs
        )

    (row,), _ = prepare("lvbench", prompt="lvbench")
    text = row["responses_create_params"]["input"][0]["content"][1]["text"]
    assert text == (
        "Which invented shape moves?\n(A) Circle\n(B) Square\n(C) Triangle\n(D) Star\n"
        "Answer with the option's letter from the given choices directly."
    )
    with pytest.raises(ValueError, match="their own benchmark"):
        prepare("lvbench", prompt="videomme_v1")
    with pytest.raises(ValueError, match="their own benchmark"):
        prepare_dataset(
            [v1_record()],
            benchmark="videomme_v1",
            dataset_revision=REVISION,
            media=media(source="/data/source.mp4", sandbox_path="video/shapes.mp4"),
            setting="agentic",
            prompt="videomme_v1",
        )


def v2_question(options: tuple[str, ...]) -> VideoQuestion:
    return VideoQuestion(
        task_id="videomme_v2:shape-1",
        dataset_source="https://example.invalid/invented-shapes",
        dataset_revision=REVISION,
        video_id="shapes",
        video_source="https://example.invalid/shapes.mp4",
        video_sha256="b" * 64,
        question="Which shape moves?",
        options=options,
    )


def test_videomme_v2_letter_prompt_names_the_available_letters() -> None:
    row = prepare_single_turn(
        v2_question(("Circle", "Square", "Triangle", "Star", "Moon", "Sun", "Cross", "None of the above")),
        expected_answer="B",
        video_url="data:video/mp4;base64,AAAA",
        prompt="videomme_v2",
    )
    assert row["responses_create_params"]["input"][0]["content"][1]["text"] == (
        "Which shape moves?\nA. Circle\nB. Square\nC. Triangle\nD. Star\nE. Moon\nF. Sun\nG. Cross\n"
        "H. None of the above\nRespond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.\n"
        "Answer: "
    )
    with pytest.raises(ValueError, match="exactly four options"):
        prepare_single_turn(
            v2_question(("Circle", "Square", "Triangle")),
            expected_answer="B",
            video_url="data:video/mp4;base64,AAAA",
            prompt="videomme_v1",
        )


async def test_videomme_v2_scores_the_judges_choice(tmp_path) -> None:
    row = prepare_single_turn(
        v2_question(("Circle", "Square", "Triangle")),
        expected_answer="B",
        video_url="data:video/mp4;base64,AAAA",
        prompt="videomme_v2",
    )
    cohort = {"dataset_revision": REVISION, "answer_policy": "llm_extraction_v1", "task_ids": [row["task_id"]]}
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(cohort))
    client, calls = scripted_judge('{"choice": "B"}')
    config = dict(name="videomme_v2", host="127.0.0.1", port=8000, entrypoint="app.py", cohort_manifest=path)
    server = VideoMMEV2ResourcesServer(
        config=VideoMMEV2ResourcesServerConfig(**config, judge_model_server=JUDGE), server_client=client
    )
    body = VideoMMEV2VerifyRequest.model_validate({**row, "response": response("The square moves. Answer: B")})
    result = await server.verify(body)
    assert (result.outcome, result.predicted_answer, result.reward, result.answer_policy) == (
        "correct",
        "B",
        1.0,
        "llm_extraction_v1",
    )
    assert "Respond with only the letter (A, B, or C)" in calls[0]["json"].messages[0]["content"]
    with pytest.raises(ValueError, match="needs judge_model_server"):
        VideoMMEV2ResourcesServer(config=VideoMMEV2ResourcesServerConfig(**config), server_client=client)


@pytest.mark.parametrize("server_name", ["videomme_v2", "videomme_v1", "lvbench"])
def test_llm_extraction_configs_name_an_openai_compatible_judge(tmp_path, server_name) -> None:
    path = Path(__file__).parents[2] / server_name / "configs" / f"{server_name}_single_turn_llm_extraction.yaml"
    plain = OmegaConf.load(path.with_name(f"{server_name}_single_turn.yaml"))
    values = {
        f"{server_name}_cohort_manifest": str(tmp_path / "cohort.json"),
        "judge_base_url": "https://judge.example.invalid/v1",
        "judge_api_key": "placeholder",
        "judge_model_name": "judge-model",
    }
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(path), OmegaConf.create(values)), resolve=True)
    config_type = VideoMMEV2ResourcesServerConfig if server_name == "videomme_v2" else VideoMCQAResourcesServerConfig
    resource = config_type(
        name=server_name, host="127.0.0.1", port=8001, **config[server_name]["resources_servers"][server_name]
    )
    assert resource.judge_model_server == ModelServerRef(type="responses_api_models", name="video_mcqa_answer_judge")
    assert config["video_mcqa_answer_judge"]["responses_api_models"]["openai_model"]["openai_model"] == "judge-model"
    # The agent and Environment Server are the single-turn config's, so runs name the same agent.
    for name in (f"{server_name}_single_turn", f"{server_name}_single_turn_environment"):
        assert config[name] == OmegaConf.to_container(plain[name], resolve=True)
