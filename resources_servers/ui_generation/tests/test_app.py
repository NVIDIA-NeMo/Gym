# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymResponse
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding
from nemo_gym.verification import JudgeError, exercise_verifier_fixture

from ..app import (
    SCHEMA_VERSION,
    VERDICT_KEYS,
    VERIFIER_FIXTURE,
    UiGenerationResourcesServer,
    UiGenerationResourcesServerConfig,
    UiGenerationVerifyRequest,
    final_answer_text,
    parse_judge_completion,
    prompt_text,
    validate_verdict,
)
from ..verifier_fixture import fixture_judge_reply


REFERENCE = "```tsx\nexport default function Page() { return <main>Hello</main>; }\n```"
CANDIDATE = "Here is the page.\n\n```tsx\nexport default function Page() { return <h1>Hi – there</h1>; }\n```"


def verdict(adherence_score: int = 3, quality_score: int = 2, **overrides) -> dict:
    return {
        "schema_version": "ui-code-judge-v1",
        "prompt_adherence": {"score": adherence_score, "rationale": "Implements the page."},
        "quality": {"score": quality_score, "rationale": "Plausible code with gaps."},
        "critical_issues": ["No dark mode toggle."],
        "reference_differences": [{"difference": "Uses an h1", "material_to_prompt": False}],
        "uncertainty": [],
        "execution_verified": False,
    } | overrides


def policy_response(*items: dict, status: str = "completed", **fields) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="policy",
        created_at=0,
        model="policy",
        object="response",
        status=status,
        output=list(items),
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        **fields,
    )


def message(text: str) -> dict:
    return {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def reasoning(text: str) -> dict:
    return {"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def make_request(response: NeMoGymResponse | None = None, input_messages=None) -> UiGenerationVerifyRequest:
    return UiGenerationVerifyRequest.model_validate(
        {
            "responses_create_params": {
                "input": input_messages or [{"role": "user", "content": "Build a page. Use NextJS in one tsx file."}]
            },
            "response": (response or policy_response(message(CANDIDATE))).model_dump(),
            "verifier_metadata": {"id": "task-1", "reference_answer": REFERENCE, "category": "nextjs"},
        }
    )


def make_server(**config) -> UiGenerationResourcesServer:
    server_config = UiGenerationResourcesServerConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        name="ui_generation",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_chat_completions_create_params={"messages": [], "temperature": 1.0, "top_p": 1.0, "max_tokens": 65536},
        **config,
    )
    return UiGenerationResourcesServer(config=server_config, server_client=MagicMock(spec=ServerClient))


def completion(content: str, finish_reason: str = "stop") -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-1",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
            ],
        }
    )


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


def test_the_rubric_asks_for_the_verdict_the_parser_accepts() -> None:
    rubric = make_server()._judge_rubric
    # The schema the rubric shows the judge must be the one validate_verdict enforces.
    example = json.loads(rubric[rubric.index("{") : rubric.rindex("}") + 1])
    assert set(example) == VERDICT_KEYS
    assert example["schema_version"] == SCHEMA_VERSION
    assert validate_verdict(example) is None
    # The case keys judge_request sends are the ones the rubric names.
    for key in ("prompt", "original_guideline", "candidate_answer"):
        assert f'"{key}"' in rubric


def test_the_judge_request_carries_the_rubric_and_the_case_as_json() -> None:
    server = make_server()
    request = server.judge_request("Build a page.", REFERENCE, CANDIDATE)
    system, user = request.messages
    assert system == {"role": "system", "content": server._judge_rubric}
    assert user["role"] == "user"
    # The keys the rubric names, in this order, with non-ASCII kept as is.
    assert user["content"] == json.dumps(
        {"prompt": "Build a page.", "original_guideline": REFERENCE, "candidate_answer": CANDIDATE}, ensure_ascii=False
    )
    assert "–" in user["content"]
    assert (request.temperature, request.top_p, request.max_tokens) == (1.0, 1.0, 65536)
    assert server.config.judge_chat_completions_create_params.messages == []


def test_the_prompt_joins_every_user_message() -> None:
    request = make_request(
        input_messages=[
            {"role": "system", "content": "You write UIs."},
            {"role": "user", "content": "Part one."},
            {"role": "user", "content": [{"type": "input_text", "text": "Part two."}]},
        ]
    )
    assert prompt_text(request) == "Part one.\n\nPart two."
    as_string = request.model_copy(update={"responses_create_params": request.responses_create_params.model_copy()})
    as_string.responses_create_params.input = "Just this."
    assert prompt_text(as_string) == "Just this."


def test_the_candidate_is_the_last_assistant_message_without_reasoning() -> None:
    response = policy_response(
        reasoning("thinking about layout"),
        message("first attempt"),
        message("<think>hidden plan</think>\n```tsx\nfinal\n```"),
    )
    assert final_answer_text(response) == "```tsx\nfinal\n```"
    assert final_answer_text(policy_response(reasoning("only thoughts"))) == ""


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(verdict()),
        "```json\n" + json.dumps(verdict(), indent=2) + "\n```",
        "  ```\n" + json.dumps(verdict()) + "\n```\n",
    ],
)
def test_a_valid_verdict_parses_bare_or_fenced(content: str) -> None:
    parsed, judge_reasoning = parse_judge_completion(completion(content))
    assert parsed == verdict()
    assert judge_reasoning is None


def test_a_leading_think_block_is_the_judge_reasoning() -> None:
    parsed, judge_reasoning = parse_judge_completion(
        completion("<think>\nweigh it\n</think>\n" + json.dumps(verdict()))
    )
    assert parsed == verdict()
    assert judge_reasoning == "weigh it"


@pytest.mark.parametrize(
    "content,finish_reason,error",
    [
        (json.dumps(verdict()), "length", "finish_length"),
        ("The code is good. Score 4.", "stop", "invalid_json"),
        ("```json\n" + json.dumps(verdict()) + "\n``` trailing", "stop", "invalid_json"),
        ("<think>never closed " + json.dumps(verdict()), "stop", "invalid_json"),
        (json.dumps([verdict()]), "stop", "invalid_schema"),
    ],
)
def test_a_reply_without_a_complete_verdict_is_a_judge_failure(content, finish_reason, error) -> None:
    with pytest.raises(JudgeError, match=error):
        parse_judge_completion(completion(content, finish_reason))


def test_a_reply_without_choices_is_a_judge_failure() -> None:
    reply = completion("{}").model_copy(update={"choices": []})
    with pytest.raises(JudgeError, match="malformed_response"):
        parse_judge_completion(reply)


@pytest.mark.parametrize(
    "bad,error",
    [
        ({"extra": 1}, "keys"),
        ({"schema_version": "v2"}, "schema_version"),
        ({"quality": {"score": 5, "rationale": "x"}}, "quality.score 5"),
        ({"quality": {"score": 3.0, "rationale": "x"}}, "quality.score 3.0"),
        ({"quality": {"score": True, "rationale": "x"}}, "quality.score True"),
        ({"prompt_adherence": {"score": "3", "rationale": "x"}}, "prompt_adherence.score"),
        ({"prompt_adherence": {"score": 3}}, "exactly score and rationale"),
        ({"quality": {"score": 3, "rationale": None}}, "quality.rationale"),
        ({"critical_issues": "none"}, "critical_issues"),
        ({"uncertainty": [1]}, "uncertainty"),
        ({"reference_differences": [{"difference": "x", "material_to_prompt": "no"}]}, "reference_differences"),
        ({"execution_verified": True}, "execution_verified"),
    ],
)
def test_the_verdict_schema_is_strict(bad: dict, error: str) -> None:
    obj = verdict(**bad)
    if "extra" in bad:
        obj = verdict() | bad
    assert error in validate_verdict(obj)
    with pytest.raises(JudgeError, match="invalid_schema"):
        parse_judge_completion(completion(json.dumps(obj)))


@pytest.mark.parametrize(
    "reward_score,reward", [("quality", 0.5), ("prompt_adherence", 0.75), ("mean", 0.625), ("min", 0.5)]
)
def test_the_reward_is_the_chosen_score_over_four(reward_score: str, reward: float) -> None:
    assert make_server(reward_score=reward_score).reward_from_scores(3, 2) == reward


def test_verify_scores_the_judge_verdict() -> None:
    server = make_server()
    server.server_client.post = AsyncMock(return_value=fixture_judge_reply(json.dumps(verdict(3, 2))))

    result = asyncio.run(server.verify(make_request()))

    assert result.reward == 0.5
    assert result.reward_components == {"prompt_adherence": 0.75, "quality": 0.5}
    assert (result.prompt_adherence_score, result.quality_score) == (3, 2)
    assert result.judge_verdict == verdict(3, 2)
    assert not result.mask_sample
    call = server.server_client.post.await_args
    assert call.kwargs["server_name"] == "judge"
    assert call.kwargs["url_path"] == "/v1/chat/completions"
    sent = json.loads(call.kwargs["json"].messages[1]["content"])
    assert sent == {
        "prompt": "Build a page. Use NextJS in one tsx file.",
        "original_guideline": REFERENCE,
        "candidate_answer": CANDIDATE,
    }


@pytest.mark.parametrize(
    "response,reason",
    [
        (
            policy_response(
                message("```tsx\nexport default"),
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            "policy_response_truncated",
        ),
        (policy_response(reasoning("ran out before answering")), "empty_final_answer"),
        (policy_response(message("<think>only reasoning</think>")), "empty_final_answer"),
    ],
)
def test_a_cut_off_or_empty_answer_scores_zero_without_a_judge_call(response, reason) -> None:
    server = make_server()
    server.server_client.post = AsyncMock()

    result = asyncio.run(server.verify(make_request(response)))

    assert (result.reward, result.prompt_adherence_score, result.quality_score) == (0.0, 0, 0)
    assert result.failure_reason == reason
    assert not result.mask_sample
    server.server_client.post.assert_not_awaited()


def _verify_over_http(server: UiGenerationResourcesServer) -> dict:
    client = TestClient(server.setup_webserver())
    reply = client.post("/verify", json=make_request().model_dump(mode="json"))
    assert reply.status_code == 200
    return reply.json()


def test_an_unparseable_verdict_masks_the_row_instead_of_scoring_zero() -> None:
    server = make_server()
    server.server_client.post = AsyncMock(return_value=fixture_judge_reply("Looks great, 4/4"))

    row = _verify_over_http(server)

    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"
    assert "invalid_json" in row["failure_reason"]
    assert row["response"]["id"] == "policy"


def test_a_judge_cut_off_by_its_token_cap_masks_the_row() -> None:
    server = make_server()
    server.server_client.post = AsyncMock(
        return_value=fixture_judge_reply(json.dumps(verdict()), finish_reason="length")
    )

    row = _verify_over_http(server)

    assert row["mask_sample"] is True
    assert "finish_length" in row["failure_reason"]


def test_a_judge_outage_is_retried_then_masks_the_row(monkeypatch) -> None:
    monkeypatch.setattr("nemo_gym.verification.judge._judge_retry_wait_s", lambda attempt: 0)
    server = make_server(judge_max_attempts=2)
    server.server_client.post = AsyncMock(side_effect=ConnectionError("judge down"))

    row = _verify_over_http(server)

    assert server.server_client.post.await_count == 2
    assert row["mask_sample"] is True
    assert row["failure_kind"] == "judge_failed"


def test_a_rate_limited_judge_call_is_retried(monkeypatch) -> None:
    monkeypatch.setattr("nemo_gym.verification.judge._judge_retry_wait_s", lambda attempt: 0)
    server = make_server()
    server.server_client.post = AsyncMock(
        side_effect=[ConnectionError("reset"), fixture_judge_reply(json.dumps(verdict()))]
    )

    result = asyncio.run(server.verify(make_request()))

    assert result.reward == 0.5
    assert server.server_client.post.await_count == 2


def _graded(server: UiGenerationResourcesServer):
    invocation = VerifierInvocation(invocation_id="verify-ui")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = asyncio.run(server.verify(make_request()))
    except JudgeError as error:
        result = error
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation


def test_a_parsed_verdict_is_a_completed_judge_grading() -> None:
    server = make_server()
    server.server_client.post = AsyncMock(return_value=fixture_judge_reply(json.dumps(verdict()), reply_id="judge-7"))

    result, invocation = _graded(server)

    (operation,) = invocation.grading_operations
    assert operation.kind == "llm_judge"
    assert operation.outcome == "completed"
    assert [d.response_id for d in operation.dependencies] == ["judge-7"]
    assert operation.result_binding_sha256 == grading_result_binding(result)


def test_an_invalid_verdict_is_a_failed_judge_grading() -> None:
    server = make_server()
    server.server_client.post = AsyncMock(return_value=fixture_judge_reply("not json"))

    result, invocation = _graded(server)

    assert isinstance(result, JudgeError)
    (operation,) = invocation.grading_operations
    assert operation.outcome == "failed"


def test_judge_calls_respect_the_concurrency_limit() -> None:
    server = make_server(judge_max_concurrency=2)
    active = peak = 0

    async def slow_reply(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return fixture_judge_reply(json.dumps(verdict()))

    server.server_client.post = AsyncMock(side_effect=slow_reply)

    async def run_all():
        return await asyncio.gather(*(server.verify(make_request()) for _ in range(6)))

    results = asyncio.run(run_all())
    assert [r.reward for r in results] == [0.5] * 6
    assert peak == 2


def test_metrics_report_mean_scores_per_category() -> None:
    def rollout(category, adherence, quality):
        return {
            "reward": quality / 4,
            "prompt_adherence_score": adherence,
            "quality_score": quality,
            "verifier_metadata": {"category": category},
        }

    tasks = [
        [rollout("nextjs", 4, 4), rollout("nextjs", 2, 1)],
        [rollout("html", 3, 3)],
        [{"reward": 0.0, "verifier_metadata": {}}],
    ]

    assert make_server().compute_metrics(tasks) == {
        "nextjs/mean/prompt_adherence_score": 3.0,
        "nextjs/mean/quality_score": 2.5,
        "nextjs/mean/reward": 0.625,
        "html/mean/prompt_adherence_score": 3.0,
        "html/mean/quality_score": 3.0,
        "html/mean/reward": 0.75,
    }
