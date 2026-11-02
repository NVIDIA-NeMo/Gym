# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import math
import time

import pytest

from benchmarks.vials.prepare import build_row
from nemo_gym.judge import JudgeError
from nemo_gym.verifier_fixture import exercise_verifier_fixture

from ..app import VERIFIER_FIXTURE, VialsVerifyRequest
from ..fixture_judge import FixtureJudgeClient, create_fixture_server, judge_response
from ..upstream_protocol import (
    JUDGE_SYSTEM,
    MULTIMODAL_JUDGE_SYSTEM,
    SYSTEM_PROMPT,
    USER_INSTRUCTION_TAIL,
    extract_answer_block,
)


IMAGE = {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"}


def _response(text: str = "", reasoning: str = "") -> dict:
    output = []
    if reasoning:
        output.append({"id": "r", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    if text:
        output.append(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    return {
        "id": "resp",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def _request(text: str = "", *, reasoning: str = "", images: bool = True, **metadata) -> VialsVerifyRequest:
    content = ([IMAGE, IMAGE] if images else []) + [{"type": "input_text", "text": "Q?"}]
    return VialsVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": content}]},
            "response": _response(text, reasoning),
            "verifier_metadata": {
                "id": "t1",
                "question": "Which lanes show the product?",
                "gtfa": "lanes 2 and 3",
                "domain": "Blotting",
            }
            | metadata,
        }
    )


def _verdict(equivalent: bool, reasoning: str = "same") -> str:
    return json.dumps({"reasoning": reasoning, "equivalent": equivalent})


def _verify(request: VialsVerifyRequest, client: FixtureJudgeClient, **config):
    return asyncio.run(create_fixture_server(client, **config).verify(request))


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


def test_judge_request_matches_upstream() -> None:
    client = FixtureJudgeClient()
    result = _verify(_request("REASONING: bands\nANSWER: Lanes 2 and 3"), client)

    assert result.reward == 1.0 and result.extracted_answer == "Lanes 2 and 3" and result.judge_parse_ok
    (sent,) = client.requests
    assert sent["max_output_tokens"] == 512 and "temperature" not in sent
    assert sent["text"]["format"]["type"] == "json_schema"
    assert (sent["input"][0]["role"], sent["input"][0]["content"]) == ("system", JUDGE_SYSTEM)
    assert sent["input"][1]["content"] == (
        "Question:\nWhich lanes show the product?\n\nGround-truth final answer:\nlanes 2 and 3\n\n"
        "Candidate final answer:\nLanes 2 and 3\n\n"
        "Are the candidate and ground-truth final answers equivalent?"
    )


def test_lead_bounds_reach_the_judge() -> None:
    client = FixtureJudgeClient([_verdict(True)])
    _verify(_request("ANSWER: 41", answer_lower_bound=40.0, answer_upper_bound=None), client)

    assert (
        "Lead-assigned acceptable numeric range (inclusive): lower=40.0, upper=n/a."
        in (client.requests[0]["input"][1]["content"])
    )


def test_missing_answer_scores_zero_without_a_judge_call() -> None:
    client = FixtureJudgeClient([])
    result = _verify(_request("The answer is lanes 2 and 3."), client)

    assert result.reward == 0.0 and result.extracted_answer is None and not client.requests


def test_empty_message_falls_back_to_reasoning_text() -> None:
    client = FixtureJudgeClient()
    result = _verify(_request("", reasoning="thinking...\nANSWER: lanes 2 and 3"), client)

    assert result.reward == 1.0 and result.answer_recovered_from_reasoning


def test_unverifiable_verdict_is_regraded_with_the_images() -> None:
    client = FixtureJudgeClient([_verdict(False, "UNVERIFIABLE: lane labels"), _verdict(True, "image shows it")])
    result = _verify(_request("ANSWER: the 2nd and 3rd lanes"), client)

    assert result.reward == 1.0 and result.judge_multimodal and result.judge_reasoning == "image shows it"
    regrade = client.requests[1]["input"]
    assert regrade[0]["content"] == MULTIMODAL_JUDGE_SYSTEM
    assert [part["type"] for part in regrade[1]["content"]] == ["input_image", "input_image", "input_text"]
    assert regrade[1]["content"][-1]["text"].endswith(
        "Using the image only to resolve labelling ambiguities, are the candidate and ground-truth final answers "
        "equivalent?"
    )


def test_unverifiable_verdict_stands_when_the_fallback_is_off_or_there_is_no_image() -> None:
    for request, config in (
        (_request("ANSWER: x"), {"multimodal_judge_fallback": False}),
        (_request("ANSWER: x", images=False), {}),
    ):
        client = FixtureJudgeClient([_verdict(False, "UNVERIFIABLE: labels")])
        result = _verify(request, client, **config)
        assert result.reward == 0.0 and result.judge_unverifiable and not result.judge_multimodal
        assert len(client.requests) == 1


def test_unparseable_verdict_is_retried_then_scored_incorrect() -> None:
    client = FixtureJudgeClient(["", "not json", _verdict(True)])
    assert _verify(_request("ANSWER: lanes 2 and 3"), client).reward == 1.0

    client = FixtureJudgeClient(["", "not json", "still not json"])
    result = _verify(_request("ANSWER: lanes 2 and 3"), client)
    assert result.reward == 0.0 and result.judge_parse_ok is False and len(client.requests) == 3


def test_fenced_verdict_is_parsed() -> None:
    fence = "`" * 3
    client = FixtureJudgeClient([f"{fence}json\n{_verdict(True)}\n{fence}"])
    assert _verify(_request("ANSWER: lanes 2 and 3"), client).reward == 1.0


def test_failed_judge_call_raises_judge_error() -> None:
    server = create_fixture_server()

    async def fail(**_):
        raise ConnectionError("judge down")

    server.server_client.post = fail
    with pytest.raises(JudgeError, match="judge down"):
        asyncio.run(server.verify(_request("ANSWER: lanes 2 and 3")))


def test_compute_metrics_reports_upstream_accuracy_pass_at_k_and_pass_power_k() -> None:
    def rollout(reward, domain, answer="a"):
        return {"reward": reward, "domain": domain, "extracted_answer": answer, "judge_parse_ok": True}

    tasks = [
        [rollout(1.0, "Blotting"), rollout(1.0, "Blotting"), rollout(1.0, "Blotting")],
        [rollout(1.0, "Phylogenetics"), rollout(0.0, "Phylogenetics"), rollout(0.0, "Phylogenetics", None)],
        [rollout(0.0, "Phylogenetics"), rollout(0.0, "Phylogenetics"), rollout(0.0, "Phylogenetics")],
    ]
    metrics = create_fixture_server().compute_metrics(tasks)

    assert math.isclose(metrics["pass@1[avg-of-3]/accuracy"], 100.0 * 4 / 9)
    assert math.isclose(metrics["pass@3/accuracy"], 100.0 * 2 / 3)
    assert math.isclose(metrics["pass^3/accuracy"], 100.0 / 3)
    assert math.isclose(metrics["Phylogenetics/pass@3/accuracy"], 50.0)
    assert math.isclose(metrics["rate/no_answer"], 100.0 / 9)
    assert math.isclose(metrics["accuracy"], 100.0 * 4 / 9)
    assert math.isclose(metrics["Phylogenetics/accuracy"], 100.0 / 6)
    assert math.isclose(metrics["Blotting/pass^3/accuracy"], 100.0)
    assert math.isclose(metrics["Phylogenetics/pass^3/accuracy"], 0.0)
    assert math.isclose(metrics["Phylogenetics/pass^1/accuracy"], 50.0)
    key = create_fixture_server().get_key_metrics(metrics)
    assert set(key) == {"accuracy", "pass@1[avg-of-3]/accuracy", "pass@3/accuracy", "pass^3/accuracy"}


def test_zero_imputed_rows_without_verify_fields_keep_their_task_domain() -> None:
    scored = {"reward": 1.0, "domain": "Blotting", "extracted_answer": "a", "judge_parse_ok": True}
    imputed = {"reward": 0.0, "verifier_metadata": {"domain": "Blotting"}}
    tasks = [[imputed, scored, scored], [imputed, imputed, imputed], [{"reward": 0.0}]]
    metrics = create_fixture_server().compute_metrics(tasks)

    assert math.isclose(metrics["accuracy"], 100.0 * 2 / 7)
    assert math.isclose(metrics["Blotting/accuracy"], 100.0 * 2 / 6)
    assert math.isclose(metrics["Blotting/pass@3/accuracy"], 50.0)
    assert math.isclose(metrics["Blotting/pass^3/accuracy"], 0.0)


def test_accuracy_pools_attempts_when_some_are_masked() -> None:
    def rollout(reward):
        return {"reward": reward, "domain": "Blotting", "extracted_answer": "a", "judge_parse_ok": True}

    # Task 2 lost two attempts to masking: upstream pools 2 of 4 attempts, not (1/3 + 1) / 2.
    metrics = create_fixture_server().compute_metrics([[rollout(1.0), rollout(0.0), rollout(0.0)], [rollout(1.0)]])

    assert math.isclose(metrics["accuracy"], 50.0)
    assert math.isclose(metrics["Blotting/accuracy"], 50.0)
    assert math.isclose(metrics["pass@1[avg-of-3]/accuracy"], 100.0 * 2 / 3)


def test_build_row_keeps_upstream_prompt_order_and_metadata() -> None:
    record = {
        "id": "t9",
        "question": "Is A &gt; B?",
        "answer": "A &amp; B",
        "domain": "Molecular structure",
        "images": [{"bytes": b"\x89PNG", "path": None}, {"bytes": b"\xff\xd8", "path": None}],
        "metadata": {
            "secondary_tags": ["SMILES"],
            "image_filenames": ["a.png", "b.JPG"],
            "answer_lower_bound": float("nan"),
            "answer_upper_bound": 3.0,
        },
    }
    row = build_row(record)

    system, user = row["responses_create_params"]["input"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert [part["type"] for part in user["content"]] == ["input_image", "input_image", "input_text"]
    assert user["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert user["content"][2]["text"] == f"Is A > B?\n\n{USER_INSTRUCTION_TAIL}"
    assert row["responses_create_params"]["temperature"] == 1.0
    assert row["responses_create_params"]["max_output_tokens"] == 32768
    assert row["verifier_metadata"] | {} == {
        "id": "t9",
        "question": "Is A > B?",
        "gtfa": "A & B",
        "domain": "Molecular structure",
        "secondary_tags": ["SMILES"],
        "answer_lower_bound": None,
        "answer_upper_bound": 3.0,
    }


def test_build_row_rejects_a_task_without_images() -> None:
    record = {
        "id": "t0",
        "question": "q",
        "answer": "a",
        "domain": "Other",
        "images": [],
        "metadata": {"image_filenames": []},
    }
    with pytest.raises(ValueError, match="0 images"):
        build_row(record)


@pytest.mark.parametrize(
    "text",
    [
        " " * 200_000 + "x",
        "\n".join([" \t" * 5_000] * 40),
        "ANSWER: a" + " " * 200_000 + "]",
        "ANSWER: $" + " " * 200_000 + "a",
        # Internal whitespace with no closing delimiter reaches every suffix check.
        "ANSWER: a" + " " * 200_000 + "b",
    ],
)
def test_answer_extraction_is_linear_on_long_whitespace(text: str) -> None:
    start = time.perf_counter()
    extract_answer_block(text)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("REASONING: r\n- **ANSWER:** \\[ A, B \\]", "A, B"),
        ("1. ANSWER: $$5$$", "5"),
        ("ANSWER: x  \\]", "x"),
        ("ANSWER: $$$", None),
        ("ANSWER: " + "`" * 3 + "txt\nfoo  " + "`" * 3, "foo"),
        ("the answer: 5", "5"),
        ("ANSWER: first\nANSWER: **last**", "last"),
    ],
)
def test_answer_extraction_matches_upstream_cases(text: str, expected: str | None) -> None:
    assert extract_answer_block(text) == expected


def test_judge_response_helper_round_trips() -> None:
    assert judge_response("x")["output"][0]["content"][0]["text"] == "x"
