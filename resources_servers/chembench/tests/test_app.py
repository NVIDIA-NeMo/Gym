# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.chembench.app import (
    VERIFIER_FIXTURE,
    ChembenchResourcesServer,
    ChembenchResourcesServerConfig,
    ChembenchVerifier,
    ChembenchVerifyRequest,
    extract_answer,
)
from resources_servers.chembench.task_data import TaskData


SERVER_DIR = Path(__file__).parents[1]


def response(text: str | None) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="test",
        created_at=0,
        model="test",
        object="response",
        output=[]
        if text is None
        else [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def request(text: str | None, question_type="mcq", expected_answer="A, C") -> ChembenchVerifyRequest:
    return ChembenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Chemistry question"}]},
        response=response(text),
        verifier_metadata={"question_type": question_type, "expected_answer": expected_answer, "uuid": "task-1"},
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[ANSWER]C, A, A[/ANSWER]", "A, C"),
        ("[ANSWER]\nA C\n[/ANSWER]", "A, C"),
        ("[ANSWER]A, C.[/ANSWER]", "A, C"),
        ("[ANSWER](A, C)[/ANSWER]", "A, C"),
        ("[ANSWER]A and C[/ANSWER]", "A, C"),
        ("[ANSWER]A, B, and C.[/ANSWER]", "A, B, C"),
        ("[ANSWER]The answers are A, C.[/ANSWER]", "A, C"),
        ("[ANSWER]Consider B; final answer is (C, A, A).[/ANSWER]", "A, C"),
        ("[ANS]B[/ANS]", "B"),
        ("<ANSWER>B</ANSWER>", "B"),
        ("<ans>B</ans>", "B"),
        ("Reasoning A. [ANSWER]B[/ANSWER]", "B"),
        ("[ANSWER]A[/ANSWER]\n[ANSWER]B[/ANSWER]", "B"),
        ("A, C", "A, C"),
        ("<think>[ANSWER]A[/ANSWER]</think>[ANSWER]B[/ANSWER]", "B"),
        ("<thinking>[ANSWER]A[/ANSWER]</thinking>B", "B"),
        ("hidden A</think>B", "B"),
        ("<think>[ANSWER]A[/ANSWER]", None),
        ("", None),
        ("I considered A and C.", None),
        ("[ANSWER]A or C[/ANSWER]", "C"),
        ("[ANSWER]ANSWER is A[/ANSWER]", "A"),
        ("[ANSWER]Consider A; final answer is (B).[/ANSWER]", "B"),
        ("[ANSWER]The answer is C. END[/ANSWER]", "C"),
        ("[ANSWER]ANSWER[/ANSWER]", None),
        ("[ANSWER]AB[/ANSWER]", None),
        ("[ANSWER]a B2[/ANSWER]", None),
        ("[ANSWER]The answer is A[/ANSWER] B", "A"),
        ("[ANSWER]Answer A[/ANSWER][ANSWER]Answer B[/ANSWER]", "B"),
        ("[ANSWER]Answer A[/ANSWER][ANSWER]unsure[/ANSWER]", None),
        ("<think>[ANSWER]Answer C[/ANSWER]</think>[ANSWER]Answer B[/ANSWER]", "B"),
        ("<ans>Answer: C</ans>", "C"),
        ("ANSWER is A", None),
        ("[ANSWER]a[/ANSWER]", None),
        ("[ANSWER]A[/ANS]", None),
        ("[ANSWER]A", None),
        ("[ANSWER]A[/ANSWER][ANSWER]unsure[/ANSWER]", None),
    ],
)
def test_mcq_extraction(text, expected):
    assert extract_answer(text, "mcq") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[ANSWER]5[/ANSWER]", 5.0),
        ("[ANSWER]\n-1.25e-3\n[/ANSWER]", -0.00125),
        ("<ans>6.02E23</ans>", 6.02e23),
        ("+.5", 0.5),
        ("5.", 5.0),
        ("[ANSWER]1[/ANSWER] [ANSWER]2[/ANSWER]", 2.0),
        ("<think>12</think>[ANSWER]2[/ANSWER]", 2.0),
        ("[ANSWER]1e-3 mol[/ANSWER]", 0.001),
        ("[ANSWER]5 units[/ANSWER]", 5.0),
        ("[ANSWER]Initially 4; final answer is 5 units.[/ANSWER]", 5.0),
        ("[ANSWER]The answer is -1.25e-3 mol[/ANSWER]", -0.00125),
        ("[ANSWER]Mass: +6.02E23 units[/ANSWER]", 6.02e23),
        ("[ANSWER]The answer is -.5 units[/ANSWER]", -0.5),
        ("[ANSWER]5 units[/ANSWER] 99", 5.0),
        ("[ANSWER]4 units[/ANSWER][ANSWER]5 units[/ANSWER]", 5.0),
        ("[ANSWER]4 units[/ANSWER][ANSWER]unknown[/ANSWER]", None),
        ("[ANSWER]4 then 1e999 units[/ANSWER]", None),
        ("[ANSWER]C6H12O6[/ANSWER]", None),
        ("[ANSWER]1e- units[/ANSWER]", None),
        ("<ans>5 units</ans>", 5.0),
        ("5 units", None),
        ("[ANSWER]1,234[/ANSWER]", None),
        ("[ANSWER]NaN[/ANSWER]", None),
        ("[ANSWER]inf[/ANSWER]", None),
        ("[ANSWER]1e999[/ANSWER]", None),
        ("[ANSWER]2 * 10^3[/ANSWER]", 2000.0),
        ("[ANSWER]2 or 3[/ANSWER]", 3.0),
        ("There are 5 peaks.", None),
        ("[ANSWER]5[/ANS]", None),
        ("", None),
        ("[ANSWER]3.5 × 10^-3[/ANSWER]", 0.0035),
        ("1/2", 0.5),
        ("[ANSWER]The answer is -1/2 units.[/ANSWER]", -0.5),
        ("[ANSWER]3.5 \\times 10^{-3}[/ANSWER]", 0.0035),
        ("[ANSWER]1/0[/ANSWER]", None),
        ("[ANSWER]1 / 1e999[/ANSWER]", None),
        ("[ANSWER]2 * 10^999[/ANSWER]", None),
        ("[ANSWER]2 + (3)[/ANSWER]", None),
        ("[ANSWER]2 * 3[/ANSWER]", None),
        ("[ANSWER]\\frac{1}{2}[/ANSWER]", None),
        ("[ANSWER]sqrt(4)[/ANSWER]", None),
        ("[ANSWER]2 * 10^[/ANSWER]", None),
    ],
)
def test_numeric_extraction(text, expected):
    assert extract_answer(text, "numeric") == expected


@pytest.mark.parametrize(
    ("text", "kind", "gold", "reward"),
    [
        ("[ANSWER]C, A[/ANSWER]", "mcq", "A, C", 1.0),
        ("[ANSWER]A, C[/ANSWER]", "mcq", "C, A", 1.0),
        ("[ANSWER]A[/ANSWER]", "mcq", "A, C", 0.0),
        ("[ANSWER]A, B, C[/ANSWER]", "mcq", "A, C", 0.0),
        ("[ANSWER]B[/ANSWER]", "mcq", "A, C", 0.0),
        ("[ANSWER]5.0[/ANSWER]", "numeric", "5", 1.0),
        ("[ANSWER]5.000000001[/ANSWER]", "numeric", "5", 1.0),
        ("[ANSWER]1e-3[/ANSWER]", "numeric", "0.001", 1.0),
        ("[ANSWER]1e-3 mol[/ANSWER]", "numeric", "1", 0.0),
        ("[ANSWER]ANSWER is A[/ANSWER]", "mcq", "A", 1.0),
        ("[ANSWER]ANSWER is B[/ANSWER]", "mcq", "A", 0.0),
        ("[ANSWER]A or C[/ANSWER]", "mcq", "A, C", 0.0),
        ("[ANSWER]5 units[/ANSWER]", "numeric", "5", 1.0),
        ("[ANSWER]5 units[/ANSWER]", "numeric", "6", 0.0),
        ("[ANSWER]1e-3 mol[/ANSWER]", "numeric", "0.001", 1.0),
        ("[ANSWER]First 5, then 6 units[/ANSWER]", "numeric", "5", 0.0),
        (None, "mcq", "A", 0.0),
        ("", "numeric", "5", 0.0),
        ("[ANSWER]3.5 × 10^-3[/ANSWER]", "numeric", "0.0035", 1.0),
    ],
)
async def test_verify(text, kind, gold, reward):
    body = request(text, kind, gold)
    before = body.model_dump()
    result = await ChembenchVerifier().verify(body)
    assert result.reward == reward
    assert result.no_answer == (result.predicted_answer is None)
    assert result.verifier_metadata.uuid == "task-1"
    assert result.responses_create_params == body.responses_create_params
    assert result.response == body.response
    assert body.model_dump() == before
    assert json.loads(result.model_dump_json())["reward"] == reward


@pytest.mark.parametrize("answer", ["A, C.", "(A, C)", "A and C", "The answers are A, C."])
@pytest.mark.parametrize("gold,reward", [("A, C", 1.0), ("C", 0.0), ("A, B, C", 0.0)])
async def test_multiselect_fallback_preserves_complete_answer_set(answer, gold, reward):
    result = await ChembenchVerifier().verify(request(f"[ANSWER]{answer}[/ANSWER]", "mcq", gold))
    assert result.predicted_answer == "A, C"
    assert result.reward == reward


async def test_reasoning_items_and_refusals():
    body = request(None)
    body.response.output = NeMoGymResponse.model_validate(
        {
            **body.response.model_dump(),
            "output": [
                {
                    "id": "r",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "[ANSWER]A, C[/ANSWER]"}],
                },
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "I cannot answer."}],
                },
            ],
        }
    ).output
    result = await ChembenchVerifier().verify(body)
    assert result.reward == 0.0
    assert result.predicted_answer is None


def test_server_setup():
    server = ChembenchResourcesServer(
        config=ChembenchResourcesServerConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="chembench"),
        server_client=MagicMock(spec=ServerClient),
    )
    assert any(route.path == "/verify" for route in server.setup_webserver().routes)


@pytest.mark.parametrize(
    ("kind", "gold"),
    [("mcq", ""), ("mcq", "AB"), ("numeric", "NaN"), ("numeric", "inf"), ("numeric", "five"), ("other", "5")],
)
def test_bad_task_metadata(kind, gold):
    with pytest.raises(ValidationError):
        TaskData(question_type=kind, expected_answer=gold)


async def test_verifier_fixture():
    await exercise_verifier_fixture(
        VERIFIER_FIXTURE,
        reward_range=(0.0, 1.0),
        higher_is_better=True,
        determinism="unknown",
    )


async def test_example_rows():
    rows = [json.loads(line) for line in (SERVER_DIR / "data/example.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert {row["verifier_metadata"]["question_type"] for row in rows} == {"mcq", "numeric"}
    assert any("," in row["verifier_metadata"]["expected_answer"] for row in rows)
    assert any(row["verifier_metadata"]["in_human_subset"] for row in rows)
    for row in rows:
        gold = row["verifier_metadata"]["expected_answer"]
        body = ChembenchVerifyRequest(**row, response=response(f"[ANSWER]{gold}[/ANSWER]"))
        assert (await ChembenchVerifier().verify(body)).reward == 1.0


def test_example_schema():
    adapter = load_task_data_schema(SERVER_DIR)
    path = SERVER_DIR / "data/example.jsonl"
    report = validate_jsonl_rows("chembench", adapter, str(path), path.read_text().splitlines())
    assert report.rows == 5
    assert report.error_rows == 0
    assert not report.unknown_keys
    assert not report.misplaced_keys


@pytest.mark.parametrize(
    "gold,prediction,tolerance,reward",
    [
        ("100", "100", None, 1),
        ("100", "100.5", None, 1),
        ("100", "99.5", None, 1),
        ("100", "101", None, 0),
        ("100", "99", None, 0),
        ("100", "102", None, 0),
        ("100", "105", 10, 1),  # Explicit tolerances are absolute, despite their name.
        ("100", "110", 10, 0),
        ("100", "100.05", 0.01, 0),
        ("100", "100", 0, 0),  # Zero must not fall back to the default.
        ("100", "100", -1, 0),
        ("-100", "-100", None, 0),
        ("0", "0", None, 0),
        ("-100", "-99.5", 1, 1),
        ("0", "0.5", 1, 1),
        ("100", "unknown", 10, 0),
    ],
)
async def test_upstream_numeric_tolerance(gold, prediction, tolerance, reward):
    body = request(f"[ANSWER]{prediction}[/ANSWER]", "numeric", gold)
    body.verifier_metadata.relative_tolerance = tolerance
    result = await ChembenchVerifier().verify(body)
    assert result.reward == reward
    assert result.verifier_metadata.relative_tolerance == tolerance


async def test_mcq_ignores_numeric_tolerance():
    body = request("[ANSWER]C, A[/ANSWER]")
    body.verifier_metadata.relative_tolerance = 0
    assert (await ChembenchVerifier().verify(body)).reward == 1


@pytest.mark.parametrize("tolerance", [float("inf"), float("-inf"), float("nan")])
def test_tolerance_must_be_finite(tolerance):
    with pytest.raises(ValidationError):
        TaskData(question_type="numeric", expected_answer="100", relative_tolerance=tolerance)
