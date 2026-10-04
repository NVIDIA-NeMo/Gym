# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.frontiermath.app import (
    FrontierMathConfig,
    FrontierMathResourcesServer,
    FrontierMathVerifyRequest,
)
from resources_servers.frontiermath.grading import extract_answer, grade_answer


ANSWERS = json.loads(
    (Path(__file__).resolve().parents[3] / "benchmarks/indic/frontier_math/answers.json").read_text()
)["answers"]
BMO = ANSWERS[0]["expected_answer"]


@pytest.fixture
def server() -> FrontierMathResourcesServer:
    return FrontierMathResourcesServer(
        config=FrontierMathConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="frontiermath"),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize("answer", ANSWERS, ids=lambda answer: answer["row_id"])
def test_reference_answer_and_wrong_answer(answer: dict) -> None:
    result = grade_answer(
        expected_answer=answer["expected_answer"],
        answer_type=answer["answer_type"],
        generated_answer="\\boxed{" + answer["expected_answer"] + "}",
    )
    assert result.reward == 1
    wrong = str(int(answer["expected_answer"]) + 1) if answer["answer_type"] == "integer" else r"\frac{\sqrt{3}}{36}"
    assert (
        grade_answer(
            expected_answer=answer["expected_answer"],
            answer_type=answer["answer_type"],
            generated_answer="\\boxed{" + wrong + "}",
        ).reward
        == 0
    )


@pytest.mark.parametrize("text", [r"\boxed{२१४}", r"\boxed{২১৪}", r"\boxed{۲۱۴}", r"\boxed{\frac{428}{2}}"])
def test_exact_integer_formats(text: str) -> None:
    assert grade_answer(expected_answer="214", answer_type="integer", generated_answer=text).reward == 1


def test_symbolic_equivalence() -> None:
    alternative = r"\frac{1+6\exp(-\frac{1}{6}-20\sqrt{3})}{12\sqrt{3}}"
    assert (
        grade_answer(
            expected_answer=BMO, answer_type="expression", generated_answer="\\boxed{" + alternative + "}"
        ).reward
        == 1
    )


@pytest.mark.parametrize(
    "expression,expected,status",
    [
        (r"\sum_{n=0}^{250}\binom{1015-4n}{15}", "625243878951", "incorrect"),
        (r"\sum_{n=0}^{250}\binom{1015-4n}{15}", "14006388104146849994602092635462501", "correct"),
        (r"\sum_{n=1}^{3}n^2", "14", "correct"),
        (r"\sum_{n=1}^{1001}n", "501501", "invalid_expression"),
    ],
)
def test_bounded_finite_sum(expression: str, expected: str, status: str) -> None:
    result = grade_answer(
        expected_answer=expected, answer_type="integer", generated_answer="\\boxed{" + expression + "}"
    )
    assert result.grading_status == status


@pytest.mark.parametrize("variable", ["x", "t", "q"])
def test_real_generating_function_answer(variable: str) -> None:
    expression = r"\left[x^{1000}\right]\frac{1}{(1-x)^2(1-x^4)^2(1-x^5)^2(1-x^6)}".replace("x", variable)
    for expected, reward in [("625243878951", 1), ("625243878952", 0)]:
        assert (
            grade_answer(
                expected_answer=expected, answer_type="integer", generated_answer="\\boxed{" + expression + "}"
            ).reward
            == reward
        )


@pytest.mark.parametrize(
    "expression,expected",
    [
        (r"[x^4]\frac{1+x}{1-x}", "2"),
        (r"[x^{3}]\frac{1}{2-x}", r"\frac{1}{16}"),
        (r"[x^{0}]\frac{3+x}{1-x}", "3"),
    ],
)
def test_general_coefficient_evaluation(expression: str, expected: str) -> None:
    assert (
        grade_answer(
            expected_answer=expected, answer_type="expression", generated_answer="\\boxed{" + expression + "}"
        ).reward
        == 1
    )


@pytest.mark.parametrize("expression", [r"[x^{10001}]\frac{1}{1-x}", r"[x^3]\frac{1}{x}", r"[x^3]\frac{y}{1-x}"])
def test_coefficient_rejects_unbounded_or_symbolic_inputs(expression: str) -> None:
    assert (
        grade_answer(
            expected_answer="1", answer_type="integer", generated_answer="\\boxed{" + expression + "}"
        ).grading_status
        == "invalid_expression"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "214",
        r"\boxed{}",
        r"\boxed{214",
        r"\boxed{214} then \boxed{",
        r"<think>\boxed{214}</think>",
        r"<thinking>\boxed{214}",
    ],
)
def test_missing_or_reasoning_only_answer(text: str) -> None:
    assert extract_answer(text) is None


@pytest.mark.parametrize(
    "answer", ["214.0", "214 or 215", "{214,215}", "x=214", "214+x", "nan", "\\frac{1}{0}", "\\badcommand{214}"]
)
def test_invalid_or_approximate_expression(answer: str) -> None:
    assert (
        grade_answer(expected_answer="214", answer_type="integer", generated_answer="\\boxed{" + answer + "}").reward
        == 0
    )


def test_final_answer_only() -> None:
    assert (
        grade_answer(
            expected_answer="214", answer_type="integer", generated_answer=r"<think>\boxed{214}</think>\boxed{215}"
        ).reward
        == 0
    )
    assert (
        grade_answer(expected_answer="214", answer_type="integer", generated_answer=r"\boxed{215}\boxed {214}").reward
        == 1
    )
    assert (
        grade_answer(
            expected_answer="214", answer_type="integer", generated_answer="\\boxed{" + "1" * 4097 + "}"
        ).grading_status
        == "answer_too_long"
    )


async def test_real_subprocess_and_timeout(server: FrontierMathResourcesServer) -> None:
    result = await server.grade(expected_answer=BMO, answer_type="expression", generated_answer="\\boxed{" + BMO + "}")
    assert result.reward == 1
    server.config.verifier_timeout_seconds = 0.000001
    result = await server.grade(expected_answer=BMO, answer_type="expression", generated_answer="\\boxed{" + BMO + "}")
    assert result.reward == 0 and result.grading_status == "timeout"


async def test_concurrent_grading(server: FrontierMathResourcesServer) -> None:
    results = await asyncio.gather(
        *[
            server.grade(expected_answer="214", answer_type="integer", generated_answer=f"\\boxed{{{n}}}")
            for n in [214, 215, 214, 213]
        ]
    )
    assert [result.reward for result in results] == [1, 0, 1, 0]


async def test_oversized_response(server: FrontierMathResourcesServer) -> None:
    result = await server.grade(
        expected_answer="2", answer_type="integer", generated_answer="\\boxed{" + "1" * 4097 + "}"
    )
    assert result.grading_status == "answer_too_long"


@pytest.mark.parametrize("returncode,stdout", [(1, b""), (0, b"not json"), (0, b"{}")])
async def test_worker_failure(server: FrontierMathResourcesServer, returncode: int, stdout: bytes) -> None:
    process = MagicMock(returncode=returncode)
    process.communicate = AsyncMock(return_value=(stdout, b"failure\xff"))
    with patch(
        "resources_servers.frontiermath.app.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)
    ):
        result = await server.grade(expected_answer="2", answer_type="integer", generated_answer=r"\boxed{2}")
    assert result.reward == 0 and result.grading_status == "verifier_error"


async def test_cannot_start_worker(server: FrontierMathResourcesServer) -> None:
    with patch(
        "resources_servers.frontiermath.app.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=OSError("Unavailable")),
    ):
        result = await server.grade(expected_answer="2", answer_type="integer", generated_answer=r"\boxed{2}")
    assert result.grading_status == "verifier_error"


async def test_verify_response_preserves_metadata(server: FrontierMathResourcesServer) -> None:
    row = json.loads((Path(__file__).resolve().parents[1] / "data/example.jsonl").read_text().splitlines()[0])
    request = FrontierMathVerifyRequest.model_validate(
        {
            **row,
            "response": {
                "id": "test",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "output": [
                    {
                        "id": "answer",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": "\\boxed{" + row["expected_answer"] + "}",
                            }
                        ],
                    }
                ],
            },
        }
    )
    result = await server.verify(request)
    assert result.reward == 1
    assert result.row_id == row["row_id"]
    assert result.language_code == row["language_code"]
    assert result.judge_pass_stage == row["judge_pass_stage"]
    request.response.output = []
    assert (await server.verify(request)).grading_status == "missing_answer"
