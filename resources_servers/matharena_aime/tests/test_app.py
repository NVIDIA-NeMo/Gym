# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.matharena_aime.app import (
    FormatRetryRequest,
    MathArenaAIMEConfig,
    MathArenaAIMEResourcesServer,
    MathArenaAIMEVerifyRequest,
    is_truncated,
    visible_answer,
)
from resources_servers.matharena_aime.setup_parser import DIRECTORY, ensure_parser_runtime


def response(text=r"\boxed{42}", **updates):
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 1,
            "model": "test",
            "object": "response",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            **updates,
        }
    )


def request(text=r"\boxed{42}", **updates):
    return MathArenaAIMEVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "question"},
            "response": response(text),
            "task_id": "hi:1",
            "language": "hi",
            "problem_idx": 1,
            "expected_answer": 42,
            **updates,
        }
    )


@pytest.fixture
def server():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return MathArenaAIMEResourcesServer(
        config=MathArenaAIMEConfig(
            host="0.0.0.0", port=8080, entrypoint="app.py", name="matharena_aime", parser_python=sys.executable
        ),
        server_client=client,
    )


@pytest.fixture(scope="module")
def parser_python():
    return ensure_parser_runtime(DIRECTORY / ".parser-venv")


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\boxed{42}", r"\boxed{42}"),
        (r"<think>\boxed{41}</think>\boxed{42}", r"\boxed{42}"),
        (r"orphan reasoning</thinking>\boxed{42}", r"\boxed{42}"),
        (r"<thinking>\boxed{42}", ""),
        ("", ""),
    ],
)
def test_visible_text_strips_tagged_reasoning(text, expected):
    assert visible_answer(response(text)) == expected


def test_only_last_assistant_message_is_used():
    first, final = response(r"\boxed{42}"), response(r"\boxed{43}")
    final = response(
        output=first.model_dump()["output"]
        + [{"type": "reasoning", "id": "reason", "summary": []}]
        + final.model_dump()["output"]
    )
    assert visible_answer(final) == r"\boxed{43}"
    only_reasoning = response(output=[{"type": "reasoning", "id": "reason", "summary": []}])
    assert visible_answer(only_reasoning) == ""


def test_final_empty_message_is_not_replaced_with_earlier_answer():
    combined = response("")
    combined.output = response().output + combined.output
    assert visible_answer(combined) == ""


def test_final_user_message_stops_old_answer_selection():
    combined = response()
    combined.output.append(NeMoGymEasyInputMessage(role="user", content="format repair"))
    assert visible_answer(combined) == ""


def test_truncation_signals():
    assert not is_truncated(response())
    assert is_truncated(response(status="incomplete"))
    assert is_truncated(response(incomplete_details={"reason": "max_output_tokens"}))
    incomplete = response()
    incomplete.output[0].status = "incomplete"
    assert is_truncated(incomplete)


def test_setup_autoinstalls_only_when_parser_python_unspecified(monkeypatch, server):
    installed = MagicMock(return_value=DIRECTORY / "fake-python")
    monkeypatch.setattr("resources_servers.matharena_aime.app.ensure_parser_runtime", installed)
    config = server.config.model_copy(update={"parser_python": None})
    auto = MathArenaAIMEResourcesServer(config=config, server_client=server.server_client)
    assert auto._parser_python == DIRECTORY / "fake-python"
    installed.assert_called_once_with(DIRECTORY / ".parser-venv")
    assert "/needs_format_retry" in {route.path for route in auto.setup_webserver().routes}


@pytest.mark.parametrize("gold", [-1, 1000, "42", True, 42.0])
def test_gold_is_strict_integer_in_aime_range(gold):
    with pytest.raises(ValidationError):
        request(expected_answer=gold)


@pytest.mark.asyncio
async def test_native_endpoints_use_real_isolated_parser(server, parser_python):
    server._parser_python = parser_python
    boxed = await server.verify(request())
    assert boxed.valid and boxed.reward == 1.0 and not boxed.mask_sample
    assert boxed.extracted_answer == "42" and not boxed.review_required
    check = await server.needs_format_retry(FormatRetryRequest(response=response("42")))
    assert check.valid and check.needs_format_retry and check.parser_warning == 3
    bare = await server.verify(request("42"))
    assert bare.reward == 1.0 and bare.review_required and bare.parser_warning == 3
    # This expression reaches the actual ANTLR LaTeX fallback, not the safe AST
    # rewrite. It cannot be validated in Gym's incompatible ANTLR 4.9 process.
    integral = await server.verify(request(r"\boxed{\int_{0}^{1} 84x dx}"))
    assert integral.valid and integral.reward == 1.0 and integral.extracted_answer == "42"
    unsafe = await server.verify(request(r'\boxed{__import__("os").getcwd()}'))
    assert not unsafe.valid and unsafe.mask_sample
    assert unsafe.verifier_error.startswith("unsafe_expression:")
    failed = await server.verify(request(format_retry_check={"valid": False, "verifier_error": "prior failure"}))
    assert not failed.valid and failed.verifier_error == "prior failure"


@pytest.mark.asyncio
async def test_verify_final_turn_only_and_forward_total_tokens(monkeypatch, server):
    parser = AsyncMock(return_value={"valid": True, "reward": 0.0, "parser_warning": 1, "extracted_answer": "43"})
    monkeypatch.setattr(MathArenaAIMEResourcesServer, "parse", parser)
    combined = response(
        usage={
            "input_tokens": 5,
            "output_tokens": 120000,
            "total_tokens": 120005,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 20},
        }
    )
    result = await server.verify(
        request(response=combined, turn_responses=[response(status="incomplete"), response(r"\boxed{43}")])
    )
    parser.assert_awaited_once_with(r"\boxed{43}", strict=False, expected=42, output_tokens=120000)
    assert result.candidate_truncated and result.review_required and result.valid


@pytest.mark.asyncio
async def test_invalid_format_check_without_message_is_preserved(server):
    result = await server.verify(request(format_retry_check={"valid": False}))
    assert not result.valid and result.verifier_error == "format_check_failed"


@pytest.mark.asyncio
async def test_reasoning_only_repair_never_reuses_initial_answer(monkeypatch, server):
    parser = AsyncMock(return_value={"valid": True, "reward": 0.0, "parser_warning": 3, "extracted_answer": None})
    monkeypatch.setattr(MathArenaAIMEResourcesServer, "parse", parser)
    first = response(r"\boxed{42}")
    repair = response(output=[{"type": "reasoning", "id": "reasoning", "summary": []}])
    combined = response(output=first.model_dump()["output"] + repair.model_dump()["output"])
    result = await server.verify(request(response=combined, turn_responses=[first, repair]))
    parser.assert_awaited_once_with("", strict=False, expected=42, output_tokens=0)
    assert result.reward == 0.0 and result.review_required and result.extracted_answer is None


def process_mock(stdout=b'{"valid":true}', *, returncode=0, stderr=b""):
    process = MagicMock(returncode=returncode)
    process.communicate = AsyncMock(return_value=(stdout, stderr))
    return process


@pytest.mark.parametrize(
    "stdout,returncode,error",
    [(b"bad", 0, "invalid JSON"), (b"[]", 0, "validity flag"), (b"{}", 0, "validity flag"), (b"", -9, "exit -9")],
)
@pytest.mark.asyncio
async def test_worker_errors_are_invalid_not_wrong(monkeypatch, server, stdout, returncode, error):
    process = process_mock(stdout, returncode=returncode, stderr=b"stderr\xff")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    result = await server.parse("text", strict=True)
    assert not result["valid"] and error in result["verifier_error"]


@pytest.mark.asyncio
async def test_worker_launch_error(monkeypatch, server):
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError("missing")))
    result = await server.parse("text", strict=False)
    assert not result["valid"] and "parser_launch_failed" in result["verifier_error"]


@pytest.mark.parametrize(
    "updates,expected,error",
    [
        ({"parser_warning": True}, None, "result fields"),
        ({"parser_warning": 4}, None, "result fields"),
        ({"needs_format_retry": None}, None, "result fields"),
        ({"extracted_answer": 42}, None, "result fields"),
        ({"reward": 0.5}, 42, "reward"),
        ({"reward": True}, 42, "reward"),
        ({"valid": False}, None, "error diagnosis"),
        ({"valid": False, "verifier_error": ""}, None, "error diagnosis"),
    ],
)
@pytest.mark.asyncio
async def test_worker_success_schema_is_not_optional(monkeypatch, server, updates, expected, error):
    result = {"valid": True, "parser_warning": 0, "needs_format_retry": False, "extracted_answer": "42", "reward": 1.0}
    process = process_mock(json.dumps(result | updates).encode())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    parsed = await server.parse("text", strict=expected is None, expected=expected)
    assert not parsed["valid"] and error in parsed["verifier_error"]


@pytest.mark.asyncio
async def test_worker_timeout_kills_and_drains_process(monkeypatch, server):
    process = process_mock(returncode=None)
    process.communicate = AsyncMock(side_effect=[TimeoutError(), (b"", b"")])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    result = await server.parse("text", strict=True)
    assert not result["valid"] and "parser_timeout" in result["verifier_error"]
    process.kill.assert_called_once()
    assert process.communicate.await_count == 2


@pytest.mark.asyncio
async def test_worker_cancellation_also_kills_process(monkeypatch, server):
    process = process_mock(returncode=None)
    process.communicate = AsyncMock(side_effect=[asyncio.CancelledError(), (b"", b"")])
    process.kill.side_effect = ProcessLookupError
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    with pytest.raises(asyncio.CancelledError):
        await server.parse("text", strict=True)
    process.kill.assert_called_once()


def rows_for(groups, *, repeats=4):
    expected = {
        language: {
            "questions": len(answers),
            "ids_sha256": hashlib.sha256(
                json.dumps([f"{language}:{i}" for i in range(len(answers))]).encode()
            ).hexdigest(),
        }
        for language, answers in groups.items()
    }
    return [
        {
            "task_id": f"{language}:{i}",
            "language": language,
            "reward": reward,
            "response": {},
            "valid": True,
            "mask_sample": False,
            "parser_warning": 0,
            "review_required": False,
            "candidate_truncated": False,
            "expected_groups": expected,
            TASK_INDEX_KEY_NAME: f"{language}:{i}",
            ROLLOUT_INDEX_KEY_NAME: repeat,
        }
        for language, answers in groups.items()
        for i, reward in enumerate(answers)
        for repeat in range(repeats)
    ]


def test_pass_at_four_weights_problems_then_languages_equally(server):
    rows = rows_for({"hi": [1, 1], "or": [0]})
    metrics = server.compute_metrics([rows])
    assert metrics["pass@4/accuracy"] == 100.0 * 2 / 3
    assert metrics["matharena_aime/macro_pass@4/accuracy"] == 50.0
    assert metrics["matharena_aime/selection_complete"]
    assert not metrics["matharena_aime/provisional"]
    assert metrics["matharena_aime/language/hi/pass@4/accuracy"] == 100.0
    assert server.get_key_metrics(metrics | {"unrelated": 1}) == metrics


def test_pass_at_four_succeeds_when_any_repeat_is_correct(server):
    rows = rows_for({"hi": [0]})
    rows[-1]["reward"] = 1
    metrics = server.compute_metrics([rows])
    assert metrics["pass@4/accuracy"] == 100.0
    assert metrics["matharena_aime/language/hi/pass@4/accuracy"] == 100.0


@pytest.mark.parametrize("issue", ["missing_repeat", "duplicate", "invalid", "warning", "truncated", "unknown"])
def test_incomplete_or_review_needed_results_have_no_headline(server, issue):
    rows = rows_for({"hi": [1]})
    if issue == "missing_repeat":
        rows.pop()
    elif issue == "duplicate":
        rows[-1][ROLLOUT_INDEX_KEY_NAME] = 0
    elif issue == "invalid":
        rows[0].update(valid=False, mask_sample=True)
    elif issue in {"warning", "truncated"}:
        rows[0].update(review_required=True, parser_warning=3 if issue == "warning" else 0, candidate_truncated=True)
    else:
        for row in rows:
            row.pop("expected_groups")
    metrics = server.compute_metrics([rows])
    assert metrics["matharena_aime/provisional"]
    assert "matharena_aime/macro_pass@4/accuracy" not in metrics
    assert metrics["matharena_aime/observed_macro_pass@4/accuracy"] == 100.0


def test_missing_language_and_all_invalid_are_audited(server):
    rows = rows_for({"hi": [1], "ta": [1]})[:4]
    for row in rows:
        row.update(valid=False, mask_sample=True)
    metrics = server.compute_metrics([rows])
    assert metrics["matharena_aime/language/ta/missing_questions"] == 1
    assert metrics["matharena_aime/language/ta/missing_repeats"] == 4
    assert metrics["matharena_aime/invalid_rollouts"] == 4
    assert "matharena_aime/observed_macro_pass@4/accuracy" not in metrics


def test_inconsistent_expected_groups_fail(server):
    rows = rows_for({"hi": [1]})
    rows[0]["expected_groups"] = {"hi": {"questions": 5, "ids_sha256": "different"}}
    with pytest.raises(ValueError, match="Inconsistent"):
        server.compute_metrics([rows])


@pytest.mark.asyncio
async def test_native_aggregation_includes_masked_measurements_in_completeness(server):
    rows = rows_for({"hi": [1]})
    rows[0].update(valid=False, mask_sample=True)
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert metrics.agent_metrics["matharena_aime/invalid_rollouts"] == 1
    assert not metrics.key_metrics["matharena_aime/selection_complete"]
    assert "matharena_aime/macro_pass@4/accuracy" not in metrics.key_metrics
    for row in rows:
        row.update(valid=False, mask_sample=True)
    masked = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert masked.key_metrics["matharena_aime/invalid_rollouts"] == 4
    empty = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    assert empty.key_metrics["matharena_aime/provisional"]
