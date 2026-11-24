# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perceptionbench resources server, grading observation, and schema conformance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import MODEL_CONTEXT_LIMIT, NeMoGymResponse
from nemo_gym.data import TaskDataValidator, load_task_data_schema
from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding
from nemo_gym.verification import exercise_verifier_fixture, judge_failsafe
from resources_servers.perceptionbench.app import (
    SERVER_DIR,
    VERIFIER_FIXTURE,
    JudgeParseError,
    PerceptionbenchResourcesServer,
    PerceptionbenchResourcesServerConfig,
    PerceptionbenchVerifierMetadata,
    PerceptionbenchVerifyRequest,
    PerceptionbenchVerifyResponse,
    normalize_escape,
    parse_judge_verdict,
    render_judge_prompt,
    strip_reasoning_block,
)
from resources_servers.perceptionbench.task_data import TaskData
from resources_servers.perceptionbench.verifier_fixture import fixture_judge_reply


EXPECTED_JUDGE_PROMPT_SHA256 = (
    "3736ab84d90d96a0344a25bf96a1d9ac2a0672656bbc1eeb01176f20cd3d6397"  # pragma: allowlist secret
)


def _make_response(text: str, *, metadata: dict[str, str] | None = None) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_policy",
        created_at=0,
        model="test-vlm",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        metadata=metadata,
    )


def _make_request(
    text: str = "There are 3 cups.",
    *,
    answer: str = "3",
    question: str = "How many cups are in <|image_1|>?",
    category: str = "visual_counting_error",
    source_bmk: str = "DA-2K",
    metadata: dict[str, str] | None = None,
) -> PerceptionbenchVerifyRequest:
    return PerceptionbenchVerifyRequest(
        task_id="perceptionbench_12",
        responses_create_params={"input": [{"role": "user", "content": question}]},
        response=_make_response(text, metadata=metadata),
        verifier_metadata=PerceptionbenchVerifierMetadata(
            index=12,
            question=question,
            answer=answer,
            category=category,
            source_bmk=source_bmk,
        ),
    )


def _make_server(*, max_attempts: int = 3) -> tuple[PerceptionbenchResourcesServer, MagicMock]:
    client = MagicMock(spec=ServerClient)
    config = PerceptionbenchResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="perceptionbench",
        judge_model_server=ModelServerRef(type="responses_api_models", name="perceptionbench_judge_model"),
        judge_max_attempts=max_attempts,
    )
    return PerceptionbenchResourcesServer(config=config, server_client=client), client


async def _verify_through_failsafe(
    server: PerceptionbenchResourcesServer,
    request: PerceptionbenchVerifyRequest,
) -> dict:
    wrapped = judge_failsafe(server.verify)
    result = await wrapped(body=request)
    if isinstance(result, PerceptionbenchVerifyResponse):
        return result.model_dump()
    return json.loads(result.body)


async def _verify_observed(
    server: PerceptionbenchResourcesServer,
    request: PerceptionbenchVerifyRequest,
) -> tuple[PerceptionbenchVerifyResponse | BaseException, VerifierInvocation]:
    invocation = VerifierInvocation(invocation_id="verify-perceptionbench")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = await server.verify(request)
    except BaseException as error:
        return error, invocation
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation


async def test_verifier_fixture() -> None:
    await exercise_verifier_fixture(
        VERIFIER_FIXTURE,
        reward_range=(0.0, 1.0),
        higher_is_better=True,
        determinism="unknown",
    )


def test_judge_prompt_matches_pinned_upstream_sha256() -> None:
    prompt_bytes = (SERVER_DIR / "prompt_templates" / "judge_prompt.txt").read_bytes()
    assert hashlib.sha256(prompt_bytes).hexdigest() == EXPECTED_JUDGE_PROMPT_SHA256


class TestPromptAndVerdictParsing:
    def test_true_in_reason_does_not_flip_false_verdict(self) -> None:
        passed, reason = parse_judge_verdict("[reason]\nIt is true that 4 is close to 3.\n[judge]\nFalse")
        assert passed is False
        assert reason == "It is true that 4 is close to 3."
        assert parse_judge_verdict("[reason]\nExact.\n[judge]\n  true  ")[0] is True
        assert parse_judge_verdict("[reason]\nMismatch.\n[judge]\nFALSE")[0] is False

    @pytest.mark.parametrize(
        "non_boolean_verdict",
        ["Not true", "not False", "The answer is true.", "True or False", "True (exact match)"],
        ids=["not_true", "not_false", "prose", "both_tokens", "trailing_prose"],
    )
    def test_non_boolean_verdicts_raise_parse_error(self, non_boolean_verdict: str) -> None:
        with pytest.raises(JudgeParseError, match="expected exact True or False"):
            parse_judge_verdict(f"[reason]\nExplanation.\n[judge]\n{non_boolean_verdict}")

    def test_leading_judge_thinking_block_is_stripped(self) -> None:
        passed, reason = parse_judge_verdict("<think>comparing 3 and 3</think>\n[reason]\nExact match.\n[judge]\nTrue")
        assert passed is True
        assert reason == "Exact match."
        assert strip_reasoning_block("orphan</think> final") == "final"
        assert strip_reasoning_block("<|channel>thought\nstep 1<channel|>0015400") == "0015400"
        assert strip_reasoning_block("<|channel>thought\ntruncated without closing") == ""
        with pytest.raises(JudgeParseError, match="truncated"):
            parse_judge_verdict("<think>unclosed")
        with pytest.raises(JudgeParseError, match="precede"):
            parse_judge_verdict("[judge]\nTrue\n[reason]\nlate")

    def test_single_pass_prompt_rendering_does_not_expand_placeholders_in_student_answer(self) -> None:
        template = "Q: {problem}\nS: {assistant_answer}\nR: {reference_answer}"
        rendered = render_judge_prompt(
            template,
            question="What is x?",
            candidate="Trying {reference_answer} trick",
            reference="SECRET_42",
        )
        assert "S: Trying {reference_answer} trick" in rendered
        assert rendered.count("SECRET_42") == 1

    def test_normalize_escape_matches_upstream_rules(self) -> None:
        assert normalize_escape(r"line1\nline2") == "line1\nline2"
        assert normalize_escape(r"\frac{1}{2}") == r"\frac{1}{2}"


class TestVerify:
    async def test_true_verdict_scores_one_and_records_completed_judge_grading(self) -> None:
        server, client = _make_server()
        client.post = AsyncMock(
            return_value=fixture_judge_reply("[reason]\n3 matches 3.\n[judge]\nTrue", reply_id="judge_1")
        )
        req = _make_request("<think>Let me count: 1, 2, 3.</think>\nThere are 3 cups.")
        result, invocation = await _verify_observed(server, req)
        assert isinstance(result, PerceptionbenchVerifyResponse)
        assert result.reward == 1.0
        assert result.no_answer == 0.0
        assert result.evaluation_completed is True
        assert result.extracted_answer == "There are 3 cups."
        assert result.judge_verdict is True
        assert result.judge_attempts == 1

        # Verify judge saw stripped answer and temperature 0.3
        call_kwargs = client.post.await_args.kwargs
        sent_params = call_kwargs["json"]
        assert sent_params.temperature == 0.3
        assert sent_params.max_tokens == 4096
        sent_prompt = sent_params.messages[0]["content"]
        assert "Let me count" not in sent_prompt
        assert "How many cups are in <|image_1|>?" in sent_prompt
        assert "There are 3 cups." in sent_prompt

        # Grading path A conformance: bound to response & reward, invalidated if reward mutates
        (operation,) = invocation.grading_operations
        assert (operation.kind, operation.outcome) == ("llm_judge", "completed")
        assert [(d.model_ref.name, d.response_id) for d in operation.dependencies] == [
            ("perceptionbench_judge_model", "judge_1")
        ]
        assert operation.result_binding_sha256 == grading_result_binding(result)
        tampered = result.model_copy(update={"reward": 0.0})
        assert operation.result_binding_sha256 != grading_result_binding(tampered)

    async def test_false_verdict_scores_unmasked_zero(self) -> None:
        server, client = _make_server()
        client.post = AsyncMock(return_value=fixture_judge_reply("[reason]\n4 != 3.\n[judge]\nFalse"))
        row = await _verify_through_failsafe(server, _make_request("4"))
        assert row["reward"] == 0.0
        assert row["no_answer"] == 0.0
        assert row["mask_sample"] is False
        assert row["evaluation_completed"] is True
        assert row["judge_verdict"] is False

    @pytest.mark.parametrize(
        "bad_prediction",
        ["", "   ", "<think>still counting cups without closing", "[error] timeout"],
        ids=["empty", "whitespace", "unclosed_think", "error_prefix"],
    )
    async def test_missing_or_truncated_answer_scores_zero_without_calling_judge(self, bad_prediction: str) -> None:
        server, client = _make_server()
        client.post = AsyncMock()
        result, invocation = await _verify_observed(server, _make_request(bad_prediction))
        assert isinstance(result, PerceptionbenchVerifyResponse)
        assert result.reward == 0.0
        assert result.no_answer == 1.0
        assert result.mask_sample is False
        assert result.evaluation_completed is True
        assert not invocation.grading_operations
        client.post.assert_not_awaited()

    async def test_context_window_exceeded_sets_typed_failure_kind(self) -> None:
        server, client = _make_server()
        client.post = AsyncMock()
        result = await server.verify(_make_request("", metadata={"nemo_gym_context_window_exceeded": "true"}))
        assert result.reward == 0.0
        assert result.mask_sample is False
        assert result.failure_kind == MODEL_CONTEXT_LIMIT
        client.post.assert_not_awaited()

    async def test_judge_retry_recovers_after_unparseable_first_reply(self) -> None:
        server, client = _make_server(max_attempts=3)
        client.post = AsyncMock(
            side_effect=[
                fixture_judge_reply("truncated [reason] without judge", reply_id="bad_1"),
                fixture_judge_reply("[reason]\nMatches.\n[judge]\nTrue", reply_id="good_2"),
            ]
        )
        result, invocation = await _verify_observed(server, _make_request("3"))
        assert isinstance(result, PerceptionbenchVerifyResponse)
        assert result.reward == 1.0
        assert result.judge_attempts == 2
        (operation,) = invocation.grading_operations
        assert operation.outcome == "completed"
        assert [d.response_id for d in operation.dependencies] == ["bad_1", "good_2"]

    @pytest.mark.parametrize(
        "unusable_reply",
        [
            "no markers at all",
            "[reason]\ncut off\n[judge]\n",
            "[reason]\nboth\n[judge]\nTrue or False",
            "[reason]\nThe candidate is incorrect.\n[judge]\nNot true",
            "[reason]\nNot wrong.\n[judge]\nnot False",
            "[reason]\nMatches.\n[judge]\nThe answer is true.",
            "",
        ],
        ids=["no_markers", "empty_verdict", "ambiguous_verdict", "not_true", "not_false", "prose_verdict", "blank"],
    )
    async def test_exhausted_unparseable_judge_replies_mask_row(self, unusable_reply: str) -> None:
        server, client = _make_server(max_attempts=3)
        client.post = AsyncMock(return_value=fixture_judge_reply(unusable_reply))
        row = await _verify_through_failsafe(server, _make_request("4", answer="3"))
        assert row["reward"] == 0.0
        assert row["mask_sample"] is True
        assert row["failure_kind"] == "judge_failed"
        assert row["_ng_failure_class"] == "judge_failed"
        assert client.post.await_count == 3

    async def test_judge_transport_error_masks_row_and_records_failed_grading(self, monkeypatch) -> None:
        monkeypatch.setattr("nemo_gym.verification.judge._judge_retry_wait_s", lambda _attempt: 0.0)
        server, client = _make_server()
        client.post = AsyncMock(side_effect=ConnectionError("judge endpoint unreachable"))
        error, invocation = await _verify_observed(server, _make_request("3"))
        assert isinstance(error, BaseException)
        (operation,) = invocation.grading_operations
        assert operation.outcome == "failed"

        row = await _verify_through_failsafe(server, _make_request("3"))
        assert row["mask_sample"] is True
        assert row["failure_kind"] == "judge_failed"
        assert "judge endpoint unreachable" in row["failure_reason"]


class TestMetricsAndSchema:
    def test_compute_and_key_metrics_group_by_category_and_non_na_source(self) -> None:
        server, _ = _make_server()
        tasks = [
            [{"reward": 1.0, "verifier_metadata": {"category": "ocr_error", "source_bmk": "OCRBench"}}],
            [{"reward": 0.0, "verifier_metadata": {"category": "ocr_error", "source_bmk": "NA"}}],
            [{"reward": 1.0, "verifier_metadata": {"category": "visual_counting_error", "source_bmk": "DA-2K"}}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics == {
            "by_category/ocr_error/accuracy": 0.5,
            "by_category/visual_counting_error/accuracy": 1.0,
            "macro_category_accuracy": 0.75,
            "by_source/DA-2K/accuracy": 1.0,
            "by_source/OCRBench/accuracy": 1.0,
        }
        key = server.get_key_metrics({"mean/reward": 2 / 3, "mean/no_answer": 0.0, "extra": 99.0, **metrics})
        assert key == {
            "mean/reward": 2 / 3,
            "mean/no_answer": 0.0,
            "macro_category_accuracy": 0.75,
            "by_category/ocr_error/accuracy": 0.5,
            "by_category/visual_counting_error/accuracy": 1.0,
        }

    def test_task_data_schema_matches_request_metadata_and_validates_prepared_row(self) -> None:
        def verifier_contract(model: type) -> dict:
            return {
                "extra": model.model_config.get("extra"),
                "fields": {
                    name: (
                        info.annotation,
                        info.is_required(),
                        info.default,
                        info.default_factory,
                        tuple(info.metadata),
                    )
                    for name, info in model.model_fields.items()
                    if name != "task_id"
                },
            }

        assert verifier_contract(TaskData) == verifier_contract(PerceptionbenchVerifierMetadata)
        adapter = load_task_data_schema(Path(SERVER_DIR))
        assert adapter is not None
        validator = TaskDataValidator("perceptionbench", adapter, "synthetic.jsonl")
        validator.validate_row(
            0,
            {
                "task_id": "perceptionbench_0",
                "responses_create_params": {"input": [{"role": "user", "content": "Q"}]},
                "verifier_metadata": {
                    "index": 0,
                    "question": "<|image_1|>What is shown?",
                    "answer": "NA",
                    "category": "ocr_error",
                    "source_bmk": "NA",
                    "source_idx": None,
                },
            },
        )
        assert validator.report.clean, validator.report.summary()
