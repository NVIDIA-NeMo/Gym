# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
    AABriefcaseLiteVerifyRequest,
    _BinaryJudgeHttpClient,
    _pairwise_task_prompt,
    _parse_binary_judgement,
    _requested_filenames,
    _stage_submission,
)
from resources_servers.gdpval.judge_panel import ResolvedJudge


@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize("formatting_retry", [False, True])
async def test_binary_call_requests_json_without_changing_grading_input(monkeypatch, passed, formatting_retry) -> None:
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    server = AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(dataset_dir="unused")
    )
    server._aa_binary_system = "Original AA system prompt."
    server._aa_binary_user = (
        "{task_markdown}\n{check_description}\n{score_1_criteria}\n{score_0_criteria}<<<SUBMISSION CONTENT MESSAGES>>>"
    )
    judge = ResolvedJudge(name="judge", model="model", base_url="http://upstream.invalid/v1", api_key="dummy")
    check = {
        "check_id": "test-check",
        "check_description": "Check totals.",
        "score_1_criteria": "Totals agree.",
        "score_0_criteria": "Totals differ.",
    }
    artifacts = [{"type": "text", "text": "Submitted total: 42"}]
    verdict = {"passed": passed, "reasoning": "Compared the submitted total."}
    raw = json.dumps(verdict)
    requests = []

    async def create(**kwargs):
        requests.append(deepcopy(kwargs))
        content = "Unstructured prose." if formatting_retry and len(requests) == 1 else raw
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=create)
    monkeypatch.setattr("resources_servers.aa_briefcase_lite.app.AsyncOpenAI", lambda **kwargs: client)
    assert await server._binary_call(judge, "Make a report.", check, artifacts) == (verdict, raw)
    instruction = 'Return only one JSON object with boolean key "passed" and string key "reasoning".'
    expected_messages = [
        {"role": "system", "content": "Original AA system prompt.\n\n" + instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Make a report.\nCheck totals.\nTotals agree.\nTotals differ."},
                {"type": "text", "text": "Submitted total: 42"},
                {"type": "text", "text": ""},
            ],
        },
    ]
    assert requests[0]["messages"] == expected_messages
    assert len(requests) == 1 + formatting_retry
    assert server._aa_binary_system == "Original AA system prompt."
    assert artifacts == [{"type": "text", "text": "Submitted total: 42"}]
    if formatting_retry:
        assert requests[1]["messages"] == [
            *expected_messages,
            {"role": "assistant", "content": "Unstructured prose."},
            {"role": "user", "content": instruction},
        ]


def test_parse_binary_judgement_accepts_json_and_fence() -> None:
    assert _parse_binary_judgement('{"passed": true, "reasoning": "visible evidence"}') == {
        "passed": True,
        "reasoning": "visible evidence",
    }
    assert _parse_binary_judgement('```json\n{"passed": false, "reasoning": "missing"}\n```') == {
        "passed": False,
        "reasoning": "missing",
    }


def test_parse_binary_judgement_rejects_non_boolean_and_missing_reasoning_type() -> None:
    assert _parse_binary_judgement('{"passed": 1, "reasoning": "no"}') is None
    assert _parse_binary_judgement('{"passed": true, "reasoning": []}') is None
    assert _parse_binary_judgement("not json") is None


def test_requested_filenames_splits_and_deduplicates() -> None:
    checks = [
        {"taskdoer_output_file": "market_overview.pdf, market_overview.tex"},
        {"taskdoer_output_file": "market_overview.pdf"},
    ]
    assert _requested_filenames(checks) == ["market_overview.pdf", "market_overview.tex"]


def test_stage_submission_is_read_only_view_and_records_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"pdf")
    with ExitStack() as stack:
        stage, missing = _stage_submission(str(tmp_path), ["report.pdf", "notes.txt"], stack)
        assert (stage / "report.pdf").is_symlink()
        assert (stage / "report.pdf").read_bytes() == b"pdf"
        assert missing == ["notes.txt"]


def test_pairwise_prompt_is_criterion_specific_and_source_blind() -> None:
    prompt = _pairwise_task_prompt(
        "Make a deck.",
        {
            "check_description": "Prefer stronger analysis.",
            "score_1_criteria": "A wins when its synthesis is stronger.",
            "score_0_criteria": "B wins when its synthesis is stronger.",
        },
    )
    assert "Make a deck." in prompt
    assert "Prefer stronger analysis." in prompt
    assert "A wins" in prompt and "B wins" in prompt
    assert "external source files" in prompt


@pytest.mark.parametrize("recover", [True, False])
@pytest.mark.parametrize("transport_timeout", [False, True])
async def test_binary_transport_timeout_retries_are_bounded(monkeypatch, caplog, recover, transport_timeout):
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    server = AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(dataset_dir="unused")
    )
    server._aa_binary_system = "Judge the artifact."
    server._aa_binary_user = (
        "{task_markdown} {check_description} {score_1_criteria} {score_0_criteria}<<<SUBMISSION CONTENT MESSAGES>>>"
    )
    judge = ResolvedJudge(name="judge", model="model", base_url="http://upstream.invalid/v1", api_key="dummy")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if not recover or len(requests) < 3:
            if transport_timeout:
                raise httpx.ReadTimeout("private-error-body", request=request)
            return httpx.Response(408, json={"error": {"message": "private-error-body"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"passed": false, "reasoning": "Verdict"}'},
                    }
                ],
            },
        )

    monkeypatch.setattr(
        "resources_servers.aa_briefcase_lite.app._BinaryJudgeHttpClient",
        lambda **kwargs: _BinaryJudgeHttpClient(transport=httpx.MockTransport(respond), **kwargs),
    )
    call = server._binary_call(
        judge,
        "private-prompt-text",
        {
            "check_id": "test-check",
            "check_description": "Check",
            "score_1_criteria": "Pass",
            "score_0_criteria": "Fail",
        },
        [{"type": "text", "text": "private-artifact-text"}],
    )
    if recover:
        parsed, _ = await call
        assert parsed == {"passed": False, "reasoning": "Verdict"}
    else:
        with pytest.raises(APITimeoutError if transport_timeout else APIStatusError):
            await call
    assert len(requests) == 3
    assert requests[0] == requests[1] == requests[2]
    records = [record.getMessage() for record in caplog.records if record.name.endswith(".binary_transport")]
    assert len(records) == 3
    for attempt, record in enumerate(records, 1):
        assert f"check_id=test-check model=model format_attempt=1 transport_attempt={attempt}" in record
        expected = (
            "status=200 error=None"
            if recover and attempt == 3
            else ("status=None error=ReadTimeout" if transport_timeout else "status=408 error=None")
        )
        assert expected in record
        assert float(record.split("duration_seconds=")[1]) >= 0
    assert all(
        value not in " ".join(records)
        for value in (
            "private-error-body",
            "private-prompt-text",
            "private-artifact-text",
            "dummy",
            "Authorization",
            "upstream.invalid",
        )
    )


@pytest.mark.parametrize("empty_answers", [0, 2, 3])
async def test_binary_empty_answers_retry_only_the_affected_check(monkeypatch, tmp_path, caplog, empty_answers):
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    server = AABriefcaseLiteResourcesServer.model_construct(
        config=AABriefcaseLiteResourcesServerConfig.model_construct(
            dataset_dir=str(tmp_path), preconvert_office_to_pdf=False
        )
    )
    server._aa_binary_system = "Judge the artifact."
    server._aa_binary_user = (
        "{task_markdown} {check_description} {score_1_criteria} {score_0_criteria}<<<SUBMISSION CONTENT MESSAGES>>>"
    )
    server._aa_checks = [
        {
            "task_id": "task",
            "scoring_type": "binary",
            "check_id": name,
            "check_type": "format",
            "check_description": name,
            "score_1_criteria": "Pass",
            "score_0_criteria": "Fail",
            "taskdoer_output_file": "artifact.txt",
        }
        for name in ("first check", "second check")
    ]
    (tmp_path / "artifact.txt").write_text("Submitted artifact")
    judge = ResolvedJudge(name="judge", model="model", base_url="http://upstream.invalid/v1", api_key="dummy")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        first_check = "first check" in str(requests[-1]["messages"])
        content = (
            ""
            if not first_check and len(requests) <= empty_answers + 1
            else json.dumps({"passed": first_check, "reasoning": "Verdict"})
        )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop" if content else "length",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 32768, "total_tokens": 32778},
            },
        )

    monkeypatch.setattr(
        "resources_servers.aa_briefcase_lite.app._BinaryJudgeHttpClient",
        lambda **kwargs: _BinaryJudgeHttpClient(transport=httpx.MockTransport(respond), **kwargs),
    )
    reward, results, invalid = await server._verify_binary(
        AABriefcaseLiteVerifyRequest.model_construct(task_id="task", deliverables_dir=str(tmp_path)),
        "Task",
        [judge],
    )

    assert reward == 0.5
    assert results[0]["passed"] is True
    assert results[1]["passed"] is (None if empty_answers == 3 else False)
    assert invalid == (1 if empty_answers == 3 else 0)
    assert len(requests) == 1 + min(empty_answers + 1, 3)
    assert sum("first check" in str(request["messages"]) for request in requests) == 1
    assert all(request == requests[1] for request in requests[1:])
    assert (
        sum("Invalid binary judge answer reached its token limit" in r.message for r in caplog.records)
        == empty_answers
    )
    assert "Submitted artifact" not in caplog.text
    records = [record.getMessage() for record in caplog.records if record.name.endswith(".binary_transport")]
    assert len(records) == len(requests)
    assert all("transport_attempt=1 status=200" in record for record in records)
    for attempt, record in enumerate(records[1:], 1):
        assert f"format_attempt={attempt} " in record
