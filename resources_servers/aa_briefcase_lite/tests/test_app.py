# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
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
    server._aa_binary_user = "{task_markdown}\n{check_description}\n{score_1_criteria}\n{score_0_criteria}"
    judge = ResolvedJudge(name="judge", model="model", base_url="http://upstream.invalid/v1", api_key="dummy")
    check = {
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
