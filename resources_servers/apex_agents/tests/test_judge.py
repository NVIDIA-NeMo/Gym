# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from resources_servers.apex_agents.artifacts import ArtifactChange
from resources_servers.apex_agents.judge import (
    _build_prompt,
    expected_file_type,
    grade_apex_output,
)


def _criterion() -> dict:
    return {"verifier_id": "v1", "criteria": "Correct result"}


def test_expected_file_type_uses_apex_labels() -> None:
    assert expected_file_type("message_in_console", _criterion()) == "Final Answer Only (No Files)"
    assert expected_file_type("make_new_sheet", _criterion()) == "Spreadsheets (.xlsx, .xls, .xlsm, .ods)"
    assert expected_file_type("make_new_slide_deck", _criterion()) == "Presentations (.pptx, .ppt, .odp)"


def test_expected_file_type_rejects_legacy_preprocessed_rows() -> None:
    criterion = _criterion() | {"grading_target": {"expected_file_type": "Spreadsheet"}}
    assert expected_file_type(None, criterion) == "All output (modified files and final message in console)"


def test_console_prompt_excludes_file_artifacts() -> None:
    prompt = _build_prompt(
        instruction="Answer here",
        response="The answer",
        criteria="States the answer",
        file_type="Final Answer Only (No Files)",
        changes=[],
        context_window_size=32768,
    )
    assert "The answer" in prompt
    assert "This criterion evaluates the agent's final text response only" in prompt
    assert "<ARTIFACT_STRUCTURE>" not in prompt


@pytest.mark.asyncio
async def test_apex_grading_extracts_output_and_returns_binary_score(monkeypatch, tmp_path: Path) -> None:
    final_root = tmp_path / "final"
    final_file = final_root / "filesystem" / "final.txt"
    final_file.parent.mkdir(parents=True)
    final_file.write_text("finished", encoding="utf-8")
    change = ArtifactChange(
        path="filesystem/final.txt",
        change_type="added",
        before_path=None,
        after_path=final_file,
    )

    response = object()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    async def fake_status(actual) -> None:
        assert actual is response

    async def fake_json(actual) -> dict:
        assert actual is response
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "rationale": "The required file is present.",
                                "is_criteria_true": True,
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", fake_status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", fake_json)

    reward, scores, usage = await grade_apex_output(
        server_client=client,
        model_server_name="judge-server",
        task_id="task-1",
        world_id="world-1",
        instruction="Create final.txt",
        response="Done",
        rubric=[{"verifier_id": "v1", "criteria": "final.txt exists"}],
        expected_output=None,
        artifact_changes=[change],
        final_root=final_root,
        judge_model="judge-model",
        judge_create_params_overrides={"reasoning_effort": "low"},
        judge_context_window_size=32768,
    )

    assert reward == 1.0
    assert scores["v1"]["values"]["evaluated_artifacts"] == "final.txt"
    assert "judge_trace" not in scores["v1"]["values"]
    assert usage["document_extraction"] == "local"
    request = client.post.call_args.kwargs
    assert request["server_name"] == "judge-server"
    assert request["json"]["model"] == "judge-model"
    assert request["json"]["reasoning_effort"] == "low"
    assert request["json"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_apex_grading_fails_rollout_when_any_criterion_fails(monkeypatch, tmp_path: Path) -> None:
    answers = iter([True, False])
    response = object()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    async def fake_status(_actual) -> None:
        return None

    async def fake_json(_actual) -> dict:
        passed = next(answers)
        return {"choices": [{"message": {"content": json.dumps({"rationale": "graded", "is_criteria_true": passed})}}]}

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", fake_status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", fake_json)

    reward, _scores, usage = await grade_apex_output(
        server_client=client,
        model_server_name="judge-server",
        task_id="task-1",
        world_id="world-1",
        instruction="Do both things",
        response="Done",
        rubric=[
            {"verifier_id": "v1", "criteria": "First thing"},
            {"verifier_id": "v2", "criteria": "Second thing"},
        ],
        expected_output="message_in_console",
        artifact_changes=[],
        final_root=tmp_path,
        judge_model="gemini-3-flash",
        judge_create_params_overrides={"reasoning_effort": "low"},
        judge_context_window_size=32768,
    )

    assert reward == 0.0
    values = usage["scoring"]["scoring_method_result_values"]
    assert values["criteria_pass_rate"] == 0.5
    assert values["grade_score_percentage"] == 0.0
