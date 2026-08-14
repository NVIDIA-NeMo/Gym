# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from resources_servers.apex_agents.artifacts import ArtifactChange
from resources_servers.apex_agents.judge import (
    _build_prompt,
    _structured_response_format,
    expected_file_type,
    grade_apex_output,
)


def _criterion() -> dict:
    return {"verifier_id": "v1", "criteria": "Correct result"}


def test_expected_file_type_uses_apex_labels() -> None:
    assert expected_file_type("message_in_console", _criterion()) == "Final Answer Only (No Files)"
    assert expected_file_type("make_new_sheet", _criterion()) == "Spreadsheets (.xlsx, .xls, .xlsm)"
    assert expected_file_type("make_new_slide_deck", _criterion()) == "Presentations (.pptx, .ppt)"


def test_expected_file_type_normalizes_legacy_preprocessed_rows() -> None:
    criterion = _criterion() | {"grading_target": {"expected_file_type": "Spreadsheet"}}
    assert expected_file_type(None, criterion) == "Spreadsheets (.xlsx, .xls, .xlsm)"


def test_structured_response_uses_pydantic_json_schema() -> None:
    class Grade(BaseModel):
        rationale: str
        is_criteria_true: bool

    assert _structured_response_format(Grade) == {
        "type": "json_schema",
        "json_schema": {
            "name": "Grade",
            "schema": Grade.model_json_schema(),
            "strict": True,
        },
    }


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
async def test_apex_grading_extracts_output_and_returns_fractional_score(monkeypatch, tmp_path: Path) -> None:
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
        capture_judge_traces=True,
    )

    assert reward == 1.0
    assert scores["v1"]["values"]["evaluated_artifacts"] == "final.txt"
    assert scores["v1"]["values"]["judge_trace"]["parsed_response"]["is_criteria_true"] is True
    assert usage["document_extraction"] == "local"
    request = client.post.call_args.kwargs
    assert request["server_name"] == "judge-server"
    assert request["json"]["model"] == "judge-model"
    assert request["json"]["reasoning_effort"] == "low"
    assert request["json"]["response_format"]["json_schema"]["name"] == "GradingResponse"
