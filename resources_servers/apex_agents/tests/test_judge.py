# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from resources_servers.apex_agents.artifacts import ArtifactChange
from resources_servers.apex_agents.judge import (
    _artifact_xml,
    _build_prompt,
    _selection_artifact_xml,
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
        context_window_size=1_000_000,
    )
    assert "<AGENT_OUTPUT>\n<TEXT_RESPONSE>\nThe answer\n</TEXT_RESPONSE>\n</AGENT_OUTPUT>" in prompt
    assert "This criterion evaluates the agent's final text response only" in prompt
    assert "<ARTIFACT_STRUCTURE>" not in prompt


def test_sub_artifact_xml_preserves_archipelago_metadata() -> None:
    change = ArtifactChange(
        path="filesystem/deck.pptx",
        change_type="modified",
        before_path=Path("before/deck.pptx"),
        after_path=Path("after/deck.pptx"),
        artifact_type="slide",
        index=2,
        original_index=1,
        title="Results",
        old_content="10 percent",
        new_content="12 percent",
        content_diff="--- original_1\n+++ final_2\n-10 percent\n+12 percent",
    )

    selector_xml = _selection_artifact_xml(1, change, change.content_diff or "", truncated=False)
    judge_xml = _artifact_xml([change], character_budget=10_000)

    for rendered in (selector_xml, judge_xml):
        assert '<ARTIFACT id="1" type="slide" change="modified">' in rendered
        assert "<path>deck.pptx</path>" in rendered
        assert "<title>Results</title>" in rendered
        assert "<sub_index>3</sub_index>" in rendered
        assert "<original_index>2</original_index>" in rendered


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
        judge_context_window_size=1_000_000,
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
    assert "<TEXT_RESPONSE>\nDone\n</TEXT_RESPONSE>" in request["json"]["messages"][1]["content"]
    assert "  <created_content>\n  finished\n  </created_content>" in request["json"]["messages"][1]["content"]
    assert "--- BEFORE ---" not in request["json"]["messages"][1]["content"]
    assert client.post.call_count == 1


@pytest.mark.asyncio
async def test_required_file_type_without_matching_artifact_auto_fails(tmp_path: Path) -> None:
    client = MagicMock()
    client.post = AsyncMock()

    reward, scores, usage = await grade_apex_output(
        server_client=client,
        model_server_name="judge-server",
        task_id="task-1",
        world_id="world-1",
        instruction="Create a spreadsheet",
        response="Done",
        rubric=[{"verifier_id": "v1", "criteria": "The spreadsheet contains revenue"}],
        expected_output="make_new_sheet",
        artifact_changes=[],
        final_root=tmp_path,
        judge_model="gemini-3-flash",
        judge_create_params_overrides={"reasoning_effort": "low"},
        judge_context_window_size=1_000_000,
    )

    assert reward == 0.0
    assert scores["v1"]["score"] == 0.0
    assert scores["v1"]["values"]["artifact_selection"]["reason"] == "no_matching_artifacts"
    assert "No files matching the expected type" in scores["v1"]["message"]
    assert usage["scoring"]["scoring_method_result_values"]["failed_count"] == 1
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_large_artifacts_use_gemini_selector_before_gemini_judge(monkeypatch, tmp_path: Path) -> None:
    final_root = tmp_path / "final"
    backup = final_root / "filesystem" / "_BACKUP_EuroGrid.pptx"
    deliverable = final_root / "filesystem" / "EuroGrid transmission network modernization.pptx"
    backup.parent.mkdir(parents=True)
    backup.write_text("old deck " * 200, encoding="utf-8")
    deliverable.write_text("Germany-North Thermal Overload " * 200, encoding="utf-8")
    changes = [
        ArtifactChange(
            path="filesystem/_BACKUP_EuroGrid.pptx", change_type="added", before_path=None, after_path=backup
        ),
        ArtifactChange(
            path="filesystem/EuroGrid transmission network modernization.pptx",
            change_type="added",
            before_path=None,
            after_path=deliverable,
        ),
    ]
    response = object()
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    payloads = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "rationale": "The final deliverable is relevant.",
                                    "selected_artifact_indices": [2],
                                }
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "rationale": "Germany-North is present.",
                                    "is_criteria_true": True,
                                }
                            )
                        }
                    }
                ]
            },
        ]
    )

    async def fake_status(_actual) -> None:
        return None

    async def fake_json(_actual) -> dict:
        return next(payloads)

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", fake_status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", fake_json)
    monkeypatch.setattr(
        "resources_servers.apex_agents.judge.artifact_change_content",
        lambda change: change.after_path.read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        "resources_servers.apex_agents.judge.artifact_change_text",
        lambda change, *, max_chars: change.after_path.read_text(encoding="utf-8")[:max_chars],
    )
    monkeypatch.setattr("resources_servers.apex_agents.judge._visual_blocks", lambda *_args, **_kwargs: [])

    reward, scores, _usage = await grade_apex_output(
        server_client=client,
        model_server_name="apex_judge_model",
        task_id="task-1",
        world_id="world-1",
        instruction="Update the EuroGrid deck",
        response="Done",
        rubric=[{"verifier_id": "v1", "criteria": "States Germany-North"}],
        expected_output="make_new_slide_deck",
        artifact_changes=changes,
        final_root=final_root,
        judge_model="gcp/google/gemini-3-flash-preview",
        judge_create_params_overrides={"reasoning_effort": "low"},
        judge_context_window_size=100,
    )

    assert reward == 1.0
    assert client.post.call_count == 2
    selector_request = client.post.call_args_list[0].kwargs
    judge_request = client.post.call_args_list[1].kwargs
    assert selector_request["server_name"] == judge_request["server_name"] == "apex_judge_model"
    assert selector_request["json"]["model"] == judge_request["json"]["model"]
    assert selector_request["json"]["model"] == "gcp/google/gemini-3-flash-preview"
    assert "preprocessing filter" in selector_request["json"]["messages"][0]["content"]
    assert scores["v1"]["values"]["evaluated_artifacts"] == "EuroGrid transmission network modernization.pptx"
    selection = scores["v1"]["values"]["artifact_selection"]
    assert selection["status"] == "completed"
    assert selection["selected_artifact_indices"] == [2]


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
        judge_context_window_size=1_000_000,
    )

    assert reward == 0.0
    values = usage["scoring"]["scoring_method_result_values"]
    assert values["criteria_pass_rate"] == 0.5
    assert values["grade_score_percentage"] == 0.0
