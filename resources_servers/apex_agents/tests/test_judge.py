# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.verification import JudgeError, is_judge_failure
from resources_servers.apex_agents.artifacts import ArtifactChange
from resources_servers.apex_agents.judge import (
    _IMAGE_TOKEN_ESTIMATE,
    _MAX_JUDGE_IMAGES,
    _artifact_xml,
    _budget_images,
    _build_prompt,
    _request_is_large,
    _selection_artifact_xml,
    expected_file_type,
    grade_apex_output,
    weighted_rubric_score,
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
    assert usage["scoring"]["scoring_method"] == "all_pass"
    values = usage["scoring"]["scoring_method_result_values"]
    assert values["criteria_pass_rate"] == 0.5
    assert values["grade_score_percentage"] == 0.0
    assert values["all_pass"] == 0.0
    assert values["weighted_score"] == 0.5


@pytest.mark.asyncio
async def test_weighted_average_scoring_gives_weighted_partial_credit(monkeypatch, tmp_path: Path) -> None:
    answers = iter([True, False, False])
    client = MagicMock()
    client.post = AsyncMock(return_value=object())

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
        instruction="Do three things",
        response="Done",
        rubric=[
            {"verifier_id": "v1", "criteria": "Heavy thing", "weight": 3},
            {"verifier_id": "v2", "criteria": "Light thing", "weight": 1},
            {"verifier_id": "v3", "criteria": "Unscored thing", "weight": 0},
        ],
        expected_output="message_in_console",
        artifact_changes=[],
        final_root=tmp_path,
        judge_model="gemini-3-flash",
        judge_create_params_overrides=None,
        judge_context_window_size=1_000_000,
        scoring_method="weighted_average",
    )

    assert reward == 0.75
    assert usage["scoring"]["scoring_method"] == "weighted_average"
    values = usage["scoring"]["scoring_method_result_values"]
    assert values["all_pass"] == 0.0
    assert values["weighted_score"] == 0.75
    assert values["grade_score_percentage"] == 75.0


def test_weighted_rubric_score_defaults_missing_weights_and_rejects_negative_ones() -> None:
    passed, failed = {"score": 1.0}, {"score": 0.0}

    assert weighted_rubric_score([{}, {"weight": None}], [passed, failed]) == 0.5
    assert weighted_rubric_score([{"weight": 0}, {"weight": "2"}], [failed, passed]) == 1.0
    assert weighted_rubric_score([{"weight": 0}], [passed]) == 0.0
    with pytest.raises(ValueError, match="negative weight"):
        weighted_rubric_score([{"weight": -1}], [passed])


@pytest.mark.asyncio
async def test_unknown_scoring_method_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown APEX scoring method"):
        await grade_apex_output(
            server_client=MagicMock(),
            model_server_name="judge-server",
            task_id="task-1",
            world_id="world-1",
            instruction="Do it",
            response="Done",
            rubric=[{"verifier_id": "v1", "criteria": "Thing"}],
            expected_output="message_in_console",
            artifact_changes=[],
            final_root=tmp_path,
            judge_model="gemini-3-flash",
            judge_create_params_overrides=None,
            judge_context_window_size=1_000_000,
            scoring_method="partial",
        )


def _sheet_change(title: str, content: str) -> ArtifactChange:
    return ArtifactChange(
        path="filesystem/model.xlsx",
        change_type="added",
        before_path=None,
        after_path=Path("/unused/model.xlsx"),
        artifact_type="sheet",
        index=0,
        title=title,
        new_content=content,
    )


@pytest.mark.parametrize("budget", [400, 1_000, 1_500])
def test_one_oversized_artifact_is_truncated_not_dropped(budget: int) -> None:
    rendered = _artifact_xml([_sheet_change("Summary", "value row\n" * 400)], character_budget=budget)

    assert rendered.startswith('<ARTIFACT id="1" type="sheet" change="created">')
    assert rendered.endswith("</ARTIFACT>")
    assert "<path>model.xlsx</path>" in rendered
    assert "<title>Summary</title>" in rendered
    assert "value row" in rendered
    assert len(rendered) <= budget


def test_several_oversized_artifacts_share_the_total_budget() -> None:
    changes = [_sheet_change(f"Sheet {number}", f"row {number}\n" * 500) for number in range(3)]

    # Three equal shares of the budget: each artifact is cut to its share, wrapper included.
    rendered = _artifact_xml(changes, character_budget=3_300)

    assert len(rendered) <= 3_300
    assert rendered.count("<ARTIFACT ") == rendered.count("</ARTIFACT>") == 3
    for number in range(3):
        assert f"row {number}" in rendered


def test_an_artifact_whose_wrapper_does_not_fit_is_left_out() -> None:
    assert _artifact_xml([_sheet_change("Summary", "value row\n" * 400)], character_budget=40) == ""


async def _grade(
    client,
    tmp_path: Path,
    *,
    changes: list[ArtifactChange] | None = None,
    window: int = 1_000_000,
    response: str = "Done",
):
    return await grade_apex_output(
        server_client=client,
        model_server_name="apex_judge_model",
        task_id="task-1",
        world_id="world-1",
        instruction="Do the task",
        response=response,
        rubric=[{"verifier_id": "v1", "criteria": "Correct result"}],
        expected_output=None,
        artifact_changes=changes or [],
        final_root=tmp_path,
        judge_model="judge",
        judge_create_params_overrides=None,
        judge_context_window_size=window,
    )


@pytest.mark.asyncio
async def test_a_failed_judge_call_is_a_judge_failure_not_a_wrong_answer(tmp_path: Path) -> None:
    client = MagicMock()
    client.post = AsyncMock(side_effect=ConnectionError("judge unreachable"))

    with pytest.raises(Exception) as raised:
        await _grade(client, tmp_path)

    assert is_judge_failure(raised.value)


@pytest.mark.asyncio
async def test_a_failed_criterion_stops_the_other_criteria_judge_calls(tmp_path: Path) -> None:
    cancelled = 0

    async def post(**kwargs):
        nonlocal cancelled
        if "Criterion 0" in json.dumps(kwargs.get("json")):
            # Let the other criteria reach their own judge calls first.
            await asyncio.sleep(0.05)
            raise ConnectionError("judge unreachable")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise

    client = MagicMock()
    client.post = post

    with pytest.raises(Exception) as raised:
        await asyncio.wait_for(
            grade_apex_output(
                server_client=client,
                model_server_name="apex_judge_model",
                task_id="task-1",
                world_id="world-1",
                instruction="Do the task",
                response="Done",
                rubric=[{"verifier_id": f"v{index}", "criteria": f"Criterion {index}"} for index in range(3)],
                expected_output=None,
                artifact_changes=[],
                final_root=tmp_path,
                judge_model="judge",
                judge_create_params_overrides=None,
                judge_context_window_size=1_000_000,
            ),
            10,
        )

    # The first failure keeps its type, so the verify endpoint's judge failsafe still masks the rollout ...
    assert is_judge_failure(raised.value)
    # ... and no judge call outlives the grading that started it.
    assert cancelled == 2


@pytest.mark.asyncio
async def test_a_judge_http_error_is_a_judge_failure(monkeypatch, tmp_path: Path) -> None:
    client = MagicMock()
    client.post = AsyncMock(return_value=object())

    async def unavailable(_response) -> None:
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", unavailable)

    with pytest.raises(Exception) as raised:
        await _grade(client, tmp_path)

    assert is_judge_failure(raised.value)


@pytest.mark.asyncio
async def test_a_verdict_that_cannot_be_read_is_a_judge_failure(monkeypatch, tmp_path: Path) -> None:
    client = MagicMock()
    client.post = AsyncMock(return_value=object())

    async def fake_status(_response) -> None:
        return None

    async def not_json(_response) -> dict:
        return {"choices": [{"message": {"content": "not json"}}]}

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", fake_status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", not_json)

    with pytest.raises(JudgeError, match="invalid structured output"):
        await _grade(client, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("selector_fault", ["call", "unreadable"])
async def test_a_failed_artifact_selection_is_not_graded_as_no_artifacts(
    monkeypatch, tmp_path: Path, selector_fault: str
) -> None:
    deliverable = tmp_path / "filesystem" / "notes.txt"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text("Germany-North Thermal Overload " * 200, encoding="utf-8")
    changes = [
        ArtifactChange(path="filesystem/notes.txt", change_type="added", before_path=None, after_path=deliverable)
    ]
    client = MagicMock()
    client.post = AsyncMock(return_value=object())

    async def status(_response) -> None:
        if selector_fault == "call":
            raise RuntimeError("503 Service Unavailable")

    async def unreadable(_response) -> dict:
        return {"choices": [{"message": {"content": "not json"}}]}

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", unreadable)
    monkeypatch.setattr(
        "resources_servers.apex_agents.judge.artifact_change_content",
        lambda change: change.after_path.read_text(encoding="utf-8"),
    )

    with pytest.raises(Exception) as raised:
        # A 100-token window forces the selection step before any criterion is graded.
        await _grade(client, tmp_path, changes=changes, window=100)

    assert is_judge_failure(raised.value)
    # Every request made was a selection request: no criterion was graded on an empty selection.
    for call in client.post.call_args_list:
        assert "preprocessing filter" in call.kwargs["json"]["messages"][0]["content"]


def _verdict(passed: bool) -> dict:
    return {"choices": [{"message": {"content": json.dumps({"rationale": "checked", "is_criteria_true": passed})}}]}


def _http_error(status: int, body: str):
    from aiohttp import ClientResponseError

    error = ClientResponseError(request_info=MagicMock(), history=(), status=status, message="rejected")
    error.response_content = body.encode("utf-8")
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, '{"error": {"code": "context_length_exceeded", "message": "maximum context length is 1000000 tokens"}}'),
        (
            400,
            '{"error": {"message": "The input token count (1210000) exceeds the maximum number of tokens allowed"}}',
        ),
        (413, "Request Entity Too Large"),
    ],
)
async def test_a_request_the_judge_refuses_for_its_size_scores_the_criterion_zero(
    monkeypatch, tmp_path: Path, status: int, body: str
) -> None:
    client = MagicMock()
    client.post = AsyncMock(return_value=object())

    async def too_large(_response) -> None:
        raise _http_error(status, body)

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", too_large)

    # The final answer alone is some 600,000 estimated tokens: over half of the judge's 1,000,000-token window.
    reward, scores, usage = await _grade(client, tmp_path, response="figure " * 180_000)

    # The policy's oversized output, graded: a completed zero, not a masked judge failure.
    assert reward == 0.0
    assert scores["v1"]["score"] == 0.0
    assert scores["v1"]["status"] == "completed"
    assert scores["v1"]["values"]["judge_request_rejected"] == "request_too_large"
    assert usage["scoring"]["scoring_method_result_values"]["failed_count"] == 1
    # The refusal would repeat, so the request is not retried.
    assert client.post.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, '{"error": {"message": "Unsupported parameter: reasoning_effort"}}'),
        # A small request refused in the words of a size limit: the judge's setup, not the policy's output.
        (
            400,
            "'max_tokens' is too large: 65536. This model's maximum context length is 65536 tokens "
            "and your request has 512 input tokens",
        ),
        (413, "Request Entity Too Large"),
        (500, '{"error": {"message": "maximum context length exceeded"}}'),
        (503, "Service Unavailable"),
        (429, "rate limited"),
    ],
)
async def test_other_judge_http_errors_stay_judge_failures(
    monkeypatch, tmp_path: Path, status: int, body: str
) -> None:
    client = MagicMock()
    client.post = AsyncMock(return_value=object())

    async def failing(_response) -> None:
        raise _http_error(status, body)

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", failing)

    with pytest.raises(Exception) as raised:
        await _grade(client, tmp_path)

    assert is_judge_failure(raised.value)


@pytest.mark.asyncio
async def test_a_refused_criterion_does_not_stop_the_other_criteria(monkeypatch, tmp_path: Path) -> None:
    client = MagicMock()

    async def post(**kwargs):
        return "large" if "Criterion 0" in json.dumps(kwargs["json"]) else "fine"

    client.post = post

    async def status(response) -> None:
        if response == "large":
            raise _http_error(413, "payload too large")

    async def reply(_response) -> dict:
        return _verdict(True)

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", reply)

    reward, scores, _usage = await grade_apex_output(
        server_client=client,
        model_server_name="apex_judge_model",
        task_id="task-1",
        world_id="world-1",
        instruction="Do the task",
        response="figure " * 180_000,
        rubric=[
            {"verifier_id": "v0", "criteria": "Criterion 0", "weight": 1},
            {"verifier_id": "v1", "criteria": "Criterion 1", "weight": 3},
        ],
        expected_output=None,
        artifact_changes=[],
        final_root=tmp_path,
        judge_model="judge",
        judge_create_params_overrides=None,
        judge_context_window_size=1_000_000,
        scoring_method="weighted_average",
    )

    assert (scores["v0"]["score"], scores["v1"]["score"]) == (0.0, 1.0)
    assert reward == 0.75


def _image_block(size: int = 100) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * size}}


def test_images_are_capped_by_count_with_a_note_for_the_judge() -> None:
    blocks = [{"type": "text", "text": "\n0000__deck.pptx (rendered from Office):"}]
    for page in range(_MAX_JUDGE_IMAGES + 5):
        blocks += [{"type": "text", "text": f"Page {page + 1}"}, _image_block()]

    kept, image_count = _budget_images(blocks, context_window_size=1_000_000)

    assert image_count == _MAX_JUDGE_IMAGES
    assert sum(block["type"] == "image_url" for block in kept) == _MAX_JUDGE_IMAGES
    # An omitted image takes its page label with it, and the judge is told images are missing.
    assert {"type": "text", "text": f"Page {_MAX_JUDGE_IMAGES + 1}"} not in kept
    assert kept[-1]["text"] == "\n[5 further image(s) omitted: judge request size limit]"


def test_images_are_capped_by_the_judge_window_and_by_encoded_size(monkeypatch) -> None:
    blocks = [_image_block() for _ in range(10)]

    # A quarter of a 40,000-token window holds three images at the fixed per-image cost.
    assert _budget_images(blocks, context_window_size=40_000)[1] == 10_000 // _IMAGE_TOKEN_ESTIMATE == 3
    # A window too small for any image sends none.
    assert _budget_images(blocks, context_window_size=1_000) == ([], 0)

    monkeypatch.setattr("resources_servers.apex_agents.judge._MAX_JUDGE_IMAGE_CHARS", 450)
    kept, image_count = _budget_images(blocks, context_window_size=1_000_000)
    assert image_count == 3
    assert sum(len(block["image_url"]["url"]) for block in kept if block["type"] == "image_url") <= 450


def test_the_grading_prompt_is_budgeted_at_two_characters_per_token_less_the_images() -> None:
    change = _sheet_change("Dense", "1234567890\t" * 50_000)

    def prompt(reserved_tokens: int) -> str:
        return _build_prompt(
            instruction="Do the task",
            response="Done",
            criteria="Correct result",
            file_type="Spreadsheets (.xlsx, .xls, .xlsm, .ods)",
            changes=[change],
            context_window_size=100_000,
            reserved_tokens=reserved_tokens,
        )

    # 90% of the window at two characters per token, not four: the 550,000-character sheet is cut to fit.
    assert len(prompt(0)) <= 100_000 * 0.9 * 2
    assert "1234567890" in prompt(0)
    # Tokens charged to images come out of the same window.
    assert len(prompt(0)) - len(prompt(30_000)) >= 30_000 * 2 - 100


@pytest.mark.asyncio
async def test_criteria_sharing_an_id_are_all_counted(monkeypatch, tmp_path: Path) -> None:
    client = MagicMock()
    client.post = AsyncMock(side_effect=lambda **kwargs: json.dumps(kwargs["json"]))

    async def status(_response) -> None:
        return None

    async def reply(request: str) -> dict:
        return _verdict("First criterion" not in request)

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", reply)

    reward, scores, usage = await grade_apex_output(
        server_client=client,
        model_server_name="apex_judge_model",
        task_id="task-1",
        world_id="world-1",
        instruction="Do the task",
        response="Done",
        # The failing criterion comes first: an ID-keyed mapping would keep only the passing one.
        rubric=[{"verifier_id": "v1", "criteria": "First criterion"}, {"verifier_id": "v1", "criteria": "Second"}],
        expected_output=None,
        artifact_changes=[],
        final_root=tmp_path,
        judge_model="judge",
        judge_create_params_overrides=None,
        judge_context_window_size=1_000_000,
    )

    values = usage["scoring"]["scoring_method_result_values"]
    assert reward == 0.0
    assert (values["passed_count"], values["failed_count"], values["total_count"]) == (1, 1, 2)
    assert values["all_pass"] == 0.0
    assert usage["verifier_count"] == 2
    assert {key: score["score"] for key, score in scores.items()} == {"v1": 0.0, "v1#2": 1.0}


def _slide_change(deck: Path, index: int) -> ArtifactChange:
    return ArtifactChange(
        path="filesystem/deck.pptx",
        change_type="modified",
        before_path=deck,
        after_path=deck,
        artifact_type="slide",
        index=index,
        original_index=index,
        title=f"Slide {index + 1}",
        old_content="old",
        new_content="new",
        content_diff="-old\n+new",
    )


@pytest.mark.asyncio
async def test_a_changed_file_is_rendered_once_for_all_criteria_at_its_changed_slides(
    monkeypatch, tmp_path: Path
) -> None:
    deck = tmp_path / "filesystem" / "deck.pptx"
    deck.parent.mkdir(parents=True)
    deck.write_bytes(b"deck")
    renders: list[tuple[Path, int, set[int] | None, dict]] = []

    def render(final_root: Path, path: Path, position: int, changed_indices: set[int] | None, **converter):
        assert final_root == tmp_path
        renders.append((path, position, changed_indices, converter))
        return [{"type": "text", "text": "\n0000__deck.pptx:"}, {"type": "text", "text": "Slide 14"}, _image_block()]

    client = MagicMock()
    client.post = AsyncMock(return_value=object())

    async def status(_response) -> None:
        return None

    async def reply(_response) -> dict:
        return _verdict(True)

    monkeypatch.setattr("resources_servers.apex_agents.judge._visual_blocks", render)
    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", reply)

    reward, _scores, _usage = await grade_apex_output(
        server_client=client,
        model_server_name="apex_judge_model",
        task_id="task-1",
        world_id="world-1",
        instruction="Do the task",
        response="Done",
        rubric=[{"verifier_id": f"v{index}", "criteria": f"Criterion {index}"} for index in range(4)],
        expected_output=None,
        artifact_changes=[_slide_change(deck, 13), _slide_change(deck, 2)],
        final_root=tmp_path,
        judge_model="judge",
        judge_create_params_overrides=None,
        judge_context_window_size=1_000_000,
        document_converter_image="/images/archipelago.sif",
    )

    assert reward == 1.0
    # One render for four criteria, asked for the changed slides rather than the deck's first pages.
    assert renders == [
        (deck, 0, {2, 13}, {"document_converter_image": "/images/archipelago.sif", "allow_host_document_tools": False})
    ]
    assert client.post.call_count == 4
    for call in client.post.call_args_list:
        content = call.kwargs["json"]["messages"][1]["content"]
        assert [block["type"] for block in content] == ["text", "text", "text", "image_url"]


@pytest.mark.asyncio
async def test_prompt_assembly_and_rendering_run_through_the_blocking_runner(monkeypatch, tmp_path: Path) -> None:
    deliverable = tmp_path / "filesystem" / "notes.txt"
    deliverable.parent.mkdir(parents=True)
    deliverable.write_text("finished", encoding="utf-8")
    change = ArtifactChange(path="filesystem/notes.txt", change_type="added", before_path=None, after_path=deliverable)
    offloaded: list[str] = []

    async def run_blocking(function, *args, **kwargs):
        offloaded.append(getattr(getattr(function, "func", function), "__name__", "?"))
        return await asyncio.to_thread(function, *args, **kwargs)

    client = MagicMock()
    client.post = AsyncMock(return_value=object())

    async def status(_response) -> None:
        return None

    async def reply(_response) -> dict:
        return _verdict(True)

    monkeypatch.setattr("resources_servers.apex_agents.judge.raise_for_status", status)
    monkeypatch.setattr("resources_servers.apex_agents.judge.get_response_json", reply)

    await grade_apex_output(
        server_client=client,
        model_server_name="apex_judge_model",
        task_id="task-1",
        world_id="world-1",
        instruction="Do the task",
        response="Done",
        rubric=[{"verifier_id": "v1", "criteria": "Correct result"}],
        expected_output=None,
        artifact_changes=[change],
        final_root=tmp_path,
        judge_model="judge",
        judge_create_params_overrides=None,
        judge_context_window_size=1_000_000,
        run_blocking=run_blocking,
    )

    assert offloaded == ["<lambda>", "_visual_blocks", "_build_prompt"]


def test_a_refusal_counts_against_the_policy_only_when_the_request_was_large() -> None:
    def request(text: str, images: int = 0, image_size: int = 100) -> dict:
        content = [{"type": "text", "text": text}, *[_image_block(image_size) for _ in range(images)]]
        return {
            "model": "judge",
            "messages": [{"role": "system", "content": "Grade."}, {"role": "user", "content": content}],
        }

    small = request("a short prompt")
    # 120,000 characters are 60,000 estimated tokens, over half of a 100,000-token window.
    long_text = request("x" * 120_000)
    # Twenty images are charged 60,000 tokens.
    many_images = request("a short prompt", images=20)
    # One 1.5 MiB image is few tokens but a large payload.
    heavy_payload = request("a short prompt", images=1, image_size=1_500_000)

    for status in (400, 413):
        assert not _request_is_large(small, status=status, context_window_size=100_000)
        assert _request_is_large(long_text, status=status, context_window_size=100_000)
        assert _request_is_large(many_images, status=status, context_window_size=100_000)
    # Payload size speaks only to a payload-size refusal.
    assert _request_is_large(heavy_payload, status=413, context_window_size=100_000)
    assert not _request_is_large(heavy_payload, status=400, context_window_size=100_000)
