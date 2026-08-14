# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused APEX artifact judging through Gym's configured model server."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from aiohttp import ClientTimeout
from pydantic import BaseModel, ValidationError

from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.apex_agents.artifacts import (
    ArtifactChange,
    artifact_change_text,
    visual_content_blocks,
)
from resources_servers.apex_agents.prompts import (
    ARTIFACT_STRUCTURE,
    EVAL_SCOPE_BOTH,
    EVAL_SCOPE_FILES_ONLY,
    EVAL_SCOPE_TEXT_ONLY,
    GRADING_SYSTEM_PROMPT,
    GRADING_USER_PROMPT,
)


_ALL_OUTPUT = "All output (modified files and final message in console)"
_FINAL_ANSWER_ONLY = "Final Answer Only (No Files)"
_MAX_JSON_RETRIES = 10

EXPECTED_FILE_TYPES = {
    "message_in_console": _FINAL_ANSWER_ONLY,
    "make_new_doc": "Word Documents (.docx, .doc)",
    "edit_existing_doc": "Word Documents (.docx, .doc)",
    "make_new_sheet": "Spreadsheets (.xlsx, .xls, .xlsm)",
    "edit_existing_sheet": "Spreadsheets (.xlsx, .xls, .xlsm)",
    "make_new_slide_deck": "Presentations (.pptx, .ppt)",
    "edit_existing_slide_deck": "Presentations (.pptx, .ppt)",
}

_FILE_TYPE_ALIASES = {
    "Word document": EXPECTED_FILE_TYPES["make_new_doc"],
    "Spreadsheet": EXPECTED_FILE_TYPES["make_new_sheet"],
    "Presentation": EXPECTED_FILE_TYPES["make_new_slide_deck"],
    "All output": _ALL_OUTPUT,
}

_FILE_TYPE_EXTENSIONS = {
    "Word Documents (.docx, .doc)": {".docx", ".doc"},
    "Text Documents (.docx, .doc, .txt)": {".docx", ".doc", ".txt"},
    "Spreadsheets (.xlsx, .xls, .xlsm)": {".xlsx", ".xls", ".xlsm"},
    "Presentations (.pptx, .ppt)": {".pptx", ".ppt"},
}


class GradingResponse(BaseModel):
    rationale: str
    is_criteria_true: bool


def expected_file_type(expected_output: str | None, criterion: dict[str, Any]) -> str:
    target = criterion.get("grading_target")
    if isinstance(target, dict):
        explicit = str(target.get("expected_file_type") or "").strip()
        if explicit:
            return _FILE_TYPE_ALIASES.get(explicit, explicit)
    return EXPECTED_FILE_TYPES.get(str(expected_output or ""), _ALL_OUTPUT)


def _structured_response_format(response_format: Any) -> Any:
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
                "strict": True,
            },
        }
    return response_format


def _display_path(path: str) -> str:
    parts = path.split("/", 1)
    return parts[1] if len(parts) == 2 and parts[0] in {"filesystem", ".apps_data"} else path


def _matching_changes(changes: list[ArtifactChange], file_type: str) -> list[ArtifactChange]:
    if file_type == _FINAL_ANSWER_ONLY:
        return []
    extensions = _FILE_TYPE_EXTENSIONS.get(file_type)
    if extensions is None:
        return changes
    return [change for change in changes if Path(change.path).suffix.lower() in extensions]


def _artifact_xml(changes: list[ArtifactChange], *, character_budget: int) -> str:
    if not changes:
        return ""
    per_artifact = max(1_000, character_budget // len(changes))
    remaining = character_budget
    rendered: list[str] = []
    for index, change in enumerate(changes, 1):
        content = artifact_change_text(change, max_chars=min(per_artifact, remaining))
        change_name = {"added": "created", "modified": "modified", "deleted": "deleted"}[change.change_type]
        content_tag = {
            "added": "created_content",
            "modified": "diff",
            "deleted": "deleted_content",
        }[change.change_type]
        artifact = (
            f'<ARTIFACT id="{index}" type="file" change="{change_name}">\n'
            f"  <path>{_display_path(change.path)}</path>\n"
            f"  <{content_tag}>\n{content}\n  </{content_tag}>\n"
            "</ARTIFACT>"
        )
        if len(artifact) > remaining:
            break
        rendered.append(artifact)
        remaining -= len(artifact)
    return "\n\n".join(rendered)


def _build_prompt(
    *,
    instruction: str,
    response: str,
    criteria: str,
    file_type: str,
    changes: list[ArtifactChange],
    context_window_size: int,
) -> str:
    if file_type == _FINAL_ANSWER_ONLY:
        scope = EVAL_SCOPE_TEXT_ONLY
        artifact_structure = ""
        agent_output = response
    else:
        scope = EVAL_SCOPE_BOTH if file_type == _ALL_OUTPUT else EVAL_SCOPE_FILES_ONLY
        artifact_structure = ARTIFACT_STRUCTURE
        fixed_prompt_chars = len(GRADING_SYSTEM_PROMPT) + len(instruction) + len(response) + len(criteria) + 2_000
        artifact_budget = max(4_000, int(context_window_size * 4 * 0.8) - fixed_prompt_chars)
        artifacts = _artifact_xml(changes, character_budget=artifact_budget)
        final_answer = f"<FINAL_ANSWER>\n{response}\n</FINAL_ANSWER>\n" if scope == EVAL_SCOPE_BOTH else ""
        agent_output = f"{final_answer}{artifacts}".strip()

    return GRADING_USER_PROMPT.format(
        artifact_structure=artifact_structure,
        instruction=instruction,
        agent_output=agent_output,
        criteria=criteria,
        evaluation_scope=scope,
    )


def _visual_blocks(final_root: Path, changes: list[ArtifactChange]) -> list[dict[str, Any]]:
    files = [change.after_path for change in changes if change.after_path is not None]
    if not files:
        return []
    return [block for block in visual_content_blocks(final_root, files) if block.get("type") == "image_url"]


async def _judge_criterion(
    *,
    server_client: Any,
    model_server_name: str,
    judge_model: str,
    judge_create_params_overrides: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    visual_blocks: list[dict[str, Any]],
    capture_judge_trace: bool,
) -> tuple[GradingResponse, dict[str, Any] | None]:
    user_content: str | list[dict[str, Any]] = user_prompt
    if visual_blocks:
        user_content = [{"type": "text", "text": user_prompt}, *visual_blocks]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw_content: str | None = None
    parsed: GradingResponse | None = None
    for _ in range(_MAX_JSON_RETRIES):
        body = dict(judge_create_params_overrides)
        body.update(
            {
                "model": judge_model,
                "messages": messages,
                "response_format": _structured_response_format(GradingResponse),
            }
        )
        judge_response = await server_client.post(
            server_name=model_server_name,
            url_path="/v1/chat/completions",
            json=body,
            timeout=ClientTimeout(total=3600),
        )
        await raise_for_status(judge_response)
        payload = await get_response_json(judge_response)
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        raw_content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(raw_content, str) or not raw_content:
            continue
        try:
            raw_json = json.loads(raw_content)
            if isinstance(raw_json, dict) and isinstance(raw_json.get("rationale"), dict):
                raw_json["rationale"] = json.dumps(raw_json["rationale"])
            parsed = GradingResponse.model_validate(raw_json)
            break
        except (json.JSONDecodeError, ValidationError):
            continue
    if parsed is None:
        raise ValueError(f"judge returned invalid structured output after {_MAX_JSON_RETRIES} attempts")

    trace = None
    if capture_judge_trace:
        trace = {
            "model": judge_model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "messages": messages,
            "raw_response": raw_content,
            "parsed_response": parsed.model_dump(mode="json"),
            "image_count": len(visual_blocks),
        }
    return parsed, trace


async def grade_apex_output(
    *,
    server_client: Any,
    model_server_name: str,
    task_id: str,
    world_id: str,
    instruction: str,
    response: str,
    rubric: list[dict[str, Any]],
    expected_output: str | None,
    artifact_changes: list[ArtifactChange],
    final_root: Path,
    judge_model: str,
    judge_create_params_overrides: dict[str, Any] | None,
    judge_context_window_size: int,
    capture_judge_traces: bool,
    metadata: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Grade every APEX rubric criterion and return its fractional pass rate."""
    del world_id, metadata
    overrides = dict(judge_create_params_overrides or {})

    async def grade_one(index: int, criterion: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        verifier_id = str(criterion.get("verifier_id") or "").strip()
        criteria = str(criterion.get("criteria") or "").strip()
        if not verifier_id or not criteria:
            raise ValueError(f"rubric criterion {index} must include verifier_id and criteria")
        file_type = expected_file_type(expected_output, criterion)
        selected = _matching_changes(artifact_changes, file_type)
        user_prompt = _build_prompt(
            instruction=instruction,
            response=response,
            criteria=criteria,
            file_type=file_type,
            changes=selected,
            context_window_size=judge_context_window_size,
        )
        visuals = _visual_blocks(final_root, selected)
        parsed, trace = await _judge_criterion(
            server_client=server_client,
            model_server_name=model_server_name,
            judge_model=judge_model,
            judge_create_params_overrides=overrides,
            system_prompt=GRADING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            visual_blocks=visuals,
            capture_judge_trace=capture_judge_traces,
        )
        score = 1.0 if parsed.is_criteria_true else 0.0
        values: dict[str, Any] = {
            "judge_grade": "pass" if parsed.is_criteria_true else "fail",
            "grade_rationale": parsed.rationale,
            "evaluated_artifacts": ", ".join(_display_path(change.path) for change in selected),
        }
        if trace is not None:
            values["judge_trace"] = trace
        return verifier_id, {
            "score": score,
            "status": "completed",
            "message": parsed.rationale,
            "values": values,
        }

    results = await asyncio.gather(*(grade_one(index, criterion) for index, criterion in enumerate(rubric)))
    rubric_scores = dict(results)
    passed_count = sum(score["score"] >= 0.99 for score in rubric_scores.values())
    total_count = len(rubric_scores)
    failed_count = total_count - passed_count
    reward = passed_count / total_count if total_count else 0.0
    scoring = {
        "final_score": reward,
        "scoring_method_result_values": {
            "passed_count": passed_count,
            "failed_count": failed_count,
            "total_count": total_count,
            "grade_score_percentage": reward * 100,
        },
    }
    judge_usage = {
        "ok": True,
        "grading_run_id": str(uuid.uuid4()),
        "task_id": task_id,
        "status": "completed",
        "scoring": scoring,
        "verifier_count": total_count,
        "document_extraction": "local",
    }
    return reward, rubric_scores, judge_usage
