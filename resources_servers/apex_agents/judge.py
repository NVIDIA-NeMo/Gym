# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused APEX artifact judging through Gym's configured model server."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from aiohttp import ClientTimeout
from pydantic import BaseModel, ValidationError

from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.apex_agents.artifacts import (
    ArtifactChange,
    artifact_change_content,
    artifact_change_text,
    visual_content_blocks,
)
from resources_servers.apex_agents.prompts import (
    ARTIFACT_SELECTION_SYSTEM_PROMPT,
    ARTIFACT_SELECTION_USER_PROMPT,
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
_GEMINI_CONSERVATIVE_TOKEN_MULTIPLIER = 1.9
_TOTAL_CONTENT_BUDGET_RATIO = 0.90
LOG = logging.getLogger(__name__)

EXPECTED_FILE_TYPES = {
    "message_in_console": _FINAL_ANSWER_ONLY,
    "make_new_doc": "Word Documents (.docx, .doc, .odt)",
    "edit_existing_doc": "Word Documents (.docx, .doc, .odt)",
    "make_new_sheet": "Spreadsheets (.xlsx, .xls, .xlsm, .ods)",
    "edit_existing_sheet": "Spreadsheets (.xlsx, .xls, .xlsm, .ods)",
    "make_new_slide_deck": "Presentations (.pptx, .ppt, .odp)",
    "edit_existing_slide_deck": "Presentations (.pptx, .ppt, .odp)",
}

_FILE_TYPE_EXTENSIONS = {
    "Word Documents (.docx, .doc, .odt)": {".docx", ".doc", ".odt"},
    "Text Files (.txt)": {".txt"},
    "PDF Documents (.pdf)": {".pdf"},
    "Spreadsheets (.xlsx, .xls, .xlsm, .ods)": {".xlsx", ".xls", ".xlsm", ".ods"},
    "Presentations (.pptx, .ppt, .odp)": {".pptx", ".ppt", ".odp"},
    "Python Files (.py)": {".py"},
    "JavaScript/TypeScript (.js, .ts, .jsx, .tsx)": {".js", ".ts", ".jsx", ".tsx"},
    "Markdown (.md)": {".md"},
    "JSON/YAML (.json, .yaml, .yml)": {".json", ".yaml", ".yml"},
    "Images (.png, .jpg, .jpeg, .webp)": {".png", ".jpg", ".jpeg", ".webp"},
}


class GradingResponse(BaseModel):
    rationale: str
    is_criteria_true: bool


class ArtifactSelectionResponse(BaseModel):
    rationale: str
    selected_artifact_indices: list[int]


def expected_file_type(expected_output: str | None, criterion: dict[str, Any]) -> str:
    target = criterion.get("grading_target")
    if isinstance(target, dict):
        explicit = str(target.get("expected_file_type") or "").strip()
        if explicit:
            if explicit in {_FINAL_ANSWER_ONLY, _ALL_OUTPUT, *_FILE_TYPE_EXTENSIONS}:
                return explicit
            return _ALL_OUTPUT
    return EXPECTED_FILE_TYPES.get(str(expected_output or ""), _ALL_OUTPUT)


def _display_path(path: str) -> str:
    parts = path.split("/", 1)
    return parts[1] if len(parts) == 2 and parts[0] in {"filesystem", ".apps_data"} else path


def _artifact_display_name(change: ArtifactChange) -> str:
    path = _display_path(change.path)
    if change.artifact_type in {"slide", "sheet", "page"} and change.index is not None:
        label = f"{change.artifact_type.capitalize()} {change.index + 1}"
        if change.title:
            label += f": {change.title}"
        return f"{path} ({label})"
    return path


def _artifact_metadata_xml(change: ArtifactChange) -> list[str]:
    metadata = [f"  <path>{_display_path(change.path)}</path>"]
    is_sub_artifact = change.index is not None or change.artifact_type in {"slide", "sheet", "page"}
    if is_sub_artifact:
        if change.title:
            metadata.append(f"  <title>{change.title}</title>")
        if change.index is not None:
            metadata.append(f"  <sub_index>{change.index + 1}</sub_index>")
        if change.original_index is not None:
            metadata.append(f"  <original_index>{change.original_index + 1}</original_index>")
    return metadata


def _matching_changes(changes: list[ArtifactChange], file_type: str) -> list[ArtifactChange]:
    if file_type == _FINAL_ANSWER_ONLY:
        return []
    extensions = _FILE_TYPE_EXTENSIONS.get(file_type)
    if extensions is None:
        return changes
    return [change for change in changes if Path(change.path).suffix.lower() in extensions]


def _should_auto_fail_missing_file_type(file_type: str, changes: list[ArtifactChange]) -> bool:
    return file_type in _FILE_TYPE_EXTENSIONS and not changes


def _estimate_tokens(text: str, *, model: str, conservative: bool = False) -> int:
    """Match Archipelago's LiteLLM fallback estimate without importing LiteLLM."""
    estimate = max(1, (len(text) + 3) // 4)
    if conservative and "gemini" in model.lower():
        estimate = int(estimate * _GEMINI_CONSERVATIVE_TOKEN_MULTIPLIER)
    return estimate


def _selection_artifact_xml(index: int, change: ArtifactChange, content: str, *, truncated: bool) -> str:
    change_name = {"added": "created", "modified": "modified", "deleted": "deleted"}[change.change_type]
    content_tag = {
        "added": "created_content",
        "modified": "diff",
        "deleted": "deleted_content",
    }[change.change_type]
    truncated_attribute = ' truncated="true"' if truncated else ""
    indented = "\n".join(f"    {line}" for line in content.splitlines())
    parts = [
        f'<ARTIFACT id="{index}" type="{change.artifact_type}" change="{change_name}"{truncated_attribute}>',
        *_artifact_metadata_xml(change),
        f"  <{content_tag}>",
        indented,
        f"  </{content_tag}>",
        "</ARTIFACT>",
    ]
    return "\n".join(parts)


async def _select_artifacts_if_needed(
    *,
    server_client: Any,
    model_server_name: str,
    judge_model: str,
    judge_create_params_overrides: dict[str, Any],
    instruction: str,
    criteria: str,
    changes: list[ArtifactChange],
    context_window_size: int,
) -> tuple[list[ArtifactChange], dict[str, Any]]:
    contents = [artifact_change_content(change) for change in changes]
    total_tokens = sum(_estimate_tokens(content, model=judge_model) for content in contents)
    threshold = int(context_window_size * 0.5)
    if total_tokens <= threshold:
        return changes, {
            "status": "skipped",
            "reason": "artifacts_fit_within_budget",
            "artifact_tokens": total_tokens,
            "threshold_tokens": threshold,
            "selected_count": len(changes),
            "total_count": len(changes),
        }

    task_prompt_section = f"<ORIGINAL_TASK>\n{instruction}\n</ORIGINAL_TASK>\n" if instruction else ""
    prompt_without_artifacts = ARTIFACT_SELECTION_USER_PROMPT.format(
        task_prompt_section=task_prompt_section,
        criteria=criteria,
        artifacts_list="",
    )
    base_tokens = _estimate_tokens(
        ARTIFACT_SELECTION_SYSTEM_PROMPT + "\n" + prompt_without_artifacts,
        model=judge_model,
        conservative=True,
    )
    artifact_token_budget = max(0, int(context_window_size * 0.6) - base_tokens - 500)
    per_artifact_chars = (
        max(1, int((artifact_token_budget / len(changes)) * 4 / _GEMINI_CONSERVATIVE_TOKEN_MULTIPLIER))
        if changes
        else 0
    )
    rendered: list[str] = []
    for index, (change, content) in enumerate(zip(changes, contents, strict=True), 1):
        truncated = len(content) > per_artifact_chars
        visible_content = content[:per_artifact_chars] if truncated else content
        rendered.append(_selection_artifact_xml(index, change, visible_content, truncated=truncated))
    user_prompt = ARTIFACT_SELECTION_USER_PROMPT.format(
        task_prompt_section=task_prompt_section,
        criteria=criteria,
        artifacts_list="\n\n".join(rendered),
    )
    messages = [
        {"role": "system", "content": ARTIFACT_SELECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw_content: str | None = None
    parsed: ArtifactSelectionResponse | None = None
    try:
        for _ in range(_MAX_JSON_RETRIES):
            body = dict(judge_create_params_overrides)
            body.update(
                {
                    "model": judge_model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                }
            )
            response = await server_client.post(
                server_name=model_server_name,
                url_path="/v1/chat/completions",
                json=body,
                timeout=ClientTimeout(total=3600),
            )
            await raise_for_status(response)
            payload = await get_response_json(response)
            choices = payload.get("choices") if isinstance(payload, dict) else None
            message = choices[0].get("message") if isinstance(choices, list) and choices else None
            raw_content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(raw_content, str) or not raw_content:
                continue
            try:
                parsed = ArtifactSelectionResponse.model_validate_json(raw_content)
                break
            except ValidationError:
                continue
    except Exception as exc:
        LOG.exception("APEX artifact selection failed")
        return [], {
            "status": "failed",
            "error": str(exc),
            "artifact_tokens": total_tokens,
            "threshold_tokens": threshold,
            "selected_count": 0,
            "total_count": len(changes),
        }
    if parsed is None:
        return [], {
            "status": "failed",
            "error": f"selector returned invalid structured output after {_MAX_JSON_RETRIES} attempts",
            "artifact_tokens": total_tokens,
            "threshold_tokens": threshold,
            "selected_count": 0,
            "total_count": len(changes),
        }

    selected = [changes[index - 1] for index in parsed.selected_artifact_indices if 1 <= index <= len(changes)]
    return selected, {
        "status": "completed",
        "rationale": parsed.rationale,
        "selected_artifact_indices": parsed.selected_artifact_indices,
        "artifact_tokens": total_tokens,
        "threshold_tokens": threshold,
        "selected_count": len(selected),
        "total_count": len(changes),
    }


def _artifact_xml(changes: list[ArtifactChange], *, character_budget: int) -> str:
    if not changes:
        return ""
    per_artifact = max(1_000, character_budget // len(changes))
    remaining = character_budget
    rendered: list[str] = []
    for index, change in enumerate(changes, 1):
        content = artifact_change_text(change, max_chars=min(per_artifact, remaining))
        change_name = {"added": "created", "modified": "modified", "deleted": "deleted"}[change.change_type]
        indented_content = "\n".join(f"  {line}" for line in content.splitlines())
        artifact = "\n".join(
            [
                f'<ARTIFACT id="{index}" type="{change.artifact_type}" change="{change_name}">',
                *_artifact_metadata_xml(change),
                indented_content,
                "</ARTIFACT>",
            ]
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
        agent_output = f"<TEXT_RESPONSE>\n{response}\n</TEXT_RESPONSE>"
    else:
        scope = EVAL_SCOPE_BOTH if file_type == _ALL_OUTPUT else EVAL_SCOPE_FILES_ONLY
        artifact_structure = ARTIFACT_STRUCTURE
        fixed_prompt_chars = len(GRADING_SYSTEM_PROMPT) + len(instruction) + len(response) + len(criteria) + 2_000
        artifact_budget = max(
            4_000,
            int(context_window_size * 4 * _TOTAL_CONTENT_BUDGET_RATIO) - fixed_prompt_chars,
        )
        artifacts = _artifact_xml(changes, character_budget=artifact_budget)
        final_answer = f"<TEXT_RESPONSE>\n{response}\n</TEXT_RESPONSE>\n" if scope == EVAL_SCOPE_BOTH else ""
        agent_output = f"{final_answer}{artifacts}".strip()

    return GRADING_USER_PROMPT.format(
        artifact_structure=artifact_structure,
        instruction=instruction,
        agent_output=agent_output,
        criteria=criteria,
        evaluation_scope=scope,
    )


def _visual_blocks(
    final_root: Path,
    changes: list[ArtifactChange],
    *,
    document_converter_image: str | None = None,
) -> list[dict[str, Any]]:
    files = list(dict.fromkeys(change.after_path for change in changes if change.after_path is not None))
    if not files:
        return []
    page_indices_by_path: dict[Path, set[int]] = {}
    for change in changes:
        if change.artifact_type == "page" and change.index is not None and change.after_path is not None:
            page_indices_by_path.setdefault(change.after_path, set()).add(change.index)
    blocks = visual_content_blocks(
        final_root,
        files,
        document_converter_image=document_converter_image,
        page_indices_by_path=page_indices_by_path,
    )
    selected: list[dict[str, Any]] = []
    pending_labels: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") == "text":
            pending_labels.append(block)
        elif block.get("type") == "image_url":
            selected.extend(pending_labels)
            pending_labels.clear()
            selected.append(block)
    return selected


async def _judge_criterion(
    *,
    server_client: Any,
    model_server_name: str,
    judge_model: str,
    judge_create_params_overrides: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    visual_blocks: list[dict[str, Any]],
) -> GradingResponse:
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
                "response_format": {"type": "json_object"},
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
    return parsed


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
    document_converter_image: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Grade every APEX rubric criterion; a rollout passes only when all pass."""
    del world_id, metadata
    overrides = dict(judge_create_params_overrides or {})

    async def grade_one(index: int, criterion: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        verifier_id = str(criterion.get("verifier_id") or "").strip()
        criteria = str(criterion.get("criteria") or "").strip()
        if not verifier_id or not criteria:
            raise ValueError(f"rubric criterion {index} must include verifier_id and criteria")
        file_type = expected_file_type(expected_output, criterion)
        filtered = _matching_changes(artifact_changes, file_type)
        if _should_auto_fail_missing_file_type(file_type, filtered):
            rationale = f"No files matching the expected type ({file_type}) were found."
            return verifier_id, {
                "score": 0.0,
                "status": "completed",
                "message": rationale,
                "values": {
                    "judge_grade": "fail",
                    "grade_rationale": rationale,
                    "evaluated_artifacts": "",
                    "artifact_selection": {
                        "status": "skipped",
                        "reason": "no_matching_artifacts",
                        "selected_count": 0,
                        "total_count": 0,
                    },
                },
            }
        selected, selection_metadata = await _select_artifacts_if_needed(
            server_client=server_client,
            model_server_name=model_server_name,
            judge_model=judge_model,
            judge_create_params_overrides=overrides,
            instruction=instruction,
            criteria=criteria,
            changes=filtered,
            context_window_size=judge_context_window_size,
        )
        user_prompt = _build_prompt(
            instruction=instruction,
            response=response,
            criteria=criteria,
            file_type=file_type,
            changes=selected,
            context_window_size=judge_context_window_size,
        )
        visuals = _visual_blocks(
            final_root,
            selected,
            document_converter_image=document_converter_image,
        )
        parsed = await _judge_criterion(
            server_client=server_client,
            model_server_name=model_server_name,
            judge_model=judge_model,
            judge_create_params_overrides=overrides,
            system_prompt=GRADING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            visual_blocks=visuals,
        )
        score = 1.0 if parsed.is_criteria_true else 0.0
        values: dict[str, Any] = {
            "judge_grade": "pass" if parsed.is_criteria_true else "fail",
            "grade_rationale": parsed.rationale,
            "evaluated_artifacts": ", ".join(_artifact_display_name(change) for change in selected),
            "artifact_selection": selection_metadata,
        }
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
    criteria_pass_rate = passed_count / total_count if total_count else 0.0
    reward = 1.0 if total_count and passed_count == total_count else 0.0
    scoring = {
        "final_score": reward,
        "scoring_method_result_values": {
            "passed_count": passed_count,
            "failed_count": failed_count,
            "total_count": total_count,
            "criteria_pass_rate": criteria_pass_rate,
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
