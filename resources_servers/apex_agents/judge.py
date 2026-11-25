# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused APEX artifact judging through Gym's configured model server."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import ClientTimeout
from pydantic import BaseModel, ValidationError

from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.verification import JudgeError, reraise_judge_errors
from resources_servers.apex_agents.artifacts import (
    ArtifactChange,
    artifact_change_content,
    artifact_change_text,
    visual_file_blocks,
)
from resources_servers.apex_agents.file_extraction import conversion_timeouts, renders_as_images
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
# The grading prompt is budgeted at two characters per token: dense workbooks and code tokenize far below the
# four characters per token of prose, and a prompt the judge rejects cannot be graded at all.
_CONSERVATIVE_CHARS_PER_TOKEN = 2
# Images are budgeted too: a page image is charged a fixed, generous token cost, images may take at most a quarter
# of the judge's window, and their encoded size is capped below common request-payload limits.
_IMAGE_TOKEN_ESTIMATE = 3_000
_IMAGE_BUDGET_RATIO = 0.25
_MAX_JUDGE_IMAGES = 60
_MAX_JUDGE_IMAGE_CHARS = 16 * 1024 * 1024
# A refusal is the policy's doing only when the request the verifier built was itself large: its estimated tokens
# pass this share of the judge's window or, for an HTTP 413, its encoded body passes this many bytes. A small
# request refused the same way points at the judge's setup (a wrong window, an output cap, a proxy limit).
_LARGE_REQUEST_WINDOW_RATIO = 0.5
_LARGE_REQUEST_BODY_BYTES = 1024 * 1024
# How a judge provider says a request is larger than it accepts, in the body of an HTTP 400.
_OVERSIZED_REQUEST_MARKERS = (
    "context_length_exceeded",
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
    "token count",
    "maximum number of tokens",
    "prompt is too long",
    "input is too long",
    "request too large",
    "payload size",
    "entity too large",
    "string too long",
)
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


class JudgeRequestTooLarge(Exception):
    """The judge refused, as larger than it accepts, a request that was large.

    The request's size comes from what the policy produced, and the same request is refused again on every retry,
    so this is the policy's result (a failed criterion), not a judge outage.
    """


def _estimated_request_tokens(body: dict[str, Any]) -> int:
    """The request's prompt tokens as the verifier budgets them: its text conservatively, plus a cost per image."""
    characters = images = 0
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            characters += len(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "image_url":
                images += 1
            elif isinstance(block, dict):
                characters += len(str(block.get("text") or ""))
    return characters // _CONSERVATIVE_CHARS_PER_TOKEN + images * _IMAGE_TOKEN_ESTIMATE


def _oversized_request_rejection(error: BaseException) -> tuple[int, str] | None:
    """The status and reason of a refusal for size, read from the HTTP error behind a failed judge call.

    Only an HTTP 413, or an HTTP 400 whose body names a context-length or payload limit, counts. Connection
    errors, timeouts and every other status (5xx included) are judge outages and return None.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status", None)
        if isinstance(status, int) and not isinstance(status, bool) and status in {400, 413}:
            content = getattr(current, "response_content", None)
            body = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content or "")
            detail = f"{body} {getattr(current, 'message', '')}".strip()
            if status == 413 or any(marker in detail.lower() for marker in _OVERSIZED_REQUEST_MARKERS):
                return status, f"HTTP {status}: {detail[:300]}"
        current = current.__cause__
    return None


def _request_is_large(body: dict[str, Any], *, status: int, context_window_size: int) -> bool:
    if _estimated_request_tokens(body) > context_window_size * _LARGE_REQUEST_WINDOW_RATIO:
        return True
    return status == 413 and len(json.dumps(body, default=str).encode("utf-8")) > _LARGE_REQUEST_BODY_BYTES


async def _judge_completion(
    server_client: Any, model_server_name: str, body: dict[str, Any], *, context_window_size: int
) -> Any:
    """One judge call: a failed call is a judge failure, except a large request refused for its size.

    A small request refused with the same words (an output cap larger than the judge's window, a proxy's payload
    limit) says nothing about the policy's output, so it stays a judge failure.
    """
    try:
        return await reraise_judge_errors(_post_chat_completion(server_client, model_server_name, body))
    except Exception as exc:
        rejection = _oversized_request_rejection(exc)
        if rejection is None or not _request_is_large(
            body, status=rejection[0], context_window_size=context_window_size
        ):
            raise
        raise JudgeRequestTooLarge(rejection[1]) from exc


async def _post_chat_completion(server_client: Any, model_server_name: str, body: dict[str, Any]) -> Any:
    """One judge request. Callers await it through ``reraise_judge_errors``, so a failed call is a judge failure."""
    response = await server_client.post(
        server_name=model_server_name,
        url_path="/v1/chat/completions",
        json=body,
        timeout=ClientTimeout(total=3600),
    )
    await raise_for_status(response)
    return await get_response_json(response)


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
    contents_by_change: dict[int, str] | None = None,
) -> tuple[list[ArtifactChange], dict[str, Any]]:
    # ``contents_by_change`` holds each change's selection text by ``id(change)``, rendered once for all criteria.
    contents = [
        contents_by_change[id(change)]
        if contents_by_change is not None and id(change) in contents_by_change
        else artifact_change_content(change)
        for change in changes
    ]
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
    for _ in range(_MAX_JSON_RETRIES):
        body = dict(judge_create_params_overrides)
        body.update(
            {
                "model": judge_model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
        )
        payload = await _judge_completion(
            server_client, model_server_name, body, context_window_size=context_window_size
        )
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
    if parsed is None:
        # Grading with no artifacts would score the answer on evidence the selector never chose, so a selection
        # that cannot be read is an unusable judgement, not an empty one.
        raise JudgeError(f"artifact selector returned invalid structured output after {_MAX_JSON_RETRIES} attempts")

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


_ARTIFACT_SEPARATOR = "\n\n"
_ARTIFACT_FIT_ATTEMPTS = 8


def _rendered_artifact(index: int, change: ArtifactChange, *, content_chars: int) -> str:
    content = artifact_change_text(change, max_chars=content_chars)
    change_name = {"added": "created", "modified": "modified", "deleted": "deleted"}[change.change_type]
    indented_content = "\n".join(f"  {line}" for line in content.splitlines())
    return "\n".join(
        [
            f'<ARTIFACT id="{index}" type="{change.artifact_type}" change="{change_name}">',
            *_artifact_metadata_xml(change),
            indented_content,
            "</ARTIFACT>",
        ]
    )


def _artifact_xml(changes: list[ArtifactChange], *, character_budget: int) -> str:
    if not changes:
        return ""
    per_artifact = max(1_000, character_budget // len(changes))
    remaining = character_budget
    rendered: list[str] = []
    for index, change in enumerate(changes, 1):
        available = remaining - (len(_ARTIFACT_SEPARATOR) if rendered else 0)
        # An artifact's share covers all of it: content, wrapper, metadata and indentation. Rendering the content at
        # the full share therefore overshoots, so shrink the content until the whole artifact fits. A selected
        # artifact reaches the judge truncated, never dropped, unless even its wrapper does not fit what is left.
        share = min(per_artifact, available)
        content_chars = share
        artifact: str | None = None
        for _ in range(_ARTIFACT_FIT_ATTEMPTS):
            if content_chars <= 0:
                break
            candidate = _rendered_artifact(index, change, content_chars=content_chars)
            if len(candidate) <= share:
                artifact = candidate
                break
            content_chars -= len(candidate) - share
        if artifact is None:
            break
        rendered.append(artifact)
        remaining = available - len(artifact)
    return _ARTIFACT_SEPARATOR.join(rendered)


def _build_prompt(
    *,
    instruction: str,
    response: str,
    criteria: str,
    file_type: str,
    changes: list[ArtifactChange],
    context_window_size: int,
    reserved_tokens: int = 0,
) -> str:
    """The grading prompt. ``reserved_tokens`` is what the images sent with it are charged against the window."""
    if file_type == _FINAL_ANSWER_ONLY:
        scope = EVAL_SCOPE_TEXT_ONLY
        artifact_structure = ""
        agent_output = f"<TEXT_RESPONSE>\n{response}\n</TEXT_RESPONSE>"
    else:
        scope = EVAL_SCOPE_BOTH if file_type == _ALL_OUTPUT else EVAL_SCOPE_FILES_ONLY
        artifact_structure = ARTIFACT_STRUCTURE
        fixed_prompt_chars = len(GRADING_SYSTEM_PROMPT) + len(instruction) + len(response) + len(criteria) + 2_000
        prompt_tokens = int(context_window_size * _TOTAL_CONTENT_BUDGET_RATIO) - reserved_tokens
        artifact_budget = max(4_000, prompt_tokens * _CONSERVATIVE_CHARS_PER_TOKEN - fixed_prompt_chars)
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
    path: Path,
    position: int,
    changed_indices: set[int] | None,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
) -> list[dict[str, Any]]:
    """One changed file's images with their labels; nothing for a file that yields no image."""
    if not renders_as_images(path):
        return []
    blocks = visual_file_blocks(
        final_root,
        path,
        position,
        document_converter_image=document_converter_image,
        allow_host_document_tools=allow_host_document_tools,
        changed_indices=changed_indices,
    )
    if not any(block.get("type") == "image_url" for block in blocks):
        return []
    return [block for block in blocks if block.get("type") in {"text", "image_url"}]


class _VisualEvidence:
    """Renders each changed file once per verify, however many criteria are graded on it."""

    def __init__(
        self,
        final_root: Path,
        *,
        document_converter_image: str | None,
        allow_host_document_tools: bool,
        run_blocking: Callable[..., Awaitable[Any]],
    ) -> None:
        self._final_root = final_root
        self._document_converter_image = document_converter_image
        self._allow_host_document_tools = allow_host_document_tools
        self._run_blocking = run_blocking
        self._renders: dict[tuple[int, Path, frozenset[int]], asyncio.Future[list[dict[str, Any]]]] = {}

    def _render(self, position: int, path: Path, indices: frozenset[int]) -> asyncio.Future[list[dict[str, Any]]]:
        key = (position, path, indices)
        render = self._renders.get(key)
        if render is None:
            render = asyncio.ensure_future(
                self._run_blocking(
                    functools.partial(
                        _visual_blocks,
                        self._final_root,
                        path,
                        position,
                        set(indices) or None,
                        document_converter_image=self._document_converter_image,
                        allow_host_document_tools=self._allow_host_document_tools,
                    )
                )
            )
            # A render no criterion is left waiting for must not log an unretrieved exception.
            render.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._renders[key] = render
        return render

    async def blocks(self, changes: list[ArtifactChange]) -> list[dict[str, Any]]:
        files = list(dict.fromkeys(change.after_path for change in changes if change.after_path is not None))
        indices_by_path: dict[Path, set[int]] = {}
        for change in changes:
            if (
                change.artifact_type in {"page", "slide", "sheet"}
                and change.index is not None
                and change.after_path is not None
            ):
                indices_by_path.setdefault(change.after_path, set()).add(change.index)
        renders = [
            self._render(position, path, frozenset(indices_by_path.get(path, ())))
            for position, path in enumerate(files)
        ]
        blocks: list[dict[str, Any]] = []
        for render in renders:
            # Shielded: a criterion that is cancelled must not cancel a render other criteria share.
            blocks.extend(await asyncio.shield(render))
        return blocks


def _budget_images(blocks: list[dict[str, Any]], *, context_window_size: int) -> tuple[list[dict[str, Any]], int]:
    """Keep the leading images that fit the judge request; returns the kept blocks and their image count."""
    max_images = min(_MAX_JUDGE_IMAGES, int(context_window_size * _IMAGE_BUDGET_RATIO) // _IMAGE_TOKEN_ESTIMATE)
    kept: list[dict[str, Any]] = []
    image_count = image_chars = omitted = 0
    for block in blocks:
        if block.get("type") != "image_url":
            kept.append(block)
            continue
        size = len(str((block.get("image_url") or {}).get("url", "")))
        if image_count < max_images and image_chars + size <= _MAX_JUDGE_IMAGE_CHARS:
            kept.append(block)
            image_count += 1
            image_chars += size
            continue
        omitted += 1
        # Drop the page label that introduced the omitted image; a file's own label starts on a new line.
        if kept and kept[-1].get("type") == "text" and not str(kept[-1].get("text", "")).startswith("\n"):
            kept.pop()
    if omitted:
        kept.append({"type": "text", "text": f"\n[{omitted} further image(s) omitted: judge request size limit]"})
    if not image_count:
        return [], 0
    return kept, image_count


async def _judge_criterion(
    *,
    server_client: Any,
    model_server_name: str,
    judge_model: str,
    judge_create_params_overrides: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    visual_blocks: list[dict[str, Any]],
    context_window_size: int,
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
        payload = await _judge_completion(
            server_client, model_server_name, body, context_window_size=context_window_size
        )
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
        raise JudgeError(f"judge returned invalid structured output after {_MAX_JSON_RETRIES} attempts")
    return parsed


SCORING_METHODS = ("all_pass", "weighted_average")


def weighted_rubric_score(rubric: list[dict[str, Any]], scores: list[dict[str, Any]]) -> float:
    """Weighted mean of criterion scores, in rubric order.

    A criterion's ``weight`` defaults to 1. Zero-weight criteria are skipped, as in the weighted-average scoring
    methods of Mercor's Archipelago grader.
    """
    earned = total = 0.0
    for index, (criterion, score) in enumerate(zip(rubric, scores, strict=True)):
        raw = criterion.get("weight")
        weight = 1.0 if raw is None else float(raw)
        if weight < 0:
            raise ValueError(f"rubric criterion {index} has a negative weight: {weight}")
        if weight:
            earned += weight * float(score["score"])
            total += weight
    return earned / total if total else 0.0


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
    allow_host_document_tools: bool = False,
    metadata: dict[str, Any] | None = None,
    scoring_method: str = "all_pass",
    run_blocking: Callable[..., Awaitable[Any]] | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Grade every APEX rubric criterion.

    With ``all_pass`` (the default) a rollout scores 1 only when every criterion passes. With ``weighted_average``
    it scores the weighted mean of its criterion scores (see ``weighted_rubric_score``).

    ``run_blocking`` awaits a synchronous callable off the event loop (``asyncio.to_thread`` by default): prompt
    assembly and document rendering go through it, and only the judge requests run on the loop.
    """
    del world_id, metadata
    if scoring_method not in SCORING_METHODS:
        raise ValueError(f"unknown APEX scoring method: {scoring_method!r}")
    overrides = dict(judge_create_params_overrides or {})
    run_blocking = run_blocking or asyncio.to_thread
    timed_out_conversions = conversion_timeouts()
    visual_evidence = _VisualEvidence(
        final_root,
        document_converter_image=document_converter_image,
        allow_host_document_tools=allow_host_document_tools,
        run_blocking=run_blocking,
    )
    # Each change's selection text is rendered once, not once per criterion.
    selection_contents: dict[int, str] = {}
    if any(
        _matching_changes(artifact_changes, expected_file_type(expected_output, criterion)) for criterion in rubric
    ):
        selection_contents = await run_blocking(
            lambda: {id(change): artifact_change_content(change) for change in artifact_changes}
        )

    def failed(verifier_id: str, rationale: str, selection: dict[str, Any], **values: Any) -> tuple[str, dict]:
        return verifier_id, {
            "score": 0.0,
            "status": "completed",
            "message": rationale,
            "values": {
                "judge_grade": "fail",
                "grade_rationale": rationale,
                "evaluated_artifacts": "",
                "artifact_selection": selection,
                **values,
            },
        }

    async def grade_one(index: int, criterion: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        verifier_id = str(criterion.get("verifier_id") or "").strip()
        criteria = str(criterion.get("criteria") or "").strip()
        if not verifier_id or not criteria:
            raise ValueError(f"rubric criterion {index} must include verifier_id and criteria")
        file_type = expected_file_type(expected_output, criterion)
        filtered = _matching_changes(artifact_changes, file_type)
        if _should_auto_fail_missing_file_type(file_type, filtered):
            return failed(
                verifier_id,
                f"No files matching the expected type ({file_type}) were found.",
                {"status": "skipped", "reason": "no_matching_artifacts", "selected_count": 0, "total_count": 0},
            )
        selection_metadata: dict[str, Any] = {"status": "not_run"}
        try:
            selected, selection_metadata = await _select_artifacts_if_needed(
                server_client=server_client,
                model_server_name=model_server_name,
                judge_model=judge_model,
                judge_create_params_overrides=overrides,
                instruction=instruction,
                criteria=criteria,
                changes=filtered,
                context_window_size=judge_context_window_size,
                contents_by_change=selection_contents,
            )
            visuals, image_count = _budget_images(
                await visual_evidence.blocks(selected), context_window_size=judge_context_window_size
            )
            user_prompt = await run_blocking(
                functools.partial(
                    _build_prompt,
                    instruction=instruction,
                    response=response,
                    criteria=criteria,
                    file_type=file_type,
                    changes=selected,
                    context_window_size=judge_context_window_size,
                    reserved_tokens=image_count * _IMAGE_TOKEN_ESTIMATE,
                )
            )
            parsed = await _judge_criterion(
                server_client=server_client,
                model_server_name=model_server_name,
                judge_model=judge_model,
                judge_create_params_overrides=overrides,
                system_prompt=GRADING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                visual_blocks=visuals,
                context_window_size=judge_context_window_size,
            )
        except JudgeRequestTooLarge as exc:
            # The judge will refuse this request every time, so the criterion is settled: the output it would
            # have to read is too large to grade. Other criteria are still graded.
            LOG.warning("judge refused criterion %s of task %s as too large: %s", verifier_id, task_id, exc)
            return failed(
                verifier_id,
                f"The judge rejected the grading request as larger than it accepts ({exc}); "
                "the submitted output is too large to grade.",
                selection_metadata,
                judge_request_rejected="request_too_large",
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

    tasks = [asyncio.ensure_future(grade_one(index, criterion)) for index, criterion in enumerate(rubric)]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        # gather raises the first failure and leaves the other criteria running: stop them before the caller
        # releases its concurrency slot and deletes the scratch directory they read. The first failure keeps its
        # type (not a TaskGroup's ExceptionGroup), so a judge outage still reaches the judge failsafe.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    # Counted over the criteria, not over the ID-keyed mapping: two criteria sharing an ID are both graded and
    # both count, and the second keeps its own entry under a numbered key.
    rubric_scores: dict[str, Any] = {}
    for verifier_id, score in results:
        key, repeat = verifier_id, 2
        while key in rubric_scores:
            key, repeat = f"{verifier_id}#{repeat}", repeat + 1
        rubric_scores[key] = score
    passed_count = sum(score["score"] >= 0.99 for _, score in results)
    total_count = len(results)
    failed_count = total_count - passed_count
    criteria_pass_rate = passed_count / total_count if total_count else 0.0
    all_pass = 1.0 if total_count and passed_count == total_count else 0.0
    weighted_score = weighted_rubric_score(rubric, [score for _, score in results])
    reward = weighted_score if scoring_method == "weighted_average" else all_pass
    scoring = {
        "final_score": reward,
        "scoring_method": scoring_method,
        "scoring_method_result_values": {
            "passed_count": passed_count,
            "failed_count": failed_count,
            "total_count": total_count,
            "criteria_pass_rate": criteria_pass_rate,
            "all_pass": all_pass,
            "weighted_score": weighted_score,
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
    if timed_out_conversions:
        # The files named here were graded on their extracted text: their conversion did not finish.
        judge_usage["document_conversion_timeouts"] = list(timed_out_conversions)
    return reward, rubric_scores, judge_usage
