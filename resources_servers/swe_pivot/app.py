# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SWE Pivot Verifier for PivotRL training on coding agent trajectories.

Layered verification (cheap first, expensive last):
  Level 0: Tool name match (is the rollout using the same tool type?)
  Level 1: Target file match (is it operating on the correct file?)
  Level 2: Argument similarity (is the edit content similar?)
  Level 4: Diff-size shaping (bonus for minimal edits)

No LLM judge — purely computational verification.
"""

import json
import logging
import re
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool classification — handles diversified tool names from OpenHands/OpenCode/Codex
# ---------------------------------------------------------------------------

_TOOL_CATEGORY_MAP: dict[str, str] = {}


def _register_category(category: str, names: list[str]):
    for name in names:
        _TOOL_CATEGORY_MAP[name.lower()] = category
        camel = "".join(w.capitalize() for w in name.split("_"))
        _TOOL_CATEGORY_MAP[camel.lower()] = category


_register_category(
    "edit",
    [
        "str_replace_editor",
        "text_editor",
        "file_editor",
        "code_editor",
        "edit_file",
        "modify_file",
        "update_file",
        "file_edit",
        "edit",
        "modify",
        "apply_patch",
        "patch_file",
        "apply_diff",
        "apply_changes",
        "write",
        "write_file",
        "save_file",
        "create_file",
    ],
)
_register_category(
    "bash",
    [
        "execute_bash",
        "run_bash",
        "bash_exec",
        "shell_run",
        "shell",
        "bash",
        "terminal",
        "shell_command",
        "shell_cmd",
        "run_shell_cmd",
        "exec_command",
        "exec_cmd",
    ],
)
_register_category(
    "search",
    [
        "grep",
        "search_text",
        "find_pattern",
        "text_search",
        "glob",
        "find_files",
        "file_glob",
        "match_files",
        "grep_files",
        "search_files",
        "find_in_files",
        "code_search",
        "list_dir",
        "ls",
        "show_dir",
        "ls_dir",
        "list_directory",
    ],
)
_register_category(
    "read",
    [
        "read",
        "read_file",
        "view_file",
        "cat_file",
        "view",
        "file_view",
        "inspect",
        "show_file",
        "read_file_content",
    ],
)
_register_category("finish", ["finish", "submit"])
_register_category("think", ["think"])
_register_category(
    "planning",
    [
        "task_tracker",
        "todo_tracker",
        "plan_tracker",
        "todo_read",
        "todo_write",
        "update_plan",
        "list_tasks",
        "update_tasks",
        "read_todos",
        "write_todos",
        "edit_plan",
        "modify_plan",
    ],
)


def get_tool_category(name: str) -> str:
    return _TOOL_CATEGORY_MAP.get(name.lower(), "unknown")


# ---------------------------------------------------------------------------
# Argument extraction helpers
# ---------------------------------------------------------------------------


def _parse_arguments(raw: Any) -> dict:
    """Parse tool call arguments from string or dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def extract_tool_info(tool_call: dict) -> tuple[str, str, dict]:
    """Extract (tool_name, tool_category, parsed_args) from a function_call."""
    name = tool_call.get("name", "")
    args = _parse_arguments(tool_call.get("arguments", ""))
    category = get_tool_category(name)
    return name, category, args


def extract_file_path(args: dict) -> str:
    """Extract the target file path from tool arguments."""
    for key in ("path", "file", "file_path", "filename", "file_name"):
        if key in args:
            return str(args[key]).strip()
    return ""


def _path_suffix(path: str, n_components: int = 3) -> str:
    """Return the last *n_components* of a path for comparison.

    Using the last 3 components (parent_dir/parent_dir/filename) disambiguates files
    like ``src/components/index.js`` vs ``test/fixtures/index.js`` while
    tolerating different repo-root prefixes across agent frameworks.
    """
    stripped = path.strip("/")
    components = stripped.split("/")
    return "/".join(components[-n_components:])


def extract_command(args: dict) -> str:
    """Extract shell command from bash tool arguments."""
    for key in ("command", "cmd"):
        if key in args:
            return str(args[key]).strip()
    return ""


def extract_edit_content(args: dict) -> str:
    """Extract the edit content for similarity comparison."""
    parts = []
    # str_replace_editor style
    if "new_str" in args:
        parts.append(str(args.get("old_str", "")))
        parts.append(str(args["new_str"]))
    # apply_patch style
    elif "patch" in args:
        parts.append(str(args["patch"]))
    elif "diff" in args:
        parts.append(str(args["diff"]))
    # write/create style
    elif "content" in args or "file_text" in args:
        parts.append(str(args.get("content", args.get("file_text", ""))))
    return "\n".join(parts)


def extract_command_verb(cmd: str) -> str:
    """Extract the first word/verb of a shell command."""
    cmd = cmd.strip()
    if not cmd:
        return ""
    first = cmd.split()[0]
    return first.rsplit("/", 1)[-1]  # strip path prefix


# Matches file/directory paths in shell commands
# Handles: /absolute/paths, ./relative/paths, dir/file.ext, quoted paths
_PATH_IN_CMD_RE = re.compile(
    r"""(?:^|[\s='"(])"""  # preceded by whitespace, =, quote, or start
    r"""("""
    r"""(?:/[\w._-]+)+"""  # /absolute/path
    r"""|\.\.?/[\w._/-]+"""  # ./relative or ../relative
    r"""|[\w._-]+/[\w._/-]+"""  # dir/file (must have at least one /)
    r""")"""
)


def _extract_paths_from_command(cmd: str) -> list[str]:
    """Extract file/directory paths from a shell command string."""
    if not cmd:
        return []
    paths = _PATH_IN_CMD_RE.findall(cmd)
    # Filter out common false positives
    return [p for p in paths if not p.startswith("--") and len(p) > 2]


def extract_editor_command(args: dict) -> str:
    """Extract the editor sub-command (view, str_replace, create, etc.)."""
    return str(args.get("command", ""))


def _parse_view_range(raw) -> tuple[int, int] | None:
    """Parse view_range from args. Returns (start, end) or None."""
    if raw is None:
        return None
    if isinstance(raw, list) and len(raw) == 2:
        try:
            return (int(raw[0]), int(raw[1]))
        except (ValueError, TypeError):
            return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == 2:
                return (int(parsed[0]), int(parsed[1]))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Verification levels
# ---------------------------------------------------------------------------


def verify_tool_name_match(
    rollout_name: str,
    rollout_cat: str,
    expected_name: str,
    expected_cat: str,
) -> bool:
    """Level 0: Does the rollout use the same tool, by exact name or shared category?"""
    if rollout_name.lower() == expected_name.lower():
        return True
    # Category-equivalent tools match (e.g. str_replace_editor / code_editor are both
    # "edit"). "unknown" is not a real category, so it never matches by category alone.
    return rollout_cat == expected_cat and rollout_cat != "unknown"


def verify_target_match(
    rollout_cat: str,
    rollout_args: dict,
    expected_cat: str,
    expected_args: dict,
) -> bool:
    """Level 1: Is the tool operating on the same target?"""
    if rollout_cat == "edit" and expected_cat == "edit":
        # Check editor sub-command first (view vs str_replace vs create)
        r_cmd = extract_editor_command(rollout_args)
        e_cmd = extract_editor_command(expected_args)
        if r_cmd and e_cmd and r_cmd != e_cmd:
            return False
        # Check target file (last 2 path components to avoid basename collisions)
        r_file = extract_file_path(rollout_args)
        e_file = extract_file_path(expected_args)
        if r_file and e_file:
            if _path_suffix(r_file, 2) != _path_suffix(e_file, 2):
                return False
        # For view commands: check view_range covers >= 80% of expected range
        if r_cmd == "view" and e_cmd == "view":
            r_range = _parse_view_range(rollout_args.get("view_range"))
            e_range = _parse_view_range(expected_args.get("view_range"))
            # A rollout with no view_range views the whole file, which covers any
            # expected range, so it passes. Only compare when both specify a range.
            if r_range and e_range:
                overlap_start = max(r_range[0], e_range[0])
                overlap_end = min(r_range[1], e_range[1])
                overlap_size = max(0, overlap_end - overlap_start)
                expected_size = max(1, e_range[1] - e_range[0])
                if overlap_size / expected_size < 0.8:
                    return False
        return True

    if rollout_cat == "bash" and expected_cat == "bash":
        r_cmd = extract_command(rollout_args)
        e_cmd = extract_command(expected_args)
        # Check command verb matches (grep vs pytest vs cd)
        r_verb = extract_command_verb(r_cmd)
        e_verb = extract_command_verb(e_cmd)
        if r_verb and e_verb and r_verb != e_verb:
            return False
        # Check file/directory paths in the command
        r_paths = _extract_paths_from_command(r_cmd)
        e_paths = _extract_paths_from_command(e_cmd)
        if r_paths and e_paths:
            # At least one path suffix (last 3 components) must overlap
            r_suffixes = {_path_suffix(p.rstrip("/")) for p in r_paths} - {""}
            e_suffixes = {_path_suffix(p.rstrip("/")) for p in e_paths} - {""}
            if r_suffixes and e_suffixes and not (r_suffixes & e_suffixes):
                return False
        # Minimum command similarity floor to prevent verb-only matching
        if r_cmd and e_cmd:
            if _capped_similarity(r_cmd, e_cmd) < _BASH_TARGET_SIMILARITY_FLOOR:
                return False
        return True

    if rollout_cat in ("search", "read") and expected_cat in ("search", "read"):
        # Check target path/pattern (last 2 components)
        r_file = extract_file_path(rollout_args)
        e_file = extract_file_path(expected_args)
        if r_file and e_file:
            return _path_suffix(r_file) == _path_suffix(e_file)
        return True

    return True  # other categories, pass


# Minimum command similarity for bash target match (Level 1).
# Prevents verb-only matching — e.g. any `cd` matching any other `cd`.
# 0.3 is intentionally low: we just want to reject wildly different commands,
# Level 2 handles the stricter similarity check.
_BASH_TARGET_SIMILARITY_FLOOR = 0.3

# Max characters to compare with SequenceMatcher.
# SequenceMatcher is O(n*m) — at 2K chars each that's ~4M ops, fast enough.
# Beyond this we truncate to keep verification fast during RL training.
_MAX_SIMILARITY_CHARS = 2000


def _capped_similarity(a: str, b: str) -> float:
    """SequenceMatcher with length cap to avoid O(n*m) blowup on large edits."""
    a = a[:_MAX_SIMILARITY_CHARS]
    b = b[:_MAX_SIMILARITY_CHARS]
    return SequenceMatcher(None, a, b).ratio()


def compute_argument_similarity(
    rollout_cat: str,
    rollout_args: dict,
    expected_cat: str,
    expected_args: dict,
) -> float:
    """Level 2: How similar are the tool arguments?"""
    if rollout_cat == "edit" and expected_cat == "edit":
        # Compare old_str and new_str separately
        # old_str match = "found the right code location?"
        # new_str match = "wrote the right fix?"
        r_old = str(rollout_args.get("old_str", ""))
        e_old = str(expected_args.get("old_str", ""))
        r_new = str(rollout_args.get("new_str", ""))
        e_new = str(expected_args.get("new_str", ""))

        has_str_replace = (r_old or r_new) and (e_old or e_new)

        if has_str_replace:
            old_sim = _capped_similarity(r_old, e_old) if (r_old and e_old) else (1.0 if r_old == e_old else 0.0)
            new_sim = _capped_similarity(r_new, e_new) if (r_new and e_new) else (1.0 if r_new == e_new else 0.0)
            # Both must independently pass — return the minimum
            return min(old_sim, new_sim)

        # Fallback for patch/diff/content style edits
        r_content = extract_edit_content(rollout_args)
        e_content = extract_edit_content(expected_args)
        if r_content and e_content:
            return _capped_similarity(r_content, e_content)
        if bool(r_content) != bool(e_content):
            return 0.0
        return 1.0  # both empty (view, undo_edit, etc.)

    if rollout_cat == "bash" and expected_cat == "bash":
        r_cmd = extract_command(rollout_args)
        e_cmd = extract_command(expected_args)
        if r_cmd and e_cmd:
            return _capped_similarity(r_cmd, e_cmd)
        return 1.0

    return 1.0  # non-edit/bash, similarity not applicable


def compute_diff_size_penalty(rollout_cat: str, rollout_args: dict, alpha: float) -> float:
    """Level 4: Penalty for large edits (rewards minimality).

    Returns a value in [0, alpha] to subtract from the reward.
    """
    if rollout_cat != "edit":
        return 0.0
    content = extract_edit_content(rollout_args)
    if not content:
        return 0.0
    # Normalize by a reasonable max size
    max_size = 500
    size_ratio = min(len(content) / max_size, 1.0)
    return alpha * size_ratio


# ---------------------------------------------------------------------------
# Failure codes
# ---------------------------------------------------------------------------


class FailureCode(str, Enum):
    NONE = "none"
    EXPECTED_ACTION_INVALID = "expected_action_invalid"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    TOOL_NAME_MISMATCH = "tool_name_mismatch"
    TARGET_MISMATCH = "target_mismatch"
    SIMILARITY_BELOW_THRESHOLD = "similarity_below_threshold"
    UNKNOWN_ERROR = "unknown_error"


# ---------------------------------------------------------------------------
# Server config, request/response models
# ---------------------------------------------------------------------------


class SwePivotResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "swe_pivot"
    # Reward mode: "binary" (0 or 1) or "continuous" (0.0 to 1.0 with partial credit)
    reward_mode: str = "binary"
    # Verification levels (layered)
    enable_target_match: bool = True  # Level 1
    enable_argument_similarity: bool = True  # Level 2
    similarity_threshold: float = 0.4  # Partial credit above this (continuous only)
    similarity_full_credit: float = 0.8  # Full credit above this
    enable_diff_size_shaping: bool = True  # Level 4 (continuous only)
    diff_size_alpha: float = 0.1  # Max penalty for large edits


class SwePivotRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    expected_action: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class SwePivotVerifyRequest(SwePivotRunRequest, BaseVerifyRequest):
    pass


class SwePivotVerifyResponse(BaseVerifyResponse):
    uuid: Optional[str | int] = None
    expected_answer: str
    model_output: str
    tool_name_match: bool
    target_match: Optional[bool] = None
    similarity_score: Optional[float] = None
    diff_size_penalty: Optional[float] = None
    failure_reason: Optional[FailureCode] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Extraction from Responses API format
# ---------------------------------------------------------------------------


def _extract_rollout_function_call(body: BaseVerifyRequest) -> Optional[dict]:
    """Extract the function_call from the model's response output."""
    for o in body.response.output:
        if getattr(o, "type", None) == "function_call":
            return {
                "name": getattr(o, "name", ""),
                "arguments": getattr(o, "arguments", ""),
            }
    return None


def _extract_expected_function_call(body: SwePivotRunRequest) -> Optional[dict]:
    """Extract expected function_call from request.

    Checks three locations (in priority order):
    1. body.expected_action — dict with {"type": "function_call", "name": ..., "arguments": ...}
    2. body.expected_answer — JSON string of the same
    3. body.metadata.expected_action — fallback
    """
    # Try top-level expected_action (our pivot data format)
    if body.expected_action:
        ea = body.expected_action
        if isinstance(ea, dict) and ea.get("type") == "function_call":
            return {"name": ea["name"], "arguments": ea.get("arguments", "")}

    # Try expected_answer as JSON string (terminal_pivot style)
    if body.expected_answer:
        try:
            ea = json.loads(body.expected_answer)
            if isinstance(ea, dict) and ea.get("type") == "function_call":
                return {"name": ea["name"], "arguments": ea.get("arguments", "")}
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # Try metadata.expected_action
    if body.metadata:
        ea = body.metadata.get("expected_action")
        if isinstance(ea, dict) and ea.get("type") == "function_call":
            return {"name": ea["name"], "arguments": ea.get("arguments", "")}

    return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class SwePivotResourcesServer(SimpleResourcesServer):
    config: SwePivotResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: SwePivotVerifyRequest) -> SwePivotVerifyResponse:
        reward = 0.0
        failure_reason = FailureCode.NONE
        tool_name_match = False
        target_match = None
        similarity_score = None
        diff_size_penalty = None

        expected_str = body.expected_answer or ""
        model_output_str = ""

        try:
            # Extract expected action
            expected_fc = _extract_expected_function_call(body)
            if not expected_fc:
                failure_reason = FailureCode.EXPECTED_ACTION_INVALID
                return self._build_response(
                    body,
                    reward,
                    expected_str,
                    model_output_str,
                    tool_name_match,
                    target_match,
                    similarity_score,
                    diff_size_penalty,
                    failure_reason,
                )

            e_name, e_cat, e_args = extract_tool_info(expected_fc)

            # Extract rollout action
            rollout_fc = _extract_rollout_function_call(body)
            if not rollout_fc:
                failure_reason = FailureCode.MODEL_OUTPUT_INVALID
                model_output_str = "(no function_call in response)"
                return self._build_response(
                    body,
                    reward,
                    expected_str,
                    model_output_str,
                    tool_name_match,
                    target_match,
                    similarity_score,
                    diff_size_penalty,
                    failure_reason,
                )

            r_name, r_cat, r_args = extract_tool_info(rollout_fc)
            model_output_str = f"{r_name}({json.dumps(r_args)[:500]})"

            # --- Level 0: Tool name match ---
            tool_name_match = verify_tool_name_match(r_name, r_cat, e_name, e_cat)
            if not tool_name_match:
                failure_reason = FailureCode.TOOL_NAME_MISMATCH
                return self._build_response(
                    body,
                    0.0,
                    expected_str,
                    model_output_str,
                    tool_name_match,
                    target_match,
                    similarity_score,
                    diff_size_penalty,
                    failure_reason,
                )

            reward = 1.0  # passed level 0

            # --- Level 1: Target match ---
            if self.config.enable_target_match:
                target_match = verify_target_match(r_cat, r_args, e_cat, e_args)
                if not target_match:
                    failure_reason = FailureCode.TARGET_MISMATCH
                    return self._build_response(
                        body,
                        0.0,
                        expected_str,
                        model_output_str,
                        tool_name_match,
                        target_match,
                        similarity_score,
                        diff_size_penalty,
                        failure_reason,
                    )

            # --- Level 2: Argument similarity ---
            if self.config.enable_argument_similarity:
                similarity_score = compute_argument_similarity(
                    r_cat,
                    r_args,
                    e_cat,
                    e_args,
                )
                full_credit_threshold = self.config.similarity_full_credit
                if self.config.reward_mode == "binary":
                    # Binary: 1.0 if above full_credit threshold, else 0.0
                    if similarity_score >= full_credit_threshold:
                        reward = 1.0
                    else:
                        failure_reason = FailureCode.SIMILARITY_BELOW_THRESHOLD
                        reward = 0.0
                        return self._build_response(
                            body,
                            reward,
                            expected_str,
                            model_output_str,
                            tool_name_match,
                            target_match,
                            similarity_score,
                            diff_size_penalty,
                            failure_reason,
                        )
                else:
                    # Continuous: partial credit between threshold and full_credit
                    if similarity_score >= full_credit_threshold:
                        reward = 1.0
                    elif similarity_score >= self.config.similarity_threshold:
                        t = self.config.similarity_threshold
                        f = full_credit_threshold
                        reward = 0.5 + 0.5 * (similarity_score - t) / (f - t)
                    else:
                        failure_reason = FailureCode.SIMILARITY_BELOW_THRESHOLD
                        reward = 0.0
                        return self._build_response(
                            body,
                            reward,
                            expected_str,
                            model_output_str,
                            tool_name_match,
                            target_match,
                            similarity_score,
                            diff_size_penalty,
                            failure_reason,
                        )

            # --- Level 4: Diff-size shaping (continuous mode only) ---
            if self.config.enable_diff_size_shaping and self.config.reward_mode == "continuous":
                diff_size_penalty = compute_diff_size_penalty(
                    r_cat,
                    r_args,
                    self.config.diff_size_alpha,
                )
                reward = reward * (1.0 - diff_size_penalty)

        except Exception as e:
            logger.error(f"swe_pivot verify error: {type(e).__name__} {e}")
            failure_reason = FailureCode.UNKNOWN_ERROR
            reward = 0.0

        return self._build_response(
            body,
            reward,
            expected_str,
            model_output_str,
            tool_name_match,
            target_match,
            similarity_score,
            diff_size_penalty,
            failure_reason,
        )

    def _build_response(
        self,
        body: SwePivotVerifyRequest,
        reward: float,
        expected_str: str,
        model_output_str: str,
        tool_name_match: bool,
        target_match: Optional[bool],
        similarity_score: Optional[float],
        diff_size_penalty: Optional[float],
        failure_reason: FailureCode,
    ) -> SwePivotVerifyResponse:
        logger.info(
            f"swe_pivot verify | uuid={body.uuid} reward={reward:.3f} "
            f"tool_match={tool_name_match} target={target_match} "
            f"sim={similarity_score} diff_penalty={diff_size_penalty} "
            f"failure={failure_reason}"
        )
        return SwePivotVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
            uuid=body.uuid,
            expected_answer=expected_str,
            model_output=model_output_str,
            tool_name_match=tool_name_match,
            target_match=target_match,
            similarity_score=similarity_score,
            diff_size_penalty=diff_size_penalty,
            failure_reason=failure_reason,
            metadata=body.metadata,
        )


if __name__ == "__main__":
    SwePivotResourcesServer.run_webserver()
