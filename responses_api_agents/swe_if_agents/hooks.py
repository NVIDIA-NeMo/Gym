# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for the swe_if_agents wrapper (no gym imports, so they are unit-testable with plain python).

The instruction-following (IF) rows carry everything the episode must honour in the request metadata (metadata
values are strings on the wire; structured values are JSON-encoded):

* ``tool_name_overrides``: JSON object identifier -> concrete tool name, the full binding the episode exposes,
  e.g. ``{"BASH_TOOL_NAME": "terminal", "READ_TOOL_NAME": "cat_file", ...}``.
* ``system_prompt_template_text``: the system prompt (Jinja template text) this row runs under, already containing
  the injected instruction; mounted as ``system_prompt.j2``.
* ``user_prompt_template_text``: the user prompt template (Jinja, rendered by OpenHands with the task) with the
  injected instruction; mounted as ``user_prompt.j2``.
* ``replay_observation_suffix``: JSON object ``{"text": str, "tool_call_id"?: str}``, text a data generator
  appended to one recorded tool output of a replayed trajectory (an instruction arriving mid-task); attached to
  that tool message as ``observation_suffix``, which nv-OpenHands appends to the regenerated observation.
* ``sdg_item``: JSON object with the constraints to grade (see ``if_constraints/grader.py``).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9_]*_TOOL_NAME$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def normalize_tool_name_overrides(raw: Any) -> Optional[str]:
    """Validate a per-row tool-name binding and return it as canonical JSON (sorted keys), or None when absent.

    Keys must be nv-OpenHands tool-name identifiers (``*_TOOL_NAME``), values simple identifiers. Anything else
    raises: a malformed binding must not silently fall back to the harness defaults, because the injected
    instruction names the tools.
    """
    if raw in (None, ""):
        return None
    mapping = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("tool_name_overrides must be a non-empty JSON object mapping identifiers to tool names")
    for ident, name in mapping.items():
        if not isinstance(ident, str) or not _IDENTIFIER_RE.match(ident):
            raise ValueError(f"tool_name_overrides: {ident!r} is not a tool-name identifier (e.g. BASH_TOOL_NAME)")
        if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
            raise ValueError(f"tool_name_overrides: {name!r} for {ident} is not a valid tool name")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("tool_name_overrides: two identifiers map to the same tool name")
    return json.dumps(mapping, sort_keys=True)


def tag_replay_observation_suffix(chat_messages: List[dict], spec_raw: Any) -> Optional[dict]:
    """Attach ``observation_suffix`` to the tool message named by ``spec_raw``.

    ``spec_raw`` is ``{"text", "tool_call_id"?}``; without an id the last tool message is tagged. Returns the tagged
    message, or None when no suffix was requested. Raises when a suffix is requested but no matching tool message
    exists: silently dropping it would run the episode without the injection.
    """
    if spec_raw in (None, ""):
        return None
    spec = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
    if not isinstance(spec, dict) or not isinstance(spec.get("text"), str) or not spec["text"]:
        raise ValueError("replay_observation_suffix must be a JSON object with a non-empty string field 'text'")
    tcid = spec.get("tool_call_id")
    for m in reversed(chat_messages):
        if m.get("role") == "tool" and (tcid is None or m.get("tool_call_id") == tcid):
            m["observation_suffix"] = spec["text"]
            return m
    raise ValueError(f"replay_observation_suffix: no tool message found (tool_call_id={tcid!r})")


def write_row_templates(persistent_dir: Path, system_text: Any, user_text: Any) -> Tuple[Optional[str], Optional[str]]:
    """Materialize per-row prompt templates as files under the episode's persistent directory.

    The container mounts that directory. Returns ``(system_prompt_path, user_prompt_path)``; a missing or empty
    text leaves that slot None (the YAML-level template, if any, then applies).
    """
    persistent_dir = Path(persistent_dir)
    persistent_dir.mkdir(parents=True, exist_ok=True)
    out: List[Optional[str]] = []
    for text, name in ((system_text, "row_system_prompt.j2"), (user_text, "row_user_prompt.j2")):
        if isinstance(text, str) and text.strip():
            path = persistent_dir / name
            path.write_text(text)
            out.append(str(path))
        else:
            out.append(None)
    return out[0], out[1]


def row_metadata_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Small, loggable summary of what an IF row asked for (no prompt text)."""
    md = metadata or {}
    return {
        "tool_name_overrides": bool(md.get("tool_name_overrides")),
        "system_prompt_template_text": len(md.get("system_prompt_template_text") or ""),
        "user_prompt_template_text": len(md.get("user_prompt_template_text") or ""),
        "replay_observation_suffix": bool(md.get("replay_observation_suffix")),
        "sdg_item": bool(md.get("sdg_item")),
    }
