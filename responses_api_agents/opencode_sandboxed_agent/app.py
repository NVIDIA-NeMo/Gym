# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import base64
import json
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from shlex import quote
from time import time
from traceback import format_exc
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from openai.types.responses import ResponseInputTextParam
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
    TrajectoryRecord,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from responses_api_agents.opencode_agent.observability import append_opencode_turns, scope_opencode_trajectory


def _load_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _milliseconds(value: Any) -> Optional[float]:
    """Convert OpenCode's Date.now()-based epoch milliseconds to seconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return None
    return float(value) / 1000


def parse_opencode_observations(
    db_path: Path, fallback_invocation_id: str, trajectory: Optional[TrajectoryRecord] = None
) -> AgentObservationBundle:
    """Read OpenCode's persisted session tree before its workspace is removed."""
    if not db_path.is_file():
        return AgentObservationBundle(
            source="opencode",
            records=[AgentInvocation(invocation_id=fallback_invocation_id)],
            gaps=[
                ObservationGap(code="agent_artifact_unavailable"),
                ObservationGap(code="agent_transcript_unavailable"),
                ObservationGap(code="model_call_ownership_unavailable"),
            ],
        )

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        session_rows = con.execute(
            "select id, parent_id, time_created from session order by time_created, id"
        ).fetchall()
        message_rows = con.execute(
            "select id, session_id, data, time_created from message order by time_created, id"
        ).fetchall()
        part_rows = con.execute(
            "select id, message_id, session_id, data, time_created from part order by time_created, id"
        ).fetchall()
    finally:
        con.close()

    messages = {row["id"]: _load_json(row["data"]) for row in message_rows}
    message_sessions = {row["id"]: row["session_id"] for row in message_rows}
    conversations: dict[str, list[Any]] = {row["id"]: [] for row in session_rows}
    invocation_status: dict[str, str] = {row["id"]: "unknown" for row in session_rows}
    tools: list[ToolCallObservation] = []
    child_tools: dict[str, set[str]] = {}
    child_status: dict[str, str] = {}
    compaction_parts: list[tuple[str, str, float | None, dict[str, Any]]] = []
    gaps: list[ObservationGap] = []
    summary_text: dict[str, list[str]] = {}
    summaries_by_parent: dict[str, list[str]] = {}
    first_item_id_by_message: dict[tuple[str, str], str] = {}

    for row in message_rows:
        message = messages[row["id"]]
        session_id = row["session_id"]
        if not isinstance(session_id, str) or session_id not in invocation_status:
            gaps.append(ObservationGap(code="agent_artifact_record_unowned", detail=row["id"]))
        elif message.get("role") == "assistant":
            if isinstance(message.get("error"), dict):
                invocation_status[session_id] = "failed"
            message_time = message.get("time") if isinstance(message.get("time"), dict) else {}
            if invocation_status[session_id] != "failed" and _milliseconds(message_time.get("completed")) is not None:
                invocation_status[session_id] = "completed"
        if message.get("summary") is True:
            summary_text[row["id"]] = []
            parent_id = message.get("parentID")
            if isinstance(parent_id, str):
                summaries_by_parent.setdefault(parent_id, []).append(row["id"])

    for row in part_rows:
        part = _load_json(row["data"])
        if not part:
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable"))
            continue
        ptype = part.get("type")
        message_id = row["message_id"]
        message = messages.get(message_id, {})
        session_id = row["session_id"] or message_sessions.get(message_id)
        if not isinstance(session_id, str):
            gaps.append(ObservationGap(code="agent_artifact_record_unowned"))
            continue
        conversation = conversations.setdefault(session_id, [])
        role = message.get("role")

        if ptype == "step-finish":
            continue

        text = part.get("text")
        if ptype == "text" and isinstance(text, str) and text.strip():
            if message_id in summary_text:
                summary_text[message_id].append(text)
            if role == "user" and part.get("ignored") is not True:
                conversation.append(NeMoGymEasyInputMessage(role="user", content=text))
            elif role == "assistant":
                item = NeMoGymResponseOutputMessage(
                    id=row["id"],
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
                conversation.append(item)
                first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "reasoning" and role == "assistant" and isinstance(text, str) and text.strip():
            conversation.append(
                NeMoGymResponseReasoningItem(
                    id=row["id"],
                    summary=[NeMoGymSummary(type="summary_text", text=text)],
                )
            )
            first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "tool" and role == "assistant":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            native_call_id = part.get("callID")
            observed_call_id = native_call_id if isinstance(native_call_id, str) and native_call_id else None
            call_id = observed_call_id or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            native_status = state.get("status")
            response_status = "completed" if native_status == "completed" else "incomplete"
            call = NeMoGymResponseFunctionToolCall(
                arguments=arguments,
                call_id=call_id,
                name=part.get("tool", ""),
                type="function_call",
                id=call_id,
                status=response_status,
            )
            conversation.append(call)
            first_item_id_by_message.setdefault((session_id, message_id), call_id)
            native_time = state.get("time") if isinstance(state.get("time"), dict) else {}
            # OpenCode retains raw output in SQLite but substitutes this literal in later model inputs after pruning.
            if native_status == "completed" and native_time.get("compacted") is not None:
                observed_tool_output = "[Old tool result content cleared]"
            else:
                observed_tool_output = state.get("output") if state.get("output") is not None else state.get("error")
            if observed_tool_output is not None:
                result = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=str(observed_tool_output),
                    status=response_status,
                )
                conversation.append(result)

            native_start = native_time.get("start")
            native_end = native_time.get("end")
            valid_interval = (
                isinstance(native_start, (int, float))
                and not isinstance(native_start, bool)
                and isinstance(native_end, (int, float))
                and not isinstance(native_end, bool)
                and native_end >= native_start
            )
            started_at = _milliseconds(native_start) if valid_interval else None
            completed_at = _milliseconds(native_end) if valid_interval else None
            duration_ms = float(native_end - native_start) if valid_interval else None
            status = {
                "completed": "completed",
                "error": "failed",
                "running": "incomplete",
                "pending": "incomplete",
            }.get(native_status, "unknown")
            if observed_call_id is not None:
                tools.append(
                    ToolCallObservation(
                        invocation_id=session_id,
                        tool_call_id=observed_call_id,
                        tool_name=part.get("tool") if isinstance(part.get("tool"), str) else None,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        timing_source="artifact" if started_at is not None else None,
                        status=status,
                        error_type="tool_error" if native_status == "error" else None,
                    )
                )
            else:
                gaps.append(
                    ObservationGap(
                        code="tool_call_identity_unavailable",
                        invocation_id=session_id,
                        detail=row["id"],
                    )
                )
            if observed_call_id is not None and (
                started_at is None or (native_status in {"completed", "error"} and completed_at is None)
            ):
                gaps.append(
                    ObservationGap(
                        code="tool_timing_unavailable",
                        invocation_id=session_id,
                        detail=observed_call_id,
                    )
                )
            metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
            if not metadata and isinstance(part.get("metadata"), dict):
                metadata = part["metadata"]
            child_id = metadata.get("sessionId")
            if isinstance(child_id, str) and observed_call_id is not None:
                child_tools.setdefault(child_id, set()).add(observed_call_id)
                child_status[child_id] = status
            continue
        if ptype == "compaction":
            compaction_parts.append((session_id, message_id, _milliseconds(row["time_created"]), part))
            continue

    compactions: list[ContextCompactionObservation] = []
    for session_id, message_id, observed_at, part in compaction_parts:
        summary_ids = summaries_by_parent.get(message_id, [])
        summary = "\n".join(summary_text.get(summary_ids[0], [])) if len(summary_ids) == 1 else None
        if len(summary_ids) > 1:
            gaps.append(
                ObservationGap(
                    code="compaction_summary_ambiguous",
                    invocation_id=session_id,
                )
            )
        trigger = "overflow" if part.get("overflow") is True else "automatic" if part.get("auto") is True else "manual"
        tail_start_id = part.get("tail_start_id") if isinstance(part.get("tail_start_id"), str) else None
        first_kept_item_id = (
            first_item_id_by_message.get((session_id, tail_start_id)) if tail_start_id is not None else None
        )
        compactions.append(
            ContextCompactionObservation(
                invocation_id=session_id,
                observed_at=observed_at,
                trigger=trigger,
                outcome="completed" if summary else "unknown",
                summary=summary,
                first_kept_item_id=first_kept_item_id,
            )
        )
        if tail_start_id is not None and first_kept_item_id is None:
            gaps.append(
                ObservationGap(
                    code="compaction_first_kept_item_unavailable",
                    invocation_id=session_id,
                    detail=tail_start_id,
                )
            )
        if not summary:
            gaps.append(ObservationGap(code="compaction_summary_unavailable", invocation_id=session_id))
        gaps.append(ObservationGap(code="compaction_token_counts_unavailable", invocation_id=session_id))
        gaps.append(
            ObservationGap(
                code="compaction_model_call_boundary_unavailable",
                invocation_id=session_id,
            )
        )

    session_ids = {row["id"] for row in session_rows}
    invocations = []
    for row in session_rows:
        invocation_id = row["id"]
        parent_id = row["parent_id"]
        spawn_candidates = child_tools.get(invocation_id, set())
        if parent_id is not None and parent_id not in session_ids:
            gaps.append(
                ObservationGap(
                    code="subagent_parent_unavailable",
                    invocation_id=invocation_id,
                    detail=parent_id,
                )
            )
        if len(spawn_candidates) > 1:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_ambiguous",
                    invocation_id=invocation_id,
                )
            )
        elif parent_id is not None and not spawn_candidates:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_tool_unavailable",
                    invocation_id=invocation_id,
                )
            )
        invocations.append(
            AgentInvocation(
                invocation_id=invocation_id,
                parent_invocation_id=parent_id,
                spawned_by_tool_call_id=next(iter(spawn_candidates)) if len(spawn_candidates) == 1 else None,
                status=(
                    invocation_status.get(invocation_id, "unknown")
                    if invocation_status.get(invocation_id, "unknown") != "unknown"
                    else child_status.get(invocation_id, "unknown")
                ),
                conversation=conversations.get(invocation_id, []),
            )
        )
    if not invocations:
        invocations = [AgentInvocation(invocation_id=fallback_invocation_id)]
        gaps.append(ObservationGap(code="agent_transcript_unavailable"))

    if trajectory is not None:
        append_opencode_turns(trajectory, session_ids, message_rows, part_rows)

    return AgentObservationBundle(
        source="opencode",
        records=[*invocations, *tools, *compactions],
        gaps=gaps,
    )


# Task images can carry their builder's proxy settings (gitingest's points HTTP(S)_PROXY at
# sys-proxy-rd-relay.byted.org:8118, unresolvable here). OpenCode runs on Bun, which honours them, so every model call
# failed at once and was retried every 30 s for the whole rollout: gitingest never finished in any no-internet run.
# The model server is reached directly and the agent's own commands have no internet anyway, so drop them.
UNSET_PROXY_ENV = "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy"


class IdleNudgeConfig(BaseModel):
    """After OpenCode exits cleanly with time left, and its last assistant turn made no tool call, continue the same
    session with ``text``, up to ``max_nudges`` times, unless that turn's text contains ``stop_on_text`` (the model's
    explicit "done"). A turn cut off at the output token limit gets LENGTH_NOTE before the text. A turn that ended on an
    API/provider error is not continued. Mirrors responses_api_agents/opencode_agent's IdleNudgeConfig."""

    text: str
    max_nudges: int = Field(default=3, ge=1)
    stop_on_text: Optional[str] = None
    # Put LENGTH_NOTE before `text` when the stopped turn hit the output token limit. False sends `text` alone every
    # time, as OpenHands' headless auto-continue does.
    length_note: bool = True
    # Appended to the task prompt (after a blank line) on the FIRST run, so the model knows how to finish from turn 1.
    # OpenHands' `finish` tool is in its tool list from the start; OpenCode has no such tool, so without this the model
    # learns `stop_on_text` only from the first nudge (measured 2026-11-04: 99% of NL2Repo traces carried exactly one
    # nudge, the finish handshake). The task prompt itself is left as the benchmark wrote it.
    finish_instruction: Optional[str] = None


class OpenCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    opencode_version: str
    remote_opencode_install_script_path: Optional[str] = None
    remote_opencode_binary_path: Optional[str] = None
    remote_opencode_musl_binary_path: Optional[str] = None
    opencode_config: Dict[str, Any] = Field(default_factory=dict)
    opencode_max_context_window: int
    opencode_model_call_timeout: Optional[int] = None
    # Keep OpenCode's data dir (XDG_DATA_HOME) under /tmp even when observations are off. For task images that run as a
    # user without a writable home (NL2RepoBench more-itertools runs as `appuser`), where the default $HOME/.local/share
    # fails with EACCES before the agent's first turn. Off by default: unchanged command line for everyone else.
    opencode_data_home_in_tmp: bool = False
    # Continue the session when the model stops with budget left (same model and semantics as the ProgramBench
    # opencode_agent's IdleNudgeConfig). `opencode run` exits on ANY reply without a tool call, which is also how it
    # says it is done; mini-swe-agent answers a reply without a command with a correction, and OpenHands with a
    # corrective message. Without this, a turn cut off at max_tokens mid-reasoning ends the task with an empty workspace
    # (NL2RepoBench, Super 3.5 at max_tokens 32000: 5 of 63 rollouts, 2026-11-02). Off unless configured.
    idle_nudge: Optional["IdleNudgeConfig"] = None
    # OpenHands 0.56's StuckDetector (NL2RepoBench's harness ends a run when its agent loops): stuck_detector.mjs, an
    # OpenCode plugin, checks the repeat/error/alternation/compaction scenarios after every tool call and ends the run
    # when one fires; the monologue scenario (3 identical replies with no tool call between) is checked here before a
    # nudge. The row's `opencode_stuck` names the scenario. Off by default.
    stuck_detector: bool = False
    # Where each session's raw OpenCode export (and observations database) is saved, one subdirectory per session.
    # Default: this agent's own results/ directory, shared by every run; a launcher points it into the run's directory.
    opencode_results_dir: Optional[str] = None

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float
    # Seconds past sandbox_timeout before the agent phase is cut off (OpenCode killed, the workspace graded as it is).
    wall_grace_s: float = 60.0

    debug: bool = False


class OpenCodeSandboxedAgentRunRequest(BaseRunRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


def _build_remote_opencode_install_command(
    install_script_path: str,
    binary_path: str,
    musl_binary_path: str,
) -> str:
    """Build the invocation for the network-free, libc-aware cached installer."""
    return (
        f"bash {quote(install_script_path)} "
        f"--glibc-binary {quote(binary_path)} "
        f"--musl-binary {quote(musl_binary_path)}"
    )


def _extract_opencode_session_id(session_list_stdout: str) -> str:
    """Return the newest OpenCode session ID from ``session list`` JSON output."""
    sessions = json.loads(session_list_stdout)
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("OpenCode did not return any sessions")

    session_id = sessions[0].get("id") if isinstance(sessions[0], dict) else None
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("The newest OpenCode session does not have a valid ID")
    return session_id


# Put before the idle-nudge text when the stopped turn was cut off at the output token limit (OpenCode records finish
# "length"), the case mini-swe-agent's ProgramBench format_error_template calls out on its own.
LENGTH_NOTE = (
    "Your previous response reached the output token limit before you produced a tool call, so it was cut off. "
    "Respond more concisely."
)


def _steps_used(opencode_export: Dict[str, Any]) -> int:
    """Agentic steps the session has taken: one `step-finish` part per step, as OpenCode records them."""
    return sum(
        1
        for m in opencode_export.get("messages") or []
        if (m.get("info") or {}).get("role") == "assistant"
        for p in m.get("parts") or []
        if p.get("type") == "step-finish"
    )


def _idle_reason(opencode_export: Dict[str, Any], stop_on_text: Optional[str] = None) -> Optional[str]:
    """Why the session should be continued, or None.

    "length": the last assistant turn was cut off at the output token limit without a tool call.
    "idle":   it stopped without a tool call for any other reason (reasoning only, empty, or text), unless its text
              contains ``stop_on_text``.
    None:     it called a tool, ended on an API/provider error, said ``stop_on_text``, or there is no assistant turn."""
    messages = opencode_export.get("messages") or []
    assistant = [m for m in messages if (m.get("info") or {}).get("role") == "assistant"]
    if not assistant:
        return None
    last = assistant[-1]
    info = last.get("info") or {}
    if info.get("error"):
        return None  # an API/provider error is a failure to report, not a turn to continue
    parts = last.get("parts") or []
    if any(p.get("type") == "tool" for p in parts):
        return None
    text = " ".join(str(p.get("text") or "") for p in parts if p.get("type") == "text")
    if stop_on_text and stop_on_text in text:
        return None
    if info.get("finish") == "length" or any(
        p.get("type") == "step-finish" and p.get("reason") == "length" for p in parts
    ):
        return "length"
    return "idle"


_SESSION_ID_RE = re.compile(r"\bses_[A-Za-z0-9]+")
MAX_CHILD_SESSIONS = 500


def _child_session_ids(opencode_export: Dict[str, Any]) -> List[str]:
    """Sessions this one started with the `task` tool (subagents), in call order. The task tool records the child in
    its metadata (`sessionId`); a resumed child is named by the call's `task_id`, and its output names it too
    (`<task id="ses_...">`). `opencode session list` shows only top-level sessions, so this is how children are found."""
    ids: List[str] = []
    own = (opencode_export.get("info") or {}).get("id")
    for message in opencode_export.get("messages") or []:
        for part in message.get("parts") or []:
            if part.get("type") != "tool" or part.get("tool") != "task":
                continue
            state = part.get("state") or {}
            found = [(state.get("metadata") or {}).get("sessionId"), (state.get("input") or {}).get("task_id")]
            found += _SESSION_ID_RE.findall(str(state.get("output") or ""))
            for sid in found:
                if isinstance(sid, str) and sid.startswith("ses_") and sid != own and sid not in ids:
                    ids.append(sid)
    return ids


def _token_totals(opencode_export: Dict[str, Any]) -> Dict[str, int]:
    """Token use summed over a session's model calls, as OpenCode records them per assistant message."""
    totals = {"calls": 0, "input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    for message in opencode_export.get("messages") or []:
        info = message.get("info") or {}
        tokens = info.get("tokens")
        if info.get("role") != "assistant" or not tokens:
            continue
        cache = tokens.get("cache") or {}
        totals["calls"] += 1
        totals["input"] += int(tokens.get("input") or 0)
        totals["output"] += int(tokens.get("output") or 0)
        totals["reasoning"] += int(tokens.get("reasoning") or 0)
        totals["cache_read"] += int(cache.get("read") or 0)
        totals["cache_write"] += int(cache.get("write") or 0)
    return totals


def _token_breakdown(main: Dict[str, Any], children: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Main agent vs subagents: {"main": totals, "subagents": totals summed, "per_subagent": {id: totals}}."""
    per = {sid: _token_totals(export) for sid, export in children.items()}
    summed = _token_totals({})
    for totals in per.values():
        for key in summed:
            summed[key] += totals[key]
    return {"main": _token_totals(main), "subagents": summed, "per_subagent": per}


STUCK_DIR = "/tmp/nemo_gym_stuck"
STUCK_PLUGIN = Path(__file__).with_name("stuck_detector.mjs")


def _monologue_stuck(opencode_export: Dict[str, Any]) -> bool:
    """OpenHands' monologue scenario: the last 3 agent messages (steps with no tool call) are identical, with nothing
    observed (no tool call, no compaction) between them. User messages (the nudges) are left out, as OpenHands does."""
    history: List[Optional[str]] = []  # an agent message's text, or None for an observation
    for message in opencode_export.get("messages") or []:
        parts = message.get("parts") or []
        if (message.get("info") or {}).get("role") != "assistant":
            history.extend(None for p in parts if p.get("type") == "compaction")
            continue
        step: Optional[List[Dict[str, Any]]] = None
        for part in parts + [{"type": "step-finish"}]:
            if part.get("type") == "step-start":
                step = []
            elif part.get("type") == "step-finish":
                if step is not None:
                    if any(p.get("type") == "tool" for p in step):
                        history.append(None)
                    else:
                        history.append("".join(str(p.get("text") or "") for p in step if p.get("type") == "text"))
                step = None
            elif step is not None:
                step.append(part)
    agent = [i for i, h in enumerate(history) if h is not None]
    if len(agent) < 3:
        return False
    first, *_, last = agent[-3:]
    return len({history[i] for i in agent[-3:]}) == 1 and all(h is not None for h in history[first:last])


async def _read_stuck_marker(sandbox: Any) -> Optional[str]:
    """The scenario stuck_detector.mjs recorded, or None."""
    try:
        res = await sandbox.exec(command=f"cat {STUCK_DIR}/stuck.json 2>/dev/null || true")
        return json.loads(res.stdout).get("scenario") if (res.stdout or "").strip() else None
    except Exception:
        print("Reading the stuck marker failed", format_exc(), file=sys.stderr)
        return None


class _CombinedExecResult:
    """Stdout/stderr of several `opencode run` execs (the first run plus nudged resumes), shaped like an exec result."""

    def __init__(self, first: Any) -> None:
        self.stdout = getattr(first, "stdout", "") or ""
        self.stderr = getattr(first, "stderr", "") or ""
        self.return_code = getattr(first, "return_code", None)
        self.error_type = getattr(first, "error_type", None)
        self.stuck: Optional[str] = None  # the OpenHands stuck scenario that ended the session, if any

    def add(self, nxt: Any) -> None:
        self.stdout += "\n" + (getattr(nxt, "stdout", "") or "")
        self.stderr += "\n" + (getattr(nxt, "stderr", "") or "")
        self.return_code = getattr(nxt, "return_code", None)
        self.error_type = getattr(nxt, "error_type", None)


class OpenCodeSandboxedAgentVerifyRequest(BaseVerifyRequest):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")


class OpenCodeSandboxedAgentVerifyResponse(BaseVerifyResponse):
    # Allow for benchmark params to propagate properly
    model_config = ConfigDict(extra="allow")

    opencode_results_fpath: str
    opencode_run_stdout: str
    opencode_run_stderr: str
    opencode_finished: bool
    opencode_export_found: bool
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class OpenCodeSandboxedAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: OpenCodeSandboxedAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _start_sandbox(self, sandbox_id: Optional[str] = None, workdir: Optional[str] = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        )
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        if sandbox_id:
            sandbox = await AsyncSandbox.connect(
                {"sandbox_id": sandbox_id, "workdir": workdir}, provider=resolved_sandbox_provider
            )
            return sandbox

        if self.config.debug:
            print("Creating new sandbox since one wasn't provided", file=sys.stderr)

        resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        # TODO @bxyu-nvidia: Refactor this after swapping to PTY as this should be set on the SWE Bench resources server side
        env = cpu_cap_env(resources.cpu) if self.config.sandbox_config.get("derive_cpu_env", True) else {}
        env |= dict(self.config.sandbox_config.get("env", {}))  # explicit keys win over the derived caps

        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        sandbox_spec = SandboxSpec(
            image="swebench/sweb.eval.x86_64.astropy_1776_astropy-12907",  # This is just the first SWE Bench Verified image for now
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
            },
            resources=resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )

        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(sandbox_spec)

        return sandbox

    def _agent_sandbox_observation(
        self,
        *,
        sandbox: AsyncSandbox,
        return_code: Any,
        error_type: Any,
        finished: bool,
    ) -> SandboxObservation:
        handle = getattr(sandbox, "_handle", None)
        handle_provider = getattr(handle, "provider_name", None)
        handle_sandbox_id = getattr(handle, "sandbox_id", None)
        normalized_error = error_type.lower() if isinstance(error_type, str) else ""
        if "timeout" in normalized_error:
            outcome = "timeout"
        elif normalized_error:
            outcome = "sandbox_error"
        elif return_code == 0 and finished:
            outcome = "completed"
        elif isinstance(return_code, int):
            outcome = "failed" if return_code != 0 else "unknown"
        else:
            outcome = "unknown"
        return SandboxObservation(
            role="agent",
            provider=handle_provider if isinstance(handle_provider, str) else None,
            sandbox_id=handle_sandbox_id if isinstance(handle_sandbox_id, str) else None,
            outcome=outcome,
            exit_code=return_code if not normalized_error and isinstance(return_code, int) else None,
            error_type=error_type if isinstance(error_type, str) else None,
        )

    async def _create_opencode_config(self, request: Request) -> Dict[str, Any]:
        base_url = (
            self.base_url_for_run(
                base_url=get_server_url(self.config.model_server.name),
                body=await request.json(),
            )
            + "/v1"
        )
        return {
            "model": "nemo_gym/dummy_model",
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "nemo_gym": {
                    # TODO @bxyu-nvidia: We should use @ai-sdk/openai here but there is some /v1/responses streaming error.
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "dummy_key",  # pragma: allowlist secret
                        "timeout": self.config.opencode_model_call_timeout,  # in milliseconds
                    },
                    "models": {
                        "dummy_model": {
                            "limit": {
                                "context": self.config.opencode_max_context_window,
                                "input": self.config.opencode_max_context_window,
                                # See the OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX flag below for more information.
                                "output": self.config.opencode_max_context_window,
                            },
                        },
                    },
                }
            },
            **self.config.opencode_config,
        }

    async def _nudge_idle_turns(
        self,
        *,
        sandbox: Any,
        first_result: Any,
        config_content: str,
        env_suffix: str,
        run_flags: str,
        session_env: Optional[Dict[str, str]],
        deadline: float,
    ) -> tuple[Any, int]:
        """Continue the session per ``idle_nudge`` while its last assistant turn made no tool call.

        Returns the combined exec result and the number of nudges sent. Stops at the configured limit, when an
        `opencode run` does not exit cleanly, when the session cannot be read, when under a minute of the
        sandbox_timeout budget is left, or when the session has used OpenCode's step cap (`agent.build.steps` in
        `config_content`): each `opencode run --session` would otherwise start a fresh cap, so a resume gets only the
        steps left, which makes the cap one budget for the whole task (as OpenHands' max_iterations is)."""
        base_config = json.loads(config_content)
        step_cap = ((base_config.get("agent") or {}).get("build") or {}).get("steps")
        combined = _CombinedExecResult(first_result)
        latest_stdout = combined.stdout
        nudges = 0
        check_remote = "/tmp/opencode_idle_check.json"
        nudge_config = self.config.idle_nudge
        assert nudge_config is not None
        while nudges < nudge_config.max_nudges:
            if self.config.stuck_detector:
                combined.stuck = await _read_stuck_marker(sandbox)
                if combined.stuck:
                    break  # the plugin ended the run, as OpenHands ends a stuck agent
            if "OpenCode run finished" not in latest_stdout:
                break
            remaining = deadline - time()
            if remaining < 60:
                break
            try:
                listed = await sandbox.exec(
                    command="export PATH=$HOME/.opencode/bin:$PATH && opencode session list --format json",
                    env=session_env,
                )
                session_id = _extract_opencode_session_id(listed.stdout or "")
                exported = await sandbox.exec(
                    command=(
                        "export PATH=$HOME/.opencode/bin:$PATH "
                        f"&& opencode export {quote(session_id)} > {quote(check_remote)}"
                    ),
                    env=session_env,
                )
                if exported.return_code != 0:
                    break
                with tempfile.TemporaryDirectory(prefix="nemo-gym-opencode-nudge-") as tmp:
                    local = Path(tmp) / "export.json"
                    await sandbox.download(check_remote, local)
                    session_export = json.loads(local.read_text().strip() or "{}")
            except Exception:
                print("Idle check failed; not nudging", format_exc(), file=sys.stderr)
                break
            reason = _idle_reason(session_export, nudge_config.stop_on_text)
            if reason is None:
                break
            if self.config.stuck_detector and _monologue_stuck(session_export):
                combined.stuck = "monologue"
                break
            config = base_config
            if isinstance(step_cap, int) and step_cap > 0:
                used = _steps_used(session_export)
                if used >= step_cap:
                    break  # the whole-task step budget is spent, as OpenHands stops at max_iterations
                config = json.loads(config_content)
                config.setdefault("agent", {}).setdefault("build", {})["steps"] = step_cap - used
            nudges += 1
            print(
                f"OpenCode session {session_id} stopped without a tool call ({reason}); nudge {nudges}",
                file=sys.stderr,
            )
            nudge = (
                f"{LENGTH_NOTE}\n{nudge_config.text}"
                if reason == "length" and nudge_config.length_note
                else nudge_config.text
            )
            resume = (
                f"export PATH=$HOME/.opencode/bin:$PATH && {UNSET_PROXY_ENV} "
                f"&& printf '%s' {quote(nudge)} | OPENCODE_CONFIG_CONTENT={quote(json.dumps(config))} {env_suffix} "
                f"opencode run --session {quote(session_id)} {run_flags} "
                f'&& echo "OpenCode run finished"'
            )
            try:
                resumed = await sandbox.exec(command=resume, timeout_s=remaining)
            except Exception:
                print("OpenCode resume after an idle stop hit an error", format_exc(), file=sys.stderr)
                break
            combined.add(resumed)
            latest_stdout = getattr(resumed, "stdout", "") or ""
        return combined, nudges

    def _opencode_export_to_usages(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseUsage]:
        usages: List[NeMoGymResponseUsage] = []
        for message in opencode_export["messages"]:
            if message["info"]["role"] != "assistant":
                continue

            token_info = message["info"].get("tokens")
            if not token_info:
                continue

            usage = NeMoGymResponseUsage(
                input_tokens=token_info["input"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=token_info["cache"]["read"]),
                output_tokens=token_info["output"],
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=token_info["reasoning"]),
                total_tokens=token_info.get("total", 0),  # Somehow total may be missing
            )
            usages.append(usage)

        return usages

    def _opencode_export_to_output_items(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
        messages = []
        for message in opencode_export["messages"]:
            if message["info"]["role"] == "user":
                message_parts = []
                for part in message["parts"]:
                    if part["type"] != "text":
                        continue

                    message_parts.append(ResponseInputTextParam(text=part["text"], type="input_text"))

                messages.append(NeMoGymEasyInputMessage(content=message_parts, role="user"))
            elif message["info"]["role"] == "assistant":
                converter = ResponsesConverter(return_token_id_information=True)
                for part in message["parts"]:
                    if part["type"] == "text":
                        output_items = converter.postprocess_assistant_message_dict(
                            message_dict={
                                "content": part["text"],
                                "role": "assistant",
                            }
                        )
                        messages.extend(output_items)
                    elif part["type"] == "reasoning":
                        output_items = converter.postprocess_assistant_message_dict(
                            message_dict={
                                "content": converter._wrap_reasoning_in_think_tags([part["text"]]),
                                "role": "assistant",
                            }
                        )
                        messages.extend(output_items)
                    elif part["type"] == "tool":
                        messages.append(
                            NeMoGymResponseFunctionToolCall(
                                arguments=json.dumps(part["state"]["input"]),
                                call_id=part["callID"],
                                name=part["tool"],
                            )
                        )
                        messages.append(
                            NeMoGymFunctionCallOutput(
                                call_id=part["callID"],
                                # @bxyu-nvidia: Somehow the output here may be missing...
                                output=part["state"].get("output", ""),
                            )
                        )
                    elif part["type"] in ("step-finish", "step-start", "patch"):
                        pass
                    else:
                        # @bxyu-nvidia: Defensive raise in case we're missing something.
                        raise NotImplementedError(part)
            else:
                # @bxyu-nvidia: Defensive raise in case we're missing something.
                raise NotImplementedError(message)

        return messages

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]

        query = None
        # This can be modified to handle system/developer prompts too.
        for input_item in body.input:
            if input_item.role == "user":
                assert not query, body.input
                if isinstance(input_item.content, str):
                    query = input_item.content
                elif isinstance(input_item.content, list):
                    assert len(input_item.content) == 1, body.input
                    query = input_item.content[0]["text"]

        assert query, body.input
        if self.config.idle_nudge is not None and self.config.idle_nudge.finish_instruction:
            query = f"{query}\n\n{self.config.idle_nudge.finish_instruction}"

        opencode_debug_str = ""
        if self.config.debug:
            opencode_debug_str = "--print-logs --log-level DEBUG"

        opencode_thinking_str = "--thinking"

        if self.config.remote_opencode_binary_path and self.config.remote_opencode_install_script_path:
            if self.config.remote_opencode_musl_binary_path:
                install_str = _build_remote_opencode_install_command(
                    install_script_path=self.config.remote_opencode_install_script_path,
                    binary_path=self.config.remote_opencode_binary_path,
                    musl_binary_path=self.config.remote_opencode_musl_binary_path,
                )
            else:
                install_str = (
                    f"bash {quote(self.config.remote_opencode_install_script_path)} "
                    f"--binary {quote(self.config.remote_opencode_binary_path)}"
                )
        else:
            print(
                "Downloading and installing OpenCode in the sandbox. Please consider mounting or uploading the appropriate OpenCode binary instead!",
                file=sys.stderr,
            )
            install_str = f"""installer=$(mktemp) && curl -fL -o "$installer" https://opencode.ai/install \
        && echo "Downloaded OpenCode installer to $installer" \
        && VERSION={self.config.opencode_version} bash "$installer\""""

        opencode_config = await self._create_opencode_config(request)
        stuck_setup = ""
        if self.config.stuck_detector:
            # No upload API here: the plugin rides in the run command (base64), and loads from a file:// URL
            plugin_b64 = base64.b64encode(STUCK_PLUGIN.read_bytes()).decode()
            stuck_setup = (
                f"mkdir -p {STUCK_DIR} && rm -f {STUCK_DIR}/stuck.json "
                f"&& echo {plugin_b64} | base64 -d > {STUCK_DIR}/stuck_detector.mjs && "
            )
            opencode_config = {
                **opencode_config,
                "plugin": [*opencode_config.get("plugin", []), f"file://{STUCK_DIR}/stuck_detector.mjs"],
            }
        opencode_config_content = json.dumps(opencode_config)
        observation_invocation_id = getattr(request.state, "_ng_observation_invocation_id", None)
        observation_invocation_id = observation_invocation_id if isinstance(observation_invocation_id, str) else None
        collect_observations = observation_invocation_id is not None
        trajectory = (
            TrajectoryRecord(task_id="unscoped", rollout_id=observation_invocation_id)
            if collect_observations
            else None
        )
        xdg_env: Dict[str, str] = {}
        remote_data_home = None
        # Observations read OpenCode's session database, so it goes to a known /tmp path; opencode_data_home_in_tmp does
        # the same for images whose user has no writable home (see the config field), and also moves OpenCode's state,
        # cache and config dirs there: with only XDG_DATA_HOME moved, it next fails on mkdir $HOME/.local/state.
        if collect_observations or self.config.opencode_data_home_in_tmp:
            remote_data_home = f"/tmp/nemo-gym-opencode-{uuid4().hex}"
            xdg_env["XDG_DATA_HOME"] = remote_data_home
            if self.config.opencode_data_home_in_tmp:
                for name in ("STATE", "CACHE", "CONFIG"):
                    xdg_env[f"XDG_{name}_HOME"] = f"{remote_data_home}-{name.lower()}"
        xdg_home_str = " ".join(f"{key}={value}" for key, value in xdg_env.items())

        # @bxyu-nvidia: Regarding `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000` below:
        # OpenCode defaults to 32k here https://github.com/anomalyco/opencode/blob/58a99916bb96edf5cf605dc03e1be1e4bacf9ff7/packages/opencode/src/provider/transform.ts#L21
        # and there is no way to set it to null.
        # Here, we set an exorbitantly high number that cannot ever be reached.
        # In future versions of OpenCode, this can be directly passed via maxOutputTokens in the limit config above https://github.com/anomalyco/opencode/blob/1b18a50418f730aca32630ccfcde850f2b5fc360/packages/opencode/src/provider/transform.ts#L1418
        # The prompt goes to `opencode run` on STDIN, not as an argument: OpenCode wraps any argument containing a space
        # in literal double quotes (and escapes inner quotes) before sending it, so the model saw `"According to ..."`
        # (opencode cli/cmd/run.ts). Piped input is sent as is. Same for every nudge. Changed 2026-11-04.
        command = f"""
        echo "Shell: $SHELL" \
        && {stuck_setup}{install_str} \
        && export PATH=$HOME/.opencode/bin:$PATH \
        && {UNSET_PROXY_ENV} \
        && echo "Installed OpenCode" \
        && printf '%s' {quote(query)} | OPENCODE_CONFIG_CONTENT={quote(opencode_config_content)} OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000 {xdg_home_str} \
            opencode run --title "NG dummy title" {opencode_debug_str} {opencode_thinking_str} \
        && echo "OpenCode run finished"
        """

        if self.config.debug:
            print(f"Running command:\n```bash\n{command}\n```\n", file=sys.stderr)
            print(f"OpenCode config JSON str: {opencode_config_content}", file=sys.stderr)

        run_error_type = None
        run_started = time()
        # One hard deadline for the whole agent phase (the first `opencode run` and every nudged resume): sandbox_timeout.
        # The per-exec timeouts alone did not hold it: NL2RepoBench rollouts on OpenSandbox ran ~14 h against a 6 h
        # sandbox_timeout (2026-11-03), since the provider's own backstop on a detached command is about twice its
        # timeout and each resume got a fresh one. On expiry OpenCode is killed in the sandbox and the run goes on to
        # export and grading with what is there; the row says opencode_wall_exceeded.
        phase = {"result": None, "idle_nudges": 0, "stuck": None}

        async def agent_phase() -> None:
            phase["result"] = await sandbox.exec(command=command, timeout_s=self.config.sandbox_timeout)
            if self.config.idle_nudge is not None:
                phase["result"], phase["idle_nudges"] = await self._nudge_idle_turns(
                    sandbox=sandbox,
                    first_result=phase["result"],
                    config_content=opencode_config_content,
                    env_suffix=f"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000 {xdg_home_str}",
                    run_flags=f"{opencode_debug_str} {opencode_thinking_str}",
                    session_env=dict(xdg_env) or None,
                    deadline=run_started + self.config.sandbox_timeout,
                )
                phase["stuck"] = phase["result"].stuck

        wall_exceeded = False
        try:
            await asyncio.wait_for(agent_phase(), timeout=self.config.sandbox_timeout + self.config.wall_grace_s)
        except asyncio.TimeoutError:
            wall_exceeded = True
            run_error_type = "WallTimeExceeded"
            print(f"OpenCode agent phase passed sandbox_timeout ({self.config.sandbox_timeout:g} s); stopping it",
                  file=sys.stderr)
            try:
                await asyncio.wait_for(
                    sandbox.exec(command="pkill -KILL -f 'opencode run' >/dev/null 2>&1; true", timeout_s=60), timeout=90
                )
            except Exception:
                print("Could not stop OpenCode after the wall deadline", format_exc(), file=sys.stderr)
        except Exception as exc:
            run_error_type = type(exc).__name__
            print("OpenCode exec hit error.", format_exc(), file=sys.stderr)
        result, idle_nudges, stuck = phase["result"], phase["idle_nudges"], phase["stuck"]
        if result is not None and self.config.stuck_detector and stuck is None:
            stuck = await _read_stuck_marker(sandbox)

        if self.config.debug and result:
            print("OpenCode install and run stdout:\n", result.stdout, file=sys.stderr)
            print("OpenCode install and run stderr:\n", result.stderr, file=sys.stderr)

        export_fname = "export.json"
        # Kept outside the sandbox workdir on purpose: SWE-bench-style environments set the workdir
        # to the git repo, and resources servers extract the model patch with `git add -N . && git
        # diff`, which would sweep this transcript into the patch.
        export_remote_fpath = f"/tmp/opencode_{export_fname}"
        try:
            session_env = dict(xdg_env) or None
            session_list_result = await sandbox.exec(
                command="export PATH=$HOME/.opencode/bin:$PATH && opencode session list --format json",
                env=session_env,
            )
            if session_list_result.return_code != 0:
                raise RuntimeError(f"Failed to list OpenCode sessions: {session_list_result}")
            session_id = _extract_opencode_session_id(session_list_result.stdout or "")
            export_result = await sandbox.exec(
                command=(
                    "export PATH=$HOME/.opencode/bin:$PATH "
                    f"&& opencode export {quote(session_id)} > {quote(export_remote_fpath)}"
                ),
                env=session_env,
            )
        except Exception:
            export_result = None
            print("Failed to export results", format_exc(), file=sys.stderr)
        if self.config.debug and export_result:
            print("Export stdout:\n", export_result.stdout, file=sys.stderr)
            print("Export stderr:\n", export_result.stderr, file=sys.stderr)

        results_root = Path(self.config.opencode_results_dir) if self.config.opencode_results_dir else Path(__file__).parent / "results"
        results_dir: Path = results_root / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        results_local_fpath = results_dir / export_fname
        if export_result is not None and export_result.return_code == 0:
            if self.config.debug:
                print(f"Downloading results from {export_remote_fpath} to {results_local_fpath}", file=sys.stderr)
            try:
                await sandbox.download(export_remote_fpath, results_local_fpath)
            except:
                print(f"Failed to download export results to {results_local_fpath}", format_exc(), file=sys.stderr)
                print("Export stdout:\n", export_result.stdout, file=sys.stderr)
                print("Export stderr:\n", export_result.stderr, file=sys.stderr)

        # Subagent sessions (the `task` tool), recursively: exported next to the main one, children/<session id>.json
        child_exports: Dict[str, Dict[str, Any]] = {}
        if results_local_fpath.exists():
            try:
                pending = _child_session_ids(json.loads(results_local_fpath.read_text().strip() or "{}"))
            except Exception:
                pending = []
            while pending and len(child_exports) < MAX_CHILD_SESSIONS:
                child_id = pending.pop(0)
                if child_id in child_exports:
                    continue
                child_remote = f"/tmp/opencode_child_{child_id}.json"
                child_local = results_dir / "children" / f"{child_id}.json"
                try:
                    exported = await sandbox.exec(
                        command=(
                            "export PATH=$HOME/.opencode/bin:$PATH "
                            f"&& opencode export {quote(child_id)} > {quote(child_remote)}"
                        ),
                        env=dict(xdg_env) or None,
                    )
                    if exported.return_code != 0:
                        print(f"Failed to export subagent session {child_id}: {exported.stderr}", file=sys.stderr)
                        continue
                    child_local.parent.mkdir(parents=True, exist_ok=True)
                    await sandbox.download(child_remote, child_local)
                    child_export = json.loads(child_local.read_text().strip() or "{}")
                except Exception:
                    print(f"Failed to export subagent session {child_id}", format_exc(), file=sys.stderr)
                    continue
                child_exports[child_id] = child_export
                pending.extend(c for c in _child_session_ids(child_export) if c not in child_exports)

        observations = None
        if collect_observations:
            assert remote_data_home is not None
            observations_remote_fpath = f"{remote_data_home}/opencode/opencode.db"
            snapshot_remote_fpath = f"{remote_data_home}/opencode/nemo-gym-observations.db"
            observations_local_fpath = results_dir / "opencode.db"
            observations_local_fpath.unlink(missing_ok=True)
            try:
                snapshot_script = (
                    "import sqlite3,sys;"
                    "source=sqlite3.connect(f'file:{sys.argv[1]}?mode=ro',uri=True);"
                    "destination=sqlite3.connect(sys.argv[2]);"
                    "source.backup(destination);destination.close();source.close()"
                )
                snapshot_result = await sandbox.exec(
                    command=(
                        f"python3 -c {quote(snapshot_script)} "
                        f"{quote(observations_remote_fpath)} {quote(snapshot_remote_fpath)}"
                    )
                )
                if snapshot_result.return_code != 0 or snapshot_result.error_type is not None:
                    raise RuntimeError("OpenCode database snapshot failed")
                await sandbox.download(snapshot_remote_fpath, observations_local_fpath)
                observations = parse_opencode_observations(
                    observations_local_fpath, observation_invocation_id, trajectory
                )
            except Exception:
                print("Failed to capture OpenCode observations", format_exc(), file=sys.stderr)
                trajectory.gaps.append(ObservationGap(code="turns_unavailable"))
                observations = AgentObservationBundle(
                    source="opencode",
                    records=[AgentInvocation(invocation_id=observation_invocation_id)],
                    gaps=[
                        ObservationGap(code="agent_artifact_unavailable"),
                        ObservationGap(code="agent_transcript_unavailable"),
                        ObservationGap(code="model_call_ownership_unavailable"),
                        ObservationGap(code="observation_capture_failed"),
                    ],
                )
            finally:
                observations_local_fpath.unlink(missing_ok=True)

        opencode_export = dict()
        if results_local_fpath.exists():
            opencode_export = json.loads(results_local_fpath.read_text().strip() or "{}")

        output = []
        usage = None
        opencode_export_found = False
        if opencode_export:
            opencode_export_found = True
            # Assume only one input message. May change with a system/developer message later on.
            output = self._opencode_export_to_output_items(opencode_export)[1:]
            usage = NeMoGymResponseUsage.sum_from_list(self._opencode_export_to_usages(opencode_export))

        result_stdout = (result.stdout if result else "") or ""
        result_stderr = (result.stderr if result else "") or ""
        opencode_finished = False
        std_out_split = result_stdout.rsplit("Shell: ", maxsplit=1)
        if len(std_out_split) > 1:
            opencode_finished = "OpenCode run finished" in std_out_split[1]

        if collect_observations and observations is not None:
            agent_sandbox_observation = self._agent_sandbox_observation(
                sandbox=sandbox,
                return_code=getattr(result, "return_code", None),
                error_type=getattr(result, "error_type", None) or run_error_type,
                finished=opencode_finished,
            )
            for record in observations.records:
                if isinstance(record, ToolCallObservation):
                    record.sandbox_id = agent_sandbox_observation.sandbox_id
                elif isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                    status = {
                        "completed": "completed",
                        "failed": "failed",
                        "sandbox_error": "failed",
                        "timeout": "incomplete",
                        "cancelled": "incomplete",
                    }.get(agent_sandbox_observation.outcome)
                    if status is not None:
                        record.status = status
            observations.records.append(agent_sandbox_observation)
            observations.gaps.append(ObservationGap(code="sandbox_lifecycle_timing_unavailable"))

        run_result = {
            "opencode_results_fpath": str(results_local_fpath) if opencode_export_found else "",
            "opencode_run_stdout": result_stdout,
            "opencode_run_stderr": result_stderr,
            "opencode_export_found": opencode_export_found,
            "opencode_finished": opencode_finished,
            "opencode_idle_nudges": idle_nudges,
            "opencode_stuck": stuck,
            "opencode_wall_exceeded": wall_exceeded,
            # Token use of the main agent vs its subagents (`usage` and the trajectory are the main session only)
            "opencode_tokens": _token_breakdown(opencode_export, child_exports),
            "opencode_subagent_sessions": list(child_exports),
        }
        if collect_observations:
            run_result["_ng_agent_observations"] = observations
            run_result["_ng_trajectory"] = trajectory
        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = run_result

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
        )

    async def run(
        self, request: Request, body: OpenCodeSandboxedAgentRunRequest
    ) -> OpenCodeSandboxedAgentVerifyResponse:
        cookies = request.cookies
        session_key = request.session[SESSION_ID_KEY]
        rollout_id = self.rollout_id_from_run(body)

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = cookies | seed_session_response.cookies

        # @bxyu-nvidia: "sandbox_handle" comes from resources_servers/swebench/app.py
        # Once we graduate to use the sandbox server, this will be in a generic seed_session type that can be model validated.
        seed_session_result = await seed_session_response.json()
        sandbox = await self._start_sandbox(
            sandbox_id=seed_session_result.get("sandbox_handle"),
            workdir=seed_session_result.get("workdir"),
        )
        self._sandbox_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox

        # Propagating the sandbox handle
        cookies["sandbox_id"] = session_key

        request._cookies = cookies
        request.state._ng_observation_invocation_id = rollout_id
        observations = None
        try:
            response = await self.responses(request, body.responses_create_params)
        finally:
            del request.state._ng_observation_invocation_id
            run_result = self._sandbox_id_to_run_result.get(session_key, {})
            observations = run_result.pop("_ng_agent_observations", None)

        verify_request = OpenCodeSandboxedAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)

        try:
            await sandbox.stop()
        except Exception:
            print("Failed to stop sandbox", format_exc(), file=sys.stderr)

        self._sandbox_id_to_sandbox.pop(session_key, None)

        # @bxyu-nvidia: This is scraped from the raw create params. Later on we can dynamically set this if OpenCode exports this :rofl:
        opencode_system_prompt = "You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.\n\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.\n\nIf the user asks for help or wants to give feedback inform them of the following:\n- /help: Get help with using opencode\n- To give feedback, users should report the issue at https://github.com/anomalyco/opencode/issues\n\nWhen the user directly asks about opencode (eg 'can opencode do...', 'does opencode have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from opencode docs at https://opencode.ai\n\n# Tone and style\nYou should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).\nRemember that your output will be displayed on a command line interface. Your responses can use GitHub-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.\nOutput text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.\nIf you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.\nOnly use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.\nIMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.\nIMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.\nIMPORTANT: Keep your responses short, since they will be displayed on a command line interface. You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail. Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as \"The answer is <answer>.\", \"Here is the content of the file...\" or \"Based on the information provided, the answer is...\" or \"Here is what I will do next...\". Here are some examples to demonstrate appropriate verbosity:\n<example>\nuser: what is 2+2?\nassistant: 4\n</example>\n\n<example>\nuser: is 11 a prime number?\nassistant: Yes\n</example>\n\n<example>\nuser: what command should I run to list files in the current directory?\nassistant: ls\n</example>\n\n<example>\nuser: what command should I run to watch files in the current directory?\nassistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]\nnpm run dev\n</example>\n\n<example>\nuser: what files are in the directory src/?\nassistant: [runs ls and sees foo.c, bar.c, baz.c]\nuser: which file contains the implementation of foo?\nassistant: src/foo.c\n</example>\n\n<example>\nuser: write tests for new feature\nassistant: [uses grep and glob search tools to find where similar tests are defined, uses concurrent read file tool use blocks in one tool call to read relevant files at the same time, uses edit file tool to write new tests]\n</example>\n\n# Proactiveness\nYou are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:\n1. Doing the right thing when asked, including taking actions and follow-up actions\n2. Not surprising the user with actions you take without asking\nFor example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.\n3. Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.\n\n# Following conventions\nWhen making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.\n- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).\n- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.\n- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.\n- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.\n\n# Code style\n- IMPORTANT: DO NOT ADD ***ANY*** COMMENTS unless asked\n\n# Doing tasks\nThe user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:\n- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.\n- Implement the solution using all tools available to you\n- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.\n- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (e.g. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to AGENTS.md so that you will know to run it next time.\nNEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.\n\n- Tool results and user messages may include <system-reminder> tags. <system-reminder> tags contain useful information and reminders. They are NOT part of the user's provided input or the tool result.\n\n# Tool usage policy\n- When doing file search, prefer to use the Task tool in order to reduce context usage.\n- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel. For example, if you need to run \"git status\" and \"git diff\", send a single message with two tool calls to run the calls in parallel.\n\nYou MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.\n\nIMPORTANT: Before you begin work, think about what the code you're editing is supposed to do based on the filenames directory structure.\n\n# Code References\n\nWhen referencing specific functions or pieces of code include the pattern `file_path:line_number` to allow the user to easily navigate to the source code location.\n\n<example>\nuser: Where are errors from the client handled?\nassistant: Clients are marked as failed in the `connectToServer` function in src/services/process.ts:712.\n</example>\n\nYou are powered by the model named dummy_model. The exact model ID is nemo_gym/dummy_model\nHere is some useful information about the environment you are running in:\n<env>\n  Working directory: /testbed\n  Workspace root folder: /testbed\n  Is directory a git repo: yes\n  Platform: linux\n  Today's date: Tue Aug 04 2026\n</env>\nSkills provide specialized instructions and workflows for specific tasks.\nUse the skill tool to load a skill when a task matches its description.\n<available_skills>\n  <skill>\n    <name>customize-opencode</name>\n    <description>Use ONLY when the user is editing or creating opencode's own configuration: opencode.json, opencode.jsonc, files under .opencode/, or files under ~/.config/opencode/. Also use when creating or fixing opencode agents, subagents, skills, plugins, MCP servers, or permission rules. Do not use for the user's own application code, or for any project that is not configuring opencode itself.</description>\n    <location>file:///testbed/%3Cbuilt-in%3E</location>\n  </skill>\n</available_skills>"

        response_dict = await get_response_json(verify_response)
        run_result = self._sandbox_id_to_run_result.pop(session_key)
        trajectory = run_result.pop("_ng_trajectory", None)
        if trajectory is not None and rollout_id is not None:
            response_dict["ng_trajectory"] = scope_opencode_trajectory(trajectory, body, rollout_id).model_dump(
                mode="json"
            )
        response_dict |= run_result
        raw_verifier_sandbox_observation = response_dict.pop("verifier_sandbox_observation", None)
        response_dict["responses_create_params"]["input"].insert(
            0, {"content": opencode_system_prompt, "role": "system"}
        )

        if rollout_id is not None:
            if observations is None:
                observations = AgentObservationBundle(
                    source="opencode",
                    records=[AgentInvocation(invocation_id=rollout_id)],
                    gaps=[ObservationGap(code="observation_capture_failed")],
                )
            if raw_verifier_sandbox_observation is not None:
                try:
                    verifier_observation = SandboxObservation.model_validate(raw_verifier_sandbox_observation)
                    if verifier_observation.role != "verifier":
                        raise ValueError("resources server returned a non-verifier sandbox observation")
                    observations.records.append(verifier_observation)
                except Exception:
                    observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_invalid"))
            else:
                observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_unavailable"))
            response_dict["ng_agent_observations"] = observations.model_dump(mode="json")
        return OpenCodeSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    OpenCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeSandboxedAgent.run_webserver()  # noqa: F401
