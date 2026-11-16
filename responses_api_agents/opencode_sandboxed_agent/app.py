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
import binascii
import hashlib
import json
import logging
import re
import sqlite3
import sys
import tempfile
from asyncio import Semaphore
from contextlib import nullcontext
from copy import deepcopy
from functools import cache
from pathlib import Path
from random import uniform
from shlex import quote
from tempfile import TemporaryDirectory
from time import monotonic, time
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Literal, NamedTuple, Optional
from uuid import uuid4

from anyio import CancelScope
from fastapi import Request
from openai.types.responses import ResponseInputTextParam
from pydantic import ConfigDict, Field, FilePath

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentOutcome,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
    restricted_network_policy,
    sandbox_server_url,
    seed_mcp_servers,
    verification_failure_fields,
    verify_agent_response,
)
from nemo_gym.client import SESSION_ID_KEY, get_server_url, raise_for_status
from nemo_gym.config import (
    ModelServerRef,
    ResourcesServerRef,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.core import (
    AGENT_ROUTE_FAILED,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_OOM_KILLED,
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
    NgFailureEvent,
    SandboxFailureDetails,
    agent_route_failure,
    observe_failure,
)
from nemo_gym.core.failure_kinds import PROVIDER_COMMAND_OUTCOME_UNKNOWN
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
    sandbox_failure,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.trajectory import (
    NG_AGENT_SANDBOX_OBSERVATION_KEY,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
    TrajectoryRecord,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint, is_nemo_gym_fastapi_worker
from nemo_gym.utils.exception_inspection import error_to_http_status, provider_error_reason, sandbox_failure_prefix
from responses_api_agents.opencode_agent.observability import (
    append_opencode_turns,
    opencode_tool_outcome,
    scope_opencode_trajectory,
)
from responses_api_agents.opencode_agent.runtime import (
    OBSERVABILITY_PATCH,
    OPENCODE_VERSION,
    apply_observability_patch,
    command_failure_reason,
)


LOGGER = logging.getLogger(__name__)

DEPRECATION_WARNING = (
    "DEPRECATED: opencode_sandboxed_agent runs each episode through its own legacy /run path and will be removed "
    "after the environment-server migration. Where the resources server returns sandbox_access, switch to a session "
    "harness in its allowed_agents under an environment server: opencode_agent where listed (Terminal Bench 2.1), "
    "otherwise another such as pi_agent or codex_agent (SWE-bench Pro); for vibench_agent, that is "
    "resources_servers/vibench/configs/vibench.yaml. Setups with no session path yet keep working here: SWE "
    "resources servers that return only a sandbox_handle (swebench, swe_gym, r2e_gym and others) and the HLE search "
    "presets (tool_servers). "
    "See responses_api_agents/opencode_agent/README.md#environmentserver-sessions."
)


@cache
def _warn_deprecated_once() -> None:
    # Workers of a multi-worker server rebuild it after the main process has already logged this.
    if not is_nemo_gym_fastapi_worker():
        LOGGER.warning(DEPRECATION_WARNING)


_RIPGREP_UPLOAD_MAX_ATTEMPTS = 2
_RIPGREP_UPLOAD_MAX_CONCURRENT = 2

_REMOTE_ASSISTANT_MESSAGE_PLUGIN = "/tmp/nemo-gym-opencode-assistant-message-header.js"
# Image parts of the user turn are uploaded here and handed to `opencode run -f`. Outside the task
# workdir on purpose (see the export path below) and fixed because each rollout owns its sandbox.
_REMOTE_ATTACHMENTS_DIR = "/tmp/nemo-gym-attachments"
# The image formats OpenCode 1.17.11's Read tool recognises by magic bytes (SUPPORTED_IMAGE_MIMES in
# packages/opencode/src/tool/read.ts). `opencode run -f` labels a local file text/plain and routes it
# through that tool (packages/opencode/src/cli/cmd/run.ts, session/prompt.ts), which re-sniffs the
# bytes and attaches a data: file part; any other format is read as a binary file and fails.
_ATTACHMENT_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}


class _ImageAttachment(NamedTuple):
    """One decoded `input_image` part of the user turn."""

    remote_path: str
    content: bytes


def _decode_image_part(part: Dict[str, Any], index: int) -> _ImageAttachment:
    """Turn one `input_image` part into bytes for the sandbox, failing closed on anything OpenCode cannot show."""
    image_url = part.get("image_url")
    if not image_url:
        raise ValueError(f"input_image part {index} has no image_url; file_id inputs are not supported")
    if not image_url.startswith("data:"):
        raise ValueError(
            f"input_image part {index} is a remote URL; the OpenCode sandbox has no internet access, so only "
            "data: URIs are supported (prepare the dataset with the image inlined)"
        )
    header, separator, payload = image_url[len("data:") :].partition(",")
    media_type, *parameters = header.split(";")
    if not separator or "base64" not in parameters:
        raise ValueError(f"input_image part {index} is not a base64 data URI")
    extension = _ATTACHMENT_EXTENSIONS.get(media_type.strip().lower())
    if extension is None:
        raise ValueError(
            f"input_image part {index} has media type {media_type!r}; OpenCode attaches only "
            + ", ".join(sorted(_ATTACHMENT_EXTENSIONS))
        )
    try:
        content = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"input_image part {index} carries invalid base64") from exc
    return _ImageAttachment(remote_path=f"{_REMOTE_ATTACHMENTS_DIR}/image-{index}.{extension}", content=content)


def _parse_user_turn(input_items: Any) -> tuple[str, List[_ImageAttachment]]:
    """The single user turn as OpenCode receives it: its text as the query, its images as attachments.

    A string or single `input_text` turn is passed through unchanged. System and developer items are
    not forwarded (OpenCode keeps its own system prompt); a dataset that needs instructions folds them
    into the user turn, as benchmarks/hle_diamond/prompts/opencode.yaml does.
    """
    query = None
    attachments: List[_ImageAttachment] = []
    for input_item in input_items:
        if input_item.role == "user":
            assert not query, input_items
            if isinstance(input_item.content, str):
                query = input_item.content
            elif isinstance(input_item.content, list):
                texts = [part for part in input_item.content if part["type"] == "input_text"]
                images = [part for part in input_item.content if part["type"] == "input_image"]
                assert len(texts) == 1 and len(texts) + len(images) == len(input_item.content), input_items
                query = texts[0]["text"]
                attachments = [_decode_image_part(part, index) for index, part in enumerate(images, start=1)]

    assert query, input_items
    return query, attachments


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
    db_path: Path,
    fallback_invocation_id: str,
    trajectory: Optional[TrajectoryRecord] = None,
    *,
    model_ref: ModelServerRef | None = None,
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
    summaries_by_parent: dict[tuple[str, str], list[str]] = {}
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
        if message.get("role") == "assistant" and message.get("summary") is True:
            summary_text[row["id"]] = []
            parent_id = message.get("parentID")
            if isinstance(parent_id, str):
                summaries_by_parent.setdefault((session_id, parent_id), []).append(row["id"])

    for row in part_rows:
        part = _load_json(row["data"])
        if not part:
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable"))
            # The part may have been a tool call, so the tool history read is not complete.
            gaps.append(ObservationGap(code="tool_timing_unavailable", detail=row["id"]))
            continue
        ptype = part.get("type")
        message_id = row["message_id"]
        message = messages.get(message_id, {})
        session_id = row["session_id"] or message_sessions.get(message_id)
        if not isinstance(session_id, str):
            gaps.append(ObservationGap(code="agent_artifact_record_unowned"))
            if ptype == "tool":
                # A tool part that belongs to no session is left out, so the tool history read is not complete.
                gaps.append(ObservationGap(code="tool_timing_unavailable", detail=row["id"]))
            continue
        conversation = conversations.setdefault(session_id, [])
        role = message.get("role")
        if ptype == "tool" and role not in ("assistant", "user"):
            # A tool part whose message cannot say who sent it is left out, so the tool history read is not complete.
            gaps.append(ObservationGap(code="tool_timing_unavailable", invocation_id=session_id, detail=row["id"]))

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
            outcome = opencode_tool_outcome(part.get("tool"), state)
            if outcome.output is not None:
                result = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=outcome.output,
                    status=response_status,
                )
                conversation.append(result)
            if outcome.gap is not None:
                gaps.append(ObservationGap(code=outcome.gap, invocation_id=session_id, detail=observed_call_id))

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
            status = outcome.status
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
                        error_type=outcome.error_type,
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
                # The tool ran but has no observation to time, so the tool history read is not complete.
                gaps.append(
                    ObservationGap(
                        code="tool_timing_unavailable",
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
                child_status[child_id] = {"timeout": "incomplete", "cancelled": "incomplete"}.get(status, status)
            continue
        if ptype == "compaction":
            compaction_parts.append((session_id, message_id, _milliseconds(row["time_created"]), part))
            continue

    compactions: list[ContextCompactionObservation] = []
    for session_id, message_id, observed_at, part in compaction_parts:
        summary_ids = summaries_by_parent.get((session_id, message_id), [])
        summary = "\n".join(summary_text.get(summary_ids[0], [])) if len(summary_ids) == 1 else None
        if len(summary_ids) > 1:
            gaps.append(
                ObservationGap(
                    code="compaction_summary_ambiguous",
                    invocation_id=session_id,
                    detail=",".join(summary_ids),
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
                source_message_ids=summary_ids,
                source_model_ref=model_ref,
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
        append_opencode_turns(trajectory, session_ids, message_rows, part_rows, model_ref=model_ref)

    return AgentObservationBundle(
        source="opencode",
        records=[*invocations, *tools, *compactions],
        gaps=gaps,
    )


# OpenCode's `attachment.image` limits (config schema packages/core/src/v1/config/attachment.ts at v1.17.11):
# an image within all three is forwarded unchanged. 2**31 - 1 is the largest value its PositiveInt accepts
# without a JSON precision caveat and is far above any image a chat-completions request can carry.
OPENCODE_IMAGE_LIMITS_UNRESTRICTED = {
    "max_width": 2**31 - 1,
    "max_height": 2**31 - 1,
    "max_base64_bytes": 2**31 - 1,
}


class OpenCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    opencode_version: str = OPENCODE_VERSION
    remote_opencode_install_script_path: Optional[str] = None
    remote_opencode_binary_path: Optional[str] = None
    remote_opencode_musl_binary_path: Optional[str] = None
    local_ripgrep_binary_path: FilePath | None = None
    opencode_config: Dict[str, Any] = Field(default_factory=dict)
    opencode_max_context_window: int
    concurrency: Optional[int] = Field(default=None)
    preinstalled_opencode: bool = False
    execution_failure_reward_zero: bool = False
    # Opt-in mode gives setup and agent execution separate sandbox_timeout budgets.
    report_agent_sandbox_oom: bool = False
    output_token_policy: Literal["fixed", "remaining_context"] = "fixed"
    network_access: Literal["inherit", "model_only", "model_and_tools"] = "inherit"
    tool_servers: List[ResourcesServerRef] = Field(default_factory=list)
    artifacts_dir: Optional[str] = None
    opencode_model_call_timeout: Optional[int] = None

    # Environment variables exported in front of `opencode run` inside the sandbox.
    #
    # Several opencode behaviours are reachable ONLY from the process environment, not from
    # the JSON config. The one this exists for is the builtin `websearch` tool: it is filtered
    # out of the model's tool list unless the provider id is literally "opencode" or one of
    # OPENCODE_ENABLE_EXA / OPENCODE_EXPERIMENTAL / OPENCODE_EXPERIMENTAL_EXA is set
    # (tool/registry.ts `webSearchEnabled`, fed from effect/runtime-flags.ts, i.e. env). Our
    # provider id is `nemo_gym`, so `tools: {websearch: true}` alone is a silent no-op -- the
    # model never sees the tool and falls back to webfetch-ing google.com/search.
    #
    # Values are shell-quoted individually. Keys are not validated: this is a raw passthrough.
    opencode_env: Dict[str, str] = Field(default_factory=dict)

    # Pass the user message on stdin instead of as a positional argv.
    #
    # `opencode run` re-wraps its message argv: cli/cmd/run.ts does
    #   [...args.message, ...(args["--"] ?? [])]
    #     .map((arg) => (arg.includes(" ") ? `"${arg.replace(/"/g, '\\"')}"` : arg)).join(" ")
    # so a single argv containing spaces -- which is every natural-language question -- reaches
    # the model wrapped in double quotes with every internal quote backslash-escaped. Verified
    # on this stack: all 400 BrowseComp rows arrived starting with a stray `"`, and 30 of 400
    # had internal quotes mangled. Piping instead takes run.ts's `resolveRunInput(value, piped)`
    # branch, which returns the piped text verbatim when the argv message is empty.
    opencode_query_via_stdin: bool = False

    # Drop auto-compaction summaries from the converted output items. They are assistant messages
    # like any other, so a run that ends on a compaction otherwise hands a verifier that grades the
    # last assistant message (e.g. an answer judge) the summary instead of the answer. Off by
    # default: a patch-graded benchmark is unaffected and should keep the full transcript. The raw
    # export.json is downloaded either way, so nothing is lost for offline analysis.
    exclude_compaction_summaries: bool = False

    # Plugin .js files under plugins/ to upload into the sandbox and register with opencode.
    # Plugins are how we change opencode's behaviour without rebuilding the binary: opencode
    # imports them into its own process and calls them at defined hook points, so they can edit
    # the live request, replace a builtin tool, or end a run. See plugins/ for what each does.
    opencode_plugins: List[str] = Field(default_factory=list)

    # Sandbox config
    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    sandbox_timeout: float

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


def _read_opencode_child_messages(db_path: Path, root_session_id: str) -> list[dict[str, Any]]:
    """Read each descendant message once, excluding the separately exported root."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            with recursive descendants(id) as (
                select id from session where id = ?
                union
                select session.id from session join descendants on session.parent_id = descendants.id
            )
            select message.data from message join descendants on message.session_id = descendants.id
            where message.session_id != ?
            order by message.time_created, message.id
            """,
            (root_session_id, root_session_id),
        ).fetchall()
    finally:
        con.close()
    return [{"info": json.loads(row[0])} for row in rows]


def _terminal_reason(run_output: str) -> str:
    """How the rollout ended, as reported by nemo_gym_turn_budget.js. "" when it ended normally.

    Two signals, deliberately combined here rather than in the plugin:

    `context_overflow` / `context_overflow_recovered` -- stock publishes the SAME
    ContextOverflowError event whether the overflow killed the session or merely triggered an
    emergency compaction (processor.ts:926-935 branches on compaction.auto but publishes in both).
    With compaction on, the common case is recovery, so the plugin arms a marker on the error and
    cancels it on the next model call for that session. An overflow still armed at the end is the
    only one that actually ended the run. Getting this wrong is not cosmetic: context_window masks
    a row out of scoring, so a false positive silently deletes a healthy rollout from the
    denominator.

    Precedence follows the fork's bench/terminal_error.ts prefer(): context_window outranks
    max_iteration, because a run that overflowed was cut off mid-thought by an infrastructure
    limit, which is the truer description than "spent its turn budget".
    """
    armed = 0
    max_iteration = False
    for line in run_output.splitlines():
        if "[nemo-gym-plugin] " not in line:
            continue
        if "context_overflow_recovered" in line:
            armed = max(0, armed - 1)
        elif "context_overflow " in line or line.rstrip().endswith("context_overflow"):
            armed += 1
        elif "terminal=" in line:
            reason = line.split("terminal=", 1)[1].split()[0]
            if reason.startswith("max_iteration"):
                max_iteration = True
    if armed > 0:
        return "context_window"
    if max_iteration:
        return "max_iteration"
    return ""


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
    opencode_exit_code: Optional[int] = None
    opencode_error_type: Optional[str] = None
    opencode_failed: bool = False
    # Images of the user turn uploaded into the sandbox and attached with `opencode run -f`.
    opencode_attachment_count: int = 0
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class OpenCodeSandboxedAgent(SimpleResponsesAPIAgent):
    config: OpenCodeSandboxedAgentConfig
    # The harness runs inside the task sandbox and calls the model server from there.
    calls_model_from_sandbox: ClassVar[bool] = True

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        _warn_deprecated_once()

        self._sem = Semaphore(self.config.concurrency) if self.config.concurrency else nullcontext()
        # Uvicorn workers have separate agent instances, so this bounds each process.
        self._ripgrep_upload_sem = Semaphore(_RIPGREP_UPLOAD_MAX_CONCURRENT)
        self._sandbox_id_to_sandbox: Dict[str, AsyncSandbox] = dict()
        self._sandbox_id_to_run_result: Dict[str, Dict[str, Any]] = dict()

    async def _start_sandbox(self, sandbox_id: Optional[str] = None, workdir: Optional[str] = None) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        )
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        if sandbox_id:
            if self.config.network_access != "inherit":
                raise ValueError("Cannot verify network policy on an externally supplied sandbox")
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
            image=self.config.sandbox_config.get("image", "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907"),
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=self.config.sandbox_config.get("workdir"),
            env=env,
            files=dict(self.config.sandbox_config.get("files", {})),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
            },
            resources=resources,
            entrypoint=self.config.sandbox_config.get("entrypoint"),
            provider_options=deepcopy(self.config.sandbox_config.get("provider_options", {})),
        )

        if self.config.network_access != "inherit":
            urls = [sandbox_server_url(self.config.model_server.name, require_reachable=True)]
            if self.config.network_access == "model_and_tools":
                if not self.config.tool_servers:
                    raise ValueError("model_and_tools requires tool_servers")
                urls.extend(
                    sandbox_server_url(server.name, require_reachable=True) for server in self.config.tool_servers
                )
            sandbox_spec.provider_options["network_policy"] = restricted_network_policy(
                resolved_sandbox_provider.name, urls
            )
        sandbox = AsyncSandbox(resolved_sandbox_provider)
        await sandbox.start(sandbox_spec)

        return sandbox

    async def _stage_ripgrep(self, sandbox: AsyncSandbox, source_path: Path) -> str:
        """Retry an ambiguous upload timeout without reusing its remote path."""
        size_bytes = source_path.stat().st_size
        sandbox_id = sandbox.sandbox_id
        budget = asyncio.timeout(self.config.sandbox_timeout)
        try:
            async with budget:
                for attempt in range(1, _RIPGREP_UPLOAD_MAX_ATTEMPTS + 1):
                    remote_path = f"/tmp/nemo-gym-ripgrep-{uuid4().hex}"
                    started = monotonic()
                    try:
                        async with self._ripgrep_upload_sem:
                            await sandbox.upload(source_path, remote_path)
                    except (TimeoutError, NgFailureEvent) as error:
                        if isinstance(error, NgFailureEvent) and (
                            error.boundary != "sandbox" or error.failure_kind != PROVIDER_IO_TIMEOUT
                        ):
                            raise
                        if isinstance(error, NgFailureEvent):
                            observe_failure(error)
                        LOGGER.warning(
                            "OpenCode ripgrep upload timed out; sandbox_id=%s remote_path=%s "
                            "attempt=%d/%d size_bytes=%d duration_s=%.1f",
                            sandbox_id,
                            remote_path,
                            attempt,
                            _RIPGREP_UPLOAD_MAX_ATTEMPTS,
                            size_bytes,
                            monotonic() - started,
                        )
                        if attempt == _RIPGREP_UPLOAD_MAX_ATTEMPTS:
                            raise
                        await asyncio.sleep(uniform(15.0, 30.0))
                    else:
                        LOGGER.info(
                            "OpenCode ripgrep upload completed; sandbox_id=%s remote_path=%s "
                            "attempt=%d/%d size_bytes=%d duration_s=%.1f",
                            sandbox_id,
                            remote_path,
                            attempt,
                            _RIPGREP_UPLOAD_MAX_ATTEMPTS,
                            size_bytes,
                            monotonic() - started,
                        )
                        return remote_path

                raise RuntimeError("OpenCode ripgrep upload retry loop did not run")
        except TimeoutError as error:
            reason = (
                "OpenCode ripgrep staging exceeded its setup budget"
                if budget.expired()
                else "OpenCode ripgrep upload exhausted its timeout retries"
            )
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.setup",
                stage="seed",
                failure_reason=f"{reason}; sandbox_id={sandbox_id}",
            ) from error

    def _plugin_uploads(self) -> list[tuple[Path, str]]:
        """Every opencode plugin this rollout ships, as (local file, sandbox path), in load order.

        One list with two consumers. `_create_opencode_config` registers the sandbox paths in the
        `plugin` array and `responses` uploads the files. The registration site has no sandbox,
        which is fine: a sandbox path is a deterministic function of the plugin name, so both
        sites can derive it from this method and a registered plugin cannot drift away from an
        uploaded one. That drift is the failure this guards -- opencode silently ignores a
        `plugin` entry whose file is absent.

        The two groups are plugins in exactly the same sense: `required-mcp.js` and
        `remaining-context.js` each export a hook factory and reach opencode through the same
        `plugin` array as the files under `plugins/`. They differ only in where the repo keeps
        them, which is why their sandbox paths keep distinct prefixes -- `plugins/` is
        config-named and open-ended, so a future `plugins/remaining-context.js` must not land on
        top of this directory's copy.

        Config-named plugins come first. `nemo_gym_sentinel.js` heads that list, and the point of
        the sentinel is to load ahead of the guards whose presence it attests to.
        """
        plugin_dir = Path(__file__).parent / "plugins"
        uploads: list[tuple[Path, str]] = []
        for name in self.config.opencode_plugins:
            local = plugin_dir / name
            if not local.is_file():
                raise FileNotFoundError(f"opencode plugin not found: {local}")
            uploads.append((local, f"/tmp/nemo-gym-plugin-{name}"))
        policy_plugins = []
        if self.config.output_token_policy == "remaining_context":
            policy_plugins.append("remaining-context.js")
        if self.config.tool_servers:
            policy_plugins.append("required-mcp.js")
        uploads.extend((Path(__file__).with_name(name), f"/tmp/nemo-gym-{name}") for name in policy_plugins)
        return uploads

    @staticmethod
    def _agent_sandbox_facts(sandbox: AsyncSandbox, **facts: Any) -> str:
        """The failure_reason prefix for the agent's own command: its phase, provider, sandbox and command."""
        handle = getattr(sandbox, "_handle", None)
        provider = getattr(handle, "provider_name", None)
        sandbox_id = getattr(handle, "sandbox_id", None)
        return sandbox_failure_prefix(
            phase="agent",
            provider=provider if isinstance(provider, str) else None,
            sandbox_id=sandbox_id if isinstance(sandbox_id, str) else None,
            command="opencode_run",
            **facts,
        )

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
        if "timeout" in normalized_error or (not normalized_error and return_code == 124):
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
            stage="agent",
            provider=handle_provider if isinstance(handle_provider, str) else None,
            sandbox_id=handle_sandbox_id if isinstance(handle_sandbox_id, str) else None,
            outcome=outcome,
            exit_code=return_code if not normalized_error and isinstance(return_code, int) else None,
            error_type=error_type if isinstance(error_type, str) else None,
        )

    async def _create_opencode_config(self, request: Request, *, image_input: bool = False) -> Dict[str, Any]:
        """The OpenCode config for this run; ``image_input`` declares the model vision-capable so OpenCode
        forwards attached images instead of replacing them with "this model does not support image input"
        (``unsupportedParts`` in OpenCode's provider/transform.ts reads ``capabilities.input.image``, which
        provider.ts derives from the model's ``modalities.input``)."""
        base_url = (
            self.base_url_for_run(
                base_url=sandbox_server_url(
                    self.config.model_server.name, require_reachable=self.config.network_access != "inherit"
                ),
                body=await request.json(),
            )
            + "/v1"
        )
        config: Dict[str, Any] = {
            "model": "nemo_gym/dummy_model",
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "nemo_gym": {
                    # TODO @bxyu-nvidia: We should use @ai-sdk/openai here but there is some /v1/responses streaming error.
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": "dummy_key",  # pragma: allowlist secret
                        "chunkTimeout": int(self.config.sandbox_timeout * 1000),
                        "timeout": self.config.opencode_model_call_timeout
                        if self.config.opencode_model_call_timeout is not None
                        else False,  # milliseconds
                    },
                    "models": {
                        "dummy_model": {
                            "temperature": True,
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
        }
        if image_input:
            # Only when the turn carries images: with it, OpenCode would also send any image the agent's
            # Read tool opens during the run, which a text-only model cannot take.
            config["provider"]["nemo_gym"]["models"]["dummy_model"]["modalities"] = {
                "input": ["text", "image"],
                "output": ["text"],
            }
            # OpenCode v1.17.11 image/image.ts downscales any image over 2000 px on a side or 5 MiB of
            # base64 and re-encodes it (PNG, else JPEG at quality 80..40) when its bundled photon resizer
            # loads. Lifting the limits makes `normalize` return the part untouched, so the model receives
            # the caller's bytes as a direct image_url input would carry them. `auto_resize: false` is not
            # the knob for that: it turns the resize into a SizeError that fails the turn.
            config["attachment"] = {"image": dict(OPENCODE_IMAGE_LIMITS_UNRESTRICTED)}

        # Recursive merge, not a top-level dict spread. With a spread, a benchmark that sets
        # `opencode_config.provider` (to reach e.g. models.dummy_model.limit.input, the real
        # compaction-trigger knob, or .interleaved for reasoning propagation) silently replaces
        # the whole provider block above and loses baseURL/apiKey. Matches the merge semantics on
        # Gym main. `provider` is the block that actually has generated siblings to lose; keys
        # like `agent` and `compaction` are not generated here, so they merge in whole.
        # None and "" mean "not set" and are skipped. Hydra configs are struct-locked, so a key a
        # benchmark wants to override must already exist in the base agent yaml; declaring it empty
        # there is how it stays available without being forced on every other benchmark. An empty
        # string is never a meaningful opencode value, whereas false and 0 are, so only ""/None are
        # dropped.
        def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
            for key, value in override.items():
                if value is None or value == "":
                    continue
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    _merge(base[key], value)
                elif isinstance(value, dict):
                    pruned: Dict[str, Any] = {}
                    _merge(pruned, value)
                    if pruned:
                        base[key] = pruned
                else:
                    base[key] = deepcopy(value)

        _merge(config, self.config.opencode_config)
        config.setdefault("plugin", []).extend(f"file://{remote}" for _, remote in self._plugin_uploads())
        rollout_mcp = getattr(request.state, "_ng_opencode_mcp", None)
        if isinstance(rollout_mcp, dict):
            config.setdefault("mcp", {}).update(rollout_mcp)
        return config

    async def _seed_tool_servers(self, request: Request, body: OpenCodeSandboxedAgentRunRequest) -> Dict[str, Any]:
        entries = await seed_mcp_servers(
            self.server_client,
            self.config.tool_servers,
            body,
            request.cookies,
            timeout_s=self.config.sandbox_timeout,
            require_reachable=self.config.network_access != "inherit",
        )
        return {name: {"type": "remote", **entry} for name, entry in entries.items()}

    def _resolve_opencode_env(self, query: str) -> Dict[str, str]:
        """`opencode_env` with any `EXA_API_KEYS` pool collapsed to one per-row `EXA_API_KEY`.

        opencode reads a single `EXA_API_KEY` once at module load (tool/mcp-websearch.ts builds
        `EXA_URL` from it in a module-scope const), so the pool has to be resolved out here. The
        pick is a stable hash of the question rather than a counter: every row of a benchmark then
        lands on the same key across retries and across reruns, which keeps per-key rate-limit
        behaviour reproducible instead of depending on arrival order. Pool count is decoupled from
        shard/parallelism count, which is the bug the legacy harness hit by pinning a key per shard.
        """
        # Empty values are dropped, not exported. Hydra configs are struct-locked, so every key a
        # benchmark might set has to be declared in the base agent yaml; declaring them empty there
        # is how a key stays available without being switched on for every other benchmark.
        env = {k: v for k, v in self.config.opencode_env.items() if v != ""}

        # Point websearch/webfetch at the resources server's Exa proxy when it is running one, so
        # every shard and every sandbox shares one cache. Injected here rather than set in the
        # benchmark yaml because the address is only known at run time, and this is already the
        # function that resolves a per-row value into the sandbox environment.
        #
        # Opt-in and fail-safe: if the search server has no cache root configured the route does
        # not exist, we leave the variable unset, and the plugin goes straight to Exa exactly as
        # before. A benchmark that sets NEMO_GYM_EXA_BASE_URL explicitly wins over this.
        if not env.get("NEMO_GYM_EXA_BASE_URL"):
            try:
                cache_root = (
                    get_first_server_config_dict(get_global_config_dict(), self.config.resources_server.name) or {}
                ).get("exa_cache_root")
                if cache_root:
                    env["NEMO_GYM_EXA_BASE_URL"] = get_server_url(self.config.resources_server.name) + "/mcp"
            # SystemExit is deliberate alongside Exception. get_global_config_dict() loads the
            # Hydra config, and Hydra reports a resolution failure by catching its own
            # MissingConfigException and calling sys.exit(1) -- a BaseException that `except
            # Exception` does not stop. Without SystemExit here, a best-effort cache lookup can
            # terminate the rollout it was only ever meant to optimise.
            except (Exception, SystemExit) as e:  # noqa: BLE001 - never let cache wiring break a rollout
                print(
                    f"[opencode_sandboxed_agent] exa cache URL lookup failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        pool_raw = env.pop("EXA_API_KEYS", "")
        pool = [k.strip() for k in re.split(r"[:,\s]+", pool_raw) if k.strip()]
        if pool and not env.get("EXA_API_KEY"):
            digest = hashlib.sha256(query.encode("utf-8")).digest()
            env["EXA_API_KEY"] = pool[int.from_bytes(digest[:8], "big") % len(pool)]
        return env

    def _opencode_export_to_usages(self, opencode_export: Dict[str, Any]) -> List[NeMoGymResponseUsage]:
        usages: List[NeMoGymResponseUsage] = []
        for message in opencode_export["messages"]:
            if message["info"]["role"] != "assistant":
                continue

            token_info = message["info"].get("tokens")
            if not token_info:
                continue

            # OpenCode separates cache reads and writes from uncached input; Responses includes all three.
            usage = NeMoGymResponseUsage(
                input_tokens=token_info["input"] + token_info["cache"]["read"] + token_info["cache"]["write"],
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=token_info["cache"]["read"]),
                output_tokens=token_info["output"] + token_info["reasoning"],
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
                # Compaction summaries carry summary=true and are authored by the "compaction"
                # agent; both markers are present on exports from opencode 1.17.x.
                # getattr: this converter is also exercised as a pure function with self=None.
                config = getattr(self, "config", None)
                if (
                    config is not None
                    and config.exclude_compaction_summaries
                    and (message["info"].get("summary") is True or message["info"].get("agent") == "compaction")
                ):
                    continue

                converter = ChatCompletionsConverter(return_token_id_information=True)
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

    async def _upload_plugin_verified(self, sandbox: Any, local: Path, remote: str, attempts: int = 3) -> None:
        """Upload one plugin and READ IT BACK, retrying until the bytes in the sandbox match.

        Why a read-back rather than trusting the write. opencode swallows plugin load failures, so
        a plugin that never arrived produces a run that looks entirely healthy; the sentinel gate
        then discards the row with no trajectory, and we cannot tell afterwards whether the
        plugins failed or the pod died. One 400-row run lost about a tenth of its rows that way,
        spread evenly across the dataset and arriving in bursts -- the shape of a transient
        infrastructure fault, not a bad question.

        A hard upload error already raises. What this adds is the silent case: a write that
        reports success but leaves the file absent, short or empty. On this cell sandbox creation
        returns at pod-SCHEDULED rather than pod-READY, so an early write is exactly the kind of
        operation that can be accepted and then not land.

        Verifying costs one small read per plugin -- under 40 KB across the whole list -- against
        a rollout that runs for minutes. Cheap insurance against losing an eighth of a run.
        """
        expected = local.read_bytes()
        last = ""
        for attempt in range(attempts):
            try:
                await sandbox.upload(local, remote)
                with TemporaryDirectory() as tmp:
                    check = Path(tmp) / local.name
                    await sandbox.download(remote, check)
                    got = check.read_bytes()
                if got == expected:
                    return
                last = f"size mismatch: wrote {len(expected)} bytes, read back {len(got)}"
            except Exception as e:  # noqa: BLE001 - any failure here is retryable
                last = f"{type(e).__name__}: {e}"
            print(
                f"[opencode_sandboxed_agent] plugin upload verify failed for {remote} "
                f"(attempt {attempt + 1}/{attempts}): {last}",
                file=sys.stderr,
                flush=True,
            )
            if attempt + 1 < attempts:
                await asyncio.sleep(2**attempt)
        # Fail loudly rather than letting opencode start with a plugin list pointing at a file
        # that is not there -- that is the silent mode this whole method exists to prevent.
        raise RuntimeError(f"opencode plugin {local.name} did not land in the sandbox at {remote}: {last}")

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        if self.config.tool_servers and not isinstance(getattr(request.state, "_ng_opencode_mcp", None), dict):
            raise ValueError("Configured tool servers require a seeded /run request")
        sandbox = self._sandbox_id_to_sandbox[request.cookies["sandbox_id"]]

        # Fails before anything is uploaded or run, so an unsupported image fails the row explicitly.
        query, attachments = _parse_user_turn(body.input)
        # Empty without images, so a text-only turn produces exactly the launch command it always did.
        opencode_attachment_str = "".join(f" -f {quote(attachment.remote_path)}" for attachment in attachments)

        opencode_debug_str = ""
        if self.config.debug:
            opencode_debug_str = "--print-logs --log-level DEBUG"

        opencode_thinking_str = "--thinking"

        install_str = ":"
        if not self.config.preinstalled_opencode:
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

        effective_config = await self._create_opencode_config(request, image_input=bool(attachments))
        # Uploaded per rollout rather than baked into the image, so a plugin edit ships with a
        # config change and no image rebuild. Kept in /tmp, outside any task directory, so it
        # cannot be picked up by tooling that inspects the workspace. Every plugin is read back
        # after the write, because opencode swallows a load failure: an upload that reports
        # success and does not land would otherwise leave the MCP gate, the output-token hook or
        # a guard silently absent from a run that still looks healthy and still gets scored.
        for local, remote in self._plugin_uploads():
            await self._upload_plugin_verified(sandbox, local, remote)
        build_agent = effective_config.setdefault("agent", {}).setdefault("build", {})
        for name in ("temperature", "top_p"):
            value = getattr(body, name, None)
            if value is not None:
                build_agent[name] = value
        if self._model_call_capture_enabled():
            # Keep the plugin outside the task repo so it cannot enter a generated patch.
            await sandbox.upload(OBSERVABILITY_PATCH, _REMOTE_ASSISTANT_MESSAGE_PLUGIN)
            apply_observability_patch(effective_config, plugin_path=Path(_REMOTE_ASSISTANT_MESSAGE_PLUGIN))
        if attachments:
            # Decoded on the host only for the upload; the bytes never reach the launch command or a log.
            with tempfile.TemporaryDirectory(prefix="nemo-gym-attachments-") as staging_dir:
                for attachment in attachments:
                    local_path = Path(staging_dir) / Path(attachment.remote_path).name
                    local_path.write_bytes(attachment.content)
                    # Uploads keep the host mode and may land root-owned: world-readable so the
                    # sandbox user can open the file.
                    local_path.chmod(0o644)
                    await sandbox.upload(local_path, attachment.remote_path)
        opencode_config_content = json.dumps(effective_config)
        observation_invocation_id = getattr(request.state, "_ng_observation_invocation_id", None)
        observation_invocation_id = observation_invocation_id if isinstance(observation_invocation_id, str) else None
        collect_observations = observation_invocation_id is not None
        xdg_home_str = ""
        remote_data_home = None
        if collect_observations:
            remote_data_home = f"/tmp/nemo-gym-opencode-{uuid4().hex}"
            xdg_home_str = f"XDG_DATA_HOME={remote_data_home}"

        # OpenCode's glob/grep tools otherwise download rg inside the sandbox.
        ripgrep_remote_path = None
        ripgrep_install_str = ""
        if self.config.local_ripgrep_binary_path is not None:
            ripgrep_remote_path = await self._stage_ripgrep(sandbox, self.config.local_ripgrep_binary_path)
            # Uploads may be root-owned: copy as the execution user; teardown removes the source.
            ripgrep_install_str = (
                '&& mkdir -p "$HOME/.opencode/bin" '
                f'&& cp {quote(ripgrep_remote_path)} "$HOME/.opencode/bin/rg" '
                '&& chmod 0755 "$HOME/.opencode/bin/rg" '
                '&& "$HOME/.opencode/bin/rg" --version'
            )

        extra_env_str = " ".join(f"{k}={quote(v)}" for k, v in sorted(self._resolve_opencode_env(query).items()))

        # Query delivery. argv is the default and preserves today's behaviour exactly; stdin is
        # opt-in (see `opencode_query_via_stdin`).
        #
        # The reason is run.ts's own rewrapping, NOT argument length and NOT shell escaping:
        # `quote()` already makes any question safe as a single argv, and a natural-language
        # question is nowhere near any execve argument limit. run.ts maps every argv message
        # through `arg.includes(" ") ? '"' + arg.replace(/"/g, '\\"') + '"' : arg`, so the model
        # receives a space-bearing question double-quoted with its inner quotes backslash-escaped.
        # `resolveRunInput(message, piped)` returns the piped text verbatim only when that argv
        # message is empty, which is why stdin mode passes no positional message at all.
        query_seed_str = ""
        query_arg_str = f"-- {quote(query)}"
        if self.config.opencode_query_via_stdin:
            # Piped straight into the run rather than staged through a temp file: run.ts gates on
            # `process.stdin.isTTY`, which is false for a pipe exactly as it is for a redirect, so
            # the pipe is a drop-in that leaves nothing behind in the sandbox's /tmp. printf '%s'
            # keeps the bytes exact -- no trailing newline, no shell expansion.
            #
            # Precedence: `|` binds tighter than `&&`, so the chain below reads as
            # `... && (printf ... | ... opencode run ...) && echo "OpenCode run finished"`. The
            # chain waits on the pipeline, whose status is opencode's, so the trailing `&&` still
            # gates on the run and no subshell is needed. Kept on one line because the backslashes
            # in `command` below are Python string continuations: the sandbox shell sees one
            # physical line, and a real newline here would be the only one.
            query_seed_str = f"printf '%s' {quote(query)} | "
            query_arg_str = ""

        # @bxyu-nvidia: Regarding `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000` below:
        # OpenCode defaults to 32k here https://github.com/anomalyco/opencode/blob/58a99916bb96edf5cf605dc03e1be1e4bacf9ff7/packages/opencode/src/provider/transform.ts#L21
        # and there is no way to set it to null.
        # Here, we set an exorbitantly high number that cannot ever be reached.
        # In future versions of OpenCode, this can be directly passed via maxOutputTokens in the limit config above https://github.com/anomalyco/opencode/blob/1b18a50418f730aca32630ccfcde850f2b5fc360/packages/opencode/src/provider/transform.ts#L1418
        setup_command = f"""
        echo "Shell: $SHELL" \
        && {install_str} \
        {ripgrep_install_str} \
        && export PATH="$HOME/.opencode/bin:$PATH" \
        && test "$(opencode --version)" = {quote(self.config.opencode_version)} \
        && echo "Installed OpenCode" \
        && rm -f /tmp/nemo-gym-mcp-setup-error
        """
        agent_command = f"""
        {query_seed_str}NEMO_GYM_REQUIRED_MCP_SERVERS={quote(json.dumps([s.name for s in self.config.tool_servers]))} OPENCODE_CONFIG_CONTENT={quote(opencode_config_content)} OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000 {xdg_home_str} {extra_env_str} \
            opencode run --title "NG dummy title" {opencode_debug_str} {opencode_thinking_str}{opencode_attachment_str} {query_arg_str} \
        && echo "OpenCode run finished"
        """
        if self.config.report_agent_sandbox_oom:
            try:
                setup_result = await sandbox.exec(command=setup_command, timeout_s=self.config.sandbox_timeout)
            except NgFailureEvent:
                raise
            except Exception as exc:
                raise NgFailureEvent(
                    boundary="harness",
                    failure_kind=AGENT_ROUTE_FAILED,
                    operation="opencode.setup",
                    stage="seed",
                    failure_reason="OpenCode setup failed before agent execution",
                ) from exc
            if setup_result.return_code != 0 or setup_result.error_type is not None:
                raise NgFailureEvent(
                    boundary="harness",
                    failure_kind=AGENT_ROUTE_FAILED,
                    operation="opencode.setup",
                    stage="seed",
                    failure_reason=command_failure_reason(
                        "OpenCode setup failed before agent execution", setup_result
                    ),
                )
            command = f'echo "Shell: $SHELL" && export PATH="$HOME/.opencode/bin:$PATH" && {agent_command}'
        else:
            command = f"{setup_command.strip()} && {agent_command.strip()}"

        if self.config.debug:
            print("Starting OpenCode (runtime configuration omitted to protect credentials)", file=sys.stderr)

        run_error_type = None
        try:
            result = await sandbox.exec(
                command=command,
                timeout_s=self.config.sandbox_timeout,
            )
        except NgFailureEvent as exc:
            if not (
                self.config.report_agent_sandbox_oom
                and exc.boundary == "sandbox"
                and exc.failure_kind == PROVIDER_OOM_KILLED
            ):
                raise
            observe_failure(exc)
            details = exc.details
            observation = SandboxObservation(
                role="agent",
                provider=details.provider if isinstance(details, SandboxFailureDetails) else None,
                sandbox_id=details.sandbox_id if isinstance(details, SandboxFailureDetails) else None,
                stage=exc.stage or "agent",
                outcome="oom",
                failure_id=exc.event_id,
            )
            self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = {
                "opencode_failed": True,
                "opencode_exit_code": None,
                "opencode_error_type": exc.failure_kind,
                "opencode_results_fpath": "",
                "opencode_run_stdout": "",
                "opencode_run_stderr": str(exc),
                "opencode_export_found": False,
                "opencode_finished": False,
                "_ng_agent_observations": AgentObservationBundle(
                    source="opencode",
                    records=(
                        [
                            AgentInvocation(
                                invocation_id=observation_invocation_id, status="failed", error_type="server_error"
                            )
                        ]
                        if observation_invocation_id is not None
                        else []
                    )
                    + [observation],
                    gaps=[ObservationGap(code="observation_capture_failed")],
                ),
            }
            response = NeMoGymResponse(
                id=f"resp_{uuid4().hex}",
                created_at=int(time()),
                model=body.model or self.config.model_server.name,
                object="response",
                output=[],
                status="failed",
                tool_choice=body.tool_choice,
                tools=body.tools,
                parallel_tool_calls=body.parallel_tool_calls,
                metadata={
                    NG_AGENT_SANDBOX_OBSERVATION_KEY: observation.model_dump_json(),
                    AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=exc.failure_kind).model_dump_json(),
                },
            )
            fields = verification_failure_fields()
            return response.model_copy(update=fields) if fields["failure_report"] is not None else response
        except Exception as error:
            # A foreign exec exception leaves the outcome unknown; report the failed harness operation with
            # the sandbox facts known here, rather than grade a substituted partial run.
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                failure_reason=self._agent_sandbox_facts(
                    sandbox, reason=provider_error_reason(error), http_status=error_to_http_status(error)
                )
                + f"Harness operation opencode.execute failed: {type(error).__name__}",
                operation="opencode.execute",
                stage="agent",
            ) from error

        if result.error_type != "timeout" and (result.error_type is not None or result.return_code == -1):
            # The provider returned a result that does not say how the command ended: an error type, or the -1
            # signal-death sentinel. That is sandbox evidence, not a contradiction inside the harness, so it keeps
            # the registered provider kind (an environment fault, ungraded and rerun under the attempt cap) and the
            # harness failure only routes it. Neither value is an observed process exit, so no exit code is recorded.
            handle = getattr(sandbox, "_handle", None)
            provider = getattr(handle, "provider_name", None)
            sandbox_id = getattr(handle, "sandbox_id", None)
            provider_failure = sandbox_failure(
                operation="exec",
                failure_kind=PROVIDER_COMMAND_OUTCOME_UNKNOWN,
                reason=self._agent_sandbox_facts(
                    sandbox,
                    exit_code=result.return_code,
                    reason=result.error_type if isinstance(result.error_type, str) else None,
                )
                + "Sandbox command outcome cannot be classified",
                provider=provider if isinstance(provider, str) else None,
                sandbox_id=sandbox_id if isinstance(sandbox_id, str) else None,
                provider_status_reason=result.error_type if isinstance(result.error_type, str) else None,
            )
            raise agent_route_failure(
                provider_failure, operation="opencode.execute", stage="agent"
            ) from provider_failure
        result_stdout = result.stdout or ""
        stdout_parts = result_stdout.rsplit("Shell: ", maxsplit=1)
        if len(stdout_parts) < 2:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.capture",
                failure_reason="OpenCode command output is missing its launch marker",
            )
        if result.return_code == 0 and (
            result.error_type is not None or "OpenCode run finished" not in stdout_parts[1]
        ):
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.capture",
                failure_reason="OpenCode command reported exit 0 without a trustworthy completion marker",
            )

        if self.config.debug and result:
            print("OpenCode install and run stdout:\n", result.stdout, file=sys.stderr)
            print("OpenCode install and run stderr:\n", result.stderr, file=sys.stderr)

        if self.config.tool_servers:
            mcp_check = await sandbox.exec(command="test ! -f /tmp/nemo-gym-mcp-setup-error", timeout_s=30)
            if mcp_check.return_code != 0 or mcp_check.error_type:
                raise NgFailureEvent(
                    boundary="harness",
                    failure_kind=AGENT_ROUTE_FAILED,
                    operation="opencode.setup",
                    stage="seed",
                    failure_reason="Required Gym MCP tools could not be initialized",
                )

        export_fname = "export.json"
        # Kept outside the sandbox workdir on purpose: SWE-bench-style environments set the workdir
        # to the git repo, and resources servers extract the model patch with `git add -N . && git
        # diff`, which would sweep this transcript into the patch.
        export_remote_fpath = f"/tmp/opencode_{export_fname}"
        results_root = (
            Path(self.config.artifacts_dir) if self.config.artifacts_dir else Path(__file__).parent / "results"
        )
        results_dir = results_root / request.session[SESSION_ID_KEY]
        results_dir.mkdir(parents=True, exist_ok=True)
        results_local_fpath = results_dir / export_fname
        # A previous attempt's transcript must never stand in for this run's export.
        results_local_fpath.unlink(missing_ok=True)
        session_env = {"XDG_DATA_HOME": remote_data_home} if remote_data_home is not None else None
        try:
            session_list_result = await sandbox.exec(
                command='export PATH="$HOME/.opencode/bin:$PATH" && opencode session list --format json',
                env=session_env,
                timeout_s=self.config.sandbox_timeout,
            )
        except NgFailureEvent:
            raise
        except Exception as exc:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.session_list",
                failure_reason="Failed to list OpenCode sessions",
            ) from exc
        if session_list_result.return_code != 0 or session_list_result.error_type is not None:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.session_list",
                failure_reason=command_failure_reason("Failed to list OpenCode sessions", session_list_result),
            )
        try:
            session_id = _extract_opencode_session_id(session_list_result.stdout or "")
        except (ValueError, TypeError) as exc:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.session_list",
                failure_reason=str(exc),
            ) from exc
        try:
            export_result = await sandbox.exec(
                command=(
                    'export PATH="$HOME/.opencode/bin:$PATH" '
                    f"&& opencode export {quote(session_id)} > {quote(export_remote_fpath)}"
                ),
                env=session_env,
                timeout_s=self.config.sandbox_timeout,
            )
        except NgFailureEvent:
            raise
        except Exception as exc:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.export",
                failure_reason=f"Failed to export OpenCode session {session_id}",
            ) from exc
        if export_result.return_code != 0 or export_result.error_type is not None:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.export",
                failure_reason=command_failure_reason(
                    f"Failed to export OpenCode session {session_id}", export_result
                ),
            )
        if self.config.debug:
            print("Export stdout:\n", export_result.stdout, file=sys.stderr)
            print("Export stderr:\n", export_result.stderr, file=sys.stderr)

        if self.config.debug:
            print(f"Downloading results from {export_remote_fpath} to {results_local_fpath}", file=sys.stderr)
        try:
            await sandbox.download(export_remote_fpath, results_local_fpath)
        except NgFailureEvent:
            raise
        except Exception as exc:
            results_local_fpath.unlink(missing_ok=True)
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.download",
                failure_reason=f"Failed to download OpenCode export to {results_local_fpath}",
            ) from exc
        if not results_local_fpath.is_file():
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.download",
                failure_reason=f"OpenCode export was not downloaded to {results_local_fpath}",
            )
        try:
            opencode_export = json.loads(results_local_fpath.read_text())
        except (OSError, ValueError) as exc:
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.capture",
                failure_reason=f"Failed to read OpenCode export from {results_local_fpath}",
            ) from exc
        if (
            not isinstance(opencode_export, dict)
            or not isinstance(opencode_export.get("messages"), list)
            or not opencode_export["messages"]
            or not any(
                isinstance(message, dict)
                and isinstance(message.get("info"), dict)
                and message["info"].get("role") == "assistant"
                for message in opencode_export["messages"]
            )
        ):
            raise NgFailureEvent(
                boundary="harness",
                failure_kind=AGENT_ROUTE_FAILED,
                operation="opencode.capture",
                failure_reason=f"OpenCode export at {results_local_fpath} had no assistant messages or was invalid",
            )

        observations = None
        trajectory = (
            TrajectoryRecord(task_id="", rollout_id=observation_invocation_id) if collect_observations else None
        )
        child_usages = []
        # Usage includes descendants even when detailed observation collection is disabled.
        if collect_observations or session_id is not None:
            snapshot_remote_fpath = (
                f"{remote_data_home}/opencode/nemo-gym-observations.db"
                if remote_data_home is not None
                else f"/tmp/nemo-gym-observations-{uuid4().hex}.db"
            )
            observations_local_fpath = results_dir / "opencode.db"
            observations_local_fpath.unlink(missing_ok=True)
            try:
                # OpenCode selects its own database and supplies SQLite even in images without Python.
                # VACUUM INTO includes committed WAL data in one consistent, standalone snapshot.
                snapshot_sql = "VACUUM INTO '" + snapshot_remote_fpath.replace("'", "''") + "'"
                snapshot_result = await sandbox.exec(
                    command=f'export PATH="$HOME/.opencode/bin:$PATH" && opencode db {quote(snapshot_sql)}',
                    env=session_env,
                    timeout_s=self.config.sandbox_timeout,
                )
                if snapshot_result.return_code != 0 or snapshot_result.error_type is not None:
                    raise RuntimeError(f"OpenCode database snapshot failed: {snapshot_result.stderr}")
                await sandbox.download(snapshot_remote_fpath, observations_local_fpath)
                if session_id is not None:
                    child_messages = await asyncio.to_thread(
                        _read_opencode_child_messages, observations_local_fpath, session_id
                    )
                    child_usages = self._opencode_export_to_usages({"messages": child_messages})
                if collect_observations:
                    observations = parse_opencode_observations(
                        observations_local_fpath,
                        observation_invocation_id,
                        trajectory,
                        model_ref=self.config.model_server,
                    )
            except Exception as error:
                if isinstance(error, NgFailureEvent):
                    if error.boundary != "sandbox" or error.failure_kind != PROVIDER_IO_TIMEOUT:
                        raise
                    # This optional snapshot supplements an already validated primary export.
                    observe_failure(error)
                print("Failed to capture OpenCode session usage or observations", format_exc(), file=sys.stderr)
                if collect_observations:
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

        output = []
        usage = None
        opencode_export_found = False
        length_limited = False
        if opencode_export:
            opencode_export_found = True
            # Assume only one input message. May change with a system/developer message later on.
            output = self._opencode_export_to_output_items(opencode_export)[1:]
            usage = NeMoGymResponseUsage.sum_from_list(
                [*self._opencode_export_to_usages(opencode_export), *child_usages]
            )

            assistant_infos = [
                message.get("info", {})
                for message in opencode_export.get("messages", [])
                if message.get("info", {}).get("role") == "assistant"
            ]
            length_limited = bool(assistant_infos and assistant_infos[-1].get("finish") == "length")
            terminal_error = assistant_infos[-1].get("error") if assistant_infos else None
            # The sandbox's exec timeout outranks the error the killed session left in its export,
            # so rules that match opencode_error_type == "timeout" still see it.
            if terminal_error and not run_error_type and getattr(result, "error_type", None) is None:
                run_error_type = (
                    terminal_error.get("name", "OpenCodeError")
                    if isinstance(terminal_error, dict)
                    else "OpenCodeError"
                )

        result_stderr = (result.stderr if result else "") or ""
        opencode_finished = False
        if len(stdout_parts) > 1:
            opencode_finished = "OpenCode run finished" in stdout_parts[1]

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
                        record.status = "incomplete" if status == "completed" and length_limited else status
            observations.records.append(agent_sandbox_observation)
            observations.gaps.append(ObservationGap(code="sandbox_lifecycle_timing_unavailable"))

        # Fail closed on the plugin layer. opencode swallows plugin load errors — it catches them,
        # logs where nothing surfaces, and its user-facing notification is commented out — so a run
        # whose plugins never loaded is indistinguishable from a healthy one. Without this check
        # the contamination filter and the turn budget would be silently optional and we would
        # still score the row. The sentinel plugin prints this marker at import time.
        # Both streams: the plugins write to stderr, opencode's own progress to stdout, and which
        # of the two a line lands in depends on the exec path -- the OpenSandbox background path
        # returns one combined stream in stdout, the foreground path splits them. Concatenating
        # makes the check independent of that.
        run_output = (result_stdout or "") + (result_stderr or "")
        plugins_loaded = "[nemo-gym-plugin] loaded" in run_output
        if self.config.opencode_plugins and not plugins_loaded and run_output.strip():
            # Hard-fail only when we actually captured a transcript and the marker is still absent
            # — that is a genuine silent plugin failure and must never be scored.
            #
            # Missing launch output or capture already raises an ungraded harness failure above.
            # A confirmed command deadline does not establish a usable capture. With an intact capture,
            # the verifier and its configured policy own grading after a timeout.
            # Carry a slice of the transcript into the exception. The failures sidecar keeps only
            # the exception body -- no response, no trajectory -- so without this a rejected row
            # records WHICH question failed and nothing about why, leaving the affected rows of
            # earlier runs permanently undiagnosable.
            #
            # What the slice is for: "no sentinel" has at least two causes that look identical from
            # here. Either the plugins genuinely failed to load (opencode swallows that), or
            # opencode never started at all -- a failed install, a rejected config, a killed pod --
            # which also prints something and also has no marker. Truncation is NOT a candidate:
            # across every scored row of a full run the earliest surviving sentinel was call=1,
            # including in a multi-megabyte transcript. Head and tail together separate those cases on
            # sight, and bound what a huge transcript can add to the sidecar.
            head = run_output[:800].replace("\n", "\\n")
            tail = run_output[-800:].replace("\n", "\\n")
            raise RuntimeError(
                "opencode plugins did not load (no sentinel in run output); configured: "
                f"{self.config.opencode_plugins}. Refusing to score a rollout that ran without them. "
                f"captured_chars={len(run_output)} export_found={opencode_export_found} "
                f"finished={opencode_finished} head={head!r} tail={tail!r}"
            )

        run_result = {
            "opencode_failed": bool(run_error_type or getattr(result, "error_type", None))
            or getattr(result, "return_code", None) != 0
            or not opencode_finished
            or length_limited,
            "opencode_exit_code": getattr(result, "return_code", None),
            "opencode_error_type": run_error_type or getattr(result, "error_type", None),
            "opencode_results_fpath": str(results_local_fpath),
            "opencode_run_stdout": result_stdout,
            "opencode_run_stderr": result_stderr,
            "opencode_export_found": opencode_export_found,
            "opencode_finished": opencode_finished,
            "opencode_plugins_loaded": plugins_loaded,
            # Terminal reason the turn-budget plugin reported, if any. Stock emits nothing here,
            # so an empty string means the run ended some other way (answered, or infra failure).
            "opencode_terminal_reason": _terminal_reason(run_output),
            "opencode_attachment_count": len(attachments),
        }
        if collect_observations:
            run_result["_ng_agent_observations"] = observations
            run_result["_ng_trajectory"] = trajectory
        self._sandbox_id_to_run_result[request.cookies["sandbox_id"]] = run_result

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model or self.config.model_server.name,
            object="response",
            output=output,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=usage,
            status="incomplete" if length_limited else None,
            incomplete_details={"reason": "max_output_tokens"} if length_limited else None,
        )
        return response

    async def run(
        self, request: Request, body: OpenCodeSandboxedAgentRunRequest
    ) -> OpenCodeSandboxedAgentVerifyResponse:
        async with self._sem:
            return await self._run(request, body)

    async def _run(
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

        request.state._ng_opencode_mcp = await self._seed_tool_servers(request, body)

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
            run_result = self._sandbox_id_to_run_result.get(session_key, {}).copy()
            observations = run_result.pop("_ng_agent_observations", None)
            trajectory = run_result.pop("_ng_trajectory", None)
            # Hand the verifier the terminal reason BEFORE it grades. run_result is merged into the
            # saved row further down, but that happens after /verify has already returned, so a
            # verifier that wants to act on how the rollout ended (e.g. masking a budget-exhausted
            # row rather than scoring it zero) cannot see it from the row alone. The run request is
            # extra="allow", so this rides through for verifiers that want it and is ignored by the
            # ones that do not.
            verify_body = body.model_copy(
                update={"opencode_terminal_reason": run_result.get("opencode_terminal_reason", "")}
            )
            confirmed_agent_oom = (
                self.config.report_agent_sandbox_oom
                and run_result.get("opencode_failed") is True
                and run_result.get("opencode_error_type") == PROVIDER_OOM_KILLED
                and observations is not None
                and any(
                    isinstance(record, SandboxObservation)
                    and record.role == "agent"
                    and record.stage == "agent"
                    and record.outcome == "oom"
                    and record.failure_id is not None
                    for record in observations.records
                )
            )
            if not run_result.get("opencode_export_found") and not confirmed_agent_oom:
                raise NgFailureEvent(
                    boundary="harness",
                    failure_kind=AGENT_ROUTE_FAILED,
                    operation="opencode.capture",
                    failure_reason="OpenCode export unavailable; refusing to verify an uncaptured agent response",
                )
            response_dict = await verify_agent_response(
                self.server_client,
                self.config.resources_server,
                verify_body,
                response,
                cookies,
                force_zero_reward=(
                    self.config.execution_failure_reward_zero
                    and run_result.get("opencode_failed", False)
                    and not confirmed_agent_oom
                ),
            )
        finally:
            del request.state._ng_observation_invocation_id
            del request.state._ng_opencode_mcp
            # A server cancels its handler when the caller disconnects, and OpenCode keeps running
            # in its pod regardless: only stopping the sandbox ends it. Shielded because the
            # cancellation is re-delivered at every await until the handler exits, so an unshielded
            # stop would itself be cancelled and leave the pod generating until its TTL.
            with CancelScope(shield=True):
                try:
                    await sandbox.stop()
                except Exception:
                    print("Failed to stop sandbox", format_exc(), file=sys.stderr)
                finally:
                    self._sandbox_id_to_sandbox.pop(session_key, None)
                    self._sandbox_id_to_run_result.pop(session_key, None)

        response_dict |= run_result
        if trajectory is not None:
            response_dict["ng_trajectory"] = scope_opencode_trajectory(trajectory, body, rollout_id)
        raw_verifier_sandbox_observation = response_dict.pop("verifier_sandbox_observation", None)
        raw_verifier_sandbox_observations = response_dict.pop("verifier_sandbox_observations", None)
        if raw_verifier_sandbox_observations is None:
            raw_verifier_sandbox_observations = (
                [raw_verifier_sandbox_observation] if raw_verifier_sandbox_observation is not None else []
            )
        if rollout_id is not None:
            if observations is None:
                observations = AgentObservationBundle(
                    source="opencode",
                    records=[AgentInvocation(invocation_id=rollout_id)],
                    gaps=[ObservationGap(code="observation_capture_failed")],
                )
            if not isinstance(raw_verifier_sandbox_observations, list):
                observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_invalid"))
            elif not raw_verifier_sandbox_observations:
                observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_unavailable"))
            else:
                for raw_observation in raw_verifier_sandbox_observations:
                    if raw_observation is None:
                        observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_unavailable"))
                        continue
                    try:
                        verifier_observation = SandboxObservation.model_validate(raw_observation)
                        if verifier_observation.role != "verifier":
                            raise ValueError("resources server returned a non-verifier sandbox observation")
                        observations.records.append(verifier_observation)
                    except Exception:
                        observations.gaps.append(ObservationGap(code="verifier_sandbox_observation_invalid"))
            response_dict["ng_agent_observations"] = observations.model_dump(mode="json")
        return OpenCodeSandboxedAgentVerifyResponse.model_validate(response_dict)


if __name__ == "__main__":
    OpenCodeSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeSandboxedAgent.run_webserver()  # noqa: F401
