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
import copy
import json
import logging
import os
import shlex
import shutil
import sqlite3
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    ToolCallObservation,
    pop_response_observations,
    response_with_observations,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode


LOG = logging.getLogger(__name__)


def _load_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _milliseconds(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return None
    return float(value) / 1000


def _parse_opencode_session(db_path: Path, fallback_invocation_id: str) -> AgentObservationBundle:
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

    for row in message_rows:
        message = messages[row["id"]]
        session_id = row["session_id"]
        if message.get("role") == "assistant":
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
            if role == "user":
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
            continue
        if ptype == "reasoning" and role == "assistant" and isinstance(text, str) and text.strip():
            conversation.append(
                NeMoGymResponseReasoningItem(
                    id=row["id"],
                    summary=[NeMoGymSummary(type="summary_text", text=text)],
                )
            )
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
            observed_tool_output = state.get("output") if state.get("output") is not None else state.get("error")
            if observed_tool_output is not None:
                result = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=str(observed_tool_output),
                    status=response_status,
                )
                conversation.append(result)

            native_time = state.get("time") if isinstance(state.get("time"), dict) else {}
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
        compactions.append(
            ContextCompactionObservation(
                invocation_id=session_id,
                observed_at=observed_at,
                trigger=trigger,
                outcome="completed" if summary else "unknown",
                summary=summary,
                first_kept_item_id=(part.get("tail_start_id") if isinstance(part.get("tail_start_id"), str) else None),
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
    gaps.append(ObservationGap(code="model_call_ownership_unavailable"))

    return AgentObservationBundle(
        source="opencode",
        records=[*invocations, *tools, *compactions],
        gaps=gaps,
    )


def parse_opencode_session(db_path: Path) -> tuple[list[Any], dict[str, int]]:
    """Convert an OpenCode session database into the existing Gym response shape."""
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    if not db_path.is_file():
        return output_items, {"input_tokens": 0, "output_tokens": 0}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        roles = {row["id"]: _load_json(row["data"]).get("role") for row in con.execute("select id, data from message")}
        rows = con.execute("select message_id, data from part order by time_created").fetchall()
    finally:
        con.close()

    for row in rows:
        part = _load_json(row["data"])
        ptype = part.get("type")
        if ptype == "step-finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            output_tokens += int(tokens.get("output") or 0)
        elif roles.get(row["message_id"]) == "assistant" and ptype == "text" and (part.get("text") or "").strip():
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=part["text"], annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif roles.get(row["message_id"]) == "assistant" and ptype == "tool":
            state = part.get("state") or {}
            call_id = part.get("callID") or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=part.get("tool", ""),
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            if state.get("output") is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(state["output"]),
                        status="completed",
                    )
                )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    """Return (user_message, system_message) from a responses body input list."""
    items = list(body_input)
    system_message: Optional[str] = None

    if items:
        first = items[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "system":
            content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            system_message = content or ""
            items = items[1:]

    user_message = ""
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
                )
            user_message = content or ""
            break

    return user_message, system_message


class OpenCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "opencode"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/opencode_agent/workspaces"
    repo_dir: Optional[str] = None
    thinking: bool = True
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    opencode_config: dict[str, Any] = Field(default_factory=dict)
    opencode_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class OpenCodeAgent(SimpleResponsesAPIAgent):
    """Runs the CLI (opencode run --format=json)"""

    config: OpenCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_opencode(self.config.opencode_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("opencode command %r is not on PATH yet", self.config.command)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"opencode_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _repo_dir(self, fallback: Path) -> Path:
        """Return the configured persistent repository or the temporary fallback."""
        if not self.config.repo_dir:
            return fallback
        root = Path(self.config.repo_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write_opencode_config(self, work_dir: Path, rollout_id: Optional[str] = None) -> None:
        config = copy.deepcopy(self.config.opencode_config)
        if self.config.model_server is not None:
            provider = self.config.model.partition("/")[0]
            providers = config.get("provider")
            provider_config = providers.get(provider) if isinstance(providers, dict) else None
            if isinstance(provider_config, dict):
                options = provider_config.setdefault("options", {})
                if not isinstance(options, dict):
                    raise ValueError(f"opencode_config.provider.{provider}.options must be an object")
                options["baseURL"] = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
            elif provider != "openai":
                raise ValueError(f"opencode_config.provider.{provider} is required when model_server is configured")
        if config:
            (work_dir / "opencode.json").write_text(json.dumps(config, indent=2))

    def _env(self, data_home: str, rollout_id: Optional[str] = None) -> dict[str, str]:
        env = {**os.environ, "XDG_DATA_HOME": data_home}
        base_url = self.config.openai_base_url
        if self.config.model_server is not None:
            base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if self.config.openai_api_key:
            env["OPENAI_API_KEY"] = self.config.openai_api_key
        env.update({k: v for k, v in self.config.env.items() if v})
        if self.config.model_server is not None:
            env["OPENAI_BASE_URL"] = base_url
        return env

    async def _run_opencode(
        self,
        instruction: str,
        system_prompt: Optional[str],
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> tuple[list[Any], dict[str, int], str, AgentObservationBundle]:
        """Run one headless OpenCode session and read its persisted artifact."""
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_home = work_dir / ".opencode-data"
        data_home.mkdir(parents=True, exist_ok=True)
        self._write_opencode_config(project_dir, rollout_id)
        env = self._env(str(data_home), rollout_id)

        cmd = [*self.config.command_parts, "run", "-m", self.config.model, "--dir", str(project_dir)]
        if self.config.thinking:
            cmd.append("--thinking")
        cmd.extend(self.config.extra_args)
        cmd.append(prompt)

        try:
            timed_out = False
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                _, stderr = await proc.communicate()
                timed_out = True
                LOG.warning("opencode timed out after %ds", self.config.timeout)

            if proc.returncode not in (0, None):
                LOG.warning("opencode exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            db_path = data_home / "opencode" / "opencode.db"
            invocation_id = rollout_id or f"opencode-{uuid4().hex}"
            output_items, usage = (
                ([], {"input_tokens": 0, "output_tokens": 0}) if timed_out else parse_opencode_session(db_path)
            )
            observations = AgentObservationBundle(source="opencode")
            if collect_observations:
                try:
                    observations = _parse_opencode_session(db_path, invocation_id)
                except Exception:
                    LOG.exception("failed to read OpenCode session artifact")
                    observations = AgentObservationBundle(
                        source="opencode",
                        records=[AgentInvocation(invocation_id=invocation_id)],
                        gaps=[
                            ObservationGap(code="agent_artifact_unavailable"),
                            ObservationGap(code="agent_transcript_unavailable"),
                            ObservationGap(code="model_call_ownership_unavailable"),
                        ],
                    )
            run_status = "incomplete" if timed_out else "completed" if proc.returncode == 0 else "failed"
            for invocation in observations.records:
                if not isinstance(invocation, AgentInvocation):
                    continue
                if invocation.parent_invocation_id is None:
                    invocation.status = run_status
            if timed_out:
                observations.gaps.append(ObservationGap(code="agent_run_timeout"))
            return output_items, usage, self.config.model, observations
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        output_items, usage, model_name, observations = await self._run_opencode(
            user_message,
            system_prompt,
            rollout_id=rollout_id,
            collect_observations=collect_observations,
        )

        if collect_observations:
            root = next(
                (
                    record
                    for record in observations.records
                    if isinstance(record, AgentInvocation) and record.parent_invocation_id is None
                ),
                None,
            )
            if root is not None and not any(
                getattr(item, "role", None) in {"user", "system", "developer"} for item in root.conversation
            ):
                root.conversation = [*body.input, *root.conversation]

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("OpenCode produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )
        return AgentEpisode(response=response, observations=observations)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        rollout_id = request.path_params.get("rollout_id") if request is not None else None
        episode = await self._create_episode(
            body,
            rollout_id=rollout_id,
            collect_observations=rollout_id is not None,
        )
        return response_with_observations(episode) if rollout_id is not None else episode.response

    async def responses_with_observations(
        self,
        request: Optional[Request],
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: Optional[str] = None,
    ) -> AgentEpisode:
        if rollout_id is None and request is not None:
            rollout_id = request.path_params.get("rollout_id")
        return await self._create_episode(body, rollout_id=rollout_id, collect_observations=True)

    async def run(self, request: Request, body: OpenCodeAgentRunRequest) -> OpenCodeAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            observations = pop_response_observations(agent_resp_json)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump()
                | {"response": agent_resp_json}
                | ({"rollout_id": rollout_id} if rollout_id is not None else {}),
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return OpenCodeAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
            )


if __name__ == "__main__":
    OpenCodeAgent.run_webserver()
