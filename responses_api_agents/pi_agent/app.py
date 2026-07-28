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
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    pop_response_observations,
    response_with_observations,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.pi_agent.setup_pi import ensure_pi


LOG = logging.getLogger(__name__)


def parse_pi_events(stdout: str) -> tuple[list[Any], dict[str, int]]:
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            continue

        if role == "assistant":
            usage = message.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            input_tokens += int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
            output_tokens += int(usage.get("output") or 0)
            texts = [b["text"] for b in content if isinstance(b, dict) and (b.get("text") or "").strip()]
            if texts:
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text="\n".join(texts), annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
                arguments = json.dumps(args) if isinstance(args, (dict, list)) else str(args or "")
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        name=block.get("name", ""),
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )

        elif role == "toolResult":
            call_id = message.get("toolCallId", "")
            result_text = "".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="completed",
                )
            )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


async def _read_pi_stdout(stream: asyncio.StreamReader) -> tuple[str, list[tuple[float, dict[str, Any]]]]:
    lines: list[str] = []
    events: list[tuple[float, dict[str, Any]]] = []

    def consume(line: bytes) -> None:
        observed_at = time()
        text = line.decode(errors="replace")
        lines.append(text)
        try:
            event = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            return
        if isinstance(event, dict):
            events.append((observed_at, event))

    pending = bytearray()
    while chunk := await stream.read(64 * 1024):
        pending.extend(chunk)
        while (newline := pending.find(b"\n")) >= 0:
            consume(bytes(pending[: newline + 1]))
            del pending[: newline + 1]
    if pending:
        consume(bytes(pending))
    return "".join(lines), events


def _build_pi_observations(
    events: list[tuple[float, dict[str, Any]]],
    invocation_id: str,
    model_ref: Optional[ModelServerRef],
    conversation: list[Any],
    *,
    transcript_available: bool = True,
) -> AgentObservationBundle:
    def gap(code: str, detail: Optional[str] = None) -> ObservationGap:
        return ObservationGap(code=code, invocation_id=invocation_id, detail=detail)

    gaps = [gap("subagent_hierarchy_unavailable")]
    if not transcript_available:
        gaps.append(gap("agent_transcript_unavailable"))
    model_calls: list[ModelCallRef] = []
    model_call_join_missing = False
    starts: dict[str, tuple[float, Optional[str]]] = {}
    tools: dict[str, ToolCallObservation] = {}
    compaction_start: Optional[tuple[float, Optional[str], Optional[ModelCallRef]]] = None
    compactions: list[ContextCompactionObservation] = []
    compactions_waiting_for_call: list[ContextCompactionObservation] = []
    last_model_call: Optional[ModelCallRef] = None
    invocation_status = "unknown"

    for observed_at, event in events:
        event_type = event.get("type")
        message = event.get("message")
        call_id = event.get("toolCallId")
        tool_name = event.get("toolName")
        tool_name = tool_name if isinstance(tool_name, str) and tool_name else None

        if event_type == "message_end" and isinstance(message, dict) and message.get("role") == "assistant":
            response_id = message.get("responseId")
            if model_ref is not None and isinstance(response_id, str) and response_id:
                last_model_call = ModelCallRef(model_ref=model_ref, response_id=response_id)
                model_calls.append(last_model_call)
            else:
                last_model_call = None
                model_call_join_missing = True
            for compaction in compactions_waiting_for_call:
                compaction.after_model_call = last_model_call
                if last_model_call is None:
                    gaps.append(gap("compaction_after_model_call_unavailable"))
            compactions_waiting_for_call.clear()
        elif event_type == "agent_end":
            terminal_messages = event.get("messages")
            if isinstance(terminal_messages, list):
                stop_reason = next(
                    (
                        item.get("stopReason")
                        for item in reversed(terminal_messages)
                        if isinstance(item, dict) and item.get("role") == "assistant"
                    ),
                    None,
                )
                invocation_status = {
                    "stop": "completed",
                    "error": "failed",
                    "aborted": "incomplete",
                    "length": "incomplete",
                }.get(stop_reason, "unknown")
        elif event_type == "tool_execution_start" and isinstance(call_id, str):
            starts[call_id] = (observed_at, tool_name)
        elif event_type == "tool_execution_end" and isinstance(call_id, str):
            completed_at = observed_at
            started_at, started_name = starts.pop(call_id, (None, None))
            valid_interval = started_at is not None and completed_at >= started_at
            duration_ms = (completed_at - started_at) * 1000 if started_at is not None and valid_interval else None
            tools[call_id] = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=tool_name or started_name,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                timing_source="harness",
                status=(
                    "failed"
                    if event.get("isError") is True
                    else "completed"
                    if event.get("isError") is False
                    else "unknown"
                ),
            )
            if not valid_interval:
                gaps.append(gap("tool_timing_unavailable", call_id))
            if not isinstance(event.get("isError"), bool):
                gaps.append(gap("tool_outcome_unavailable", call_id))
        elif event_type == "compaction_start":
            reason = event.get("reason")
            compaction_start = (observed_at, reason if isinstance(reason, str) else None, last_model_call)
        elif event_type == "compaction_end":
            reason = event.get("reason")
            started_at, started_reason, before_model_call = compaction_start or (observed_at, None, None)
            raw_result = event.get("result")
            result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
            before = result.get("tokensBefore")
            after = result.get("estimatedTokensAfter")
            summary = result.get("summary")
            first_kept_item_id = result.get("firstKeptEntryId")
            outcome = (
                "aborted"
                if event.get("aborted") is True
                else "completed"
                if result
                else "failed"
                if isinstance(event.get("errorMessage"), str)
                else "unknown"
            )
            compaction = ContextCompactionObservation(
                invocation_id=invocation_id,
                observed_at=started_at,
                trigger=reason if isinstance(reason, str) else started_reason,
                tokens_before=before if type(before) is int and before >= 0 else None,
                tokens_after=after if type(after) is int and after >= 0 else None,
                outcome=outcome,
                summary=summary if isinstance(summary, str) else None,
                first_kept_item_id=first_kept_item_id if isinstance(first_kept_item_id, str) else None,
                before_model_call=before_model_call,
            )
            compactions.append(compaction)
            compactions_waiting_for_call.append(compaction)
            if compaction_start is None:
                gaps.append(gap("compaction_start_unavailable"))
            if not result:
                gaps.append(gap("compaction_result_unavailable"))
            else:
                if type(before) is not int or before < 0:
                    gaps.append(gap("compaction_tokens_before_unavailable"))
                if not isinstance(summary, str):
                    gaps.append(gap("compaction_summary_unavailable"))
                if not isinstance(first_kept_item_id, str):
                    gaps.append(gap("compaction_boundary_unavailable"))
                if type(after) is not int or after < 0:
                    gaps.append(gap("compaction_tokens_after_unavailable"))
            if outcome == "unknown":
                gaps.append(gap("compaction_outcome_unavailable"))
            compaction_start = None
    if not model_calls or model_call_join_missing:
        gaps.append(gap("model_call_ownership_unavailable"))
    if invocation_status == "unknown":
        gaps.append(gap("invocation_outcome_unavailable"))

    for call_id, (started_at, tool_name) in starts.items():
        tools[call_id] = ToolCallObservation(
            invocation_id=invocation_id,
            tool_call_id=call_id,
            tool_name=tool_name,
            started_at=started_at,
            timing_source="harness",
            status="incomplete",
        )
        gaps.append(gap("tool_timing_unavailable", call_id))
    if compaction_start is not None:
        started_at, reason, before_model_call = compaction_start
        compactions.append(
            ContextCompactionObservation(
                invocation_id=invocation_id,
                observed_at=started_at,
                trigger=reason,
                before_model_call=before_model_call,
            )
        )
        gaps.append(gap("compaction_result_unavailable"))
        gaps.append(gap("compaction_outcome_unavailable"))
    for _ in compactions_waiting_for_call:
        gaps.append(gap("compaction_after_model_call_unavailable"))

    def field(item: Any, name: str) -> Any:
        return item.get(name) if isinstance(item, dict) else getattr(item, name, None)

    result_ids = {
        field(item, "call_id")
        for item in conversation
        if field(item, "type") == "function_call_output" and isinstance(field(item, "call_id"), str)
    }
    for item in conversation:
        if field(item, "type") != "function_call":
            continue
        call_id = field(item, "call_id")
        if not isinstance(call_id, str) or not call_id or call_id in tools:
            continue
        tools[call_id] = ToolCallObservation(
            invocation_id=invocation_id,
            tool_call_id=call_id,
            tool_name=field(item, "name"),
            status="unknown" if call_id in result_ids else "incomplete",
        )
        gaps.append(gap("tool_timing_unavailable", call_id))

    return AgentObservationBundle(
        source="pi",
        records=[
            AgentInvocation(
                invocation_id=invocation_id,
                status=invocation_status,
                model_calls=model_calls,
                conversation=conversation,
            ),
            *tools.values(),
            *compactions,
        ],
        gaps=gaps,
    )


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


class PiAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "pi"
    model: str = "nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct"
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/pi_agent/workspaces"
    thinking: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: int = 900
    extra_args: list[str] = []
    models_config: dict[str, Any] = Field(default_factory=dict)
    pi_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class PiAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PiAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PiAgent(SimpleResponsesAPIAgent):
    """Runs the pi CLI (pi --print --mode json --no-session)"""

    config: PiAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_pi(self.config.pi_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("pi command %r is not on PATH yet", self.config.command)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"pi_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, home: Path) -> dict[str, str]:
        env = {**os.environ, "HOME": str(home), "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    def _models_config_for_run(self, rollout_id: Optional[str]) -> dict[str, Any]:
        if self.config.model_server is None:
            return self.config.models_config

        models_config = copy.deepcopy(self.config.models_config)
        provider = self.config.model.partition("/")[0]
        providers = models_config.get("providers")
        provider_config = providers.get(provider) if isinstance(providers, dict) else None
        if not isinstance(provider_config, dict):
            raise ValueError(f"models_config.providers.{provider} is required when model_server is configured")
        provider_config["baseUrl"] = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        return models_config

    async def _run_pi(
        self,
        instruction: str,
        system_prompt: Optional[str],
        *,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> tuple[list[Any], dict[str, int], str, list[tuple[float, dict[str, Any]]]]:
        provider, _, model_id = self.config.model.partition("/")
        work_dir = self._workspace_root()
        home = work_dir / ".pi-home"
        (home / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
        models_config = self._models_config_for_run(rollout_id)
        if models_config:
            (home / ".pi" / "agent" / "models.json").write_text(json.dumps(models_config, indent=2))
        env = self._env(home)

        cmd = [*self.config.command_parts, "--print", "--mode", "json", "--no-session"]
        if provider:
            cmd += ["--provider", provider, "--model", model_id]
        else:
            cmd += ["--model", self.config.model]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += self.config.extra_args
        cmd.append(instruction)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            assert proc.stdout is not None and proc.stderr is not None
            events: list[tuple[float, dict[str, Any]]] = []
            if collect_observations:
                stdout_task = asyncio.create_task(_read_pi_stdout(proc.stdout))
                stderr_task = asyncio.create_task(proc.stderr.read())
                output_task = asyncio.gather(stdout_task, stderr_task, proc.wait())
                try:
                    (stdout, events), stderr, _ = await asyncio.wait_for(
                        asyncio.shield(output_task), timeout=self.config.timeout
                    )
                except asyncio.TimeoutError:
                    if proc.returncode is None:
                        proc.kill()
                    (_, events), _, _ = await output_task
                    LOG.warning("pi timed out after %ds", self.config.timeout)
                    return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model, events
            else:
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    LOG.warning("pi timed out after %ds", self.config.timeout)
                    return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model, events

            if proc.returncode not in (0, None):
                LOG.warning("pi exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])
            output_items, usage = parse_pi_events(stdout)
            return output_items, usage, self.config.model, events
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

        output_items, usage, model_name, events = await self._run_pi(
            user_message,
            system_prompt,
            rollout_id=rollout_id,
            collect_observations=collect_observations,
        )
        observed_output_items = list(output_items)
        if not observed_output_items and events:
            observed_output_items, _ = parse_pi_events("\n".join(json.dumps(event) for _, event in events))

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("pi produced no assistant message; padding empty output")
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
        observations = AgentObservationBundle(source="pi")
        if collect_observations:
            invocation_id = rollout_id or response.id
            try:
                observations = _build_pi_observations(
                    events,
                    invocation_id,
                    self.config.model_server,
                    [*body.input, *observed_output_items],
                    transcript_available=bool(observed_output_items),
                )
            except Exception:
                LOG.exception("failed to build Pi observations")
                observations = AgentObservationBundle(
                    source="pi", gaps=[ObservationGap(code="observation_parse_failed")]
                )
            observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))
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

    async def run(self, request: Request, body: PiAgentRunRequest) -> PiAgentVerifyResponse:
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

            return PiAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
            )


if __name__ == "__main__":
    PiAgent.run_webserver()
