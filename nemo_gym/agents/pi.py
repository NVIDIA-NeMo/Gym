# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import logging
import os
import shlex
import shutil
import signal
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from nemo_gym.agents.config import AgentHarnessConfig
from nemo_gym.agents.responses import extract_instruction as _extract_instruction
from nemo_gym.config_types import ModelServerRef
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
)


LOG = logging.getLogger(__name__)


def _kill_process_group(proc: Any) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def parse_pi_events(stdout: str | bytes) -> tuple[list[Any], dict[str, int]]:
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
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
                started_at=started_at if valid_interval else None,
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


class PiHarness:
    def __init__(self, config: AgentHarnessConfig):
        model = config.model.model
        if model is None:
            raise ValueError("PiHarness requires a model")
        if config.max_turns is not None:
            raise ValueError("PiHarness does not support max_turns")
        self.config = config
        self.model = model

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.config.settings.get("command", "pi"))

    def _workspace(self) -> tuple[Path, bool]:
        if self.config.workspace is not None:
            self.config.workspace.mkdir(parents=True, exist_ok=True)
            return self.config.workspace, False

        workspace_root = self.config.settings.get("workspace_root", "outputs/pi_agent/workspaces")
        root = Path(workspace_root).expanduser() / f"pi_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root, True

    def _env(self, home: Path) -> dict[str, str]:
        env = {**os.environ, "HOME": str(home), "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}
        env.update({key: value for key, value in self.config.settings.get("env", {}).items() if value})
        return env

    def _effective_model(self, model_base_url: Optional[str]) -> str:
        if model_base_url:
            return f"{self.config.model.provider or 'nemo'}/{self.model}"
        return self.model

    def _build_models_config(self, model_base_url: Optional[str]) -> dict[str, Any]:
        config = copy.deepcopy(self.config.settings.get("models_config", {}))
        if not model_base_url:
            return config
        provider_name = self.config.model.provider or "nemo"
        providers = config.setdefault("providers", {})
        providers[provider_name] = {
            "baseUrl": model_base_url,
            "api": "openai-completions",
            "apiKey": self.config.model.api_key or "EMPTY",  # pragma: allowlist secret
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [
                {
                    "id": self.model,
                    "reasoning": True,
                    "input": ["text"],
                    "contextWindow": self.config.model.settings.get("context_window", 262144),
                    "maxTokens": self.config.model.settings.get("max_output_tokens", 131072),
                }
            ],
        }
        return config

    async def _run_pi(
        self,
        instruction: str,
        system_prompt: Optional[str],
        *,
        model_base_url: Optional[str] = None,
        collect_observations: bool = True,
    ) -> tuple[list[Any], dict[str, int], str, list[tuple[float, dict[str, Any]]]]:
        effective_model = self._effective_model(model_base_url)
        provider, _, model_id = effective_model.partition("/")
        work_dir, remove_work_dir = self._workspace()
        home = work_dir / f".pi-home-{uuid4().hex}"
        (home / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
        models_config = self._build_models_config(model_base_url)
        if models_config:
            (home / ".pi" / "agent" / "models.json").write_text(json.dumps(models_config, indent=2))
        env = self._env(home)

        cmd = [*self.command_parts, "--print", "--mode", "json", "--no-session"]
        if provider:
            cmd += ["--provider", provider, "--model", model_id]
        else:
            cmd += ["--model", self.model]
        thinking = self.config.settings.get("thinking")
        if thinking:
            cmd += ["--thinking", thinking]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += self.config.settings.get("extra_args", [])
        cmd.append(instruction)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            assert proc.stdout is not None and proc.stderr is not None
            events: list[tuple[float, dict[str, Any]]] = []
            output_task = None
            try:
                if collect_observations:
                    stdout_task = asyncio.create_task(_read_pi_stdout(proc.stdout))
                    stderr_task = asyncio.create_task(proc.stderr.read())
                    output_task = asyncio.gather(stdout_task, stderr_task, proc.wait())
                    (stdout, events), stderr, _ = await asyncio.wait_for(
                        asyncio.shield(output_task), timeout=self.config.timeout_seconds
                    )
                else:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout_seconds)
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    _kill_process_group(proc)
                if output_task is not None:
                    (_, events), _, _ = await output_task
                else:
                    await proc.communicate()
                LOG.warning("pi timed out after %ss", self.config.timeout_seconds)
                return [], {"input_tokens": 0, "output_tokens": 0}, self.model, events
            except asyncio.CancelledError:
                if proc.returncode is None:
                    _kill_process_group(proc)
                if output_task is not None:
                    await output_task
                else:
                    await proc.communicate()
                raise

            if proc.returncode not in (0, None):
                LOG.warning("pi exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])
            output_items, usage = parse_pi_events(stdout)
            return output_items, usage, self.model, events
        finally:
            if remove_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                shutil.rmtree(home, ignore_errors=True)

    async def run_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
        model_ref: Optional[ModelServerRef] = None,
        rollout_id: Optional[str] = None,
        collect_observations: bool = True,
    ) -> AgentEpisode:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        conversation_input = (
            [NeMoGymEasyInputMessage(role="system", content=system_prompt)] if system_prompt is not None else []
        )
        conversation_input.append(NeMoGymEasyInputMessage(role="user", content=user_message))

        effective_base_url = model_base_url if model_base_url is not None else self.config.model.base_url
        output_items, usage, model_name, events = await self._run_pi(
            user_message,
            system_prompt,
            model_base_url=effective_base_url,
            collect_observations=collect_observations,
        )
        observed_output_items = list(output_items)
        if not observed_output_items and events:
            observed_output_items, _ = parse_pi_events("\n".join(json.dumps(event) for _, event in events))

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("pi produced no assistant message. Padding empty output")
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
                    model_ref,
                    [*conversation_input, *observed_output_items],
                    transcript_available=bool(observed_output_items),
                )
            except Exception:
                LOG.exception("failed to build Pi observations")
                observations = AgentObservationBundle(
                    source="pi", gaps=[ObservationGap(code="observation_parse_failed")]
                )
            observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))
        return AgentEpisode(response=response, observations=observations)

    async def run(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
    ) -> NeMoGymResponse:
        episode = await self.run_episode(
            body,
            model_base_url=model_base_url,
            collect_observations=False,
        )
        return episode.response
