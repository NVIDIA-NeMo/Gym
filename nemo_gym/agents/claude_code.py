# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path
from time import monotonic, time
from typing import Any, Callable, Optional
from uuid import uuid4

from nemo_gym.agents.claude_code_observability import extract_claude_code_observations
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
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, ObservationGap
from nemo_gym.skills import stage_skills


LOG = logging.getLogger(__name__)


def _extract_text(content: list[Any]) -> str:
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _extract_thinking(content: list[Any]) -> str:
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("thinking", "reasoning"):
            parts.append(b.get("thinking") or b.get("text") or "")
    return "\n".join(p for p in parts if p)


def parse_stream_json(stdout: str) -> tuple[list[Any], dict]:
    raw_events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    output_items: list[Any] = []
    pending_calls: dict[str, dict] = {}
    buffered_think: str | None = None
    total_input = 0
    total_output = 0
    num_turns: Optional[int] = None
    result_metadata: dict[str, Any] = {}
    compacting_sessions: set[str] = set()
    compaction_attempts: list[dict[str, str]] = []

    for event in raw_events:
        etype = event.get("type")

        if etype == "result":
            usage = event.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
            # Claude Code's authoritative turn counter (what --max-turns bounds).
            if event.get("num_turns") is not None:
                num_turns = int(event["num_turns"])
            if isinstance(event.get("subtype"), str):
                result_metadata["subtype"] = event["subtype"]
            if isinstance(event.get("is_error"), bool):
                result_metadata["is_error"] = event["is_error"]
            duration_ms = event.get("duration_ms")
            if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool) and duration_ms >= 0:
                result_metadata["duration_ms"] = float(duration_ms)

        elif etype == "assistant":
            message = event.get("message", {})
            content = message.get("content") or []
            usage = message.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)

            if not isinstance(content, list):
                content = []

            think = _extract_thinking(content)
            if think:
                buffered_think = (buffered_think + "\n" + think) if buffered_think else think

            text = _extract_text(content)
            if text:
                if buffered_think:
                    text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                    buffered_think = None
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                input_data = block.get("input") or {}
                arguments = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                pending_calls[call_id] = {"name": block.get("name", ""), "call_id": call_id, "arguments": arguments}

        elif etype == "user":
            message = event.get("message", {})
            content = message.get("content") or []
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id", "")
                call_info = pending_calls.pop(tool_id, None)
                if call_info:
                    output_items.append(
                        NeMoGymResponseFunctionToolCall(
                            arguments=call_info["arguments"],
                            call_id=tool_id,
                            name=call_info["name"],
                            type="function_call",
                            id=tool_id,
                            status="completed",
                        )
                    )
                result_content = block.get("content") or ""
                if isinstance(result_content, list):
                    result_text = _extract_text(result_content)
                else:
                    result_text = str(result_content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=tool_id,
                        output=result_text,
                        status="completed",
                    )
                )

        elif etype == "system" and event.get("subtype") == "status":
            session_id = event.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                continue
            if event.get("status") == "compacting":
                compacting_sessions.add(session_id)
                continue
            compact_result = event.get("compact_result")
            if compact_result in {"failed", "success"}:
                if compact_result == "failed":
                    compaction_attempts.append({"invocation_id": session_id, "outcome": "failed"})
                compacting_sessions.discard(session_id)

    compaction_attempts.extend(
        {"invocation_id": session_id, "outcome": "unknown"} for session_id in compacting_sessions
    )
    metadata: dict = {"input_tokens": total_input, "output_tokens": total_output}
    if num_turns is not None:
        metadata["num_turns"] = num_turns
    if compaction_attempts:
        metadata["compaction_attempts"] = compaction_attempts
    metadata.update(result_metadata)
    return output_items, metadata


def _invocation_outcome(metadata: dict[str, Any], returncode: int | None) -> tuple[str, str | None]:
    subtype = metadata.get("subtype")
    if subtype == "error_max_turns":
        return "incomplete", subtype
    if metadata.get("is_error") is True or (isinstance(subtype, str) and subtype.startswith("error_")):
        return "failed", subtype if isinstance(subtype, str) else "agent_error"
    if returncode not in (0, None):
        return "failed", f"process_exit_{returncode}"
    if subtype == "success":
        return "completed", None
    return "incomplete", "result_missing"


class ClaudeCodeHarness:
    def __init__(self, config: AgentHarnessConfig):
        model = config.model.model
        if model is None:
            raise ValueError("ClaudeCodeHarness requires a model")
        self.config = config
        self.model = model
        self._static_mcp_config: Optional[dict[str, Any]] = None

    def _build_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "env": {
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        }
        settings_file = self.config.settings.get("settings_file")
        if settings_file:
            user_settings = json.loads(Path(settings_file).expanduser().read_text())
            user_env = user_settings.get("env") or {}
            settings = {**settings, **user_settings, "env": {**settings["env"], **user_env}}
        return settings

    def _setup_config_dir(self, skills_path: Optional[str] = None) -> Path:
        claude_config_dir = Path.home() / ".claude_code_agent" / uuid4().hex
        claude_config_dir.mkdir(parents=True)
        try:
            (claude_config_dir / "settings.json").write_text(json.dumps(self._build_settings()))
            if skills_path:
                stage_skills(skills_path, claude_config_dir / "skills")
        except Exception:
            shutil.rmtree(claude_config_dir, ignore_errors=True)
            raise
        return claude_config_dir

    def _build_command(
        self,
        model: str,
        instruction: str,
        system_prompt: Optional[str] = None,
        mcp_config: Optional[str] = None,
        skills_active: bool = False,
    ) -> list[str]:
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        bare = self.config.settings.get("bare", True)
        if bare and skills_active:
            LOG.warning(
                "skills are active for this request. Ignoring bare=True so Claude Code can discover them. "
                "Note this re-enables ALL native auto-discovery, not just skills (hooks, plugins, MCP servers, "
                "memory, and CLAUDE.md), so the runtime broadens versus a bare baseline."
            )
        if bare and not skills_active:
            cmd.append("--bare")
        cmd += ["--model", model]
        effective_mcp_config = mcp_config if mcp_config is not None else self.config.settings.get("mcp_config")
        if effective_mcp_config:
            cmd += ["--mcp-config", effective_mcp_config]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        allowed_tools = self.config.settings.get("allowed_tools")
        if allowed_tools:
            cmd += ["--allowedTools", allowed_tools]
        disallowed_tools = self.config.settings.get("disallowed_tools")
        if disallowed_tools:
            cmd += ["--disallowedTools", disallowed_tools]
        thinking = self.config.settings.get("thinking")
        if thinking:
            cmd += ["--thinking", thinking]
        max_thinking_tokens = self.config.settings.get("max_thinking_tokens")
        if max_thinking_tokens is not None:
            cmd += ["--max-thinking-tokens", str(max_thinking_tokens)]
        if self.config.max_turns is not None:
            cmd += ["--max-turns", str(self.config.max_turns)]
        cmd += ["--", instruction]
        return cmd

    async def _run_claude_code(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
        model_base_url: Optional[str] = None,
        observation_collector: Optional[Callable[[Path, dict[str, Any]], None]] = None,
    ) -> tuple[list[Any], str, dict[str, Any]]:
        base_url = model_base_url if model_base_url is not None else self.config.model.base_url
        # Anthropic uses short model names. Custom endpoints keep the full name.
        model = self.model if base_url else self.model.split("/")[-1]
        api_key = self.config.model.api_key

        claude_config_dir = None
        run_metadata: dict[str, Any] = {"status": "unknown"}
        try:
            claude_config_dir = self._setup_config_dir(skills_path=skills_path)
            env = {
                **os.environ,
                "ANTHROPIC_API_KEY": api_key,  # pragma: allowlist secret
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "CLAUDE_CODE_SUBAGENT_MODEL": model,
                "IS_SANDBOX": "1",
                "CLAUDE_CONFIG_DIR": str(claude_config_dir),
            }
            if base_url:
                env["ANTHROPIC_BASE_URL"] = base_url
                env["ANTHROPIC_AUTH_TOKEN"] = api_key or "local"

            cmd = self._build_command(
                model,
                instruction,
                system_prompt=system_prompt,
                mcp_config=mcp_config,
                skills_active=bool(skills_path),
            )

            process_started_at = monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.workspace,
            )
            communication = asyncio.create_task(proc.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
                stdout, _ = await communication
                LOG.warning("claude-code timed out after %ss", self.config.timeout_seconds)
                _, run_metadata = parse_stream_json(stdout.decode(errors="replace"))
                run_metadata.update(
                    status="incomplete",
                    error_type="timeout",
                    duration_ms=(monotonic() - process_started_at) * 1000,
                )
                return [], model, run_metadata
            except asyncio.CancelledError:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
                await asyncio.gather(communication, return_exceptions=True)
                raise

            if proc.returncode not in (0, None):
                LOG.warning("claude-code exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            stdout_text = stdout.decode(errors="replace")
            LOG.debug("claude-code stdout (%d chars): %s", len(stdout), stdout_text[:2000])
            output_items, run_metadata = parse_stream_json(stdout_text)
            run_metadata.setdefault("duration_ms", (monotonic() - process_started_at) * 1000)
            status, error_type = _invocation_outcome(run_metadata, proc.returncode)
            run_metadata["status"] = status
            if error_type is not None:
                run_metadata["error_type"] = error_type
            return output_items, model, run_metadata
        finally:
            if claude_config_dir is not None:
                try:
                    if observation_collector is not None:
                        await asyncio.to_thread(observation_collector, claude_config_dir, run_metadata)
                except Exception:
                    LOG.exception("failed to collect Claude Code observations")
                finally:
                    shutil.rmtree(claude_config_dir, ignore_errors=True)

    def _load_static_mcp_config(self) -> dict[str, Any]:
        mcp_config = self.config.settings.get("mcp_config")
        if not mcp_config:
            return {"mcpServers": {}}

        config_path = Path(mcp_config).expanduser()
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Claude Code mcp_config must be a JSON object: {config_path}")
        mcp_servers = config.setdefault("mcpServers", {})
        if not isinstance(mcp_servers, dict):
            raise ValueError(f"Claude Code mcp_config has non-object mcpServers: {config_path}")
        return config

    def _get_static_mcp_config(self) -> dict[str, Any]:
        if self._static_mcp_config is None:
            self._static_mcp_config = self._load_static_mcp_config()
        return self._static_mcp_config

    def write_mcp_config(
        self,
        *,
        server_name: str,
        url: str,
        output_dir: Path,
        transport: str = "http",
        headers: Optional[dict[str, str]] = None,
    ) -> str:
        entry: dict[str, Any] = {
            "type": transport,
            "url": url,
        }
        if headers:
            entry["headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning(
                "MCP seed metadata for %r has no headers. The tool endpoint will be called without a "
                "session token and will reject the calls.",
                server_name,
            )

        config = copy.deepcopy(self._get_static_mcp_config())
        config.setdefault("mcpServers", {})[server_name] = entry

        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "gym_mcp_config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        return str(config_path)

    async def run(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
        model_base_url: Optional[str] = None,
        observation_collector: Optional[Callable[[Path, dict[str, Any]], None]] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [part for part in [self.config.system_prompt, input_system] if part]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        output_items, model_name, run_metadata = await self._run_claude_code(
            user_message,
            system_prompt=system_prompt,
            mcp_config=mcp_config,
            skills_path=skills_path,
            model_base_url=model_base_url,
            observation_collector=observation_collector,
        )

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("claude-code produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = run_metadata.get("input_tokens", 0)
        output_tokens = run_metadata.get("output_tokens", 0)

        return NeMoGymResponse(
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

    async def run_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
        model_base_url: Optional[str] = None,
        model_ref: Optional[ModelServerRef] = None,
    ) -> AgentEpisode:
        observations: Optional[AgentObservationBundle] = None

        def collect(config_dir: Path, run_metadata: dict[str, Any]) -> None:
            nonlocal observations
            try:
                observations = extract_claude_code_observations(
                    config_dir,
                    model_ref=model_ref,
                    root_status=run_metadata["status"],
                    root_duration_ms=run_metadata.get("duration_ms"),
                    root_error_type=run_metadata.get("error_type"),
                    compaction_attempts=run_metadata.get("compaction_attempts"),
                )
                if model_ref is None:
                    observations.gaps.append(ObservationGap(code="model_call_ownership_unavailable"))
            except Exception:
                LOG.exception("failed to extract Claude Code observations")
                observations = AgentObservationBundle(
                    source="claude_code",
                    gaps=[ObservationGap(code="observation_parse_failed")],
                )

        response = await self.run(
            body,
            mcp_config=mcp_config,
            skills_path=skills_path,
            model_base_url=model_base_url,
            observation_collector=collect,
        )
        if observations is None:
            observations = AgentObservationBundle(
                source="claude_code",
                gaps=[ObservationGap(code="agent_transcript_unavailable")],
            )
        observations.gaps.append(ObservationGap(code="no_sandbox_runtime"))
        return AgentEpisode(response=response, observations=observations)
