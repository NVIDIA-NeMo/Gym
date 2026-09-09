# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
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


LOG = logging.getLogger(__name__)

# `cline auth` only accepts custom base URLs for OpenAI-compatible providers.
OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"


def _message(index: int, text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _reasoning_token_count(usage: dict[str, Any], *, running_total: bool = False) -> int:
    direct_keys = (
        ("totalReasoningTokens", "reasoningTokens")
        if running_total
        else (
            "reasoningTokens",
            "totalReasoningTokens",
        )
    )
    for key in direct_keys:
        value = usage.get(key)
        if value is not None:
            return int(value or 0)

    details = usage.get("outputTokenDetails") or usage.get("outputTokensDetails") or {}
    if isinstance(details, dict):
        return int(details.get("reasoningTokens") or details.get("reasoning_tokens") or 0)
    return 0


def parse_cline_events(stdout: str) -> tuple[list[Any], dict[str, Any]]:
    output_items: list[Any] = []
    metadata: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    text_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    reasoning_consumed = False
    open_tool_calls: dict[str, str] = {}
    saw_usage = False

    def flush_text() -> None:
        nonlocal reasoning_consumed
        text = "".join(text_chunks)
        text_chunks.clear()
        think = "".join(reasoning_chunks).strip()
        if not text.strip():
            return
        if think:
            text = f"<think>\n{think}\n</think>\n\n{text}"
            reasoning_chunks.clear()
            reasoning_consumed = True
        output_items.append(_message(len(output_items), text))

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue

        rtype = record.get("type")

        if rtype == "run_result":
            metadata["finish_reason"] = record.get("finishReason")
            metadata["iterations"] = record.get("iterations")
            usage = record.get("aggregateUsage") or record.get("usage") or {}
            if isinstance(usage, dict) and usage:
                # The session aggregate survives truncated event streams.
                metadata["input_tokens"] = int(usage.get("inputTokens") or 0) + int(usage.get("cacheReadTokens") or 0)
                metadata["output_tokens"] = int(usage.get("outputTokens") or 0)
                metadata["reasoning_tokens"] = _reasoning_token_count(usage)
                saw_usage = True
            model = record.get("model")
            if isinstance(model, dict) and model.get("id"):
                metadata["model"] = model["id"]
            elif isinstance(model, str) and model:
                metadata["model"] = model
            continue

        if rtype == "error":
            message = record.get("message")
            if message:
                metadata.setdefault("error", str(message))
                LOG.warning("cline reported an error: %s", str(message)[:500])
            continue

        if rtype != "agent_event":
            continue

        event = record.get("event")
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        content_type = event.get("contentType")

        if etype == "content_start" and content_type == "text":
            text_chunks.append(event.get("text") or "")

        elif etype == "content_end" and content_type == "text":
            final = event.get("text")
            if final is not None:
                text_chunks.clear()
                text_chunks.append(final)
            flush_text()

        elif etype == "content_start" and content_type == "reasoning":
            if not event.get("redacted"):
                reasoning_chunks.append(event.get("reasoning") or "")
                reasoning_consumed = False

        elif etype == "content_end" and content_type == "reasoning":
            if reasoning_consumed:
                reasoning_chunks.clear()
                reasoning_consumed = False
            else:
                final = event.get("reasoning")
                if final:
                    reasoning_chunks.clear()
                    reasoning_chunks.append(final)

        elif etype == "content_start" and content_type == "tool":
            flush_text()
            call_id = str(event.get("toolCallId") or f"call-{uuid4().hex[:8]}")
            name = event.get("toolName") or ""
            tool_input = event.get("input")
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else _stringify(tool_input)
            open_tool_calls[call_id] = name
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=name,
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )

        elif etype == "content_end" and content_type == "tool":
            call_id = str(event.get("toolCallId") or "")
            if call_id and call_id not in open_tool_calls:
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments="{}",
                        call_id=call_id,
                        name=event.get("toolName") or "",
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )
            open_tool_calls.pop(call_id, None)
            error = event.get("error")
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id or f"call-{uuid4().hex[:8]}",
                    output=_stringify(error if error else event.get("output")),
                    status="completed",
                )
            )

        elif etype == "usage":
            metadata["input_tokens"] = int(event.get("totalInputTokens") or 0) + int(
                event.get("totalCacheReadTokens") or 0
            )
            metadata["output_tokens"] = int(event.get("totalOutputTokens") or 0)
            metadata["reasoning_tokens"] = _reasoning_token_count(event, running_total=True)
            saw_usage = True

        elif etype == "done":
            metadata.setdefault("finish_reason", event.get("reason"))
            metadata.setdefault("iterations", event.get("iterations"))

        elif etype == "error":
            error = event.get("error")
            text = error.get("message") if isinstance(error, dict) else _stringify(error)
            if text:
                metadata.setdefault("error", text)
                LOG.warning("cline agent error event: %s", str(text)[:500])

    flush_text()
    trailing_think = "".join(reasoning_chunks).strip()
    if trailing_think:
        output_items.append(_message(len(output_items), f"<think>\n{trailing_think}\n</think>"))

    if not saw_usage:
        LOG.debug("cline stream carried no usage events. Token counts reported as 0")

    return output_items, metadata


def quote_prompt(prompt: str) -> str:
    # Cline parses a single word as a command unless the argument contains whitespace.
    return prompt if any(ch.isspace() for ch in prompt) else f"{prompt} "


class ClineHarness:
    def __init__(self, config: AgentHarnessConfig):
        if config.max_turns is not None:
            raise ValueError("ClineHarness does not support max_turns")
        self.config = config

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.config.settings.get("command", "cline"))

    def _workspace_root(self) -> Path:
        root = Path(self.config.settings.get("workspace_root", "outputs/cline_agent/workspaces")).expanduser()
        root /= f"cline_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _repo_dir(self, fallback: Path) -> Path:
        if self.config.workspace is None:
            return fallback
        root = self.config.workspace.expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _effective_provider(self) -> str:
        return self.config.model.provider or OPENAI_COMPATIBLE_PROVIDER

    def _env(self, data_dir: Path, model_base_url: str = "") -> dict[str, str]:
        data = str(data_dir)
        env = {
            **os.environ,
            "CLINE_DATA_DIR": data,
            "CLINE_PROVIDER_SETTINGS_PATH": str(data_dir / "settings" / "providers.json"),
            "CLINE_GLOBAL_SETTINGS_PATH": str(data_dir / "settings" / "global-settings.json"),
            "CLINE_MCP_SETTINGS_PATH": str(data_dir / "settings" / "cline_mcp_settings.json"),
            "CLINE_SESSION_DATA_DIR": str(data_dir / "sessions"),
            "CLINE_DB_DATA_DIR": str(data_dir / "db"),
            "CLINE_TEAM_DATA_DIR": str(data_dir / "teams"),
            "CLINE_HOOKS_LOG_PATH": str(data_dir / "logs" / "hooks.jsonl"),
            # A shared hub daemon would reuse process state across rollouts.
            "CLINE_SESSION_BACKEND_MODE": "local",
        }
        base_url = model_base_url or self.config.model.base_url
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_KEY"] = (
                "EMPTY" if model_base_url else self.config.model.api_key or "EMPTY"
            )  # pragma: allowlist secret
        elif self.config.model.api_key:
            env["OPENAI_API_KEY"] = self.config.model.api_key
        command_permissions = self.config.settings.get("command_permissions", {})
        if command_permissions:
            env["CLINE_COMMAND_PERMISSIONS"] = json.dumps(command_permissions)
        env.update({k: v for k, v in self.config.settings.get("env", {}).items() if v})
        return env

    def _build_auth_command(self, data_dir: Path, model_base_url: str) -> Optional[list[str]]:
        if not model_base_url:
            return None
        if not self.config.model.model:
            raise ValueError("cline_agent requires `model` to be set when `model_server` is configured")
        return [
            *self.command_parts,
            "auth",
            OPENAI_COMPATIBLE_PROVIDER,
            # Cline requires a value, but the Gym model server holds the real credential.
            "--apikey",
            "EMPTY",  # pragma: allowlist secret
            "--modelid",
            self.config.model.model,
            "--baseurl",
            model_base_url,
            "--data-dir",
            str(data_dir),
        ]

    def _build_command(self, project_dir: Path, data_dir: Path, prompt: str) -> list[str]:
        cmd = [
            *self.command_parts,
            "--json",
            "--auto-approve",
            "true",
            "--cwd",
            str(project_dir),
            "--data-dir",
            str(data_dir),
            "--timeout",
            f"{self.config.timeout_seconds:g}",
        ]
        if self.config.model.model:
            cmd += ["-m", self.config.model.model]
        cmd += ["-P", self._effective_provider()]
        thinking = self.config.settings.get("thinking")
        if thinking:
            cmd += ["--thinking", thinking]
        compaction = self.config.settings.get("compaction")
        if compaction:
            cmd += ["--compaction", compaction]
        retries = self.config.settings.get("retries")
        if retries is not None:
            cmd += ["--retries", str(retries)]
        system_prompt_override = self.config.settings.get("system_prompt_override")
        if system_prompt_override:
            cmd += ["-s", system_prompt_override]
        cmd.extend(self.config.settings.get("extra_args", []))
        cmd += ["--", quote_prompt(prompt)]
        return cmd

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process") -> None:
        # Kill the npm shim and native child together so neither keeps stdout open.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    async def _spawn(self, cmd: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[str, str, int]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.CancelledError:
            self._kill_process_group(proc)
            await proc.communicate()
            raise
        except asyncio.TimeoutError:
            self._kill_process_group(proc)
            stdout, stderr = await proc.communicate()
            return stdout.decode(errors="replace"), stderr.decode(errors="replace"), -1
        return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode or 0

    async def _run_cline(
        self, instruction: str, system_prompt: Optional[str], model_base_url: str = ""
    ) -> tuple[list[Any], dict[str, Any], str]:
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_dir = work_dir / ".cline-data"
        data_dir.mkdir(parents=True, exist_ok=True)
        env = self._env(data_dir, model_base_url)

        try:
            auth_cmd = self._build_auth_command(data_dir, model_base_url)
            if auth_cmd:
                _, auth_err, auth_rc = await self._spawn(
                    auth_cmd,
                    project_dir,
                    env,
                    self.config.settings.get("setup_timeout", 300),
                )
                if auth_rc != 0:
                    LOG.warning("cline auth exited %d: %s", auth_rc, auth_err[:500])

            cmd = self._build_command(project_dir, data_dir, prompt)
            stdout, stderr, returncode = await self._spawn(
                cmd,
                project_dir,
                env,
                self.config.timeout_seconds,
            )
            if returncode == -1:
                LOG.warning("cline timed out after %ss", self.config.timeout_seconds)
            elif returncode != 0:
                LOG.warning("cline exited %d: %s", returncode, stderr[:500])

            output_items, metadata = parse_cline_events(stdout)
            if returncode == -1:
                metadata["timed_out"] = True
            model_name = metadata.get("model") or self.config.model.model or ""
            return output_items, metadata, model_name
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def run(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        output_items, metadata, model_name = await self._run_cline(
            user_message,
            system_prompt,
            model_base_url or "",
        )

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("Cline produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = metadata.get("input_tokens", 0)
        output_tokens = metadata.get("output_tokens", 0)
        reasoning_tokens = metadata.get("reasoning_tokens", 0)

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
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=input_tokens + output_tokens,
            ),
        )
