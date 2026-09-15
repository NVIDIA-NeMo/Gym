# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import tempfile
from copy import deepcopy
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
from nemo_gym.skills import stage_skills


LOG = logging.getLogger(__name__)


def _toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def toml_dumps(data: dict[str, Any], _prefix: str = "") -> str:
    lines: list[str] = []
    tables: list[tuple[str, dict]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    chunks = ["\n".join(lines)] if lines else []
    for key, value in tables:
        full_key = f"{_prefix}.{_toml_key(key)}" if _prefix else _toml_key(key)
        body = toml_dumps(value, full_key)
        chunks.append(f"[{full_key}]" + (f"\n{body}" if body else ""))
    return "\n\n".join(chunks)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mcp_result_text(item: dict[str, Any]) -> str:
    if item.get("error"):
        return f"error: {item['error']}"
    result = item.get("result")
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if any(texts):
                return "".join(texts)
        return json.dumps(result)
    return "" if result is None else str(result)


def parse_exec_jsonl(stdout: str) -> tuple[list[Any], dict]:
    output_items: list[Any] = []
    buffered_think: Optional[str] = None
    metadata: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
    errors: list[str] = []

    def _add_tool_pair(item: dict[str, Any], name: str, arguments: dict[str, Any], output: str) -> None:
        call_id = str(item.get("id") or f"call-{uuid4().hex[:8]}")
        status = "completed" if item.get("status") != "failed" else "incomplete"
        output_items.append(
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(arguments),
                call_id=call_id,
                name=name,
                type="function_call",
                id=call_id,
                status=status,
            )
        )
        output_items.append(
            NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=output, status="completed")
        )

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        etype = event.get("type")

        if etype == "turn.completed":
            usage = event.get("usage") or {}
            metadata["input_tokens"] += int(usage.get("input_tokens") or 0)
            metadata["output_tokens"] += int(usage.get("output_tokens") or 0)
            metadata["cached_input_tokens"] += int(usage.get("cached_input_tokens") or 0)
            metadata["reasoning_tokens"] += int(usage.get("reasoning_output_tokens") or 0)
            continue

        if etype == "turn.failed":
            message = (event.get("error") or {}).get("message") or "turn failed"
            errors.append(message)
            continue

        if etype != "item.completed":
            continue

        item = event.get("item")
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == "agent_message":
            text = item.get("text") or ""
            if buffered_think:
                text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                buffered_think = None
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=str(item.get("id") or f"msg-{len(output_items)}"),
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
        elif itype == "reasoning":
            think = item.get("text") or ""
            if think:
                buffered_think = (buffered_think + "\n" + think) if buffered_think else think
        elif itype == "command_execution":
            output = item.get("aggregated_output") or ""
            exit_code = item.get("exit_code")
            if exit_code not in (None, 0):
                output = f"{output}\n[exit code: {exit_code}]"
            _add_tool_pair(item, "exec_command", {"cmd": item.get("command") or ""}, output)
        elif itype == "mcp_tool_call":
            _add_tool_pair(item, str(item.get("tool") or ""), item.get("arguments") or {}, _mcp_result_text(item))
        elif itype == "file_change":
            _add_tool_pair(item, "apply_patch", {"changes": item.get("changes")}, item.get("status") or "completed")
        elif itype == "web_search":
            _add_tool_pair(item, "web_search", {"query": item.get("query") or ""}, "")
        elif itype == "todo_list":
            _add_tool_pair(item, "update_plan", {"items": item.get("items") or []}, "")
        elif itype == "error":
            errors.append(item.get("message") or "unknown error")

    # Some reasoning parsers route the final answer through the reasoning channel.
    if buffered_think:
        output_items.append(
            NeMoGymResponseOutputMessage(
                id=f"msg-{len(output_items)}",
                content=[
                    NeMoGymResponseOutputText(
                        type="output_text", text=f"<think>\n{buffered_think}\n</think>", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    if errors:
        metadata["errors"] = errors
    return output_items, metadata


def _kill_process_group(proc: Any) -> None:
    # Kill the npm shim and native child together so neither keeps stdout open.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


class CodexHarness:
    def __init__(self, config: AgentHarnessConfig):
        if config.max_turns is not None:
            raise ValueError("CodexHarness does not support max_turns")
        self.config = config

    def _effective_model(self) -> Optional[str]:
        return self.config.model.model

    def _build_config(
        self,
        base_url: str,
        developer_instructions: Optional[str] = None,
        mcp_servers: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model_provider": "gym",
            "approval_policy": "never",
            "sandbox_mode": self.config.settings.get("sandbox_mode", "danger-full-access"),
            "web_search": "disabled",
            "check_for_update_on_startup": False,
            "analytics": {"enabled": False},
            "history": {"persistence": "none"},
            # Gym model servers cannot execute the tool shapes these features add.
            "features": {"multi_agent": False, "code_mode": False},
            "model_providers": {
                "gym": {
                    "name": "gym",
                    "base_url": base_url,
                    # The subprocess receives this key without using `codex login`.
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                    "stream_idle_timeout_ms": int(
                        self.config.model.settings.get("stream_idle_timeout_ms") or self.config.timeout_seconds * 1000
                    ),
                }
            },
        }
        model = self._effective_model()
        if model:
            config["model"] = model
        reasoning_effort = self.config.model.settings.get("reasoning_effort")
        if reasoning_effort:
            config["model_reasoning_effort"] = reasoning_effort
        if developer_instructions:
            config["developer_instructions"] = developer_instructions
        extra_config = self.config.settings.get("extra_config", {})
        if extra_config:
            config = _deep_merge(config, deepcopy(extra_config))
        if mcp_servers:
            config["mcp_servers"] = {**config.get("mcp_servers", {}), **mcp_servers}
        return config

    def _setup_codex_home(self, config: dict[str, Any], skills_path: Optional[str] = None) -> Path:
        codex_home = Path.home() / ".codex_agent" / uuid4().hex
        codex_home.mkdir(parents=True)
        try:
            (codex_home / "config.toml").write_text(toml_dumps(config))
            if skills_path:
                stage_skills(skills_path, codex_home / "skills")
        except Exception:
            shutil.rmtree(codex_home, ignore_errors=True)
            raise
        return codex_home

    def _build_command(self, instruction: str, cwd: str) -> list[str]:
        return [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            cwd,
            "--",
            instruction,
        ]

    async def _run_codex(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        model_base_url: Optional[str] = None,
    ) -> tuple[str, str]:
        base_url = model_base_url or self.config.model.base_url or "https://api.openai.com/v1"
        model = self._effective_model() or "codex-default"

        config = self._build_config(base_url, developer_instructions=system_prompt, mcp_servers=mcp_servers)

        codex_home: Optional[Path] = None
        scratch_cwd: Optional[str] = None
        try:
            codex_home = self._setup_codex_home(config, skills_path=skills_path)
            cwd = str(self.config.workspace) if self.config.workspace is not None else None
            if cwd is None:
                cwd = scratch_cwd = tempfile.mkdtemp(prefix="nemo_gym_codex_ws_")

            env = {
                **os.environ,
                "CODEX_HOME": str(codex_home),
                "OPENAI_API_KEY": self.config.model.api_key or "local",  # pragma: allowlist secret
            }

            proc = await asyncio.create_subprocess_exec(
                *self._build_command(instruction, cwd),
                stdin=asyncio.subprocess.DEVNULL,  # codex appends piped stdin to the prompt and blocks on it
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Keep the npm shim and native child in one killable group.
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout_seconds)
            except asyncio.TimeoutError:
                _kill_process_group(proc)
                await proc.communicate()
                LOG.warning("codex timed out after %ss", self.config.timeout_seconds)
                return "", model
            except asyncio.CancelledError:
                _kill_process_group(proc)
                await proc.communicate()
                raise

            if proc.returncode not in (0, None):
                LOG.warning("codex exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            LOG.debug("codex stdout (%d chars): %s", len(stdout), stdout[:2000].decode(errors="replace"))
            return stdout.decode(errors="replace"), model
        finally:
            if codex_home is not None:
                shutil.rmtree(codex_home, ignore_errors=True)
            if scratch_cwd is not None:
                shutil.rmtree(scratch_cwd, ignore_errors=True)

    async def run(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        model_base_url: Optional[str] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [part for part in [self.config.system_prompt, input_system] if part]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        stdout, model_name = await self._run_codex(
            user_message,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            skills_path=skills_path,
            model_base_url=model_base_url,
        )
        output_items, usage = parse_exec_jsonl(stdout)

        if usage.get("errors"):
            LOG.warning("codex reported errors: %s", usage["errors"])

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("codex produced no assistant message. Padding empty output")
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
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cached_tokens=usage.get("cached_input_tokens", 0)
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(
                    reasoning_tokens=usage.get("reasoning_tokens", 0)
                ),
                total_tokens=input_tokens + output_tokens,
            ),
        )
