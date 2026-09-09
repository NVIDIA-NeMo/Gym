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


def _think_message(index: int, text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def parse_kilo_events(stdout: str) -> tuple[list[Any], dict[str, int]]:
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    buffered_think: Optional[str] = None
    # Kilo 7.4.15 emits each part twice with the same id.
    seen_part_ids: set[str] = set()
    warned_empty_length = False

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
        part = event.get("part") or {}
        if not isinstance(part, dict):
            part = {}

        part_id = part.get("id")
        if part_id is not None:
            if part_id in seen_part_ids:
                continue
            seen_part_ids.add(part_id)

        if etype == "step_finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            step_output = int(tokens.get("output") or 0)
            output_tokens += step_output
            if part.get("reason") == "length" and not step_output and not warned_empty_length:
                warned_empty_length = True
                LOG.warning(
                    "kilo step stopped on 'length' with no output tokens. max_output_tokens is likely "
                    "too large for the model server's context window"
                )

        elif etype == "reasoning":
            text = (part.get("text") or "").strip()
            if text:
                buffered_think = (buffered_think + "\n" + text) if buffered_think else text

        elif etype == "text":
            text = part.get("text") or ""
            if not text.strip():
                continue
            if buffered_think:
                text = f"<think>\n{buffered_think}\n</think>\n\n{text}"
                buffered_think = None
            output_items.append(_think_message(len(output_items), text))

        elif etype == "tool_use":
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
            output = state.get("output")
            if output is None and state.get("status") == "error":
                output = state.get("error") or ""
            if output is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(output),
                        status="completed",
                    )
                )

        elif etype == "error":
            LOG.warning("kilo run reported error event: %s", str(event.get("error"))[:500])

    # Preserve answers routed through the reasoning channel.
    if buffered_think:
        output_items.append(_think_message(len(output_items), f"<think>\n{buffered_think}\n</think>"))

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


class KiloCodeHarness:
    def __init__(self, config: AgentHarnessConfig):
        if config.model.model is None:
            raise ValueError("KiloCodeHarness requires a model")
        if config.max_turns is not None:
            raise ValueError("KiloCodeHarness does not support max_turns")
        self.config = config

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.config.settings.get("command", "kilo"))

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                KiloCodeHarness._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _workspace_root(self) -> Path:
        root = Path(self.config.settings.get("workspace_root", "outputs/kilocode_agent/workspaces")).expanduser()
        root /= f"kilo_{uuid4().hex[:8]}"
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

    def _effective_model(self) -> str:
        model = self.config.model.model or ""
        provider = self.config.model.provider
        return f"{provider}/{model}" if provider else model

    def _build_kilo_config(self, model_base_url: str = "") -> dict[str, Any]:
        # Kilo resolves configured model names against the provider's model map.
        config = self._deep_merge({}, copy.deepcopy(self.config.settings.get("kilo_config", {})))

        if model_base_url:
            provider_name = self.config.model.provider or "nemo"
            provider = config.setdefault("provider", {}).setdefault(provider_name, {})
            provider.setdefault("npm", "@ai-sdk/openai-compatible")
            options = provider.setdefault("options", {})
            options.setdefault("apiKey", self.config.model.api_key or "EMPTY")  # pragma: allowlist secret
            options["baseURL"] = model_base_url
            model_name = self.config.model.model or ""
            model = provider.setdefault("models", {}).setdefault(model_name, {})
            model.setdefault("name", model_name)
            model.setdefault(
                "limit",
                {
                    "context": self.config.model.settings.get("context_window", 32768),
                    "output": self.config.model.settings.get("max_output_tokens", 8192),
                },
            )
            reasoning_field = self.config.model.settings.get("reasoning_field", "reasoning_content")
            if reasoning_field:
                model.setdefault("interleaved", {"field": reasoning_field})
        else:
            provider_id, _, model_name = (self.config.model.model or "").partition("/")
            provider = (config.get("provider") or {}).get(provider_id)
            if model_name and isinstance(provider, dict):
                provider.setdefault("models", {}).setdefault(model_name, {})
        return config

    def _env(self, data_home: str, config_home: str, model_base_url: str = "") -> dict[str, str]:
        # Kilo otherwise shares daemon, session, and config state across runs.
        env = {
            **os.environ,
            "KILO_NO_DAEMON": "1",
            "KILO_DB": ":memory:",
            "XDG_DATA_HOME": data_home,
            "XDG_CONFIG_HOME": config_home,
        }
        base_url = model_base_url or self.config.model.base_url
        api_key = "EMPTY" if model_base_url else self.config.model.api_key  # pragma: allowlist secret
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        env.update({k: v for k, v in self.config.settings.get("env", {}).items() if v})
        kilo_config = self._build_kilo_config(model_base_url)
        if kilo_config:
            # Environment config wins without changing the repository's kilo.json.
            env["KILO_CONFIG_CONTENT"] = json.dumps(kilo_config)
        return env

    def _build_command(self, project_dir: Path, prompt: str) -> list[str]:
        cmd = [
            *self.command_parts,
            "run",
            "--auto",
            # Disable external plugins and codebase indexing.
            "--pure",
            "--format",
            "json",
            "-m",
            self._effective_model(),
            "--dir",
            str(project_dir),
        ]
        if self.config.settings.get("thinking", False):
            cmd.append("--thinking")
        cmd.extend(self.config.settings.get("extra_args", []))
        cmd += ["--", prompt]
        return cmd

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process") -> None:
        # Kill the npm shim and native child together so neither keeps stdout open.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    async def _run_kilo(
        self, instruction: str, system_prompt: Optional[str], model_base_url: str = ""
    ) -> tuple[list[Any], dict[str, int], str]:
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_home = work_dir / ".kilo-data"
        config_home = work_dir / ".kilo-config"
        data_home.mkdir(parents=True, exist_ok=True)
        config_home.mkdir(parents=True, exist_ok=True)
        env = self._env(str(data_home), str(config_home), model_base_url)
        cmd = self._build_command(project_dir, prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout_seconds)
            except asyncio.TimeoutError:
                self._kill_process_group(proc)
                await proc.communicate()
                LOG.warning("kilo timed out after %ss", self.config.timeout_seconds)
                return [], {"input_tokens": 0, "output_tokens": 0}, self.config.model.model or ""
            except asyncio.CancelledError:
                self._kill_process_group(proc)
                await proc.communicate()
                raise

            if proc.returncode not in (0, None):
                LOG.warning("kilo exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            output_items, usage = parse_kilo_events(stdout.decode(errors="replace"))
            return output_items, usage, self.config.model.model or ""
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

        output_items, usage, model_name = await self._run_kilo(
            user_message,
            system_prompt,
            model_base_url or "",
        )

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("Kilo produced no assistant message. Padding empty output")
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
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )
