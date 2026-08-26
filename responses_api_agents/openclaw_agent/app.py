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
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import Any, Callable, ClassVar, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    NEMO_GYM_MCP_METADATA_KEY,
    NEMO_GYM_MCP_TOOL_CALL_PROVENANCE_KEY,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
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
    AgentObservationBundle,
    ObservationGap,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.openclaw_agent.observability import (
    OPENCLAW_OBSERVATION_SOURCE,
    OpenClawSessionTree,
    build_openclaw_observation_tree,
    build_openclaw_observations,
    discover_openclaw_session_tree,
)
from responses_api_agents.openclaw_agent.setup_openclaw import ensure_openclaw


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


def _decode_last_json_dict_suffix(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and not text[start + consumed :].strip():
            return obj
    return None


def _text_from_openclaw_payloads(envelope: dict[str, Any]) -> str:
    payloads = envelope.get("payloads")
    if not isinstance(payloads, list):
        payloads = []
    parts = [p["text"].strip() for p in payloads if isinstance(p, dict) and (p.get("text") or "").strip()]
    if parts:
        return "\n\n".join(parts)
    final = (envelope.get("meta") or {}).get("finalAssistantVisibleText")
    return final.strip() if isinstance(final, str) else ""


def parse_openclaw_output(stdout: str) -> tuple[list[Any], dict[str, int]]:
    envelope = _decode_last_json_dict_suffix(stdout)
    if not envelope:
        return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    text = _text_from_openclaw_payloads(envelope)
    output_items: list[Any] = []
    if text:
        output_items.append(
            NeMoGymResponseOutputMessage(
                id="msg-0",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    agent_meta = meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    usage = agent_meta.get("usage") if isinstance(agent_meta.get("usage"), dict) else {}
    cache_read = int(usage.get("cacheRead") or 0)
    input_tokens = int(usage.get("input") or 0) + cache_read
    output_tokens = int(usage.get("output") or 0)
    return output_items, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cache_read,
    }


def parse_openclaw_session_items(events: list[dict[str, Any]], *, include_input: bool = False) -> list[Any]:
    """Convert OpenClaw session events into Gym conversation items."""
    output_items: list[Any] = []
    for event in events:
        event_id = event.get("id")
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if include_input and role in {"user", "system", "developer"}:
            text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"]
                )
            if text:
                output_items.append(NeMoGymEasyInputMessage(role=role, content=text))
            continue
        if not include_input and not isinstance(content, list):
            continue

        if role == "assistant":
            reasoning = []
            for key in ("reasoning_content", "reasoning_text", "thinking"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    reasoning.append(value)
            message_reasoning = message.get("reasoning")
            if isinstance(message_reasoning, str) and message_reasoning:
                reasoning.append(message_reasoning)
            elif isinstance(message_reasoning, dict):
                for key in ("content", "text", "summary"):
                    value = message_reasoning.get(key)
                    if isinstance(value, str) and value:
                        reasoning.append(value)
            if isinstance(content, list):
                reasoning.extend(
                    text
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") in {"thinking", "reasoning"}
                    and isinstance((text := block.get("thinking") or block.get("text") or block.get("reasoning")), str)
                    and text
                )
            if include_input and reasoning:
                output_items.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{event_id or len(output_items)}",
                        summary=[NeMoGymSummary(text="\n".join(reasoning), type="summary_text")],
                    )
                )

            texts = [content] if include_input and isinstance(content, str) and content else []
            if isinstance(content, list):
                texts = [
                    block["text"] for block in content if isinstance(block, dict) and (block.get("text") or "").strip()
                ]
                if include_input:
                    texts = [
                        block["text"]
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") not in {"thinking", "reasoning", "toolCall"}
                        and isinstance(block.get("text"), str)
                        and block["text"].strip()
                    ]
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
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
                if include_input and args is None:
                    args = block.get("partialArgs")
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
            call_id = message.get("toolCallId") or (message.get("tool_call_id") if include_input else "") or ""
            result_text = content if include_input and isinstance(content, str) else ""
            if isinstance(content, list):
                result_text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if include_input and not result_text and message.get("details") is not None:
                result_text = json.dumps(message["details"], ensure_ascii=False)
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="completed",
                )
            )

    return output_items


def openclaw_session_conversation(
    events: list[dict[str, Any]],
    *,
    input_items: list[Any] | None = None,
    fallback_output: list[Any] | None = None,
) -> list[Any]:
    """Prefer retained transcript items and fill only evidence missing from the artifact."""
    conversation = parse_openclaw_session_items(events, include_input=True)
    inputs = input_items or []
    fallback = fallback_output or []
    if not conversation:
        return [*inputs, *fallback]
    retained_roles = {
        role for item in conversation if (role := getattr(item, "role", None)) in {"user", "system", "developer"}
    }
    missing_inputs = (
        [item for item in inputs if getattr(item, "role", None) not in retained_roles] if retained_roles else inputs
    )
    if missing_inputs:
        conversation = [*missing_inputs, *conversation]
    if fallback and not any(
        getattr(item, "role", None) == "assistant"
        or getattr(item, "type", None) in {"reasoning", "function_call", "function_call_output"}
        for item in conversation
    ):
        conversation.extend(fallback)
    return conversation


def parse_openclaw_session(session_text: str) -> list[Any]:
    """Convert an OpenClaw session .jsonl into Gym output items, including tool calls."""
    return parse_openclaw_session_items(parse_openclaw_session_events(session_text))


def parse_openclaw_session_events(session_text: str) -> list[dict[str, Any]]:
    events = []
    for line in session_text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            event = {"raw": line}
        events.append(event if isinstance(event, dict) else {"raw": line})
    return events


def openclaw_mcp_tool_call_provenance(events: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Extract canonical MCP identities from OpenClaw's tool-result metadata."""
    provenance: dict[str, dict[str, str]] = {}
    for event in events:
        if event.get("type") != "message":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "toolResult":
            continue
        call_id = message.get("toolCallId") or message.get("tool_call_id")
        details = message.get("details")
        if not isinstance(call_id, str) or not call_id or not isinstance(details, dict):
            continue
        server_name = details.get("mcpServer")
        tool_name = details.get("mcpTool")
        if isinstance(server_name, str) and server_name and isinstance(tool_name, str) and tool_name:
            provenance[call_id] = {
                "server_name": server_name,
                "tool_name": tool_name,
            }
    return provenance


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


class OpenClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    command: str = "openclaw"
    model: str = "nvinf/nvidia/meta/llama-3.3-70b-instruct"
    node_bin_dir: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/openclaw_agent/workspaces"
    openclaw_agent_id: str = "main"
    thinking: str = "off"
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    openclaw_config: dict[str, Any] = Field(default_factory=dict)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # required: every config must pin an explicit version so runs are reproducible and cannot silently drift
    openclaw_version: str

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenClawAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenClawAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class OpenClawAgent(SimpleResponsesAPIAgent):
    """Runs the OpenClaw CLI (openclaw agent --local --json)"""

    config: OpenClawAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # deny the interactive "message" channel so the headless agent finishes
    _HEADLESS_TOOL_DENY: ClassVar[tuple[str, ...]] = ("message",)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_openclaw(self.config.openclaw_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("openclaw command %r is not on PATH yet", self.config.command)

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                OpenClawAgent._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _merge_headless_tool_denies(self, cfg: dict[str, Any]) -> None:
        tools = cfg.setdefault("tools", {})
        deny = tools.get("deny")
        if not isinstance(deny, list):
            deny = []
        merged = list(dict.fromkeys([item for item in deny if isinstance(item, str)] + list(self._HEADLESS_TOOL_DENY)))
        tools["deny"] = merged

    def _build_openclaw_config(
        self,
        base: dict[str, Any],
        rollout_id: Optional[str] = None,
        *,
        openclaw_mcp_servers: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        cfg = copy.deepcopy(base)
        self._deep_merge(cfg, copy.deepcopy(self.config.openclaw_config))
        if self.config.model_server:
            providers = cfg.setdefault("models", {}).setdefault("providers", {})
            nemo = providers.setdefault("nemo", {})
            model_entry = {
                "id": self.config.model,
                "name": self.config.model,
                "api": "openai-completions",
                "reasoning": True,
                "input": ["text"],
            }
            if self.config.context_window is not None:
                model_entry["contextWindow"] = self.config.context_window
            if self.config.max_output_tokens is not None:
                model_entry["maxTokens"] = self.config.max_output_tokens
            nemo.update(
                {
                    "api": "openai-completions",
                    "baseUrl": self._resolve_model_base_url(rollout_id),
                    "apiKey": "EMPTY",  # pragma: allowlist secret
                    "models": [model_entry],
                }
            )
        if openclaw_mcp_servers:
            mcp = cfg.setdefault("mcp", {})
            if not isinstance(mcp, dict):
                raise ValueError("OpenClaw config field mcp must be an object")
            servers = mcp.setdefault("servers", {})
            if not isinstance(servers, dict):
                raise ValueError("OpenClaw config field mcp.servers must be an object")
            servers.update(copy.deepcopy(openclaw_mcp_servers))
        self._merge_headless_tool_denies(cfg)
        return cfg

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _resources_server_base_url(self) -> str:
        cfg = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(cfg)

    def _openclaw_mcp_servers_from_seed(self, seed_response_json: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Translate rollout-specific Gym MCP metadata into OpenClaw config."""
        if NEMO_GYM_MCP_METADATA_KEY not in seed_response_json:
            return None
        metadata = seed_response_json[NEMO_GYM_MCP_METADATA_KEY]
        if not isinstance(metadata, dict):
            raise ValueError("MCP seed metadata must be an object")

        server_name = metadata.get("server_name") or self.config.resources_server.name
        if not isinstance(server_name, str) or not server_name:
            raise ValueError("MCP seed metadata server_name must be a non-empty string")

        url_path = metadata.get("url_path") or "/mcp"
        if not isinstance(url_path, str):
            raise ValueError("MCP seed metadata url_path must be a string")

        transport = metadata.get("transport") or "http"
        if not isinstance(transport, str):
            raise ValueError("MCP seed metadata transport must be a string")
        openclaw_transport = {
            "http": "streamable-http",
            "streamable-http": "streamable-http",
            "sse": "sse",
        }.get(transport)
        if openclaw_transport is None:
            raise ValueError("MCP seed metadata transport is not supported by OpenClaw")

        server_config: dict[str, Any] = {
            "url": f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}",
            "transport": openclaw_transport,
        }
        headers = metadata.get("headers")
        if headers is not None and not isinstance(headers, dict):
            raise ValueError("MCP seed metadata headers must be an object")
        normalized_headers: dict[str, str] = {}
        if isinstance(headers, dict):
            for key, value in headers.items():
                if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
                    raise ValueError("MCP seed metadata headers must contain scalar values")
                normalized_headers[key] = str(value)
        if normalized_headers:
            server_config["headers"] = normalized_headers
        else:
            LOG.warning(
                "MCP seed metadata for %r has no headers; the tool endpoint will be called without a "
                "session token and will reject the calls.",
                server_name,
            )
        return {server_name: server_config}

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"openclaw_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, home: Path) -> dict[str, str]:
        env = {
            **os.environ,
            "HOME": str(home),
            "OPENCLAW_TELEMETRY": "0",
            "CLAWHUB_DISABLE_TELEMETRY": "1",
        }
        if self.config.node_bin_dir:
            env["PATH"] = f"{self.config.node_bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    async def _run_exec(
        self, args: list[str], *, cwd: Optional[str], env: dict[str, str], timeout: int
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"Timed out after {timeout}s: {shlex.join(args)}") from None
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    @staticmethod
    def _session_file(envelope: Optional[dict[str, Any]]) -> Optional[Path]:
        meta = (envelope or {}).get("meta") if isinstance(envelope, dict) else None
        agent_meta = meta.get("agentMeta") if isinstance(meta, dict) else None
        session_file = agent_meta.get("sessionFile") if isinstance(agent_meta, dict) else None
        return Path(session_file) if isinstance(session_file, str) and session_file else None

    async def _run_openclaw(
        self,
        instruction: str,
        system_prompt: Optional[str],
        rollout_id: Optional[str] = None,
        observation_collector: Optional[
            Callable[[str, list[dict[str, Any]], OpenClawSessionTree, list[ObservationGap]], None]
        ] = None,
        openclaw_mcp_servers: Optional[dict[str, Any]] = None,
    ) -> tuple[list[Any], dict[str, int], str, dict[str, dict[str, str]]]:
        """Set up and run OpenClaw, returning output, usage, model, and MCP provenance."""
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        home = work_dir / ".openclaw-home"
        home.mkdir(parents=True, exist_ok=True)
        env = self._env(home)

        try:
            code, _, stderr = await self._run_exec(
                [*self.config.command_parts, "onboard", "--non-interactive", "--accept-risk", "--skip-health"],
                cwd=str(work_dir),
                env=env,
                timeout=self.config.setup_timeout,
            )
            if code:
                raise RuntimeError(f"openclaw onboard exited {code}: {stderr}")

            config_path = home / ".openclaw" / "openclaw.json"
            if not config_path.is_file():
                raise RuntimeError(f"openclaw onboard did not produce a config at {config_path}: {stderr}")
            base_cfg = json.loads(config_path.read_text())
            config_path.write_text(
                json.dumps(
                    self._build_openclaw_config(
                        base_cfg,
                        rollout_id,
                        openclaw_mcp_servers=openclaw_mcp_servers,
                    ),
                    indent=2,
                )
                + "\n"
            )

            cmd = [
                *self.config.command_parts,
                "agent",
                "--local",
                "--json",
                "--agent",
                self.config.openclaw_agent_id,
                "--thinking",
                self.config.thinking,
                "--model",
                self._effective_model(),
                "--message",
                prompt,
                *self.config.extra_args,
            ]
            code, stdout, stderr = await self._run_exec(cmd, cwd=str(work_dir), env=env, timeout=self.config.timeout)
            if code:
                LOG.warning("openclaw exited %d: %s", code, stderr)
            LOG.debug("openclaw stdout (%d chars): %s", len(stdout), stdout[:2000])

            fallback_items, usage = parse_openclaw_output(stdout)
            envelope = _decode_last_json_dict_suffix(stdout)

            output_items: list[Any] = []
            mcp_tool_call_provenance: dict[str, dict[str, str]] = {}
            session_path = self._session_file(envelope)
            if session_path and session_path.is_file():
                session_text = session_path.read_text(errors="replace")
                session_events = parse_openclaw_session_events(session_text)
                output_items = parse_openclaw_session_items(session_events)
                mcp_tool_call_provenance = openclaw_mcp_tool_call_provenance(session_events)
                if observation_collector is not None:
                    try:
                        native_session_id = next(
                            (
                                event.get("id")
                                for event in session_events
                                if event.get("type") == "session" and isinstance(event.get("id"), str)
                            ),
                            session_path.stem,
                        )
                        session_tree, tree_gaps = discover_openclaw_session_tree(
                            home / ".openclaw" / "agents",
                            native_session_id,
                        )
                        observation_collector(native_session_id, session_events, session_tree, tree_gaps)
                    except Exception:
                        LOG.exception("failed to record OpenClaw session artifact")
            if not output_items:
                output_items = fallback_items
            return output_items, usage, self.config.model, mcp_tool_call_provenance
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[
            Callable[[str, list[dict[str, Any]], OpenClawSessionTree, list[ObservationGap]], None]
        ] = None,
        output_collector: Optional[Callable[[list[Any]], None]] = None,
        openclaw_mcp_servers: Optional[dict[str, Any]] = None,
    ) -> tuple[NeMoGymResponse, dict[str, dict[str, str]]]:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        try:
            output_items, usage, model_name, mcp_tool_call_provenance = await self._run_openclaw(
                user_message,
                system_prompt,
                rollout_id=rollout_id,
                observation_collector=observation_collector,
                openclaw_mcp_servers=openclaw_mcp_servers,
            )
        except TimeoutError:
            LOG.warning("OpenClaw timed out, padding empty output so the rollout scores instead of erroring")
            output_items, usage, model_name = [], {"input_tokens": 0, "output_tokens": 0}, self.config.model
            mcp_tool_call_provenance = {}

        if output_collector is not None:
            output_collector(list(output_items))
        if not output_items:
            LOG.warning("OpenClaw produced no assistant message. Padding empty output")
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
        cached_tokens = usage.get("cached_tokens", 0)

        return (
            NeMoGymResponse(
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
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                    output_tokens=output_tokens,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=input_tokens + output_tokens,
                ),
            ),
            mcp_tool_call_provenance,
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if not isinstance(rollout_id, str):
            response, _ = await self._create_response(body)
            return response

        episode, _ = await self._create_episode(
            body,
            rollout_id=rollout_id,
        )
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str,
        openclaw_mcp_servers: Optional[dict[str, Any]] = None,
    ) -> tuple[AgentEpisode, dict[str, dict[str, str]]]:
        session_id: Optional[str] = None
        session_events: list[dict[str, Any]] = []
        session_tree: OpenClawSessionTree = []
        tree_gaps: list[ObservationGap] = []
        input_items: list[Any] = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)]
            if isinstance(body.input, str)
            else list(body.input)
        )
        observed_output: list[Any] = []

        def collect(
            value: str,
            events: list[dict[str, Any]],
            tree: OpenClawSessionTree,
            gaps: list[ObservationGap],
        ) -> None:
            nonlocal session_id, session_events, session_tree, tree_gaps
            session_id = value
            session_events = events
            session_tree = tree
            tree_gaps = gaps

        def collect_output(value: list[Any]) -> None:
            observed_output.extend(value)

        response, mcp_tool_call_provenance = await self._create_response(
            body,
            rollout_id=rollout_id,
            observation_collector=collect,
            output_collector=collect_output,
            openclaw_mcp_servers=openclaw_mcp_servers,
        )
        try:
            if session_tree:
                tree_inputs = []
                for invocation_id, parent_id, events in session_tree:
                    conversation = openclaw_session_conversation(
                        events,
                        input_items=input_items if parent_id is None else None,
                        fallback_output=observed_output if parent_id is None else None,
                    )
                    tree_inputs.append((invocation_id, parent_id, conversation, events))
                observations = build_openclaw_observation_tree(
                    tree_inputs,
                    model_ref=self.config.model_server,
                )
                observations.gaps.extend(tree_gaps)
            else:
                transcript_available = any(event.get("type") == "message" for event in session_events)
                observations = build_openclaw_observations(
                    session_id or response.id,
                    openclaw_session_conversation(
                        session_events,
                        input_items=input_items,
                        fallback_output=observed_output,
                    ),
                    session_events,
                    transcript_available=transcript_available,
                    model_ref=self.config.model_server,
                )
                if any(gap.code == "subagent_hierarchy_unavailable" for gap in tree_gaps):
                    observations.gaps = [
                        gap for gap in observations.gaps if gap.code != "subagent_hierarchy_unavailable"
                    ]
                observations.gaps.extend(tree_gaps)
        except Exception:
            LOG.exception("failed to build OpenClaw observations")
            observations = AgentObservationBundle(
                source=OPENCLAW_OBSERVATION_SOURCE,
                gaps=[ObservationGap(code="observation_capture_failed")],
            )
        return AgentEpisode(response=response, observations=observations), mcp_tool_call_provenance

    async def run(self, request: Request, body: OpenClawAgentRunRequest) -> OpenClawAgentVerifyResponse:
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
            seed_resp_json = await get_response_json(seed_resp)

            rollout_id = self.rollout_id_from_run(body)
            openclaw_mcp_servers = self._openclaw_mcp_servers_from_seed(seed_resp_json)

            if rollout_id is not None:
                episode, mcp_tool_call_provenance = await self._create_episode(
                    body.responses_create_params,
                    rollout_id=rollout_id,
                    openclaw_mcp_servers=openclaw_mcp_servers,
                )
                agent_resp, observations = episode.response, episode.observations
            else:
                agent_resp, mcp_tool_call_provenance = await self._create_response(
                    body.responses_create_params,
                    openclaw_mcp_servers=openclaw_mcp_servers,
                )
                observations = None
            agent_resp_json = agent_resp.model_dump(mode="json")

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump()
                | {
                    "response": agent_resp_json,
                    NEMO_GYM_MCP_TOOL_CALL_PROVENANCE_KEY: mcp_tool_call_provenance,
                },
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

            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return OpenClawAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    OpenClawAgent.run_webserver()
