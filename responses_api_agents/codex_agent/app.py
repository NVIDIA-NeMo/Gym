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
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
from asyncio import Semaphore
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import SKILLS_REF_KEY_NAME, get_first_server_config_dict
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
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.skills import stage_skills
from responses_api_agents.codex_agent.setup_codex import ensure_codex


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
        # JSON string escaping is a valid TOML basic string.
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def toml_dumps(data: dict[str, Any], _prefix: str = "") -> str:
    """Serialize a nested dict of scalars/lists/dicts to TOML (the subset Codex config uses)."""
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
    """Convert ``codex exec --json`` JSONL stdout into (output_items, metadata).

    Codex emits ``item.completed`` events for each unit of work (assistant messages, reasoning,
    shell commands, MCP tool calls, file changes, ...) and a terminal ``turn.completed`` carrying
    token usage summed over every model call in the turn. Tool-shaped items are mapped to a
    ``function_call`` + ``function_call_output`` pair so verifiers see one uniform trajectory
    shape across agent harnesses; reasoning is buffered and prepended to the next assistant
    message inside ``<think>`` tags (mirroring the Claude Code agent).
    """
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

    # Some backends route the final answer through the reasoning channel (e.g. a vLLM reasoning
    # parser labeling the closing message as reasoning). If the run ends on buffered reasoning with
    # no assistant message after it, surface it as a think-tagged message rather than dropping it.
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
    """Kill the codex subprocess and every child in its process group.

    Killing only the direct child leaves the npm shim's vendored-binary child alive, holding the
    stdout pipe open — the post-kill ``communicate()`` would then block until the orphan exits.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


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


class CodexAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # When model_server is set, the Codex model provider's base_url is resolved from the Gym model
    # server's URL (every Gym model server speaks the streaming Responses dialect on /v1/responses).
    # When None, openai_base_url is used directly (default: the real OpenAI API).
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    # None -> omit `model` from the generated config and use the Codex CLI's own default. Gym model
    # servers substitute their configured model anyway; set explicitly for direct endpoints.
    model: Optional[str] = None
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"] = "danger-full-access"
    timeout: int = 600
    system_prompt: Optional[str] = None
    reasoning_effort: Optional[str] = None
    # Required: every config pins an explicit npm version so auto-install is reproducible and cannot
    # silently drift as new Codex releases land. Version bumps are then explicit, tested changes.
    codex_version: str
    # Working root handed to `codex exec --cd`. None -> a fresh temp dir per request, removed
    # afterwards, so rollouts cannot see each other's files.
    cwd: Optional[str] = None
    # Provider stream idle timeout. Gym model servers emit the synthesized SSE only once the full
    # response is computed, so the idle budget must cover an entire generation; None -> timeout * 1000.
    stream_idle_timeout_ms: Optional[int] = None
    # Extra config.toml content deep-merged over the generated base config (mcp_servers, features,
    # tools, model_verbosity, ...). Per-rollout Gym MCP entries take precedence on name collisions.
    extra_config: dict[str, Any] = Field(default_factory=dict)


class CodexAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CodexAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class CodexAgent(SimpleResponsesAPIAgent):
    config: CodexAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_codex(self.config.codex_version)
        try:
            ver = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            LOG.warning("codex version: %s", ver or "(unknown)")
        except Exception as exc:
            LOG.warning("could not determine codex version: %s", exc)

    def _resolve_call_base_url(self, rollout_id: Optional[str]) -> str:
        """Provider base_url for the CLI's model calls (Codex appends ``/responses`` to it).

        A Gym model server gets the per-rollout capture prefix plus the ``/v1`` suffix; a direct
        endpoint (``model_server`` unset) is used verbatim and never prefixed — it has no
        prefix-stripping middleware, so a prefix would 404 every call.
        """
        if self.config.model_server:
            return self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        # Mirrors claude_code_agent's null anthropic_base_url: null means the provider's real API.
        return self.config.openai_base_url or "https://api.openai.com/v1"

    def _effective_model(self) -> Optional[str]:
        """The model name written into the generated config (and reported on the response).

        An explicit (even unknown) model name keeps Codex from applying model-family feature
        gating: for models it recognizes, Codex may switch tools into code-mode carriers that
        models served through a Gym model server cannot drive. Gym model servers substitute their
        own configured model anyway, so a placeholder never reaches the backend. Returns None only
        for a direct endpoint with no configured model, where Codex uses its own default.
        """
        if self.config.model:
            return self.config.model
        if self.config.model_server:
            return "gym-policy-model"
        return None

    def _build_config(
        self,
        base_url: str,
        developer_instructions: Optional[str] = None,
        mcp_servers: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Assemble the per-run CODEX_HOME/config.toml content.

        The base config pins a Gym-owned model provider (bypassing Codex's login flow), disables
        everything that would make a rollout depend on ambient host state or phone home (analytics,
        update checks, on-disk history), and turns off the tools a Gym-served model cannot execute
        (server-side web search, multi-agent). ``extra_config`` is deep-merged on top; the
        per-rollout Gym MCP entries are applied last so they win name collisions.
        """
        config: dict[str, Any] = {
            "model_provider": "gym",
            "approval_policy": "never",
            "sandbox_mode": self.config.sandbox_mode,
            "web_search": "disabled",
            "check_for_update_on_startup": False,
            "analytics": {"enabled": False},
            "history": {"persistence": "none"},
            # multi_agent and code_mode add tool shapes (namespace fan-out, custom JS-exec tools)
            # that models served through a Gym model server cannot execute or express.
            "features": {"multi_agent": False, "code_mode": False},
            "model_providers": {
                "gym": {
                    "name": "gym",
                    "base_url": base_url,
                    # A custom provider reads its API key only from the env var named here; the
                    # agent sets it on the codex subprocess from `openai_api_key` (see _run_codex),
                    # so no `codex login` is ever needed.
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                    "stream_idle_timeout_ms": self.config.stream_idle_timeout_ms or self.config.timeout * 1000,
                }
            },
        }
        model = self._effective_model()
        if model:
            config["model"] = model
        if self.config.reasoning_effort:
            config["model_reasoning_effort"] = self.config.reasoning_effort
        if developer_instructions:
            config["developer_instructions"] = developer_instructions
        if self.config.extra_config:
            config = _deep_merge(config, deepcopy(self.config.extra_config))
        if mcp_servers:
            config["mcp_servers"] = {**config.get("mcp_servers", {}), **mcp_servers}
        return config

    def _setup_codex_home(self, config: dict[str, Any], skills_path: Optional[str] = None) -> Path:
        """Create a per-run CODEX_HOME and stage config.toml (and optionally skills) into it.

        The directory lives for the duration of a single ``_run_codex`` call. When ``skills_path``
        is provided, the directory of skills is copied into ``<home>/skills/`` where Codex's native
        skill discovery picks them up. Each request gets its own ephemeral copy, so concurrent
        requests with different skills do not contaminate one another. If setup fails partway
        (e.g. a bad ``skills_path``), the partially-created dir is removed before re-raising.
        """
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
        """Construct the ``codex exec`` argv.

        ``--json`` emits machine-readable JSONL events; ``--ephemeral`` skips session persistence;
        ``--skip-git-repo-check`` allows running in the per-rollout scratch dir. Sandboxing and
        approvals are pinned in the generated config.toml (``approval_policy = "never"``), not argv.
        The ``--`` separator keeps prompts from being parsed as flags or subcommands.
        """
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
        rollout_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """Run ``codex exec --json`` and return (stdout, model_name).

        When ``rollout_id`` is set and a model server is configured, the per-rollout capture prefix
        is applied to the provider base_url so the CLI's streaming /v1/responses calls correlate to
        this rollout.
        """
        base_url = self._resolve_call_base_url(rollout_id)
        # Report the name the config actually pins (so response.model matches what Codex was told);
        # falls back to a sentinel only for a direct endpoint that lets Codex pick its own default.
        model = self._effective_model() or "codex-default"

        config = self._build_config(base_url, developer_instructions=system_prompt, mcp_servers=mcp_servers)

        codex_home: Optional[Path] = None
        scratch_cwd: Optional[str] = None
        try:
            # Inside the try so a bad skills_path (raising in stage_skills) still cleans up the
            # partially-created home in the finally rather than leaking it per failing request.
            codex_home = self._setup_codex_home(config, skills_path=skills_path)
            cwd = self.config.cwd
            if cwd is None:
                cwd = scratch_cwd = tempfile.mkdtemp(prefix="nemo_gym_codex_ws_")

            env = {
                **os.environ,
                "CODEX_HOME": str(codex_home),
                # The provider's `env_key` in the generated config.toml; always set from config so
                # a key inherited from the server's environment can never leak into a rollout.
                "OPENAI_API_KEY": self.config.openai_api_key or "local",  # pragma: allowlist secret
            }

            proc = await asyncio.create_subprocess_exec(
                *self._build_command(instruction, cwd),
                stdin=asyncio.subprocess.DEVNULL,  # codex appends piped stdin to the prompt and blocks on it
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Own process group: `codex` on PATH is an npm shim whose child (the vendored
                # binary) must die with it, or it keeps the stdout pipe open past the kill below.
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                _kill_process_group(proc)
                await proc.communicate()
                LOG.warning("codex timed out after %ds", self.config.timeout)
                return "", model

            if proc.returncode not in (0, None):
                LOG.warning("codex exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            LOG.debug("codex stdout (%d chars): %s", len(stdout), stdout[:2000].decode(errors="replace"))
            return stdout.decode(errors="replace"), model
        finally:
            if codex_home is not None:
                shutil.rmtree(codex_home, ignore_errors=True)
            if scratch_cwd is not None:
                shutil.rmtree(scratch_cwd, ignore_errors=True)

    def _resources_server_base_url(self) -> str:
        cfg = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(cfg)

    def _rollout_mcp_servers(self, seed_response_json: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Per-rollout ``mcp_servers`` config.toml entries from /seed_session MCP metadata.

        Codex reaches Gym MCP tools over streamable HTTP; the per-rollout session token rides on a
        custom header via ``http_headers``.
        """
        metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None

        server_name = str(metadata.get("server_name") or self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        entry: dict[str, Any] = {
            "url": f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}",
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            entry["http_headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning(
                "MCP seed metadata for %r has no headers; the tool endpoint will be called without a "
                "session token and will reject the calls.",
                server_name,
            )
        return {server_name: entry}

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        stdout, model_name = await self._run_codex(
            user_message,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            skills_path=skills_path,
            rollout_id=rollout_id,
        )
        output_items, usage = parse_exec_jsonl(stdout)

        if usage.get("errors"):
            LOG.warning("codex reported errors: %s", usage["errors"])

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("codex produced no assistant message; padding empty output")
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

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        return await self._create_response(body)

    async def run(self, request: Request, body: CodexAgentRunRequest) -> CodexAgentVerifyResponse:
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

            # The run-level skills_ref (stamped by rollout collection) rides on the request body
            # (extra="allow"). Pass its path straight into _create_response so the CLI invocation
            # can stage the skills into its per-request CODEX_HOME.
            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")
            rollout_id = self.rollout_id_from_run(body)

            agent_resp = await self._create_response(
                body.responses_create_params,
                mcp_servers=self._rollout_mcp_servers(seed_resp_json),
                skills_path=skills_path,
                rollout_id=rollout_id,
            )
            agent_resp_json = agent_resp.model_dump(mode="json")

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
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

            return CodexAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    CodexAgent.run_webserver()
