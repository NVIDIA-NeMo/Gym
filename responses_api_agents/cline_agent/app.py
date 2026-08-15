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
import shlex
import shutil
import signal
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
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.cline_agent.setup_cline import ensure_cline


LOG = logging.getLogger(__name__)

# Provider id the agent authenticates when a Gym model server is configured. `cline auth` only
# accepts a base URL for the OpenAI and OpenAI-compatible providers, and every Gym model server
# speaks the OpenAI Chat Completions dialect, so this is the provider that can be pointed at one.
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
    """Render a tool output/error payload as text for a function_call_output item."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def parse_cline_events(stdout: str) -> tuple[list[Any], dict[str, Any]]:
    """Convert ``cline --json`` stdout into (output_items, metadata).

    ``cline --json`` writes one JSON object per line. Two record types carry the trajectory:

    - ``{"type": "agent_event", "event": {...}}`` — the agent loop's own events. ``content_start``
      with ``contentType: "text"`` streams assistant text in chunks; the matching ``content_end``
      carries the final text for the turn, so the message is taken from ``content_end`` and the
      chunks only serve as a fallback for a truncated stream. ``contentType: "tool"`` brackets one
      tool call: ``content_start`` has ``toolName``/``toolCallId``/``input``, ``content_end`` has
      ``output`` or ``error``. ``usage`` events carry running token totals, ``done`` the finish
      reason.
    - ``{"type": "run_result", ...}`` — the final summary (``finishReason``, ``iterations``,
      aggregate ``usage``, final ``text``, resolved ``model``).

    ``hook_event`` records duplicate tool-call boundaries the agent events already cover and are
    ignored. Reasoning arrives as ``contentType: "reasoning"`` and becomes a ``<think>`` block on
    the assistant message of the turn that produced it, matching the other CLI agents. Verified
    against cline 3.0.55.
    """
    output_items: list[Any] = []
    metadata: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    # Text/reasoning chunks for the turn being streamed; the matching content_end supersedes them.
    text_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    # True once the current turn's reasoning has been folded into a message, so the trailing
    # content_end for that same reasoning is not attached a second time (see below).
    reasoning_consumed = False
    # toolCallId -> tool name for calls whose content_start was seen. Cline emits start/end pairs,
    # but a stream cut short can leave a start without an end and a malformed one an end without a
    # start, so both halves are tracked rather than assumed.
    open_tool_calls: dict[str, str] = {}
    saw_usage = False

    def flush_text() -> None:
        """Emit the buffered assistant text for the current turn, with any reasoning attached.

        Cline closes the turn's reasoning *after* its text (``content_end`` for text, then
        ``content_end`` for reasoning), so the think block is taken from the reasoning chunks
        streamed so far rather than from that trailing event, which would otherwise land on the
        following turn.
        """
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
                # The aggregate is authoritative: Cline computes it from the session rather than
                # summing per-turn events, which a truncated stream can drop.
                metadata["input_tokens"] = int(usage.get("inputTokens") or 0) + int(usage.get("cacheReadTokens") or 0)
                metadata["output_tokens"] = int(usage.get("outputTokens") or 0)
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
            # hook_event records mirror tool boundaries the agent events already carry.
            continue

        event = record.get("event")
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        content_type = event.get("contentType")

        if etype == "content_start" and content_type == "text":
            text_chunks.append(event.get("text") or "")

        elif etype == "content_end" and content_type == "text":
            # The turn's final text; it supersedes the streamed chunks.
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
            # Closes reasoning the turn's message already carries (flush_text ran first), so this
            # only matters when there was no text to attach it to: keep the final text for a
            # reasoning-only turn, which some vLLM parsers produce by routing the whole answer
            # through the reasoning channel.
            if reasoning_consumed:
                reasoning_chunks.clear()
                reasoning_consumed = False
            else:
                final = event.get("reasoning")
                if final:
                    reasoning_chunks.clear()
                    reasoning_chunks.append(final)

        elif etype == "content_start" and content_type == "tool":
            # Text streamed before a tool call belongs to the turn that made it, so it is emitted
            # ahead of the call to keep the trajectory ordered.
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
                # A result with no recorded call: emit the call so the output is not orphaned.
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
            # Running totals for the session so far; run_result's aggregate wins when the run ends
            # normally, and these are what remains when it does not.
            metadata["input_tokens"] = int(event.get("totalInputTokens") or 0) + int(
                event.get("totalCacheReadTokens") or 0
            )
            metadata["output_tokens"] = int(event.get("totalOutputTokens") or 0)
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

    # A stream cut off mid-turn leaves text with no content_end; surface it rather than drop it.
    flush_text()
    trailing_think = "".join(reasoning_chunks).strip()
    if trailing_think:
        # Reasoning with no message to attach it to (a reasoning-only turn, or a truncated
        # stream) is surfaced on its own so the trajectory does not silently lose it.
        output_items.append(_message(len(output_items), f"<think>\n{trailing_think}\n</think>"))

    if not saw_usage:
        LOG.debug("cline stream carried no usage events; token counts reported as 0")

    return output_items, metadata


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


def quote_prompt(prompt: str) -> str:
    """Return a prompt Cline's CLI accepts as a prompt rather than as a command.

    Shells strip quotes before argv, so Cline decides a bare argument was quoted by looking for
    whitespace in it (``promptArgLooksQuoted`` in the CLI). A single-word prompt like ``hello`` is
    rejected with "Unknown command or unquoted prompt" — including after ``--``, which does not
    exempt it. Padding with a trailing space is the documented shape's observable property and
    leaves the prompt text itself unchanged. Verified against cline 3.0.55.
    """
    return prompt if any(ch.isspace() for ch in prompt) else f"{prompt} "


class ClineAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # When set, Cline's model calls go through this Gym model server instead of straight to a
    # provider, so they are captured. The agent then configures Cline's `openai-compatible`
    # provider with the server's URL, and `model` is the bare model name that server serves.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    # The Cline command, split on spaces so a multi-word launcher works (e.g. `npx cline`).
    command: str = "cline"
    # The model id handed to Cline. Required with a model server, because `cline auth` refuses to
    # write provider settings without one; None otherwise leaves whatever the configured provider
    # already selects (`-m` is omitted).
    model: Optional[str] = None
    # Provider id passed as `-P`. With a model server this is forced to `openai-compatible`, the
    # only provider besides OpenAI whose base URL `cline auth` accepts.
    provider: str = OPENAI_COMPATIBLE_PROVIDER
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    # extra env vars for the subprocess e.g. API keys
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/cline_agent/workspaces"
    # Optional persistent project dir to run in; None -> an ephemeral per-request dir.
    repo_dir: Optional[str] = None
    # Prepended to the user message, like the other CLI agent harnesses. This adds task framing
    # without disturbing Cline's own operating instructions.
    system_prompt: Optional[str] = None
    # Passed as `-s`, which *replaces* Cline's built-in system prompt outright rather than adding
    # to it (buildClineSystemPrompt returns the override alone). That drops the instructions its
    # tools are described by, so prefer `system_prompt` unless a full replacement is the point.
    system_prompt_override: Optional[str] = None
    # Reasoning effort: none|low|medium|high|xhigh. None leaves the provider default (the flag is
    # omitted entirely).
    thinking: Optional[str] = None
    # Context compaction mode: agentic|basic|off. None leaves Cline's default (agentic).
    compaction: Optional[str] = None
    # Max consecutive mistakes before Cline halts, passed as --retries. None leaves its default.
    retries: Optional[int] = None
    # Seconds for the `cline` call. Cline's own --timeout is also set from this so the CLI winds
    # the session down itself before the subprocess is killed; see _build_command.
    timeout: int = 900
    # Seconds for the one-off `cline auth` call that writes the provider settings.
    setup_timeout: int = 300
    extra_args: list[str] = []
    # JSON policy restricting the shell commands Cline may run, passed as CLINE_COMMAND_PERMISSIONS
    # (e.g. {"allow": ["python3 *"], "deny": ["sudo *"]}). Empty -> no restriction.
    command_permissions: dict[str, Any] = Field(default_factory=dict)
    # npm version of the `cline` package installed on a clean machine (the parser was validated
    # against the pinned version, so treat a bump as a deliberate, retested change). None installs
    # @latest.
    cline_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class ClineAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClineAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class ClineAgent(SimpleResponsesAPIAgent):
    """Runs the Cline CLI headlessly (``cline --json``).

    Cline runs its own tools internally; its newline-delimited JSON event stream is parsed into Gym
    format and the resources server verifies the result. Cline's model calls go through
    ``model_server`` when one is set — see that field. Eval-only either way: token IDs and logprobs
    are not wired up.
    """

    config: ClineAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_cline(self.config.cline_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("cline command %r is not on PATH yet", self.config.command)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"cline_{uuid4().hex[:8]}"
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

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        """The Gym model server's ``/v1`` URL with the per-rollout capture prefix, "" if unset."""
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    def _effective_provider(self) -> str:
        """The provider id for `-P`: forced to openai-compatible on the model-server path."""
        return OPENAI_COMPATIBLE_PROVIDER if self.config.model_server else self.config.provider

    def _env(self, data_dir: Path, model_base_url: str = "") -> dict[str, str]:
        """Environment for the cline subprocesses, isolated to this run's data dir.

        ``--data-dir`` alone is not enough: the CLI resolves several state paths from
        ``CLINE_DATA_DIR`` and friends, and an ambient value (set by an outer Cline session, for
        instance) would otherwise leak a shared provider settings file and session db across
        concurrent rollouts. Pinning them here keeps each run's provider config, sessions, and
        logs inside its own directory, and never touches the user's ``~/.cline``.
        """
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
            # Local session backend: a shared hub daemon would hand concurrent rollouts the same
            # process and state, which is exactly what the per-run data dir is avoiding.
            "CLINE_SESSION_BACKEND_MODE": "local",
        }
        # A Gym model server wins over openai_base_url, so a config carrying both cannot point the
        # subprocess at the provider the model server is meant to replace.
        base_url = model_base_url or self.config.openai_base_url
        api_key = "EMPTY" if model_base_url else self.config.openai_api_key  # pragma: allowlist secret
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        if self.config.command_permissions:
            env["CLINE_COMMAND_PERMISSIONS"] = json.dumps(self.config.command_permissions)
        env.update({k: v for k, v in self.config.env.items() if v})
        return env

    def _build_auth_command(self, data_dir: Path, model_base_url: str) -> Optional[list[str]]:
        """``cline auth`` argv that writes this run's provider settings, or None when not needed.

        Cline reads its provider, key, model, and base URL from the provider settings file rather
        than from run flags (``-k`` overrides only the key, and no flag sets a base URL), so the
        model-server path has to write that file once per run before the task runs.
        """
        if not model_base_url:
            return None
        if not self.config.model:
            raise ValueError("cline_agent requires `model` to be set when `model_server` is configured")
        return [
            *self.config.command_parts,
            "auth",
            OPENAI_COMPATIBLE_PROVIDER,
            # `cline auth` requires a key; the Gym model server does not check it, and the real
            # credential (if any) lives on the model server, not here.
            "--apikey",
            "EMPTY",  # pragma: allowlist secret
            "--modelid",
            self.config.model,
            "--baseurl",
            model_base_url,
            "--data-dir",
            str(data_dir),
        ]

    def _build_command(self, project_dir: Path, data_dir: Path, prompt: str) -> list[str]:
        """``cline`` argv for one headless run.

        ``--json`` selects the newline-delimited JSON event stream (and implies headless);
        ``--auto-approve true`` runs the tools without prompting; ``--cwd`` scopes the run to the
        per-request project dir; ``--timeout`` lets Cline wind the session down itself just before
        the subprocess deadline. The prompt goes last, after ``--``, so one starting with ``-`` is
        not parsed as a flag.
        """
        cmd = [
            *self.config.command_parts,
            "--json",
            "--auto-approve",
            "true",
            "--cwd",
            str(project_dir),
            "--data-dir",
            str(data_dir),
            "--timeout",
            str(self.config.timeout),
        ]
        if self.config.model:
            cmd += ["-m", self.config.model]
        cmd += ["-P", self._effective_provider()]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if self.config.compaction:
            cmd += ["--compaction", self.config.compaction]
        if self.config.retries is not None:
            cmd += ["--retries", str(self.config.retries)]
        if self.config.system_prompt_override:
            cmd += ["-s", self.config.system_prompt_override]
        cmd.extend(self.config.extra_args)
        cmd += ["--", quote_prompt(prompt)]
        return cmd

    @staticmethod
    def _kill_process_group(proc: "asyncio.subprocess.Process") -> None:
        # `cline` is an npm shim whose child is the real binary; killing only the shim orphans the
        # child holding the stdout pipe, so kill the whole process group (start_new_session=True).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()

    async def _spawn(self, cmd: list[str], cwd: Path, env: dict[str, str], timeout: int) -> tuple[str, str, int]:
        """Run one subprocess to completion; on timeout kill its group and report what it wrote."""
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
        except asyncio.TimeoutError:
            self._kill_process_group(proc)
            stdout, stderr = await proc.communicate()
            return stdout.decode(errors="replace"), stderr.decode(errors="replace"), -1
        return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode or 0

    async def _run_cline(
        self, instruction: str, system_prompt: Optional[str], rollout_id: Optional[str] = None
    ) -> tuple[list[Any], dict[str, Any], str]:
        """Run one headless cline task. Returns (output_items, metadata, model_name)."""
        prompt = instruction if not system_prompt else f"{system_prompt}\n\n{instruction}"
        work_dir = self._workspace_root()
        project_dir = self._repo_dir(work_dir)
        data_dir = work_dir / ".cline-data"
        data_dir.mkdir(parents=True, exist_ok=True)
        model_base_url = self._resolve_model_base_url(rollout_id)
        env = self._env(data_dir, model_base_url)

        try:
            auth_cmd = self._build_auth_command(data_dir, model_base_url)
            if auth_cmd:
                _, auth_err, auth_rc = await self._spawn(auth_cmd, project_dir, env, self.config.setup_timeout)
                if auth_rc != 0:
                    LOG.warning("cline auth exited %d: %s", auth_rc, auth_err[:500])

            cmd = self._build_command(project_dir, data_dir, prompt)
            stdout, stderr, returncode = await self._spawn(cmd, project_dir, env, self.config.timeout)
            if returncode == -1:
                LOG.warning("cline timed out after %ds", self.config.timeout)
            elif returncode != 0:
                LOG.warning("cline exited %d: %s", returncode, stderr[:500])

            output_items, metadata = parse_cline_events(stdout)
            if returncode == -1:
                metadata["timed_out"] = True
            model_name = metadata.get("model") or self.config.model or ""
            return output_items, metadata, model_name
        finally:
            # The run dir holds the ephemeral data dir (and the workspace, unless repo_dir moved
            # the project elsewhere), so nothing survives to leak into the next rollout.
            shutil.rmtree(work_dir, ignore_errors=True)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # run() reaches this handler through the /ng-rollout/<id> self-call route, which carries
        # the correlation id Cline's model calls are tagged with. Absent on a direct
        # /v1/responses call.
        rollout_id = request.path_params.get("rollout_id")
        output_items, metadata, model_name = await self._run_cline(user_message, system_prompt, rollout_id)

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

    async def run(self, request: Request, body: ClineAgentRunRequest) -> ClineAgentVerifyResponse:
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

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

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

            return ClineAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    ClineAgent.run_webserver()
