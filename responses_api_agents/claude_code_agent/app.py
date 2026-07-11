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
import shutil
import subprocess
import tempfile
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import SKILLS_REF_KEY_NAME, get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.skills import stage_skills
from nemo_gym.trajectory import Trajectory, to_response_output
from responses_api_agents.claude_code_agent.setup_claude_code import ensure_claude_code
from responses_api_agents.claude_code_agent.trajectory import build_trajectory, decode_jsonl


LOG = logging.getLogger(__name__)


def _usage_metadata(trajectory: Trajectory) -> dict:
    """Derive the response usage metadata from a trajectory.

    The provider's end-of-run report is the authoritative total when present (summing it
    with the per-step usage would double count); otherwise the per-agent-step sums are used.
    """
    if trajectory.provider_usage:
        metadata = {
            "input_tokens": int(trajectory.provider_usage.get("input_tokens") or 0),
            "output_tokens": int(trajectory.provider_usage.get("output_tokens") or 0),
            "cached_tokens": int(trajectory.provider_usage.get("cache_read_input_tokens") or 0),
        }
    else:
        metadata = {
            "input_tokens": trajectory.usage.input_tokens,
            "output_tokens": trajectory.usage.output_tokens,
            "cached_tokens": trajectory.usage.input_tokens_details.cached_tokens,
        }
    if trajectory.num_agent_steps is not None:
        # provider dialect: Claude Code reports agent steps as num_turns; keep the key
        metadata["num_turns"] = trajectory.num_agent_steps
    return metadata


def parse_stream_json(stdout: str) -> tuple[list[Any], dict]:
    """Convert claude -p --output-format=stream-json stdout into (output_items, usage).

    The stdout is parsed once into a trajectory and the response items are derived from
    it, so the response and the trajectory telemetry can never drift apart.
    """
    trajectory = build_trajectory(decode_jsonl(stdout), [])
    return to_response_output(trajectory), _usage_metadata(trajectory)


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


class ClaudeCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # When model_server is set, ANTHROPIC_BASE_URL is resolved from the Gym model
    # server's URL (requires the server to expose POST /v1/messages).
    # When None, anthropic_base_url is used directly.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # pragma: allowlist secret
    anthropic_base_url: Optional[str] = None
    max_turns: Optional[int] = 30  # None -> unlimited turns
    timeout: int = 300
    system_prompt: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    claude_code_version: Optional[str] = None
    thinking: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    # Runtime capability knobs. The default (bare=True, no mcp_config/settings) reproduces the original
    # isolated behavior: Claude Code skips hooks, LSP, plugin sync, attribution, auto-memory, background
    # prefetches, keychain reads, and CLAUDE.md auto-discovery (skills still resolve via /skill-name).
    bare: bool = True
    mcp_config: Optional[str] = None
    settings: Optional[str] = None
    # When True, the session transcript Claude Code writes under the per-run CLAUDE_CONFIG_DIR
    # is harvested before cleanup and attached to run() results as a standardized `trajectory`
    # (see trajectory.py for the schema and reconstruction semantics).
    capture_trajectory: bool = True


class ClaudeCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    # Standardized trajectory (trajectory.ClaudeCodeTrajectory, serialized); None when
    # capture is disabled or no artifacts were produced (e.g. hard timeout before startup).
    trajectory: Optional[dict[str, Any]] = None


class ClaudeCodeAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeAgentConfig
    sem: Semaphore = None
    _static_mcp_config: Optional[dict[str, Any]] = PrivateAttr(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_claude_code(self.config.claude_code_version)
        try:
            ver = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            LOG.warning("claude-code version: %s", ver or "(unknown)")
        except Exception as exc:
            LOG.warning("could not determine claude-code version: %s", exc)

    def _resolve_base_url(self) -> str:
        if self.config.model_server:
            cfg = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return self.server_client._build_server_base_url(cfg)
        return self.config.anthropic_base_url or ""

    def _build_settings(self) -> dict[str, Any]:
        """Settings written into the run's CLAUDE_CONFIG_DIR.

        The base settings disable telemetry/attribution. When ``config.settings`` points at a
        JSON file, its contents are layered on top: top-level keys override, and the ``env`` block
        is shallow-merged so the telemetry defaults are preserved unless explicitly overridden.
        """
        settings: dict[str, Any] = {
            "env": {
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
        }
        if self.config.settings:
            user_settings = json.loads(Path(self.config.settings).expanduser().read_text())
            user_env = user_settings.get("env") or {}
            settings = {**settings, **user_settings, "env": {**settings["env"], **user_env}}
        return settings

    def _setup_config_dir(self, skills_path: Optional[str] = None) -> Path:
        """Create a per-run CLAUDE_CONFIG_DIR and stage settings (and optionally skills) into it.

        The directory lives for the duration of a single ``_run_claude_code`` call. When
        ``skills_path`` is provided, the directory of skills is copied into ``<dir>/skills/`` so
        Claude Code's native discovery can pick them up. Each request gets its own ephemeral copy,
        so concurrent requests with different skills do not contaminate one another. The caller is
        responsible for removing the directory on success; if setup fails partway (e.g. a bad
        ``skills_path``), this method cleans up the partially-created dir before re-raising so it
        does not leak (the caller never receives the path in that case).
        """
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
        """Construct the ``claude`` CLI argv from config.

        ``--bare`` is only passed when ``config.bare`` is True; it skips hooks, LSP, plugin sync,
        attribution, auto-memory, background prefetches, keychain reads, and CLAUDE.md auto-discovery
        (skills still resolve via /skill-name). Explicit capabilities like ``--mcp-config`` are passed
        regardless of ``--bare`` since they are not auto-discovered.

        When ``skills_active`` is True (skills were staged into CLAUDE_CONFIG_DIR for this request),
        ``--bare`` is forced off so Claude Code's native filesystem discovery picks the skills up.
        """
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.config.bare and skills_active:
            LOG.warning(
                "skills are active for this request; ignoring bare=True so Claude Code can discover them. "
                "Note this re-enables ALL native auto-discovery, not just skills (hooks, plugins, MCP servers, "
                "memory, and CLAUDE.md), so the runtime broadens versus a bare baseline."
            )
        if self.config.bare and not skills_active:
            cmd.append("--bare")
        cmd += ["--model", model]
        effective_mcp_config = mcp_config if mcp_config is not None else self.config.mcp_config
        if effective_mcp_config:
            cmd += ["--mcp-config", effective_mcp_config]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        if self.config.allowed_tools:
            cmd += ["--allowedTools", self.config.allowed_tools]
        if self.config.disallowed_tools:
            cmd += ["--disallowedTools", self.config.disallowed_tools]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if self.config.max_thinking_tokens is not None:
            cmd += ["--max-thinking-tokens", str(self.config.max_thinking_tokens)]
        if self.config.max_turns is not None:
            cmd += ["--max-turns", str(self.config.max_turns)]
        cmd += ["--", instruction]
        return cmd

    def _collect_transcript_records(self, claude_config_dir: Path) -> list[dict]:
        """Harvest the session transcript(s) Claude Code wrote under the per-run config dir.

        Claude Code persists every session event (with timestamps, request ids, per-call
        usage, and tool execution metadata) to ``<config_dir>/projects/<cwd-slug>/*.jsonl``.
        The per-run dir is removed after each request, so this runs just before cleanup.
        """
        records: list[dict] = []
        try:
            projects_dir = claude_config_dir / "projects"
            if projects_dir.is_dir():
                for transcript in sorted(projects_dir.glob("*/*.jsonl")):
                    records.extend(decode_jsonl(transcript.read_text(errors="replace")))
        except OSError as exc:
            LOG.warning("failed to read Claude Code transcript from %s: %s", claude_config_dir, exc)
        return records

    async def _run_claude_code(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
    ) -> tuple[str, str, list[dict]]:
        """Run claude -p --output-format=stream-json; return (stdout, model_name, transcript_records)."""
        base_url = self._resolve_base_url()
        # Keep full model name for local/custom endpoints; strip provider prefix for real Anthropic API.
        model = self.config.model if base_url else self.config.model.split("/")[-1]
        api_key = self.config.anthropic_api_key

        claude_config_dir = None
        try:
            # Inside the try so a bad skills.path (raising in stage_skills) still cleans up the
            # partially-created config dir in the finally rather than leaking it per failing request.
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

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                LOG.warning("claude-code timed out after %ds", self.config.timeout)
                # The partial transcript is still on disk and is the only record of what
                # happened before the kill — harvest it for debugging.
                return "", model, self._maybe_collect_transcript(claude_config_dir)

            if proc.returncode not in (0, None):
                LOG.warning("claude-code exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])

            LOG.debug("claude-code stdout (%d chars): %s", len(stdout), stdout[:2000].decode(errors="replace"))
            return stdout.decode(errors="replace"), model, self._maybe_collect_transcript(claude_config_dir)
        finally:
            if claude_config_dir is not None:
                shutil.rmtree(claude_config_dir, ignore_errors=True)

    def _maybe_collect_transcript(self, claude_config_dir: Path) -> list[dict]:
        return self._collect_transcript_records(claude_config_dir) if self.config.capture_trajectory else []

    def _resources_server_base_url(self) -> str:
        cfg = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(cfg)

    def _load_static_mcp_config(self) -> dict[str, Any]:
        if not self.config.mcp_config:
            return {"mcpServers": {}}

        config_path = Path(self.config.mcp_config).expanduser()
        config = json.loads(config_path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Claude Code mcp_config must be a JSON object: {config_path}")
        mcp_servers = config.setdefault("mcpServers", {})
        if not isinstance(mcp_servers, dict):
            raise ValueError(f"Claude Code mcp_config has non-object mcpServers: {config_path}")
        return config

    def _get_static_mcp_config(self) -> dict[str, Any]:
        # The static mcp_config is immutable, so read it from disk at most once and reuse the cached
        # copy for every rollout instead of re-reading the file each time.
        if self._static_mcp_config is None:
            self._static_mcp_config = self._load_static_mcp_config()
        return self._static_mcp_config

    def _write_rollout_mcp_config(self, seed_response_json: dict[str, Any], output_dir: Path) -> Optional[str]:
        metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None

        server_name = metadata.get("server_name") or self.config.resources_server.name
        url_path = str(metadata.get("url_path") or "/mcp")
        url = f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}"

        entry: dict[str, Any] = {
            "type": metadata.get("transport") or "http",
            "url": url,
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            entry["headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning(
                "MCP seed metadata for %r has no headers; the tool endpoint will be called without a "
                "session token and will reject the calls.",
                server_name,
            )

        # Start from a copy of the (cached) static config and add the per-rollout Gym entry. If a static
        # mcp_config server already uses this name, the per-rollout Gym entry takes precedence over it.
        config = copy.deepcopy(self._get_static_mcp_config())
        config.setdefault("mcpServers", {})[str(server_name)] = entry

        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "gym_mcp_config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        return str(config_path)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        mcp_config: Optional[str] = None,
        skills_path: Optional[str] = None,
    ) -> tuple[NeMoGymResponse, Optional[dict[str, Any]]]:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        stdout, model_name, transcript_records = await self._run_claude_code(
            user_message,
            system_prompt=system_prompt,
            mcp_config=mcp_config,
            skills_path=skills_path,
        )
        # One parse for both artifacts: the response the verifier scores is derived from the
        # stream-json trajectory; the attached telemetry prefers the richer transcript source.
        stream_events = decode_jsonl(stdout)
        stream_trajectory = build_trajectory(stream_events, [])
        output_items = to_response_output(stream_trajectory)
        usage = _usage_metadata(stream_trajectory)

        trajectory: Optional[dict[str, Any]] = None
        if self.config.capture_trajectory:
            try:
                telemetry = (
                    build_trajectory(stream_events, transcript_records) if transcript_records else stream_trajectory
                )
                trajectory = telemetry.model_dump(mode="json")
            except Exception as exc:
                LOG.warning("failed to build trajectory: %s", exc)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("claude-code produced no assistant message; padding empty output")
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
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=usage.get("cached_tokens", 0)),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )
        return response, trajectory

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        response, _ = await self._create_response(body)
        return response

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
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
            # can stage the skills into its per-request CLAUDE_CONFIG_DIR. run() calls _create_response
            # in-process, so no metadata side-channel is needed (unlike the schema-forbidden HTTP path).
            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")

            with tempfile.TemporaryDirectory(prefix="nemo_gym_claude_mcp_") as mcp_config_dir:
                mcp_config = self._write_rollout_mcp_config(seed_resp_json, Path(mcp_config_dir))
                agent_resp, trajectory = await self._create_response(
                    body.responses_create_params, mcp_config=mcp_config, skills_path=skills_path
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

            return ClaudeCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally, "trajectory": trajectory}
            )


if __name__ == "__main__":
    ClaudeCodeAgent.run_webserver()
