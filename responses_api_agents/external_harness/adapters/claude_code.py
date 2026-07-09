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

"""Adapter for the Claude Code CLI.

This mirrors the option surface of the in-process claude_code_agent, but runs
the CLI inside a sandbox. Endpoint config rides launch()'s env (Claude Code's
SDK appends /v1/messages to ANTHROPIC_BASE_URL). prepare() writes the harness's
config files into the sandbox: a settings.json under CLAUDE_CONFIG_DIR (with the
telemetry/attribution defaults, plus any user settings), optional skills staged
for native discovery, and a merged .mcp.json (static servers plus any tools the
environment lends over MCP). launch() assembles the argv from the same knobs the
in-process agent exposes (bare, max_turns, model, allowed/disallowed tools,
append-system-prompt, thinking, max-thinking-tokens).
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from responses_api_agents.external_harness.sandbox_utils import sandbox_write_text

from .base import HarnessSpec, RolloutEndpoints, RuntimeRequirement, SeedResult


_MCP_CONFIG_NAME = ".mcp.json"
# Telemetry/attribution off, matching the in-process claude_code_agent baseline.
_BASE_SETTINGS_ENV = {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
# Behavioral defaults for the launched CLI: telemetry/updater/bug-reporting off,
# and IS_SANDBOX so --dangerously-skip-permissions works when running as root in
# a container. Users tune these through the single extra_env knob, which merges
# over both these defaults and the connection/routing vars.
_DEFAULT_LAUNCH_ENV = {
    "IS_SANDBOX": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_BUG_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
}


class ClaudeCodeSpec(HarnessSpec):
    name: str = "claude_code"
    # Model name sent to the endpoint; defaults to the model server's name.
    model: Optional[str] = None
    # Capped so Claude Code's large default (32k) fits the served context window.
    max_output_tokens: int = 4096
    max_turns: int = 30
    # --bare skips hooks, LSP, plugin sync, attribution, auto-memory, background
    # prefetches, keychain reads, and CLAUDE.md auto-discovery. Forced off when
    # skills are staged, so Claude Code's native discovery can pick them up.
    bare: bool = True
    output_format: str = "json"  # json | stream-json
    system_prompt_append: Optional[str] = None  # --append-system-prompt
    allowed_tools: Optional[str] = None  # --allowedTools (merged with env-lent MCP grants)
    disallowed_tools: Optional[str] = None  # --disallowedTools
    thinking: Optional[str] = None  # --thinking
    max_thinking_tokens: Optional[int] = None  # --max-thinking-tokens
    # settings.json contents layered over the telemetry-off defaults (the env
    # block is shallow-merged so defaults survive unless explicitly overridden).
    settings: dict[str, Any] = Field(default_factory=dict)
    # Static MCP servers, merged with any the environment lends (env-lent wins).
    mcp_servers: dict[str, dict] = Field(default_factory=dict)
    # Host directory of skills copied into the sandbox config dir for discovery.
    skills_dir: Optional[str] = None
    # CLAUDE_CONFIG_DIR inside the sandbox (holds settings.json and skills).
    # Defaults to a dir under the rollout workdir so it is unique per rollout and
    # torn down with it (important for the no-isolation local_subprocess provider).
    config_dir: Optional[str] = None


class ClaudeCodeAdapter:
    name = "claude_code"
    spec_model = ClaudeCodeSpec

    def runtime(self, spec: HarnessSpec) -> RuntimeRequirement:
        pkg = "@anthropic-ai/claude-code"
        if spec.pinned_version:
            pkg = f"{pkg}@{spec.pinned_version}"
        # Idempotent: skip the install when claude is already on PATH (baked image
        # or an already-provisioned host), otherwise install it via npm.
        guarded = f"command -v claude >/dev/null 2>&1 || npm install -g {pkg}"
        return RuntimeRequirement(setup_cmds=[["bash", "-lc", guarded]])

    def _config_dir(self, spec: ClaudeCodeSpec, workdir: str) -> str:
        return spec.config_dir or posixpath.join(workdir, ".claude_code_agent")

    def _build_settings(self, spec: ClaudeCodeSpec) -> dict[str, Any]:
        settings: dict[str, Any] = {"env": dict(_BASE_SETTINGS_ENV)}
        if spec.settings:
            user_env = spec.settings.get("env") or {}
            settings = {**settings, **spec.settings, "env": {**settings["env"], **user_env}}
        return settings

    def _merged_mcp_servers(self, spec: ClaudeCodeSpec, seed: SeedResult) -> dict[str, dict]:
        servers: dict[str, dict] = {}
        servers.update(spec.mcp_servers or {})
        servers.update(seed.mcp_servers or {})  # env-lent entries take precedence
        return servers

    async def _stage_skills(self, sandbox: Any, host_dir: str, dest_dir: str) -> None:
        root = Path(host_dir).expanduser()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                await sandbox_write_text(sandbox, posixpath.join(dest_dir, rel), path.read_text())

    async def prepare(
        self, sandbox: Any, spec: HarnessSpec, seed: SeedResult, endpoints: RolloutEndpoints, workdir: str
    ) -> None:
        assert isinstance(spec, ClaudeCodeSpec)
        config_dir = self._config_dir(spec, workdir)
        await sandbox_write_text(
            sandbox, posixpath.join(config_dir, "settings.json"), json.dumps(self._build_settings(spec), indent=2)
        )
        if spec.skills_dir:
            await self._stage_skills(sandbox, spec.skills_dir, posixpath.join(config_dir, "skills"))
        servers = self._merged_mcp_servers(spec, seed)
        if servers:
            config = {"mcpServers": servers}
            await sandbox_write_text(sandbox, posixpath.join(workdir, _MCP_CONFIG_NAME), json.dumps(config, indent=2))

    def launch(
        self,
        spec: HarnessSpec,
        seed: SeedResult,
        endpoints: RolloutEndpoints,
        task_prompt: str,
        workdir: str,
    ) -> tuple[list[str], dict[str, str]]:
        assert isinstance(spec, ClaudeCodeSpec)
        model = spec.model or endpoints.model_name
        prompt = f"{task_prompt}\n\nThe project files are in your working directory: {workdir}"
        servers = self._merged_mcp_servers(spec, seed)
        skills_active = bool(spec.skills_dir)

        cmd = ["claude", "-p", "--output-format", spec.output_format, "--dangerously-skip-permissions"]
        if spec.output_format == "stream-json":
            cmd.append("--verbose")  # required with stream-json in -p mode
        # Explicit capabilities (--mcp-config) are honored regardless of --bare;
        # skills are auto-discovered, so --bare is dropped when they are staged.
        if spec.bare and not skills_active:
            cmd.append("--bare")
        cmd += ["--max-turns", str(spec.max_turns)]
        if model:
            cmd += ["--model", model]
        if servers:
            cmd += ["--mcp-config", posixpath.join(workdir, _MCP_CONFIG_NAME)]
        allowed = [spec.allowed_tools] if spec.allowed_tools else []
        # Grant every env-lent server's tools (mcp__<server> matches all its tools).
        allowed += [f"mcp__{name}" for name in (seed.mcp_servers or {})]
        if allowed:
            cmd += ["--allowedTools", ",".join(allowed)]
        if spec.disallowed_tools:
            cmd += ["--disallowedTools", spec.disallowed_tools]
        if spec.system_prompt_append:
            cmd += ["--append-system-prompt", spec.system_prompt_append]
        if spec.thinking:
            cmd += ["--thinking", spec.thinking]
        if spec.max_thinking_tokens is not None:
            cmd += ["--max-thinking-tokens", str(spec.max_thinking_tokens)]
        cmd += ["--", prompt]

        # Behavioral defaults first, so connection/routing set below cannot be
        # clobbered by them.
        env = dict(_DEFAULT_LAUNCH_ENV)
        # Connection + config the adapter always controls (ANTHROPIC_BASE_URL is
        # the gate root; the Anthropic SDK appends /v1/messages).
        env.update(
            {
                "ANTHROPIC_BASE_URL": endpoints.model_base_url,
                "ANTHROPIC_API_KEY": endpoints.api_key or "local",
                "ANTHROPIC_AUTH_TOKEN": endpoints.api_key or "local",
                "CLAUDE_CONFIG_DIR": self._config_dir(spec, workdir),
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(spec.max_output_tokens),
            }
        )
        if model:
            # Route the main call and every subagent/haiku/sonnet/opus call to the
            # same served model, so nothing falls back to the real Anthropic API.
            env["ANTHROPIC_MODEL"] = model
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = model
        # The single user knob: extra_env merges over defaults and connection.
        env.update(spec.extra_env)
        return cmd, env


ADAPTER = ClaudeCodeAdapter()
