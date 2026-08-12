# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
import sys
from asyncio import Semaphore
from shlex import join, quote
from time import time
from traceback import format_exc
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    apply_rollout_prefix,
    get_response_json,
    get_server_url,
    raise_for_status,
)
from responses_api_agents.claude_code_agent.app import (
    _contains_cli_api_error,
    _extract_instruction,
    _invocation_outcome,
    parse_stream_json,
)


class ClaudeCodeSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    model: str = "model"
    claude_code_version: Optional[str] = None
    system_prompt: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    max_turns: Optional[int] = 100
    thinking: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    bare: bool = True

    sandbox_provider: str
    sandbox_timeout: float
    concurrency: int = 32
    settings: Dict[str, Any] = Field(default_factory=dict)
    debug: bool = False


class ClaudeCodeSandboxedAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeSandboxedAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeSandboxedAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    claude_code_run_stdout: str = ""
    claude_code_run_stderr: str = ""
    claude_code_finished: bool = False
    turns_used: int = 0


def _settings_json(overrides: Dict[str, Any]) -> str:
    defaults = {
        "env": {
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    }
    user_env = overrides.get("env") or {}
    return json.dumps({**defaults, **overrides, "env": {**defaults["env"], **user_env}})


class ClaudeCodeSandboxedAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeSandboxedAgentConfig

    _sem: Semaphore = PrivateAttr()
    _sandboxes: Dict[str, AsyncSandbox] = PrivateAttr(default_factory=dict)
    _run_results: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        self._sem = Semaphore(self.config.concurrency)
        super().model_post_init(context)

    async def _connect_sandbox(self, handle: Any) -> AsyncSandbox:
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        descriptor = handle if isinstance(handle, dict) else {"sandbox_id": handle}
        return await AsyncSandbox.connect(descriptor, provider=provider)

    def _install_command(self) -> str:
        version = f"@{self.config.claude_code_version}" if self.config.claude_code_version else ""
        package = f"@anthropic-ai/claude-code{version}"
        return f"""
if ! command -v claude >/dev/null 2>&1
then
  runtime=/tmp/nemo-gym-claude-runtime
  mkdir -p "$runtime"
  if ! command -v npm >/dev/null 2>&1
  then
    node_machine="$(uname -m)"
    if [ "$node_machine" = x86_64 ] || [ "$node_machine" = amd64 ]
    then
      node_arch=x64
    elif [ "$node_machine" = arm64 ] || [ "$node_machine" = aarch64 ]
    then
      node_arch=arm64
    else
      echo "unsupported architecture: $node_machine" >&2
      exit 1
    fi
    curl -fsSL "https://nodejs.org/dist/v22.15.0/node-v22.15.0-linux-${{node_arch}}.tar.xz" \
      | tar xJ -C "$runtime" --strip-components=1
  fi
  export PATH="$runtime/bin:$PATH"
  npm install -g --prefix "$runtime" {quote(package)}
fi
"""

    def _command(self, model: str, instruction: str, system_prompt: Optional[str]) -> str:
        args = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.config.bare:
            args.append("--bare")
        args.extend(["--model", model])
        if system_prompt:
            args.extend(["--append-system-prompt", system_prompt])
        if self.config.allowed_tools:
            args.extend(["--allowedTools", self.config.allowed_tools])
        if self.config.disallowed_tools:
            args.extend(["--disallowedTools", self.config.disallowed_tools])
        if self.config.thinking:
            args.extend(["--thinking", self.config.thinking])
        if self.config.max_thinking_tokens is not None:
            args.extend(["--max-thinking-tokens", str(self.config.max_thinking_tokens)])
        if self.config.max_turns is not None:
            args.extend(["--max-turns", str(self.config.max_turns)])
        args.extend(["--", instruction])
        return self._install_command() + "\n" + join(args)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        session_id = request.cookies["sandbox_id"]
        sandbox = self._sandboxes[session_id]
        instruction, input_system = _extract_instruction(body.input)
        system_prompt = "\n\n".join(part for part in (self.config.system_prompt, input_system) if part) or None
        model = body.model or self.config.model
        rollout_id = request.path_params.get("rollout_id")
        model_url = apply_rollout_prefix(get_server_url(self.config.model_server.name), rollout_id)
        config_dir = f"/tmp/nemo-gym-claude-config-{uuid4().hex}"
        settings = base64.b64encode(_settings_json(self.config.settings).encode()).decode()
        command = (
            f"mkdir -p {config_dir} && "
            f"printf %s {settings} | base64 -d > {config_dir}/settings.json && "
            + self._command(model, instruction, system_prompt)
        )
        env = {
            "ANTHROPIC_API_KEY": "local",
            "ANTHROPIC_AUTH_TOKEN": "local",
            "ANTHROPIC_BASE_URL": model_url,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "CLAUDE_CONFIG_DIR": config_dir,
            "IS_SANDBOX": "1",
        }
        timeout = self.config.sandbox_timeout
        if body.metadata and body.metadata.get("agent_timeout_sec") is not None:
            timeout = min(timeout, float(body.metadata["agent_timeout_sec"]))

        result = await sandbox.exec(command, env=env, timeout_s=timeout)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output, metadata = parse_stream_json(stdout)
        error_type = getattr(result, "error_type", None)
        if error_type == "timeout":
            raise TimeoutError(f"Claude Code timed out after {timeout:g} seconds")
        if error_type:
            raise RuntimeError(f"Claude Code sandbox failed: {error_type}")
        status, invocation_error = _invocation_outcome(metadata, result.return_code)
        if _contains_cli_api_error(output):
            status, invocation_error = "failed", "model_api_error"
        if status == "failed":
            raise RuntimeError(f"Claude Code failed: {invocation_error or 'agent_error'}")

        finished = status == "completed" and bool(output)
        self._run_results[session_id] = {
            "claude_code_run_stdout": stdout,
            "claude_code_run_stderr": stderr,
            "claude_code_finished": finished,
            "turns_used": int(metadata.get("num_turns") or 0),
        }
        if self.config.debug:
            print(stdout, file=sys.stderr)
            print(stderr, file=sys.stderr)

        input_tokens = int(metadata.get("input_tokens") or 0)
        output_tokens = int(metadata.get("output_tokens") or 0)
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model,
            object="response",
            output=output,
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

    async def run(
        self,
        request: Request,
        body: ClaudeCodeSandboxedAgentRunRequest,
    ) -> ClaudeCodeSandboxedAgentVerifyResponse:
        async with self._sem:
            cookies = request.cookies
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = cookies | seed_response.cookies
            seed_result = await seed_response.json()

            session_id = request.session[SESSION_ID_KEY]
            sandbox = await self._connect_sandbox(seed_result["sandbox_handle"])
            self._sandboxes[session_id] = sandbox
            cookies["sandbox_id"] = session_id
            try:
                response = await self.server_client.post(
                    server_name=self.config.name,
                    url_path=self.url_path_for_run("/v1/responses", body),
                    json=body.responses_create_params,
                    cookies=cookies,
                )
                await raise_for_status(response)
                cookies = cookies | response.cookies
                response_json = await get_response_json(response)

                verify_request = ClaudeCodeSandboxedAgentVerifyRequest.model_validate(
                    body.model_dump() | {"response": response_json}
                )
                verify_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=verify_request.model_dump(),
                    cookies=cookies,
                )
                await raise_for_status(verify_response)
                result = await get_response_json(verify_response)
                result.update(self._run_results.get(session_id, {}))
                return ClaudeCodeSandboxedAgentVerifyResponse.model_validate(result)
            finally:
                try:
                    await sandbox.stop()
                except Exception:
                    print("Failed to stop sandbox", format_exc(), file=sys.stderr)
                self._sandboxes.pop(session_id, None)
                self._run_results.pop(session_id, None)


if __name__ == "__main__":
    ClaudeCodeSandboxedAgent.run_webserver()
