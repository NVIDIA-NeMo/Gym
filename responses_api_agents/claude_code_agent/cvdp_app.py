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
import shutil
import signal
import subprocess
import tempfile
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
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
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.claude_code_agent.setup_claude_code import ensure_claude_code


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
    """Convert claude -p --output-format=stream-json stdout into (output_items, usage)."""
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

    for event in raw_events:
        etype = event.get("type")

        if etype == "result":
            usage = event.get("usage") or {}
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)

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

    return output_items, {"input_tokens": total_input, "output_tokens": total_output}


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
    # server's URL (requires the server to expose POST /v1/messages. None is pushed yet).
    # When None, anthropic_base_url is used directly.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # pragma: allowlist secret
    anthropic_base_url: Optional[str] = None
    max_turns: int = 30
    timeout: int = 300

    # --- Apptainer execution ---
    # For tasks with target files, run() executes claude inside an Apptainer
    # container built from `sim_image` (or an explicit `sif_path`). The agent owns
    # a per-rollout workspace seeded with the task's `context_files` ONLY (docs +
    # companion RTL — never the hidden harness/tests); claude may read those files
    # and self-test with the in-container EDA tools (iverilog/vvp), then returns
    # its solution as text per the system prompt. That response text is sent to
    # /verify, where the resources server extracts the RTL (no on-disk readback),
    # keeping fidelity with the original CVDP benchmark.
    #
    # `claude` itself is not in the EDA image, so the host's self-contained
    # `.claude_node` (Node + Claude Code, created by setup_claude_code) is bind
    # mounted into the container at /opt/claude_node.
    sim_image: str = "ghcr.io/hdl/sim/osvb"
    sif_path: Optional[str] = None  # explicit .sif; if unset, pull/convert sim_image
    sif_cache_dir: str = ""  # defaults to ~/.cache/nemo-gym/sif
    claude_node_dir: str = ""  # host Node+Claude prefix to bind (defaults to ./.claude_node)
    container_workdir: str = "/code"  # workspace mount point + cwd inside the container
    harness_workspace_dir: str = ""  # optional host parent dir for per-rollout workspaces

    system_prompt: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    claude_code_version: Optional[str] = None
    thinking: Optional[str] = None
    max_thinking_tokens: Optional[int] = None


class ClaudeCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class ClaudeCodeAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        ensure_claude_code(self.config.claude_code_version)
        try:
            ver = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            LOG.warning("claude-code version: %s", ver or "(unknown)")
        except Exception as exc:
            LOG.warning("could not determine claude-code version: %s", exc)

        # Apptainer setup
        self._sif_locks = {}
        self._sif_lock_guard = asyncio.Lock()
        self._sif_cache_dir = self.config.sif_cache_dir or os.path.join(Path.home(), ".cache", "nemo-gym", "sif")
        os.makedirs(self._sif_cache_dir, exist_ok=True)
        self._node_bind_dir = self._resolve_node_dir()
        LOG.warning("apptainer node bind dir=%s, sif cache=%s", self._node_bind_dir, self._sif_cache_dir)

    def _resolve_node_dir(self) -> str:
        """Locate a self-contained Node + Claude Code prefix to bind into the container."""
        node_dir = self.config.claude_node_dir or str(Path(__file__).parent / ".claude_node")
        if not os.path.isdir(os.path.join(node_dir, "bin")):
            raise RuntimeError(
                f"claude_node_dir not found at {node_dir!r}; expected a self-contained Node+Claude "
                "prefix (bin/, lib/). Run setup_claude_code.ensure_claude_code() or set claude_node_dir."
            )
        return node_dir

    async def _ensure_sif(self, image: str) -> str:
        """Return a cached SIF for the Docker image, pulling it on first use."""
        safe_name = image.replace("/", "_").replace(":", "_") + ".sif"
        sif_path = os.path.join(self._sif_cache_dir, safe_name)
        if os.path.exists(sif_path):
            return sif_path
        async with self._sif_lock_guard:
            lock = self._sif_locks.setdefault(image, asyncio.Lock())
        async with lock:
            if os.path.exists(sif_path):
                return sif_path
            tmp_path = sif_path + ".pulling"
            proc = await asyncio.create_subprocess_exec(
                "apptainer",
                "pull",
                "--force",
                tmp_path,
                f"docker://{image}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise RuntimeError(
                    f"apptainer pull failed for {image} (exit {proc.returncode}): {stderr.decode(errors='replace')}"
                )
            os.rename(tmp_path, sif_path)
            return sif_path

    async def _resolve_sif(self) -> str:
        if self.config.sif_path:
            return self.config.sif_path
        return await self._ensure_sif(self.config.sim_image)

    def _seed_workspace(self, workdir: Path, context_files: Dict[str, str]) -> None:
        """Create the CVDP workspace layout and write context files only (no harness/tests)."""
        for d in ("rtl", "verif", "docs", "src", "rundir"):
            (workdir / d).mkdir(parents=True, exist_ok=True)
        for filepath, content in (context_files or {}).items():
            if content is None:
                continue
            dest = workdir / filepath
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_text(content, encoding="utf-8")
            except Exception:
                LOG.warning("failed to seed context file %s", filepath)

    def _resolve_base_url(self) -> str:
        if self.config.model_server:
            cfg = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return self.server_client._build_server_base_url(cfg)
        return self.config.anthropic_base_url or ""

    async def _run_claude_code(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        workdir: Optional[str] = None,
    ) -> tuple[str, str]:
        """Run ``claude -p --output-format=stream-json`` and return (stdout, model_name).

        When ``workdir`` is given, claude is executed inside an Apptainer container
        bound to ``workdir`` (at ``container_workdir``) with the host Node+Claude
        install bind mounted in; otherwise it runs on the host (used for tasks with
        no target files, e.g. code-comprehension Q&A).
        """
        base_url = self._resolve_base_url()
        # Keep full model name for local/custom endpoints; strip provider prefix for real Anthropic API.
        model = self.config.model if base_url else self.config.model.split("/")[-1]
        api_key = self.config.anthropic_api_key

        sandbox = bool(workdir)

        # Config dir lives on the host. For sandbox runs it sits inside the bound
        # workspace so claude can read it at <container_workdir>/.claude_config.
        if sandbox:
            claude_config_dir = Path(workdir) / ".claude_config"
        else:
            claude_config_dir = Path.home() / ".claude_code_agent" / uuid4().hex
        claude_config_dir.mkdir(parents=True, exist_ok=True)
        try:
            (claude_config_dir / "settings.json").write_text(
                json.dumps(
                    {
                        "env": {
                            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                        }
                    }
                )
            )

            env_pairs = {
                "ANTHROPIC_API_KEY": api_key,  # pragma: allowlist secret
                "ANTHROPIC_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                "CLAUDE_CODE_SUBAGENT_MODEL": model,
                "IS_SANDBOX": "1",
            }
            if base_url:
                env_pairs["ANTHROPIC_BASE_URL"] = base_url
                env_pairs["ANTHROPIC_AUTH_TOKEN"] = api_key or "local"

            claude_args = [
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--dangerously-skip-permissions",
                "--bare",
                "--max-turns",
                str(self.config.max_turns),
                "--model",
                model,
            ]
            if system_prompt:
                claude_args += ["--append-system-prompt", system_prompt]
            if self.config.allowed_tools:
                claude_args += ["--allowedTools", self.config.allowed_tools]
            if self.config.disallowed_tools:
                claude_args += ["--disallowedTools", self.config.disallowed_tools]
            if self.config.thinking:
                claude_args += ["--thinking", self.config.thinking]
            if self.config.max_thinking_tokens is not None:
                claude_args += ["--max-thinking-tokens", str(self.config.max_thinking_tokens)]
            claude_args += ["--", instruction]

            if sandbox:
                wd = self.config.container_workdir
                sif_path = await self._resolve_sif()
                container_env = dict(env_pairs)
                # --cleanenv is intentionally not used: we set HOME/PATH/CLAUDE_CONFIG_DIR
                # plus the ANTHROPIC_* vars explicitly via --env below.
                container_env["HOME"] = wd
                container_env["CLAUDE_CONFIG_DIR"] = f"{wd}/.claude_config"
                container_env["PATH"] = (
                    "/opt/claude_node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                )
                env_args: list = []
                for key, value in container_env.items():
                    env_args += ["--env", f"{key}={value}"]
                cmd = [
                    "apptainer",
                    "exec",
                    "--writable-tmpfs",
                    "--no-mount",
                    "home",
                    "--pwd",
                    wd,
                    "--bind",
                    f"{workdir}:{wd}",
                    "--bind",
                    f"{self._node_bind_dir}:/opt/claude_node:ro",
                    *env_args,
                    sif_path,
                    "/opt/claude_node/bin/claude",
                    *claude_args,
                ]
                run_env = os.environ.copy()
            else:
                run_env = {**os.environ, **env_pairs, "CLAUDE_CONFIG_DIR": str(claude_config_dir)}
                cmd = ["claude", *claude_args]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=run_env,
                start_new_session=sandbox,  # own process group so we can kill apptainer + children
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                if sandbox:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                await proc.communicate()
                LOG.warning("claude-code timed out after %ds", self.config.timeout)
                return "", model

            if proc.returncode not in (0, None):
                LOG.warning("claude-code exited %s: %s", proc.returncode, stderr.decode(errors="replace")[:800])

            LOG.debug("claude-code stdout (%d chars): %s", len(stdout), stdout[:2000].decode(errors="replace"))
            return stdout.decode(errors="replace"), model
        finally:
            # In sandbox mode the config dir lives inside workdir, which run() cleans up.
            if not sandbox:
                shutil.rmtree(claude_config_dir, ignore_errors=True)

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

        stdout, model_name = await self._run_claude_code(user_message, system_prompt=system_prompt)
        output_items, usage = parse_stream_json(stdout)

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

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
        async with self.sem:
            meta = (body.model_extra or {}).get("verifier_metadata") or {}
            target_files = meta.get("target_files") or []
            if target_files:
                return await self._run_sandboxed(request, body, meta, target_files)

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
                url_path="/v1/responses",
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

            return ClaudeCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )

    async def _run_sandboxed(
        self,
        request: Request,
        body: ClaudeCodeAgentRunRequest,
        meta: dict,
        target_files: list,
    ) -> ClaudeCodeAgentVerifyResponse:
        """Agent-owned sandbox flow: seed a workspace with context files only and
        run claude inside the container so it can read the spec / companion RTL and
        self-test with the in-container EDA tools (iverilog/vvp). Grading stays
        faithful to the original CVDP benchmark: claude returns its solution as
        text per the system prompt, and that text is sent to /verify, where the
        resources server extracts the RTL from the response (no on-disk readback)."""
        cookies = request.cookies
        context_files = meta.get("context_files") or {}

        inp = body.responses_create_params.input
        if isinstance(inp, str):
            user_message, input_system = inp, None
        else:
            user_message, input_system = _extract_instruction(inp)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        tmp_root = self.config.harness_workspace_dir.strip()
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="cvdp_agent_", dir=tmp_root or None)
        try:
            self._seed_workspace(Path(workdir), context_files)

            stdout, model_name = await self._run_claude_code(
                user_message, system_prompt=system_prompt, workdir=workdir
            )
            output_items, usage = parse_stream_json(stdout)

            assistant_msgs = [
                it
                for it in output_items
                if getattr(it, "type", None) == "message" and getattr(it, "role", None) == "assistant"
            ]
            turns = len(assistant_msgs)
            naturally = bool(assistant_msgs)
            if not assistant_msgs:
                LOG.warning("claude-code produced no assistant message; padding empty output")
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(type="output_text", text="", annotations=[])],
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
                tool_choice=body.responses_create_params.tool_choice,
                tools=body.responses_create_params.tools,
                parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
                usage=NeMoGymResponseUsage(
                    input_tokens=input_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                    output_tokens=output_tokens,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=input_tokens + output_tokens,
                ),
            )
            agent_resp_json = response.model_dump()

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            return ClaudeCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    ClaudeCodeAgent.run_webserver()
