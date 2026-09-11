# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym control process; all Hermes imports and tools execute in the seeded sandbox."""

import asyncio
import json
import logging
from pathlib import Path
from shlex import quote
from time import monotonic, time
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


LOG = logging.getLogger(__name__)


class HermesSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model: str
    sandbox_provider: str = "sandbox"
    runtime_python: str = "/opt/hermes/bin/python3"
    remote_run_root: str = "/tmp"
    results_dir: str = "responses_api_agents/hermes_sandboxed_agent/results"
    concurrency: int = Field(default=4, ge=1)
    sandbox_timeout: float = Field(default=10800, gt=0)
    api_timeout: float = Field(default=1800, gt=0)
    max_turns: int = Field(default=90, gt=0)
    max_tokens: int | None = None
    temperature: float = 1.0
    terminal_timeout: int = 180
    enabled_toolsets: list[str] = Field(default_factory=lambda: ["terminal", "file"])
    compression_enabled: bool = True
    system_prompt: str | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class HermesSandboxedRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesSandboxedVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    # Gym's agent_run_error contract carries diagnostics, but no score or response.
    reward: float | None = Field(default=None, exclude_if=lambda value: value is None)
    response: NeMoGymResponse | None = Field(default=None, exclude_if=lambda value: value is None)
    hermes_result_path: str
    hermes_return_code: int | None
    hermes_error_type: str | None
    hermes_finished: bool
    turns_used: int


def trajectory_response(result, body, model, error_type=None):
    output = []
    for message in (result.get("messages") or [])[result.get("n_input", 0) :]:
        role = message.get("role")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content)
        if role == "assistant":
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                output.append(
                    {
                        "type": "reasoning",
                        "id": f"rs_{uuid4().hex}",
                        "summary": [{"type": "summary_text", "text": reasoning}],
                    }
                )
            if content:
                output.append(
                    {
                        "type": "message",
                        "id": f"msg_{uuid4().hex}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
            for call in message.get("tool_calls") or []:
                output.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        elif role == "tool":
            output.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": content})
    failed = bool(error_type or result.get("failed") or result.get("error"))
    completed = bool(result.get("completed")) and not result.get("interrupted")
    usage = result.get("usage") or {}
    inputs, outputs = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return NeMoGymResponse.model_validate(
        {
            "id": f"resp_{uuid4().hex}",
            "created_at": int(time()),
            "object": "response",
            "model": model,
            "status": "failed" if failed else "completed" if completed else "incomplete",
            "error": {"code": "server_error", "message": str(result.get("error") or error_type)} if failed else None,
            "output": output,
            "tool_choice": body.tool_choice,
            "tools": body.tools,
            "parallel_tool_calls": body.parallel_tool_calls,
            "usage": {
                "input_tokens": inputs,
                "output_tokens": outputs,
                "total_tokens": inputs + outputs,
                "input_tokens_details": {"cached_tokens": usage.get("cached_tokens", 0)},
                "output_tokens_details": {"reasoning_tokens": usage.get("reasoning_tokens", 0)},
            },
        }
    )


class HermesSandboxedAgent(SimpleResponsesAPIAgent):
    config: HermesSandboxedAgentConfig

    def model_post_init(self, context):
        super().model_post_init(context)
        self._sem = asyncio.Semaphore(self.config.concurrency)

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(400, "Use /run: Hermes requires a resources-server seeded sandbox")

    async def _run_in_sandbox(self, sandbox, body, rollout_id):
        started = monotonic()
        run_id = uuid4().hex
        local = Path(self.config.results_dir).resolve() / run_id
        local.mkdir(parents=True)
        remote = f"{self.config.remote_run_root.rstrip('/')}/nemo-hermes-{run_id}"
        result = {}
        return_code = None
        error_type = None
        stdout = stderr = ""
        try:
            cwd_result = await sandbox.exec("pwd", timeout_s=30)
            if (
                cwd_result.return_code
                or cwd_result.error_type
                or not (cwd_result.stdout or "").strip().startswith("/")
            ):
                raise RuntimeError(f"Cannot determine benchmark workdir: {cwd_result}")
            params = {
                key: getattr(self.config, key)
                for key in (
                    "model",
                    "max_turns",
                    "max_tokens",
                    "temperature",
                    "terminal_timeout",
                    "api_timeout",
                    "enabled_toolsets",
                    "compression_enabled",
                    "system_prompt",
                    "chat_template_kwargs",
                )
            } | {
                "run_dir": remote,
                "workdir": cwd_result.stdout.strip(),
                "base_url": self.resolve_model_base_url(self.config.model_server.name, rollout_id),
                "input": body.model_dump(mode="json")["input"],
            }
            # Only model input goes into the task container. Benchmark gold/test metadata stays outside.
            (local / "request.json").write_text(json.dumps(params))
            await sandbox.upload(local / "request.json", f"{remote}/request.json")
            await sandbox.upload(Path(__file__).with_name("runner.py"), f"{remote}/runner.py")
            executed = await sandbox.exec(
                f"{quote(self.config.runtime_python)} -I {quote(remote + '/runner.py')} {quote(remote + '/request.json')}",
                cwd=remote,
                timeout_s=self.config.sandbox_timeout,
            )
            return_code, error_type = executed.return_code, executed.error_type
            stdout, stderr = executed.stdout or "", executed.stderr or ""
            await sandbox.download(f"{remote}/result.json", local / "result.json")
            result = json.loads((local / "result.json").read_text())
            if return_code and not error_type:
                error_type = result.get("error_type") or "runner_exit"
        except Exception as exc:
            error_type = error_type or type(exc).__name__
            result = result | {"completed": False, "error": str(exc)}
            LOG.exception("Hermes sandbox execution failed; artifacts: %s", local)
        finally:
            (local / "agent_result.json").write_text(
                json.dumps(
                    {
                        "return_code": return_code,
                        "error_type": error_type,
                        "stdout": stdout,
                        "stderr": stderr,
                        "result": result,
                    },
                    indent=2,
                )
            )
        response = trajectory_response(result, body, self.config.model, error_type)
        return response, {
            "agent_run_time": monotonic() - started,
            "hermes_result_path": str(local / "agent_result.json"),
            "hermes_return_code": return_code,
            "hermes_error_type": error_type,
            "hermes_finished": response.status == "completed",
            "turns_used": result.get("api_calls", 0),
        }

    async def run(self, request: Request, body: HermesSandboxedRunRequest) -> HermesSandboxedVerifyResponse:
        async with self._sem:
            cookies = dict(request.cookies)
            seeded = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(mode="json") | {"create_pty": False},
                cookies=cookies,
            )
            await raise_for_status(seeded)
            cookies.update(seeded.cookies)
            sandbox = None
            try:
                seed = await get_response_json(seeded)
                descriptor = seed.get("sandbox_descriptor")
                if not isinstance(descriptor, dict) or not descriptor:
                    raise ValueError(
                        f"{self.config.resources_server.name} must return sandbox_descriptor from sandbox.serialize(); "
                        "a bare sandbox_handle is not sufficient for this agent"
                    )
                provider = create_provider(
                    resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
                )
                sandbox = await AsyncSandbox.connect(descriptor, provider=provider)
                response, metrics = await self._run_in_sandbox(
                    sandbox, body.responses_create_params, self.rollout_id_from_run(body)
                )
                verified = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/verify",
                    json=body.model_dump(mode="json") | {"response": response.model_dump(mode="json")},
                    cookies=cookies,
                )
                await raise_for_status(verified)
                result = await get_response_json(verified)
                result["agent_image_provenance"] = seed.get("image_provenance")
                result["verifier_reward"] = result["reward"]
                if response.status != "completed":
                    failure = f"Hermes response status: {response.status}"
                elif result.get("evaluation_completed") is False:
                    failure = result.get("error") or "Verification did not complete"
                else:
                    failure = None
                if failure:
                    # The collector routes this to its failures sidecar and excludes it from metrics.
                    # Preserve raw verifier_reward and the on-disk Hermes trajectory for diagnosis.
                    result.update(
                        reward=None,
                        response=None,
                        _ng_failure_class="agent_run_error",
                        _ng_failure_message=failure,
                    )
                return HermesSandboxedVerifyResponse.model_validate(result | metrics)
            finally:
                try:
                    cleaned = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/close_session",
                        json={},
                        cookies=cookies,
                    )
                    await raise_for_status(cleaned)
                except Exception:
                    LOG.exception("Failed to stop Hermes sandbox")
                    if sandbox is not None:
                        try:
                            await sandbox.stop()
                        except Exception:
                            LOG.exception("Sandbox fallback cleanup also failed")


if __name__ == "__main__":
    HermesSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = HermesSandboxedAgent.run_webserver()
