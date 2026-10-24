# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from pathlib import Path
from shlex import quote
from time import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.deepseek_harness_agent.trajectory import convert_events


class DeepSeekHarnessAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model: str
    sandbox_provider: str
    profile: Literal["sdk", "sdk-minimal"] = "sdk"
    reasoning_effort: str | None = None
    max_tokens: int = Field(default=32768, gt=0)
    timeout_s: float = Field(default=1800, gt=0)
    max_concurrency: int = Field(default=4, gt=0)
    python_executable: str = "python3"
    results_dir: Path = Path(__file__).parent / "results"
    # Also clean up failed runs and resources that delegate sandbox cleanup.
    stop_sandbox: bool = True


class DeepSeekHarnessRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class DeepSeekHarnessVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    dsh_finish_reason: str | None
    dsh_error: str | None = None
    dsh_artifacts: str


def task_text(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    if body.instructions or body.tools or body.previous_response_id:
        raise HTTPException(422, "DSH accepts a fresh text task; configure its instructions and tools in its profile.")
    if isinstance(body.input, str):
        prompt = body.input
    else:
        if len(body.input) != 1 or getattr(body.input[0], "role", None) != "user":
            raise HTTPException(422, "DSH requires one user task, without conversation history.")
        content = body.input[0].content
        if isinstance(content, str):
            prompt = content
        else:
            if any(part.get("type") != "input_text" for part in content):
                raise HTTPException(422, "DSH currently supports text tasks only.")
            prompt = "\n".join(part["text"] for part in content)
    if not prompt.strip():
        raise HTTPException(422, "DSH requires a non-empty task.")
    return prompt


class DeepSeekHarnessAgent(SimpleResponsesAPIAgent):
    config: DeepSeekHarnessAgentConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        sandbox = getattr(request.state, "dsh_sandbox", None)
        if sandbox is None:
            raise HTTPException(400, "Use /run to prepare the task sandbox before running DSH.")
        session_id = uuid4().hex
        local_dir = self.config.results_dir / session_id
        local_dir.mkdir(parents=True)
        remote_dir = f"/tmp/nemo-gym-dsh-{session_id}"
        harness_config = {
            "model": body.model or self.config.model,
            "profile": self.config.profile,
            "base_url": self.resolve_model_base_url(self.config.model_server.name, request.state.dsh_rollout_id),
            "api_key": "EMPTY",
            # Gym owns container isolation; avoid nested host sandbox/approval dependencies.
            "env": {"DSH_PERMISSION_MODE": "danger-full-access"},
            "reasoning_effort": self.config.reasoning_effort,
            "max_tokens": body.max_output_tokens or self.config.max_tokens,
            "initialize_timeout_seconds": min(120, self.config.timeout_s),
            "request_timeout_seconds": self.config.timeout_s,
        }
        input_path = local_dir / "input.json"
        input_path.write_text(
            json.dumps({"prompt": task_text(body), "session_id": session_id, "harness": harness_config})
        )
        result = await sandbox.exec(f"mkdir -m 700 {quote(remote_dir)}")
        if result.return_code != 0:
            raise RuntimeError("Could not create the DSH run directory in the task sandbox")
        await sandbox.upload(Path(__file__).with_name("runner.py"), f"{remote_dir}/runner.py")
        await sandbox.upload(input_path, f"{remote_dir}/input.json")
        execution_error = None
        try:
            result = await sandbox.exec(
                f"{quote(self.config.python_executable)} {remote_dir}/runner.py {remote_dir}/input.json",
                timeout_s=self.config.timeout_s,
            )
            (local_dir / "stdout.log").write_text(result.stdout or "")
            (local_dir / "stderr.log").write_text(result.stderr or "")
            if result.return_code != 0:
                execution_error = f"DSH runner exited with code {result.return_code}"
        except Exception as exc:
            execution_error = f"{type(exc).__name__}: {exc}"
        finally:
            for name in ("events.jsonl", "result.json"):
                try:
                    await sandbox.download(f"{remote_dir}/{name}", local_dir / name)
                except Exception:
                    logging.getLogger(__name__).warning("Could not retrieve DSH artifact %s", name, exc_info=True)
        result_path = local_dir / "result.json"
        try:
            summary = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = {"finish_reason": "error", "error": execution_error or "Missing or invalid DSH result"}
        if summary["error"]:
            summary["finish_reason"] = "error"
        notifications = []
        events_path = local_dir / "events.jsonl"
        if not events_path.exists():
            summary.update(finish_reason="error", error=summary.get("error") or "Missing DSH event log")
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                try:
                    notifications.append(json.loads(line))
                except json.JSONDecodeError:
                    # A terminated runner can leave a partial final JSONL record.
                    summary["error"] = "Incomplete DSH event log"
                    summary["finish_reason"] = "error"
        output, usage = convert_events(notifications, session_id)
        request.state.dsh_result = {
            "dsh_finish_reason": summary["finish_reason"],
            "dsh_error": summary["error"],
            "dsh_artifacts": str(local_dir),
        }
        return NeMoGymResponse(
            id=f"resp_{session_id}",
            created_at=int(time()),
            model=harness_config["model"],
            object="response",
            output=output,
            usage=usage,
            status="completed" if summary["finish_reason"] == "completed" else "incomplete",
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
        )

    async def run(self, request: Request, body: DeepSeekHarnessRunRequest) -> DeepSeekHarnessVerifyResponse:
        task_text(body.responses_create_params)
        async with self._semaphore:
            seed = await self.server_client.post(
                self.config.resources_server.name, "/seed_session", json=body.model_dump(), cookies=request.cookies
            )
            await raise_for_status(seed)
            cookies = request.cookies | seed.cookies
            state = await get_response_json(seed)
            descriptor = state.get("sandbox_descriptor")
            if descriptor is None:
                # The existing sandboxed-agent wire protocol carries an opaque sandbox id.
                descriptor = {"sandbox_id": state["sandbox_handle"]}
            provider = create_provider(
                resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
            )
            sandbox = None
            try:
                sandbox = await AsyncSandbox.connect(descriptor, provider=provider)
                request.state.dsh_sandbox = sandbox
                request.state.dsh_rollout_id = self.rollout_id_from_run(body)
                response = await self.responses(request, body.responses_create_params)
                verify_body = body.model_dump() | {"response": response.model_dump()}
                if self.config.skip_verification:
                    verified = verify_body | {"reward": self.config.skip_verification_reward}
                else:
                    verification = await self.server_client.post(
                        self.config.resources_server.name, "/verify", json=verify_body, cookies=cookies
                    )
                    await raise_for_status(verification)
                    verified = await get_response_json(verification)
                return DeepSeekHarnessVerifyResponse.model_validate(verified | request.state.dsh_result)
            finally:
                request.state.dsh_sandbox = None
                try:
                    if sandbox is not None and self.config.stop_sandbox:
                        await sandbox.stop()
                    else:
                        await provider.aclose()
                except Exception:
                    logging.getLogger(__name__).warning("Could not clean up DSH sandbox connection", exc_info=True)


if __name__ == "__main__":
    DeepSeekHarnessAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepSeekHarnessAgent.run_webserver()
