# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run original OpenClaw episodes through Gym's rollout-correlated policy proxy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.episode_context import maybe_rollout_id_from_run_body
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.utils.server_process import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.openclaw_agent.app import openclaw_session_conversation
from responses_api_agents.openclaw_agent.observability import build_openclaw_observations
from responses_api_agents.wildclawbench_agent.metrics import benchmark_metrics


class WildClawConfig(BaseResponsesAPIAgentConfig):
    source_root: str
    worker_python: str
    output_root: str
    model_server: ModelServerRef
    model_name: str
    context_window: int = Field(default=131072, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    thinking: str = "high"
    sandbox_runtime: str = "enroot"

    @model_validator(mode="after")
    def validate_execution(self) -> WildClawConfig:
        if self.context_window <= 4 * self.max_tokens:
            raise ValueError("Context must exceed four completion budgets for native compaction")
        if self.sandbox_runtime not in {"docker", "enroot"}:
            raise ValueError("sandbox_runtime must be docker or enroot")
        return self


class WildClawRunRequest(BaseRunRequest):
    verifier_metadata: dict


class WildClawVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_sha256: str
    source_revision: str
    native_reward: float | None
    artifact_dir: str
    errors: list[str]


def worker_environment(config: WildClawConfig, output: Path) -> dict[str, str]:
    """Keep cache/home mounts away from the host's home directories."""
    env = dict(os.environ)
    storage = output / "storage"
    for name, folder in {
        "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "XDG_STATE_HOME": "state",
        "HF_HOME": "huggingface",
        "UV_CACHE_DIR": "uv",
        "PIP_CACHE_DIR": "pip",
        "TMPDIR": "tmp",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        directory = storage / folder
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PATH"] = f"{Path(config.worker_python).parent}:{env['PATH']}"
    env["DOCKER_IMAGE"] = "wildclawbench-ubuntu:v1.3"
    env["DOCKER_NETWORK_MODE"] = "host"
    if config.sandbox_runtime == "enroot":
        env["PATH"] = f"{config.source_root}/cluster/bin:{env['PATH']}"
        env["WILDCLAW_ENROOT_IMAGE"] = str(Path(config.source_root) / ".cache/wildclawbench-ubuntu_v1.3.sqsh")
        env["ENROOT_MOUNT_HOME"] = "n"
        for name, folder in {
            "WILDCLAW_ENROOT_STATE_DIR": "state",
            "ENROOT_CACHE_PATH": "cache",
            "ENROOT_DATA_PATH": "data",
            "ENROOT_RUNTIME_PATH": "runtime",
            "ENROOT_TEMP_PATH": "tmp",
            "ENROOT_CONFIG_PATH": "config",
        }.items():
            directory = storage / "enroot" / folder
            directory.mkdir(parents=True, exist_ok=True)
            env[name] = str(directory)
    return env


class WildClawAgent(SimpleResponsesAPIAgent):
    config: WildClawConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _inflight: dict[str, tuple[str, asyncio.Task]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        # Warmup services can use fixed ports: serialize native episodes on each worker.
        self._semaphore = asyncio.Semaphore(1)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(400, "Use /run with prepared WildClawBench task metadata")

    async def run(self, body: WildClawRunRequest) -> WildClawVerifyResponse:
        rollout_id = maybe_rollout_id_from_run_body(body)
        if not rollout_id:
            raise HTTPException(400, "WildClawBench requires Gym observability and a rollout ID")
        identity = hashlib.sha256(rollout_id.encode()).hexdigest()
        fingerprint = hashlib.sha256(
            json.dumps(
                dict(request=body.model_dump(), config=self.config.model_dump()),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        existing = self._inflight.get(identity)
        if existing is not None and existing[0] != fingerprint:
            raise HTTPException(409, "Rollout ID reused with another task or configuration")
        if existing is None:
            task = asyncio.create_task(self._episode(body, rollout_id, identity, fingerprint))
            self._inflight[identity] = (fingerprint, task)
        else:
            task = existing[1]
        try:
            return await task
        finally:
            if task.done():
                self._inflight.pop(identity, None)

    async def _episode(
        self,
        body: WildClawRunRequest,
        rollout_id: str,
        identity: str,
        fingerprint: str,
    ) -> WildClawVerifyResponse:
        async with self._semaphore:
            # absolute(), rather than resolve(), preserves the path mounted into task containers.
            output = Path(self.config.output_root).absolute() / identity
            output.mkdir(parents=True, exist_ok=True)
            output.chmod(0o700)
            try:
                with (output / "binding.json").open("x") as stream:
                    json.dump(dict(fingerprint=fingerprint, rollout_id=rollout_id), stream)
            except FileExistsError:
                binding = json.loads((output / "binding.json").read_text())
                if binding["fingerprint"] != fingerprint or not (output / "result.json").is_file():
                    raise HTTPException(
                        409, "Changed or incomplete episode; inspect artifacts before an explicit retry"
                    )
                return self._response(body, json.loads((output / "result.json").read_text()), rollout_id)
            inputs = body.responses_create_params.input
            if not isinstance(inputs, list) or len(inputs) != 1:
                raise HTTPException(400, "Use the unmodified prepared single user prompt")
            item = inputs[0].model_dump() if hasattr(inputs[0], "model_dump") else inputs[0]
            if item.get("role") != "user" or not isinstance(item.get("content"), str):
                raise HTTPException(400, "Use the unmodified prepared user prompt")
            worker_config = dict(
                source_root=self.config.source_root,
                output=str(output),
                metadata=body.verifier_metadata,
                input_prompt=item["content"],
                model_name=self.config.model_name,
                policy_base_url=self.resolve_model_base_url(self.config.model_server.name, rollout_id),
                context_window=self.config.context_window,
                max_tokens=self.config.max_tokens,
                thinking=self.config.thinking,
            )
            config_file = output / "worker.json"
            config_file.write_text(json.dumps(worker_config, indent=2))
            env = worker_environment(self.config, output)
            command = [
                self.config.worker_python,
                "-m",
                "responses_api_agents.wildclawbench_agent.worker",
                str(config_file),
            ]
            process = None
            with (output / "worker.log").open("w") as log:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        env=env,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    # Includes native warmups, the original task budget, verification and cleanup.
                    await asyncio.wait_for(
                        process.wait(), timeout=int(body.verifier_metadata["timeout_seconds"]) + 900
                    )
                finally:
                    if process is not None and process.returncode is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=45)
                        except asyncio.TimeoutError:
                            os.killpg(process.pid, signal.SIGKILL)
                            await process.wait()
                    cleanup = await asyncio.create_subprocess_exec(
                        *command,
                        "--cleanup",
                        env=env,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    try:
                        await asyncio.wait_for(cleanup.wait(), timeout=60)
                    except asyncio.TimeoutError:
                        cleanup.kill()
                        await cleanup.wait()
                        raise RuntimeError(f"Sandbox cleanup timed out; inspect {output}") from None
                    if cleanup.returncode:
                        raise RuntimeError(f"Sandbox cleanup failed; inspect {output}")
            if process.returncode or not (output / "result.json").is_file():
                raise RuntimeError(f"Worker failed without a complete result; inspect {output}")
            return self._response(body, json.loads((output / "result.json").read_text()), rollout_id)

    def _response(self, body: WildClawRunRequest, result: dict, rollout_id: str) -> WildClawVerifyResponse:
        events = result.pop("events", [])
        conversation = openclaw_session_conversation(events)
        output_items = [
            item for item in conversation if getattr(item, "role", None) not in {"user", "system", "developer"}
        ]
        response = NeMoGymResponse.model_validate(
            dict(
                id="resp_" + uuid4().hex,
                created_at=int(time.time()),
                model=self.config.model_name,
                object="response",
                status="incomplete" if result["mask_sample"] else "completed",
                output=output_items,
                parallel_tool_calls=False,
                tools=[],
                tool_choice="auto",
            )
        )
        observations = build_openclaw_observations(
            rollout_id,
            conversation,
            events,
            transcript_available=bool(events),
            prefer_openclaw_session_id=False,
            capture_correlated=True,
            model_ref=self.config.model_server,
        )
        return WildClawVerifyResponse.model_validate(
            dict(
                **body.model_dump(),
                **result,
                response=response,
                ng_agent_observations=observations,
            )
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        metrics = benchmark_metrics(body.verify_responses)
        return AggregateMetrics(
            agent_metrics=metrics,
            key_metrics={key: value for key, value in metrics.items() if key == "wildclawbench/overall_score"},
        )


if __name__ == "__main__":
    WildClawAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = WildClawAgent.run_webserver()
