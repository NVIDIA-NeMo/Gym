# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym run endpoint for original-harness and OpenCode AgentVista episodes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    maybe_rollout_id_from_run_body,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.agentvista_agent.runtime.paths import output_path


class AgentVistaConfig(BaseResponsesAPIAgentConfig):
    source_root: str
    dataset: str
    image_folder: str
    task_selector: str = "all"
    model_server: ModelServerRef
    model_name: str
    output_root: str
    worker_python: str | None = None
    code_sandbox_python: str | None = None
    harness: Literal["opencode", "default"] = "opencode"
    default_max_turns: int = Field(default=30, ge=1)
    default_max_images: int = Field(default=100, ge=1)
    command: str = "opencode"
    opencode_version: str = "1.17.11"
    concurrency: int = Field(default=1, ge=1)
    max_steps: int = Field(default=150, ge=1)
    context_tokens: int = 65536
    output_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    mcp_timeout_ms: int = Field(default=180000, ge=1000)
    agent_timeout: int = Field(default=5400, ge=1)
    episode_timeout: int = Field(default=7200, ge=1)


class AgentVistaRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, str]


class AgentVistaVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    category_id: str
    artifacts: str
    verify_status: str
    trajectory: list[dict] = Field(default_factory=list)
    native_trajectory: list[dict] = Field(default_factory=list)
    execution_health: dict = Field(default_factory=dict)
    harness_version: str | None = None
    verifier_diagnostics: dict = Field(default_factory=dict)
    judge_response: str | None = None
    execution_error: str | None = None


class AgentVistaAgent(SimpleResponsesAPIAgent):
    """Execute episodes in separate processes so runtime globals cannot cross slots."""

    config: AgentVistaConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)
    _fingerprints: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        source = Path(self.config.source_root).resolve(strict=True)
        if not (source / "infer.py").is_file():
            raise ValueError(f"Missing upstream AgentVista source at {source}")
        # The AgentVista checkout stays outside Gym's temporary checkout/venv.
        sys.path.insert(0, str(source))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(400, "Use /run with prepared AgentVista verifier_metadata")

    async def run(self, body: AgentVistaRunRequest) -> AgentVistaVerifyResponse:
        # Capture IDs distinguish repeats and make HTTP retries idempotent.
        rollout_id = maybe_rollout_id_from_run_body(body)
        if not rollout_id:
            raise HTTPException(400, "AgentVista Bench requires _ng_rollout_id; enable Gym observability")
        identity = hashlib.sha256(rollout_id.encode()).hexdigest()
        from responses_api_agents.agentvista_agent.runtime import HARNESS_VERSION

        requested_version = body.verifier_metadata.get("harness_version")
        if requested_version and requested_version != HARNESS_VERSION:
            raise HTTPException(
                409,
                "Prepared dataset targets a different harness version; prepare it again",
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "request": body.model_dump(),
                    "config": self.config.model_dump(),
                    "harness_version": HARNESS_VERSION,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        output = output_path(self.config.output_root) / identity
        if identity in self._fingerprints and self._fingerprints[identity] != fingerprint:
            raise HTTPException(409, "Rollout ID reused with different request")
        if identity not in self._inflight:
            self._fingerprints[identity] = fingerprint
            self._inflight[identity] = asyncio.create_task(self._episode(body, output, fingerprint, rollout_id))
        # Duplicate requests share completion; cancelling the owning request cleans up the episode.
        task = self._inflight[identity]
        try:
            return await task
        finally:
            if task.done():
                self._inflight.pop(identity, None)
                self._fingerprints.pop(identity, None)

    async def _episode(self, body, output: Path, fingerprint: str, rollout_id: str) -> AgentVistaVerifyResponse:
        async with self._semaphore:
            output.mkdir(parents=True, exist_ok=True)
            # A persisted binding prevents retries from silently changing the task.
            binding = output / "binding.json"
            try:
                with binding.open("x") as stream:
                    json.dump({"fingerprint": fingerprint, "rollout_id": rollout_id}, stream)
            except FileExistsError:
                if json.loads(binding.read_text())["fingerprint"] != fingerprint:
                    raise HTTPException(409, "Rollout ID reused with different request")
                result_path = output / "result.json"
                if not result_path.exists():
                    raise HTTPException(
                        409,
                        "Incomplete episode requires inspection; use a new rollout ID to retry",
                    )
                return self._response(body, json.loads(result_path.read_text()))
            config = self.config.model_dump()
            # Row sampling controls from the public CLI override the configured defaults.
            sampling = body.responses_create_params
            for name, target in (
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_output_tokens", "output_tokens"),
            ):
                value = getattr(sampling, name, None)
                if value is not None:
                    config[target] = value
            config.update(
                output=str(output),
                code_sandbox_python=self.config.code_sandbox_python or sys.executable,
                task_id=body.verifier_metadata["task_id"],
                source_sha256=body.verifier_metadata["source_sha256"],
                model=self.config.model_name,
                endpoint=self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            )
            config_path = output / "worker.json"
            config_path.write_text(json.dumps(config, indent=2))
            env = dict(os.environ)
            gym_root = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = gym_root + os.pathsep + str(Path(self.config.source_root).resolve())
            env["AGENTVISTA_ROOT"] = str(Path(self.config.source_root).resolve())
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for key, folder in {
                "HOME": "home",
                "XDG_CACHE_HOME": "cache",
                "XDG_CONFIG_HOME": "config",
                "XDG_DATA_HOME": "data",
                "XDG_STATE_HOME": "state",
                "HF_HOME": "huggingface",
                "MPLCONFIGDIR": "matplotlib",
                "IPYTHONDIR": "ipython",
                "JUPYTER_DATA_DIR": "jupyter-data",
                "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
                "TMPDIR": "tmp",
            }.items():
                directory = output / "storage" / folder
                directory.mkdir(parents=True, exist_ok=True)
                env[key] = str(directory)
            process = None
            with (output / "worker.log").open("w") as log:
                try:
                    process = await asyncio.create_subprocess_exec(
                        self.config.worker_python or sys.executable,
                        "-m",
                        "responses_api_agents.agentvista_agent.runtime.default_worker"
                        if self.config.harness == "default"
                        else "responses_api_agents.agentvista_agent.runtime.worker",
                        str(config_path),
                        env=env,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    await asyncio.wait_for(process.wait(), timeout=self.config.episode_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    if process and process.returncode is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=45)
                        except asyncio.TimeoutError:
                            os.killpg(process.pid, signal.SIGKILL)
                            await process.wait()
                    raise
            result_path = output / "result.json"
            if not result_path.is_file():
                raise HTTPException(500, f"Worker exited without result; inspect {output}")
            return self._response(body, json.loads(result_path.read_text()))

    def _response(self, body: AgentVistaRunRequest, result: dict) -> AgentVistaVerifyResponse:
        from responses_api_agents.agentvista_agent.runtime.transcript import response_items

        masked = result.get("mask_sample", True)
        reward = 0.0 if masked else result["verify"]["reward"]
        # Earlier episodes may have a projected trajectory plus native events.
        # Always expose the original OpenCode events, without browser-use fields.
        native = result.get("native_trajectory", result.get("trajectory", []))
        items = result.get("response_items") or response_items(native)
        if not items:
            items = [
                {
                    "id": "msg_" + uuid4().hex,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": result.get("agent_output", ""),
                            "annotations": [],
                        }
                    ],
                }
            ]
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp_" + uuid4().hex,
                "created_at": int(time.time()),
                "model": self.config.model_name,
                "object": "response",
                "status": "incomplete"
                if result.get("execution_health", {}).get("termination")
                in {
                    "malformed_tool_call",
                    "model_protocol_error",
                    "timeout",
                    "native_error",
                    "output_budget_exhausted",
                }
                else "completed",
                "output": items,
                "parallel_tool_calls": body.responses_create_params.parallel_tool_calls,
                "tool_choice": body.responses_create_params.tool_choice,
                "tools": body.responses_create_params.tools,
            }
        )
        return AgentVistaVerifyResponse.model_validate(
            {
                **body.model_dump(),
                "response": response,
                "reward": reward,
                "harness": self.config.harness,
                "mask_sample": masked,
                "failure_kind": result.get("failure_kind"),
                "failure_reason": result.get("error") or result.get("execution_error"),
                "execution_error": result.get("execution_error"),
                "task_id": body.verifier_metadata["task_id"],
                "category_id": body.verifier_metadata["category_id"],
                "artifacts": result["artifacts"],
                "verify_status": result.get("verify", {}).get("status", "ERROR"),
                "trajectory": native,
                "native_trajectory": native,
                "execution_health": result.get("execution_health", {}),
                "harness_version": result.get("harness_version"),
                "source_provenance": result.get("source_provenance", {}),
                "verifier_diagnostics": result.get("verify", {}),
                "judge_response": result.get("verify", {}).get("analysis"),
                "task_success": result.get("task_success"),
                "reward_kind": "llm_judged",
                "deterministically_verified": False,
            }
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        rows = body.verify_responses
        valid = [r for r in rows if not r.get("mask_sample", False)]
        metrics = {
            "agentvista/num_scored": len(valid),
            "agentvista/num_masked": len(rows) - len(valid),
            "agentvista/num_verifier_errors": sum(r.get("failure_kind") == "agentvista:verifier" for r in rows),
        }
        health = [r["execution_health"] for r in rows if r.get("execution_health")]
        metrics["agentvista/num_execution_health_unobserved"] = len(rows) - len(health)
        if health:
            metrics["agentvista/tool_calls"] = sum(h["tool_calls"] for h in health)
            metrics["agentvista/tool_errors"] = sum(h["tool_errors"] for h in health)
            metrics["agentvista/execution_healthy_rate"] = sum(h["execution_healthy"] for h in health) / len(health)
            metrics["agentvista/malformed_terminations"] = sum(
                h["termination"] in {"malformed_tool_call", "model_protocol_error"} for h in health
            )
            metrics["agentvista/incomplete_tasks"] = sum(h.get("task_success") is False for h in health)
            metrics["agentvista/output_budget_exhausted"] = sum(
                h["termination"] == "output_budget_exhausted" for h in health
            )
        if valid:
            metrics["agentvista/mean_score"] = sum(r["reward"] for r in valid) / len(valid)
            metrics["agentvista/pass_rate"] = sum(r.get("verify_status") == "PASS" for r in valid) / len(valid)
            domains = [
                (r.get("verifier_metadata") or {}).get("category_id", r.get("category_id", "agentvista"))
                for r in valid
            ]
            for domain in sorted(set(domains)):
                subset = [r for r, row_domain in zip(valid, domains, strict=True) if row_domain == domain]
                metrics[f"agentvista/{domain}/mean_score"] = sum(r["reward"] for r in subset) / len(subset)
        return AggregateMetrics(
            agent_metrics=metrics,
            key_metrics={k: v for k, v in metrics.items() if k in {"agentvista/mean_score", "agentvista/pass_rate"}},
        )


if __name__ == "__main__":
    AgentVistaAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentVistaAgent.run_webserver()
