# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://gitlab-master.nvidia.com/pritamb/saas-bench/-/blob/ad48579c1af66bb7100501f45460bb3e9ec5eaae/integrations/gym/responses_api_agents/saas_bench_agent/app.py (Apache-2.0).
"""Gym run endpoint: own one native SaaS lifecycle and direct OpenCode process."""

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
from nemo_gym.core.failure_kinds import AGENT_RUN_ERROR, VERIFIER_ERROR
from nemo_gym.server_runtime import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class SaasBenchConfig(BaseResponsesAPIAgentConfig):
    num_workers: int = 1
    source_root: str
    tasks_dir: str = "tasks"
    task_selector: str = "all"
    model_server: ModelServerRef
    model_name: str
    output_root: str
    worker_python: str | None = None
    command: str = "opencode"
    opencode_version: Literal["1.17.11"] = "1.17.11"
    concurrency: int = Field(default=1, ge=1)
    slot_capacity: int = Field(default=4, ge=1, le=40)
    slot_lock_root: str = "/tmp/saas-opencode-slots"
    hostname: str = "localhost"
    max_steps: int = Field(default=200, ge=1)
    context_tokens: int = 262144
    output_tokens: int = 32768
    viewport_width: int = 1280
    viewport_height: int = 960
    mcp_timeout_ms: int = Field(default=180000, ge=1000)
    agent_timeout: int = Field(default=3600, ge=1)
    episode_timeout: int = Field(default=7200, ge=1)


class SaasBenchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, str]


class SaasBenchVerifyResponse(BaseVerifyResponse):
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
    execution_error: str | None = None


class SaasBenchAgent(SimpleResponsesAPIAgent):
    """Execute episodes in separate processes so runtime globals cannot cross slots."""

    config: SaasBenchConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)
    _fingerprints: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        if self.config.num_workers != 1 or self.config.concurrency > self.config.slot_capacity:
            raise ValueError("SaaS episodes require num_workers=1 and concurrency <= slot_capacity")
        if self.config.command == "opencode":
            from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode

            ensure_opencode(self.config.opencode_version)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        source = Path(self.config.source_root).resolve(strict=True)
        if not (source / "saas_bench/opencode/worker.py").is_file():
            raise ValueError(f"Missing SaaS OpenCode implementation at {source}")
        # The SaaS checkout stays outside Gym's temporary checkout/venv.
        sys.path.insert(0, str(source))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(400, "Use /run with prepared SaaS verifier_metadata")

    async def run(self, body: SaasBenchRunRequest) -> SaasBenchVerifyResponse:
        # Capture IDs distinguish repeats and make HTTP retries idempotent.
        rollout_id = maybe_rollout_id_from_run_body(body)
        if not rollout_id:
            raise HTTPException(400, "SaaS Bench requires _ng_rollout_id; enable Gym observability")
        identity = hashlib.sha256(rollout_id.encode()).hexdigest()
        from saas_bench.opencode.core import HARNESS_VERSION

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
        output = Path(self.config.output_root).resolve() / identity
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

    async def _episode(self, body, output: Path, fingerprint: str, rollout_id: str) -> SaasBenchVerifyResponse:
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
            config.update(
                output=str(output),
                task_id=body.verifier_metadata["task_id"],
                source_sha256=body.verifier_metadata["source_sha256"],
                model=self.config.model_name,
                endpoint=self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            )
            config_path = output / "worker.json"
            config_path.write_text(json.dumps(config, indent=2))
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(self.config.source_root).resolve())
            process = None
            with (output / "worker.log").open("w") as log:
                try:
                    process = await asyncio.create_subprocess_exec(
                        self.config.worker_python or sys.executable,
                        "-m",
                        "saas_bench.opencode.worker",
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

    def _response(self, body: SaasBenchRunRequest, result: dict) -> SaasBenchVerifyResponse:
        from saas_bench.opencode.core import measured_score, response_items
        from saas_bench.opencode.observability import native_trajectory

        masked = result.get("mask_sample", True)
        reward = 0.0 if masked else measured_score(result["verify"])
        # Earlier episodes may have a projected trajectory plus native events.
        # Always expose the original OpenCode events, without browser-use fields.
        native = result.get("native_trajectory", result.get("trajectory", []))
        items = response_items(native)
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
                }
                else "completed",
                "output": items,
                "parallel_tool_calls": body.responses_create_params.parallel_tool_calls,
                "tool_choice": body.responses_create_params.tool_choice,
                "tools": body.responses_create_params.tools,
            }
        )
        return SaasBenchVerifyResponse.model_validate(
            {
                **body.model_dump(),
                "response": response,
                "reward": reward,
                "mask_sample": masked,
                "failure_kind": (
                    VERIFIER_ERROR
                    if result.get("failure_kind") == "saas_bench:verifier"
                    else AGENT_RUN_ERROR
                    if result.get("failure_kind")
                    else None
                ),
                "native_failure_kind": result.get("failure_kind"),
                "native_metrics": {} if masked else {"weighted_score": reward, "execution": 1.0},
                "native_diagnostics": {}
                if masked
                else {
                    "task_id": body.verifier_metadata["task_id"],
                    "source_sha256": body.verifier_metadata["source_sha256"],
                    "rollout_id": maybe_rollout_id_from_run_body(body),
                    "verifier": result["verify"],
                    "graded_response_id": response.id,
                    "graded_response_sha256": hashlib.sha256(response.model_dump_json().encode()).hexdigest(),
                },
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
                "verifier_diagnostics": result.get("verifier_diagnostics", {}),
                "ng_trajectory": native_trajectory(
                    native,
                    body.verifier_metadata["task_id"],
                    maybe_rollout_id_from_run_body(body) or "unknown",
                    result.get("execution_health", {}),
                ),
            }
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        rows = body.verify_responses
        valid = [r for r in rows if not r.get("mask_sample", False)]
        metrics = {
            "saas_bench/num_scored": len(valid),
            "saas_bench/num_masked": len(rows) - len(valid),
            "saas_bench/num_verifier_errors": sum(
                r.get("failure_kind") in {"saas_bench:verifier", VERIFIER_ERROR} for r in rows
            ),
        }
        health = [r["execution_health"] for r in rows if r.get("execution_health")]
        metrics["saas_bench/num_execution_health_unobserved"] = len(rows) - len(health)
        if health:
            metrics["saas_bench/tool_calls"] = sum(h["tool_calls"] for h in health)
            metrics["saas_bench/tool_errors"] = sum(h["tool_errors"] for h in health)
            metrics["saas_bench/execution_healthy_rate"] = sum(h["execution_healthy"] for h in health) / len(health)
            metrics["saas_bench/malformed_terminations"] = sum(
                h["termination"] in {"malformed_tool_call", "model_protocol_error"} for h in health
            )
            metrics["saas_bench/incomplete_tasks"] = sum(h.get("task_success") is False for h in health)
        if valid:
            metrics["saas_bench/mean_score"] = sum(r["reward"] for r in valid) / len(valid)
            metrics["saas_bench/pass_rate"] = sum(r["verify_status"] == "PASS" for r in valid) / len(valid)
            for domain in sorted({r["category_id"] for r in valid}):
                subset = [r for r in valid if r["category_id"] == domain]
                metrics[f"saas_bench/{domain}/mean_score"] = sum(r["reward"] for r in subset) / len(subset)
        return AggregateMetrics(
            agent_metrics=metrics,
            key_metrics={k: v for k, v in metrics.items() if k in {"saas_bench/mean_score", "saas_bench/pass_rate"}},
        )


if __name__ == "__main__":
    SaasBenchAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SaasBenchAgent.run_webserver()
