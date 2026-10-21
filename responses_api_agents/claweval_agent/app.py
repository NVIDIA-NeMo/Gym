# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate native Claw-Eval tasks through Gym's /run and /aggregate_metrics APIs."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
import sys
from collections import defaultdict
from pathlib import Path
from time import monotonic, time
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.claweval_agent.task_data import TaskData as ClawEvalMetadata
from responses_api_agents.claweval_agent.trajectory import (
    ClawEvalExecutionError,
    trace_observations,
    trace_to_response,
)
from responses_api_agents.claweval_agent.worker import source_root, task_path


class ClawEvalAgentConfig(BaseResponsesAPIAgentConfig):
    claweval_root: str
    claweval_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    claweval_config: str = "config_multimodal.yaml"
    python_executable: str = sys.executable
    fixture_root: str = ""
    sandbox_image: str
    sandbox_dependencies: str
    sandbox_port: int = Field(default=18080, ge=1024, le=45536)
    sandbox_ready_timeout: float = Field(default=600, gt=0)
    workspace_root: str = "outputs/claweval_agent"
    concurrency: int = Field(default=1, ge=1)
    timeout: float = Field(default=3600, gt=0)
    shutdown_grace: float = Field(default=45, gt=0)
    seed_base: int = 1001
    no_judge: bool = False
    model_server: ModelServerRef | None = None
    policy_base_url: str = ""
    policy_model_name: str = ""
    policy_api_key: str = ""
    model_overrides: dict[str, Any] = Field(default_factory=dict)
    judge_overrides: dict[str, Any] = Field(default_factory=dict)
    user_agent_model_overrides: dict[str, Any] = Field(default_factory=dict)


class ClawEvalRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    verifier_metadata: ClawEvalMetadata
    rollout_index: int = Field(default=0, ge=0, le=2, alias=ROLLOUT_INDEX_KEY_NAME)


class ClawEvalVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    passed: bool
    status: str = "completed"
    native_result: dict[str, Any]


def benchmark_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row[ROLLOUT_INDEX_KEY_NAME] not in (0, 1, 2):
            raise ValueError("Claw-Eval trial index must be 0, 1, or 2; apply repetition only during collection")
    masked = sum(bool(row.get("mask_sample")) for row in rows)
    rows = [row for row in rows if not row.get("mask_sample")]
    for row in rows:
        groups[row["task_id"]].append(row)
    complete: list[list[dict[str, Any]]] = []
    for task_id, trials in groups.items():
        indices = [row[ROLLOUT_INDEX_KEY_NAME] for row in trials]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate Claw-Eval trial for {task_id}")
        if set(indices) == {0, 1, 2}:
            complete.append(trials)
    metrics = {
        "claweval/tasks": len(groups),
        "claweval/trials": len(rows),
        "claweval/masked_trials": masked,
        "claweval/complete_three_trial_tasks": len(complete),
        "claweval/incomplete_three_trial_tasks": len(groups) - len(complete),
    }
    if rows:
        metrics["claweval/mean_task_score"] = sum(row["reward"] for row in rows) / len(rows)
        metrics["claweval/pass_at_1"] = sum(row["passed"] for row in rows) / len(rows)
    # Never present a subset of complete tasks as the full selected run's Pass³.
    if complete and len(complete) == len(groups) and not masked:
        metrics["claweval/pass_at_3"] = sum(any(row["passed"] for row in group) for group in complete) / len(complete)
        metrics["claweval/strict_pass_3"] = sum(all(row["passed"] for row in group) for group in complete) / len(
            complete
        )
    return metrics


async def stop_worker(process: asyncio.subprocess.Process, grace: float) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace)
    except asyncio.TimeoutError:
        pass
    finally:
        # Also reap any descendant that survived a worker exit (service or srun).
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


class ClawEvalAgent(SimpleResponsesAPIAgent):
    config: ClawEvalAgentConfig
    ray_enabled = False

    def model_post_init(self, context: Any) -> None:
        self._sem = asyncio.Semaphore(self.config.concurrency)
        super().model_post_init(context)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise HTTPException(status_code=400, detail="Claw-Eval requires task metadata; use /run")

    async def run(self, body: ClawEvalRunRequest = Body()) -> ClawEvalVerifyResponse:
        async with self._sem:
            started = monotonic()
            root = source_root(self.config.claweval_root, expected_revision=self.config.claweval_revision)
            meta = body.verifier_metadata
            task_path(root, meta.task_id, meta.task_sha256)
            params = body.responses_create_params
            inputs = params.input
            if isinstance(inputs, str):
                prompt = inputs
            else:
                items = [item.model_dump() if hasattr(item, "model_dump") else item for item in inputs]
                if len(items) != 1 or items[0].get("role") != "user" or not isinstance(items[0].get("content"), str):
                    raise ValueError("Claw-Eval expects the single native user prompt produced by prepare.py")
                prompt = items[0]["content"]
            output_dir = Path(self.config.workspace_root).expanduser().resolve() / f"{meta.task_id}-{uuid4().hex}"
            output_dir.mkdir(parents=True)
            invocation_id = f"claweval-{uuid4().hex}"
            payload = self.config.model_dump()
            payload.update(
                {
                    "operation": "run",
                    "invocation_id": invocation_id,
                    "claweval_root": str(root),
                    "output_dir": str(output_dir),
                    "verifier_metadata": meta.model_dump(),
                    "prompt": prompt,
                    "seed": self.config.seed_base + body.rollout_index,
                    "sandbox_dependencies": str(Path(self.config.sandbox_dependencies).expanduser().resolve()),
                }
            )
            # Gym's per-request generation overrides should reach the native model.
            model_overrides = dict(self.config.model_overrides)
            for key, value in (
                ("base_url", self.config.policy_base_url),
                ("model_id", self.config.policy_model_name),
                ("api_key", self.config.policy_api_key),
            ):
                if value:
                    model_overrides[key] = value
            if self.config.model_server is not None:
                model_overrides["base_url"] = self.resolve_model_base_url(
                    self.config.model_server.name,
                    self.rollout_id_from_run(
                        {**body.model_dump(by_alias=True), "_ng_rollout_id": body.capture_rollout_id}
                    ),
                )
                model_overrides["api_key"] = "gym-local"
            elif self._model_call_capture_enabled():
                raise ValueError("ClawEval observability requires model_server pointing to a Gym policy model server")
            for key in ("temperature", "top_p", "max_output_tokens"):
                value = getattr(params, key, None)
                if value is None:
                    continue
                if key == "temperature":
                    model_overrides[key] = value
                else:
                    extra = dict(model_overrides.get("extra_body") or {})
                    extra["max_tokens" if key == "max_output_tokens" else key] = value
                    model_overrides["extra_body"] = extra
            payload["model_overrides"] = model_overrides
            try:
                await self._launch_worker(payload, output_dir)
                result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            except (OSError, RuntimeError) as exc:
                return self._failed_response(
                    body,
                    output_dir,
                    invocation_id,
                    prompt,
                    started,
                    {
                        "status": "failed",
                        "failure_kind": "agent_timeout" if isinstance(exc, TimeoutError) else "agent_run_error",
                        "failure_reason": f"{type(exc).__name__}; inspect {output_dir / 'worker.log'}",
                    },
                )
            if result["task_id"] != meta.task_id:
                raise ValueError("Claw-Eval worker returned a mismatched task")
            if result.get("failure_kind") == "claweval:contract_error":
                raise ValueError(result["failure_reason"])
            if result.get("status") == "failed" and result.get("failure_kind"):
                return self._failed_response(body, output_dir, invocation_id, prompt, started, result)
            score = float(result["task_score"])
            if result["task_id"] != meta.task_id or result["status"] != "completed":
                raise ValueError("Claw-Eval worker returned an incomplete or mismatched task")
            if not math.isfinite(score) or not 0 <= score <= 1 or result["passed"] != (score >= 0.75):
                raise ValueError("Invalid native Claw-Eval score or pass flag")
            trace = Path(result["trace"]).resolve()
            if not trace.is_relative_to(output_dir):
                raise ValueError("Claw-Eval returned a trace outside this rollout workspace")
            try:
                response = trace_to_response(trace, meta.task_id, result["model"], expected_score=score)
            except ClawEvalExecutionError as exc:
                return self._failed_response(
                    body,
                    output_dir,
                    invocation_id,
                    prompt,
                    started,
                    {
                        **result,
                        "status": "failed",
                        "failure_kind": "claweval:execution_error",
                        "failure_reason": str(exc),
                    },
                )
            return ClawEvalVerifyResponse.model_validate(
                {
                    **body.model_dump(by_alias=True),
                    "response": response,
                    "reward": score,
                    "task_id": meta.task_id,
                    "passed": result["passed"],
                    "native_result": result,
                    "evaluation_completed": True,
                    "ng_agent_observations": trace_observations(
                        trace,
                        invocation_id=invocation_id,
                        response=response,
                        prompt=prompt,
                        output_dir=output_dir,
                        duration_ms=(monotonic() - started) * 1000,
                    ),
                }
            )

    def _failed_response(
        self,
        body: ClawEvalRunRequest,
        output_dir: Path,
        invocation_id: str,
        prompt: str,
        started: float,
        result: dict[str, Any],
    ) -> ClawEvalVerifyResponse:
        response = NeMoGymResponse.model_validate(
            {
                "id": invocation_id,
                "created_at": int(time()),
                "object": "response",
                "model": "unknown",
                "output": [],
                "parallel_tool_calls": False,
                "tools": [],
                "tool_choice": "auto",
            }
        )
        return ClawEvalVerifyResponse.model_validate(
            {
                **body.model_dump(by_alias=True),
                "response": response,
                "reward": 0,
                "task_id": body.verifier_metadata.task_id,
                "passed": False,
                "status": "failed",
                "evaluation_completed": False,
                "mask_sample": True,
                "failure_kind": result["failure_kind"],
                "failure_reason": result["failure_reason"],
                "native_result": result,
                "ng_agent_observations": trace_observations(
                    None,
                    invocation_id=invocation_id,
                    response=response,
                    prompt=prompt,
                    output_dir=output_dir,
                    duration_ms=(monotonic() - started) * 1000,
                    failed=True,
                ),
            }
        )

    async def _launch_worker(self, payload: dict[str, Any], output_dir: Path) -> None:
        worker = Path(__file__).with_name("worker.py")
        log_path = output_dir / "worker.log"
        # Invoking a venv Python directly does not activate its companion tools
        # (notably ffprobe/ffmpeg used by the native video graders).
        native_python = shutil.which(self.config.python_executable) or self.config.python_executable
        native_bin = str(Path(native_python).absolute().parent)
        # Credentials travel over stdin, never argv or a generated config file.
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                self.config.python_executable,
                str(worker),
                cwd=payload["claweval_root"],
                env={
                    **os.environ,
                    "PATH": native_bin + os.pathsep + os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CLAW_AGENT_HARNESS": "native",
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.communicate(json.dumps(payload).encode()), timeout=self.config.timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.shield(stop_worker(process, self.config.shutdown_grace))
                raise
            if process.returncode:
                raise RuntimeError(f"Claw-Eval worker exited {process.returncode}; inspect {log_path}")

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return benchmark_metrics([row for task in tasks for row in task])

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("claweval/")}


if __name__ == "__main__":
    ClawEvalAgent.run_webserver()
