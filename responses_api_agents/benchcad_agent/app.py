# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BenchCAD with single-turn or OpenCode inference and the upstream CAD scorer."""

import asyncio
import base64
import json
import logging
from pathlib import Path
from time import time
from typing import Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import Request
from pydantic import ConfigDict, Field, JsonValue, field_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    TrajectoryRecord,
    TrajectoryTurn,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from responses_api_agents.benchcad_agent.runtime import UPSTREAM_REVISION, run_worker
from responses_api_agents.benchcad_agent.worker import DATASET_REVISION, OPENCODE_INSTRUCTIONS
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


LOG = logging.getLogger(__name__)
GYM_ROOT = Path(__file__).resolve().parents[2]
Task = Literal["vision2code", "codeedit", "vision_qa", "code_qa"]


class BenchCADModelTimeout(TimeoutError):
    """The single model invocation exceeded its configured time budget."""


class BenchCADConfig(OpenCodeSandboxedAgentConfig):
    """The CAD runtime is provisioned once by benchmark preparation."""

    resources_server: ResourcesServerRef | None = None
    inference_mode: Literal["opencode", "single_turn"] = "opencode"
    single_turn_max_output_tokens: int | None = Field(default=None, gt=0)
    single_turn_timeout: float = Field(default=3600, gt=0)
    benchmark_root: Path = GYM_ROOT / "benchmarks/benchcad/.cache/upstream"
    dataset_root: Path = GYM_ROOT / "benchmarks/benchcad/data"
    results_dir: Path = GYM_ROOT / "results/benchcad"
    execution_timeout: float = Field(default=300, gt=0)
    scoring_timeout: float = Field(default=900, gt=0)

    @field_validator("benchmark_root", "dataset_root", "results_dir")
    @classmethod
    def resolve_paths(cls, value: Path) -> Path:
        """Resolve asset paths independently of Gym's per-server working directory."""
        return (GYM_ROOT / value).resolve()

    @field_validator("artifacts_dir")
    @classmethod
    def resolve_artifacts(cls, value: str | None) -> str | None:
        """Keep OpenCode and verifier artifacts under the same configured root."""
        return str((GYM_ROOT / value).resolve()) if value is not None else None


class BenchCADRunRequest(BaseRunRequest):
    """A part is one trial; QA trials contain all questions about that part."""

    model_config = ConfigDict(extra="allow")

    task: Task
    record_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    family: str = ""


class BenchCADVerifyResponse(BaseVerifyResponse):
    """Native fractional score and diagnostics for either inference protocol."""

    model_config = ConfigDict(extra="allow")

    task: Task
    record_id: str
    family: str
    status: str
    protocol: str = "benchcad-opencode"
    upstream_revision: str = UPSTREAM_REVISION
    dataset_revision: str = DATASET_REVISION
    artifact_dir: str
    iou: float | None = None
    question_scores: list[float] = Field(default_factory=list)
    opencode_execution: dict[str, JsonValue] = Field(default_factory=dict)
    ng_agent_observations: AgentObservationBundle | None = Field(default=None, exclude_if=lambda value: value is None)
    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


def assistant_answer(response: NeMoGymResponse) -> str:
    """Grade the last assistant answer, excluding tool output and earlier attempts."""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "\n".join(part.text for part in item.content if part.type == "output_text")
    return ""


def task_directory(root: Path, body: BenchCADRunRequest) -> Path:
    """Keep task lookup within prepared assets, including when symlinks are present."""
    root = root.resolve()
    directory = (root / body.task / body.record_id).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("BenchCAD task path escapes dataset_root")
    return directory


def single_turn_params(
    params: NeMoGymResponseCreateParamsNonStreaming,
    *,
    directory: Path,
    image_names: list[str],
    max_output_tokens: int | None,
) -> NeMoGymResponseCreateParamsNonStreaming:
    """Reuse prepared prompts, replacing the OpenCode suffix with direct images."""
    values = params.model_dump(exclude_none=True)
    if params.previous_response_id is not None or params.conversation is not None:
        raise ValueError("Single-turn BenchCAD requires a standalone conversation")
    messages = values["input"]
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    suffix = "".join(f"\nUse the read tool to inspect /workspace/{name}." for name in image_names)
    suffix += OPENCODE_INSTRUCTIONS
    user_message = None
    for message in messages:
        if message.get("role") != "user":
            continue
        user_message = message
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "input_text", "text": content}]
            message["content"] = content
        for part in content:
            if part["type"] == "input_text":
                part["text"] = part["text"].removesuffix(suffix)
    if user_message is None:
        raise ValueError("Single-turn BenchCAD requires a user message")
    for name in image_names:
        image = (directory / name).resolve()
        if not image.is_relative_to(directory.resolve()) or Path(name).name != name:
            raise ValueError("Invalid BenchCAD image path")
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        user_message["content"].append(
            {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "auto"}
        )
    values.update(
        input=messages,
        tools=[],
        tool_choice="none",
        parallel_tool_calls=False,
        max_output_tokens=params.max_output_tokens or max_output_tokens,
    )
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(values)


class BenchCADAgent(OpenCodeSandboxedAgent):
    """Select direct or OpenCode inference and share isolated CAD verification.

    Agent and prediction execution use distinct disposable sandboxes. Only the trusted
    scorer can see the reference geometry; neither sandbox mounts the dataset.
    """

    ray_enabled = False
    config: BenchCADConfig

    async def _single_turn_response(
        self,
        request: Request,
        body: BenchCADRunRequest,
        *,
        directory: Path,
        image_names: list[str],
        artifacts: Path,
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None]:
        params = single_turn_params(
            body.responses_create_params,
            directory=directory,
            image_names=image_names,
            max_output_tokens=self.config.single_turn_max_output_tokens,
        )
        (artifacts / "model_input.json").write_text(params.model_dump_json())
        started = time()
        try:
            async with asyncio.timeout(self.config.single_turn_timeout):
                raw = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path=self.url_path_for_run("/v1/responses", body),
                    json=params,
                    cookies=request.cookies,
                )
                await raise_for_status(raw)
                response = NeMoGymResponse.model_validate(await get_response_json(raw))
        except TimeoutError as exc:
            raise BenchCADModelTimeout("Single-turn model request timed out") from exc
        if response.status == "failed" or response.error is not None:
            raise RuntimeError(f"BenchCAD model request failed: {response.error}")
        trajectory = None
        rollout_id = self.rollout_id_from_run(body)
        if self._model_call_capture_enabled() and rollout_id is not None:
            calls = [ModelCallRef(model_ref=self.config.model_server, response_id=response.id)]
            extra = body.model_extra or {}
            task_id = str(extra.get("task_id") or body.record_id)
            trajectory = TrajectoryRecord(
                task_id=task_id,
                rollout_id=rollout_id,
                invocations=[
                    AgentInvocation(
                        invocation_id=rollout_id,
                        status="incomplete" if response.status == "incomplete" else "completed",
                        model_calls=calls,
                        conversation=[*params.input, *response.output],
                    )
                ],
                turns=[
                    TrajectoryTurn(
                        invocation_id=rollout_id,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        turn_no=1,
                        timestamp=started,
                        question=params.input,
                        answer=[item for item in response.output if item.type != "reasoning"],
                        reasoning_content=[
                            item.model_dump(mode="json") for item in response.output if item.type == "reasoning"
                        ]
                        or None,
                        step_count=0,
                        model_calls=calls,
                    )
                ],
            )
        return response, trajectory

    async def _worker(self, *args: str) -> dict:
        root = self.config.benchmark_root.resolve()
        output = await run_worker(root / ".venv/bin/python", root, *args, timeout=self.config.scoring_timeout)
        return json.loads(output)

    async def _grade(self, directory: Path, artifacts: Path, answer: str) -> dict:
        answer_path = artifacts / "answer.txt"
        answer_path.write_text(answer)
        metadata = json.loads((directory / "task.json").read_text())
        score_args = ("score", "--task-dir", str(directory), "--answer", str(answer_path))
        if metadata["task"] in {"vision_qa", "code_qa"}:
            return await self._worker(*score_args)
        program = artifacts / "prediction.py"
        extracted = await self._worker("patch", "--answer", str(answer_path), "--output", str(program))
        if not extracted["has_code"]:
            return {"reward": 0.0, "status": "no_code"}
        sandbox = await self._start_sandbox()
        try:
            await sandbox.upload(program, "/workspace/prediction.py")
            result = await sandbox.exec(
                "python /workspace/prediction.py",
                timeout_s=self.config.execution_timeout,
            )
            (artifacts / "execution.stdout").write_text(result.stdout or "")
            (artifacts / "execution.stderr").write_text(result.stderr or "")
            if result.error_type == "timeout":
                return {"reward": 0.0, "status": "exec_timeout"}
            if result.error_type:
                raise RuntimeError(f"Prediction sandbox failed: {result.error_type}: {result.stderr}")
            if result.return_code != 0:
                return {"reward": 0.0, "status": "exec_fail"}
            exists = await sandbox.exec("test -s /workspace/prediction.step", timeout_s=30)
            if exists.error_type:
                raise RuntimeError(f"Prediction collection failed: {exists.error_type}: {exists.stderr}")
            if exists.return_code != 0:
                return {"reward": 0.0, "status": "missing_step"}
            await sandbox.download("/workspace/prediction.step", artifacts / "prediction.step")
        except TimeoutError:
            return {"reward": 0.0, "status": "exec_timeout"}
        finally:
            with CancelScope(shield=True):
                await sandbox.stop()
        return await self._worker(*score_args, "--step", str(artifacts / "prediction.step"))

    async def run(self, request: Request, body: BenchCADRunRequest) -> BenchCADVerifyResponse:
        """Run the selected inference protocol and return the native BenchCAD score."""
        async with self._sem:
            directory = task_directory(self.config.dataset_root, body)
            metadata = json.loads((directory / "task.json").read_text())
            if (metadata["task"], metadata["record_id"]) != (body.task, body.record_id):
                raise ValueError("BenchCAD task metadata does not match the requested part")
            artifacts = (self.config.results_dir / uuid4().hex).resolve()
            artifacts.mkdir(parents=True)
            (artifacts / "input.json").write_text(body.model_dump_json())
            key = request.session[SESSION_ID_KEY]
            request._cookies = request.cookies | {"sandbox_id": key}
            rollout_id = self.rollout_id_from_run(body)
            request.state._ng_observation_invocation_id = rollout_id
            response = NeMoGymResponse(
                id=uuid4().hex,
                created_at=0,
                model="benchcad",
                object="response",
                output=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
            )
            sandbox = None
            run_result = {}
            observations = None
            trajectory = None
            try:
                if self.config.inference_mode == "single_turn":
                    response, trajectory = await self._single_turn_response(
                        request, body, directory=directory, image_names=metadata["images"], artifacts=artifacts
                    )
                else:
                    sandbox = await self._start_sandbox()
                    self._sandbox_id_to_sandbox[key] = sandbox
                    for name in metadata["images"]:
                        image = (directory / name).resolve()
                        if not image.is_relative_to(directory) or Path(name).name != name:
                            raise ValueError("Invalid BenchCAD image path")
                        await sandbox.upload(image, f"/workspace/{name}")
                    response = await self.responses(request, body.responses_create_params)
                    run_result = self._sandbox_id_to_run_result.get(key, {}).copy()
                    observations = run_result.pop("_ng_agent_observations", None)
                    trajectory = run_result.pop("_ng_trajectory", None)
                    if trajectory is not None:
                        trajectory = scope_opencode_trajectory(trajectory, body, rollout_id)
                (artifacts / "response.json").write_text(response.model_dump_json())
                if self.config.inference_mode == "single_turn" and response.status == "incomplete":
                    reason = response.incomplete_details.reason if response.incomplete_details else None
                    status = "output_limit" if reason == "max_output_tokens" else "model_incomplete"
                    result = {"reward": 0.0, "status": status, "failure_kind": f"benchcad:{status}"}
                elif self.config.inference_mode == "single_turn" and any(
                    item.type == "function_call" for item in response.output
                ):
                    result = {"reward": 0.0, "status": "unexpected_tool_call"}
                elif run_result.get("opencode_failed"):
                    timed_out = (
                        run_result.get("opencode_error_type") == "timeout"
                        or run_result.get("opencode_exit_code") == 124
                    )
                    budget_exhausted = timed_out or response.status == "incomplete"
                    status = "agent_timeout" if timed_out else "context_limit" if budget_exhausted else "agent_error"
                    result = {
                        "reward": 0.0,
                        "status": status,
                        "mask_sample": not budget_exhausted,
                        "failure_kind": f"benchcad:{status}",
                    }
                else:
                    # No agent process survives into verification.
                    if sandbox is not None:
                        await sandbox.stop()
                        sandbox = None
                    result = await self._grade(directory, artifacts, assistant_answer(response))
            except BenchCADModelTimeout:
                result = {"reward": 0.0, "status": "model_timeout", "failure_kind": "benchcad:model_timeout"}
            except Exception as exc:
                LOG.exception("BenchCAD infrastructure failure for %s/%s", body.task, body.record_id)
                result = {
                    "reward": 0.0,
                    "status": "infrastructure_error",
                    "mask_sample": True,
                    "failure_kind": "benchcad:infrastructure_error",
                    "failure_reason": str(exc),
                }
            finally:
                with CancelScope(shield=True):
                    try:
                        if sandbox is not None:
                            await sandbox.stop()
                    except Exception:
                        LOG.exception("Failed to stop BenchCAD agent sandbox for %s", key)
                    finally:
                        self._sandbox_id_to_sandbox.pop(key, None)
                        self._sandbox_id_to_run_result.pop(key, None)
                        del request.state._ng_observation_invocation_id
            verified = BenchCADVerifyResponse(
                **(body.model_dump() | {"protocol": f"benchcad-{self.config.inference_mode.replace('_', '-')}"}),
                response=response,
                artifact_dir=str(artifacts),
                opencode_execution=run_result,
                ng_agent_observations=observations,
                ng_trajectory=trajectory,
                **result,
            )
            (artifacts / "result.json").write_text(verified.model_dump_json())
            return verified


if __name__ == "__main__":
    BenchCADAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BenchCADAgent.run_webserver()
