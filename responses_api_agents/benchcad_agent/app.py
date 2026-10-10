# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BenchCAD with Gym's OpenCode harness and the pinned upstream CAD scorer."""

import json
import logging
from pathlib import Path
from typing import Literal
from uuid import uuid4

from anyio import CancelScope
from fastapi import Request
from pydantic import ConfigDict, Field, JsonValue, field_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from responses_api_agents.benchcad_agent.runtime import UPSTREAM_REVISION, run_worker
from responses_api_agents.benchcad_agent.worker import DATASET_REVISION
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


LOG = logging.getLogger(__name__)
GYM_ROOT = Path(__file__).resolve().parents[2]
Task = Literal["vision2code", "codeedit", "vision_qa", "code_qa"]


class BenchCADConfig(OpenCodeSandboxedAgentConfig):
    """The CAD runtime is provisioned once by benchmark preparation."""

    resources_server: ResourcesServerRef | None = None
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
    """Native fractional score and diagnostics for an OpenCode trial."""

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


class BenchCADAgent(OpenCodeSandboxedAgent):
    """Reuse OpenCode's generation/trajectory implementation and own CAD verification.

    Agent and prediction execution use distinct disposable sandboxes. Only the trusted
    scorer can see the reference geometry; neither sandbox mounts the dataset.
    """

    ray_enabled = False
    config: BenchCADConfig

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
        """Run one isolated OpenCode episode and return the native BenchCAD score."""
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
                if run_result.get("opencode_failed"):
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
                    await sandbox.stop()
                    sandbox = None
                    result = await self._grade(directory, artifacts, assistant_answer(response))
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
                **body.model_dump(),
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
