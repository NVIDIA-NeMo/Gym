# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orthographic CAD reconstruction environment for NeMo Gym."""

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec
from nemo_gym.server_runtime.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from resources_servers.deepcad.mixed_iou_config import MixedIoUConfig
from resources_servers.deepcad.operations import OPERATION_NAMES, operation_bonus
from resources_servers.deepcad.task_data import TaskData


LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


class RewardWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Historical weight key: volume IoU from iou_backend.
    voxel_iou: float = Field(default=1.0, ge=0, allow_inf_nan=False)


class OperationBonusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    max_fraction: float = Field(default=0.10, ge=0, le=0.2, allow_inf_nan=False)


class DeepCADResourcesServerConfig(BaseResourcesServerConfig):
    operation_bonus: OperationBonusConfig = Field(default_factory=OperationBonusConfig)
    verification_metric: Literal["geometry", "mixed_iou"] = "geometry"
    mixed_iou: MixedIoUConfig | None = None
    reference_root: str = "cad-data/references"
    sandbox_provider: str | dict[str, Any] = "sandbox"
    sandbox_image: str = "deepcad-cadquery:2.8.0"
    sandbox_python: str = "python"
    preload_cq: bool = False  # Supply cq to generated code for legacy-reward experiments.
    max_concurrency: int = Field(default=4, ge=1)
    execution_timeout_s: int = Field(default=60, ge=1)
    geometry_timeout_s: int = Field(default=120, ge=1)
    geometry_library_dirs: list[str] = Field(default_factory=list)
    max_code_bytes: int = Field(default=65536, ge=1)
    max_step_bytes: int = Field(default=32 * 1024 * 1024, ge=1)
    iou_backend: Literal["brep", "voxel"] = "brep"
    alignment: Literal["rescale"] = "rescale"
    voxel_resolution: int = Field(default=64, ge=8, le=256)
    reference_cache_root: str | None = None
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)

    @model_validator(mode="after")
    def validate_metric(self) -> "DeepCADResourcesServerConfig":
        if self.verification_metric == "mixed_iou":
            if self.mixed_iou is None:
                raise ValueError("verification_metric=mixed_iou requires mixed_iou configuration")
            self.mixed_iou.validate_resolution(self.voxel_resolution)
            if self.alignment != "rescale" or self.iou_backend != "brep":
                raise ValueError("Mixed IoU requires rescale alignment and B-rep backend")
            if any(self.reward_weights.model_dump().values()):
                raise ValueError("Mixed IoU replaces geometry reward_weights; set them to zero")
        return self


class DeepCADVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class DeepCADVerifyResponse(BaseMultiRewardVerifyResponse):
    executed_operations: list[str] = Field(default_factory=list)
    matched_operations: list[str] = Field(default_factory=list)
    operation_bonus: float = 0.0
    verification_metric: Literal["geometry", "mixed_iou"] = "geometry"
    brep_iou: float | None = None
    octree_iou: float | None = None
    boundary_iou: float | None = None
    mixed_iou: float | None = None
    reference_time_s: float | None = None
    iou_backend: Literal["brep", "voxel"] | None = None
    alignment: Literal["rescale"] = "rescale"
    raw_metrics: dict[str, float | None] = Field(default_factory=dict)
    volume_iou: float = 0.0
    intersection_volume: float | None = None
    union_volume: float | None = None
    prediction_volume: float | None = None
    reference_volume: float | None = None
    scoring_method: str | None = None
    voxel_resolution: int | None = None
    intersection_voxels: int | None = None
    union_voxels: int | None = None
    prediction_voxels: int | None = None
    reference_voxels: int | None = None
    voxel_iou: float = 0.0
    reference_cache_hit: bool | None = None
    scorer_version: str | None = None
    scoring_time_s: float | None = None


def extract_code(text: str) -> str:
    """Accept Python fences or raw code; ignore model reasoning blocks."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    text = re.split(r"</think(?:ing)?>", text, flags=re.I)[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    return (blocks[-1] if blocks else text).strip()


def assistant_text(body: BaseVerifyRequest) -> str:
    messages = [item for item in body.response.output if item.type == "message" and item.role == "assistant"]
    if not messages:
        return ""
    return "\n".join(part.text for part in messages[-1].content if part.type == "output_text")


class DeepCADResourcesServer(SimpleResourcesServer):
    config: DeepCADResourcesServerConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    def _geometry_worker_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.geometry_library_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [*self.config.geometry_library_dirs, *filter(None, [env.get("LD_LIBRARY_PATH")])]
            )
        env["PYTHONPATH"] = str(HERE.parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def setup_webserver(self) -> FastAPI:
        # Validate deployment-specific native libraries before admitting rollouts.
        if self.config.geometry_library_dirs:
            for directory in self.config.geometry_library_dirs:
                if not directory or not Path(directory).is_dir():
                    raise RuntimeError(f"CAD native library directory is missing: {directory!r}")
            worker = "mixed_iou" if self.config.verification_metric == "mixed_iou" else "geometry"
            probe = subprocess.run(
                [sys.executable, "-c", f"import resources_servers.deepcad.{worker}"],
                env=self._geometry_worker_env(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if probe.returncode:
                raise RuntimeError(f"CAD geometry runtime preflight failed: {probe.stderr[-2000:]}")
        return super().setup_webserver()

    async def verify(self, body: DeepCADVerifyRequest) -> DeepCADVerifyResponse:
        """Execute code in isolation, then score the exported solid in a trusted worker."""
        result: dict[str, Any]
        code = self.code_from_response(body)
        if not code or len(code.encode()) > self.config.max_code_bytes:
            result = {"reward": 0.0, "failure_kind": "deepcad:code", "failure_reason": "Empty or oversized code"}
        else:
            try:
                compile(code, "model.py", "exec")
            except SyntaxError as exc:
                result = {"reward": 0.0, "failure_kind": "deepcad:syntax", "failure_reason": str(exc)}
            else:
                async with self._semaphore:
                    result = await self._evaluate(
                        code, body.verifier_metadata.reference_step, view_mode=body.verifier_metadata.view_mode
                    )
        result["verification_metric"] = self.config.verification_metric
        if self.config.verification_metric == "mixed_iou":
            # Stable weighted columns on failures as well as successful rollouts.
            result.setdefault(
                "reward_components", {name: 0.0 for name, weight in self.config.mixed_iou.weights.model_dump().items()}
            )
        if "reward_components" not in result:
            result["voxel_iou"] = result["reward"]
            result["reward_components"] = {
                "voxel_iou": result["reward"],
            }
        result.setdefault("iou_backend", self.config.iou_backend)
        result.setdefault("alignment", self.config.alignment)
        result.setdefault("volume_iou", result.get("voxel_iou", 0.0))
        # Raw telemetry is independent of reward weights and never enters advantages.
        result["raw_metrics"] = {
            "volume_iou": float(result["volume_iou"]),
            "voxel_iou": float(result.get("voxel_iou", 0.0)),
        }
        if self.config.verification_metric == "mixed_iou":
            for name in (
                "brep_iou",
                "octree_iou",
                "boundary_iou",
                "mixed_iou",
                "reference_time_s",
                "prediction_prepare_time_s",
                "brep_time_s",
                "octree_time_s",
                "boundary_time_s",
                "scoring_time_s",
                "reference_cache_hit",
            ):
                value = result.get(name)
                result["raw_metrics"][name] = float(value) if value is not None else None
        if self.config.operation_bonus.enabled:
            settings = self.config.operation_bonus
            bonus, matched = operation_bonus(
                float(result["reward"]),
                body.verifier_metadata.reference_operations,
                result.get("executed_operations", []),
                max_fraction=settings.max_fraction,
            )
            if result.get("failure_kind") or result.get("mask_sample"):
                bonus, matched = 0.0, []
            result["operation_bonus"] = bonus
            result["matched_operations"] = matched
            result["reward_components"]["operation_bonus"] = bonus
            result["reward"] += bonus
            reference_count = len(set(body.verifier_metadata.reference_operations) & OPERATION_NAMES)
            result["raw_metrics"].update(
                operation_bonus=bonus,
                operation_matched_count=float(len(matched)),
                operation_reference_count=float(reference_count),
                operation_recall=len(matched) / reference_count if reference_count else 0.0,
            )
        return DeepCADVerifyResponse(**body.model_dump(), **result)

    def code_from_response(self, body: DeepCADVerifyRequest) -> str:
        return extract_code(assistant_text(body))

    async def _evaluate(self, code: str, reference_step: str, *, view_mode: str = "axis") -> dict[str, Any]:
        root = Path(self.config.reference_root).expanduser().resolve()
        reference = (root / reference_step).resolve()
        if not reference.is_relative_to(root) or not reference.is_file():
            return {
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "deepcad:reference",
                "failure_reason": "Reference STEP missing or outside reference_root",
            }
        try:
            provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
            spec = SandboxSpec(
                image=self.config.sandbox_image,
                workdir="/sandbox",
                env={
                    # Enroot can inherit the launcher's environment, including serialized Gym config.
                    **{key: "" for key in os.environ if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)},
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "ENROOT_MOUNT_HOME": "no",
                },
                resources=SandboxResources(cpu=1, memory_mib=2048),
                files={
                    "/sandbox/model.py": code,
                    "/sandbox/runner.py": (HERE / "runner.py").read_text(),
                    **(
                        {"/sandbox/operations.py": (HERE / "operations.py").read_text()}
                        if self.config.operation_bonus.enabled
                        else {}
                    ),
                },
            )
            with tempfile.TemporaryDirectory(prefix="deepcad-verify-") as directory:
                prediction = Path(directory) / "prediction.step"
                async with AsyncSandbox(provider, spec) as sandbox:
                    await sandbox.start()
                    python = shlex.quote(self.config.sandbox_python)
                    probe = await sandbox.exec(f"{python} -c 'import cadquery'", timeout_s=30)
                    if probe.return_code or probe.error_type:
                        raise RuntimeError("Sandbox image cannot import CadQuery")
                    trace_flag = " --trace-operations" if self.config.operation_bonus.enabled else ""
                    preload_flag = " --preload-cq" if self.config.preload_cq else ""
                    execution = await sandbox.exec(
                        f"{python} /sandbox/runner.py{trace_flag}{preload_flag} > /sandbox/run.log 2>&1",
                        timeout_s=self.config.execution_timeout_s,
                    )
                    if execution.error_type and execution.error_type != "timeout":
                        raise RuntimeError(f"Sandbox execution failed: {execution.error_type}")
                    if execution.return_code or execution.error_type:
                        return {
                            "reward": 0.0,
                            "failure_kind": "deepcad:execution",
                            "failure_reason": "Code timed out or did not produce valid solid geometry",
                        }
                    check = await sandbox.exec(
                        "test -f /sandbox/prediction.step && wc -c < /sandbox/prediction.step", timeout_s=10
                    )
                    try:
                        size = int((check.stdout or "").strip())
                    except ValueError:
                        size = 0
                    if check.return_code or not 0 < size <= self.config.max_step_bytes:
                        return {
                            "reward": 0.0,
                            "failure_kind": "deepcad:artifact",
                            "failure_reason": "Missing or oversized STEP artifact",
                        }
                    await sandbox.download("/sandbox/prediction.step", prediction)
                    executed_operations = []
                    if self.config.operation_bonus.enabled:
                        trace_result = await sandbox.exec("head -c 16384 /sandbox/operations.json", timeout_s=10)
                        if trace_result.return_code or trace_result.error_type:
                            raise RuntimeError("Missing sandbox operation trace")
                        executed_operations = json.loads(trace_result.stdout)
                        if not isinstance(executed_operations, list) or any(
                            not isinstance(name, str) or name not in OPERATION_NAMES for name in executed_operations
                        ):
                            raise ValueError("Malformed sandbox operation trace")
                scored = await self._score(prediction, reference, view_mode=view_mode, artifact_dir=None)
                if self.config.operation_bonus.enabled:
                    scored["executed_operations"] = executed_operations
                return scored
        except Exception as exc:
            LOGGER.exception("DeepCAD infrastructure failure")
            return {
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "deepcad:infrastructure",
                "failure_reason": str(exc),
            }

    async def _score(
        self, prediction: Path, reference: Path, *, view_mode: str = "axis", artifact_dir: Path | None = None
    ) -> dict[str, Any]:
        return await self._score_geometry(prediction, reference, view_mode=view_mode, artifact_dir=artifact_dir)

    async def _score_geometry(
        self, prediction: Path, reference: Path, *, view_mode: str = "axis", artifact_dir: Path | None = None
    ) -> dict[str, Any]:
        options = {
            "iou_backend": self.config.iou_backend,
            "alignment": self.config.alignment,
            "view_mode": view_mode,
            "cache_root": self.config.reference_cache_root,
            "reward_weights": self.config.reward_weights.model_dump(),
        }
        worker = "geometry.py"
        if self.config.verification_metric == "mixed_iou":
            worker = "mixed_iou.py"
            options["mixed_iou"] = self.config.mixed_iou.model_dump()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(HERE / worker),
            str(prediction),
            str(reference),
            str(self.config.voxel_resolution),
            json.dumps(options),
            env=self._geometry_worker_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), self.config.geometry_timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"Geometry worker failed: {stderr.decode(errors='replace')[-2000:]}")
        return json.loads(stdout.decode(errors="replace"))


if __name__ == "__main__":
    DeepCADResourcesServer.run_webserver()
