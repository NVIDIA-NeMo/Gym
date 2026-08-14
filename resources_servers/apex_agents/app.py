# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apex Agents held-out rubric and artifact verifier."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import FileResponse

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from resources_servers.apex_agents.artifacts import (
    safe_extract_snapshot,
    snapshot_changes,
)
from resources_servers.apex_agents.judge import grade_apex_output


_WORLD_ID_RE = re.compile(r"^world_[0-9a-f]{32}$")
_SESSION_WORLD_KEY = "apex_world_id"


class ApexResourcesServerConfig(BaseResourcesServerConfig):
    # The inline binary snapshot is consumed but intentionally omitted from the
    # rollout JSONL to avoid multiplying large office artifacts by every repeat.
    REVERIFY_MODE = ReverifyMode.UNSUPPORTED

    judge_model_server: ModelServerRef
    judge_model: str = "judge"
    judge_create_params_overrides: Dict[str, Any] = Field(default_factory=dict)
    judge_context_window_size: int = Field(default=32768, gt=0)
    num_processes: int = 8
    capture_judge_traces: bool = True
    max_snapshot_bytes: Optional[int] = Field(default=None, gt=0)
    max_uncompressed_bytes: int = 512 * 1024 * 1024
    max_artifact_files: int = 2000
    dataset_repo: str = "mercor/apex-agents"
    world_cache_dir: str = "benchmarks/apex_agents/data/world_cache"
    artifact_output_dir: Optional[str] = "results/apex_agents_artifacts"


class ApexSeedSessionRequest(BaseSeedSessionRequest):
    task_id: str
    world_id: str = Field(pattern=r"^world_[0-9a-f]{32}$")


class ApexSeedSessionResponse(BaseSeedSessionResponse):
    world_ready: bool = True


class ApexVerifierMetadata(BaseModel):
    rubric: List[Dict[str, Any]]
    expected_output: Optional[str] = None
    gold_response: Optional[str] = None
    gold_response_type: Optional[str] = None
    task_name: Optional[str] = None


class ApexVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    world_id: str
    domain: Optional[str] = None
    verifier_metadata: Dict[str, Any]
    initial_artifact_snapshot_b64: str = Field(default="", exclude=True)
    artifact_snapshot_b64: str = Field(default="", exclude=True)
    artifact_manifest: List[str] = Field(default_factory=list)
    apex_trajectory: List[Dict[str, Any]] = Field(default_factory=list)


class ApexVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    world_id: str
    rubric_scores: Dict[str, Any] = Field(default_factory=dict)
    judge_response: Optional[Dict[str, Any]] = None
    artifact_paths: List[str] = Field(default_factory=list)
    artifact_output_dir: Optional[str] = None
    initial_snapshot_path: Optional[str] = None
    final_snapshot_path: Optional[str] = None
    invalid_judge_response: bool = False
    verifier_error: Optional[str] = None


class ApexResourcesServer(SimpleResourcesServer):
    config: ApexResourcesServerConfig
    _semaphore: Any = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/world")(self.world)
        app.router.on_startup.append(self._preflight_world_cache)
        return app

    def _world_cache_dir(self) -> Path:
        path = Path(self.config.world_cache_dir).expanduser()
        if not path.is_absolute():
            path = PARENT_DIR / path
        return path.resolve()

    async def _preflight_world_cache(self) -> None:
        """Require preprocessing before this server accepts rollouts."""
        cache_dir = self._world_cache_dir()
        if not cache_dir.is_dir():
            raise RuntimeError(
                f"Apex world cache does not exist at {cache_dir}; run "
                "`gym eval prepare --benchmark apex_agents` before starting the environment"
            )

    async def seed_session(
        self,
        request: Request,
        body: ApexSeedSessionRequest,
    ) -> ApexSeedSessionResponse:
        try:
            async with self._semaphore:
                await asyncio.to_thread(self._download_world, body.world_id)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"world {body.world_id} is missing from the preprocessed Apex cache; run "
                    "`gym eval prepare --benchmark apex_agents`"
                ),
            ) from exc
        request.session[_SESSION_WORLD_KEY] = body.world_id
        return ApexSeedSessionResponse()

    async def world(self, request: Request) -> FileResponse:
        world_id = str(request.session.pop(_SESSION_WORLD_KEY, ""))
        if not _WORLD_ID_RE.fullmatch(world_id):
            raise HTTPException(status_code=404, detail="Call seed_session before requesting the world")
        async with self._semaphore:
            world_path = Path(await asyncio.to_thread(self._download_world, world_id))
        if world_path is None or not world_path.is_file():
            raise HTTPException(status_code=404, detail="Call seed_session before requesting the world")
        return FileResponse(world_path, media_type="application/zip", filename="world.zip")

    def _download_world(self, world_id: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self.config.dataset_repo,
            filename=f"world_files_zipped/{world_id}.zip",
            repo_type="dataset",
            cache_dir=str(self._world_cache_dir()),
            local_files_only=True,
        )

    def _failure(
        self,
        body: ApexVerifyRequest,
        error: str,
        *,
        artifact_paths: list[str] | None = None,
        persisted_output_dir: Path | None = None,
    ) -> ApexVerifyResponse:
        payload = body.model_dump(exclude={"initial_artifact_snapshot_b64", "artifact_snapshot_b64"})
        return ApexVerifyResponse(
            **payload,
            reward=0.0,
            artifact_paths=artifact_paths or [],
            artifact_output_dir=str(persisted_output_dir) if persisted_output_dir else None,
            initial_snapshot_path=(
                str(persisted_output_dir / "initial_snapshot.zip") if persisted_output_dir else None
            ),
            final_snapshot_path=(str(persisted_output_dir / "final_snapshot.zip") if persisted_output_dir else None),
            invalid_judge_response=True,
            verifier_error=error,
        )

    def _persist_submission(
        self,
        body: ApexVerifyRequest,
        initial_archive_path: Path,
        final_archive_path: Path,
        final_root: Path,
        changed_paths: list[str],
    ) -> Optional[Path]:
        """Persist a rubric-free copy of one submission outside the sandbox."""
        if not self.config.artifact_output_dir:
            return None
        root = Path(self.config.artifact_output_dir).expanduser()
        if not root.is_absolute():
            root = PARENT_DIR / root
        task_dir = root.resolve() / re.sub(r"[^A-Za-z0-9_-]", "_", body.task_id)[:128]
        payload = body.model_dump(
            exclude={"initial_artifact_snapshot_b64", "artifact_snapshot_b64", "verifier_metadata"}
        )
        rollout_index = payload.get("_ng_rollout_index", 0)
        attempt_index = payload.get("_ng_attempt_index", 0)
        run_name = f"rollout_{rollout_index}_attempt_{attempt_index}_{uuid.uuid4().hex[:8]}"
        output_dir = task_dir / run_name
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)
        shutil.copy2(initial_archive_path, output_dir / "initial_snapshot.zip")
        shutil.copy2(final_archive_path, output_dir / "final_snapshot.zip")
        for relative_name in changed_paths:
            source = final_root / relative_name
            if not source.is_file():
                continue
            relative = source.relative_to(final_root)
            destination = artifacts_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (output_dir / "submission.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return output_dir

    @staticmethod
    def _persist_grading_result(
        output_dir: Path | None,
        *,
        reward: float | None = None,
        rubric_scores: dict[str, Any] | None = None,
        judge_response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if output_dir is None:
            return
        payload: dict[str, Any] = {
            "reward": reward,
            "rubric_scores": rubric_scores or {},
            "judge_response": judge_response,
        }
        if error:
            payload["error"] = error
        (output_dir / "grading.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    async def verify(self, body: ApexVerifyRequest) -> ApexVerifyResponse:
        try:
            metadata = ApexVerifierMetadata.model_validate(body.verifier_metadata)
        except Exception as exc:
            return self._failure(body, f"invalid verifier metadata: {exc}")
        if not metadata.rubric:
            return self._failure(body, "rubric is empty")

        snapshots: dict[str, bytes] = {}
        for name, encoded in {
            "initial": body.initial_artifact_snapshot_b64 or "",
            "final": body.artifact_snapshot_b64 or "",
        }.items():
            if not encoded:
                return self._failure(body, f"{name} artifact snapshot is missing")
            if (
                self.config.max_snapshot_bytes is not None
                and len(encoded) > ((self.config.max_snapshot_bytes + 2) // 3) * 4 + 4
            ):
                return self._failure(body, f"{name} artifact snapshot exceeds configured size limit")
            try:
                snapshot = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                return self._failure(body, f"invalid {name} artifact snapshot encoding: {exc}")
            if self.config.max_snapshot_bytes is not None and len(snapshot) > self.config.max_snapshot_bytes:
                return self._failure(body, f"{name} artifact snapshot exceeds configured size limit")
            snapshots[name] = snapshot

        task_prompt = ""
        persisted_output_dir: Optional[Path] = None
        artifact_paths: list[str] = []
        params_input = body.responses_create_params.input
        if isinstance(params_input, str):
            task_prompt = params_input
        else:
            for item in params_input:
                payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
                if payload.get("role") == "user" and isinstance(payload.get("content"), str):
                    task_prompt += ("\n\n" if task_prompt else "") + payload["content"]

        async with self._semaphore:
            try:
                with tempfile.TemporaryDirectory(prefix="apex-verify-") as scratch:
                    root = Path(scratch)
                    initial_archive_path = root / "initial.zip"
                    final_archive_path = root / "final.zip"
                    initial_root = root / "initial"
                    final_root = root / "final"
                    initial_root.mkdir()
                    final_root.mkdir()
                    initial_archive_path.write_bytes(snapshots["initial"])
                    final_archive_path.write_bytes(snapshots["final"])
                    initial_files = safe_extract_snapshot(
                        initial_archive_path,
                        initial_root,
                        max_files=self.config.max_artifact_files,
                        max_uncompressed_bytes=self.config.max_uncompressed_bytes,
                    )
                    final_files = safe_extract_snapshot(
                        final_archive_path,
                        final_root,
                        max_files=self.config.max_artifact_files,
                        max_uncompressed_bytes=self.config.max_uncompressed_bytes,
                    )
                    changes = snapshot_changes(initial_root, initial_files, final_root, final_files)
                    artifact_paths = [change.path for change in changes]
                    persisted_output_dir = self._persist_submission(
                        body,
                        initial_archive_path,
                        final_archive_path,
                        final_root,
                        artifact_paths,
                    )
                    reward, rubric_scores, judge = await grade_apex_output(
                        server_client=self.server_client,
                        model_server_name=self.config.judge_model_server.name,
                        task_id=body.task_id,
                        world_id=body.world_id,
                        instruction=task_prompt,
                        response=body.response.output_text,
                        rubric=metadata.rubric,
                        expected_output=metadata.expected_output,
                        artifact_changes=changes,
                        final_root=final_root,
                        judge_model=self.config.judge_model,
                        judge_create_params_overrides=self.config.judge_create_params_overrides,
                        judge_context_window_size=self.config.judge_context_window_size,
                        capture_judge_traces=self.config.capture_judge_traces,
                        metadata={
                            "task_id": body.task_id,
                            "execution": {
                                "rollout_slot_id": str((body.__pydantic_extra__ or {}).get("_ng_rollout_index", ""))
                            },
                        },
                    )
                    self._persist_grading_result(
                        persisted_output_dir,
                        reward=reward,
                        rubric_scores=rubric_scores,
                        judge_response=judge,
                    )
            except Exception as exc:
                self._persist_grading_result(persisted_output_dir, error=str(exc))
                return self._failure(
                    body,
                    str(exc),
                    artifact_paths=artifact_paths,
                    persisted_output_dir=persisted_output_dir,
                )

        payload = body.model_dump(exclude={"initial_artifact_snapshot_b64", "artifact_snapshot_b64"})
        return ApexVerifyResponse(
            **payload,
            reward=reward,
            rubric_scores=rubric_scores,
            judge_response=judge,
            artifact_paths=artifact_paths,
            artifact_output_dir=str(persisted_output_dir) if persisted_output_dir else None,
            initial_snapshot_path=(
                str(persisted_output_dir / "initial_snapshot.zip") if persisted_output_dir else None
            ),
            final_snapshot_path=(str(persisted_output_dir / "final_snapshot.zip") if persisted_output_dir else None),
            invalid_judge_response=False,
            verifier_error=None,
        )


if __name__ == "__main__":
    ApexResourcesServer.run_webserver()
