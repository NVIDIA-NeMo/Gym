# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apex Agents held-out rubric and artifact verifier."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextvars
import functools
import json
import re
import shutil
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import FileResponse

from nemo_gym import PARENT_DIR
from nemo_gym.config import ModelServerRef
from nemo_gym.core import VERIFIER_ERROR
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import VerifierFixture, is_judge_failure, is_judge_provider_stop
from resources_servers.apex_agents.artifacts import (
    ArtifactChange,
    SnapshotLimitError,
    safe_extract_snapshot,
    snapshot_changes,
)
from resources_servers.apex_agents.file_extraction import (
    converter_setup_problem,
    resolve_converter_image,
    start_conversion_record,
)
from resources_servers.apex_agents.judge import grade_apex_output
from resources_servers.apex_agents.verifier_fixture import create_fixture_server, invoke_fixture, reseed_fixture


_WORLD_ID_RE = re.compile(r"^world_[0-9a-f]{32}$")
_TASK_ID_RE = re.compile(r"^task[-_][A-Za-z0-9_-]+$")
_TASK_INPUT_FILES_RE = re.compile(r"^snap_[0-9a-f]{32}$")
_SESSION_WORLD_KEY = "apex_world_id"
_SESSION_TASK_KEY = "apex_task_id"
_SESSION_TASK_INPUT_FILES_KEY = "apex_task_input_files"


class ApexResourcesServerConfig(BaseResourcesServerConfig):
    # The inline binary snapshot is consumed but intentionally omitted from the
    # rollout JSONL to avoid multiplying large office artifacts by every repeat.
    REVERIFY_MODE = ReverifyMode.UNSUPPORTED

    judge_model_server: ModelServerRef
    judge_model: str = "gemini-3-flash"
    judge_create_params_overrides: Dict[str, Any] = Field(default_factory=lambda: {"reasoning_effort": "low"})
    judge_context_window_size: int = Field(default=1_000_000, gt=0)
    num_processes: int = 8
    max_snapshot_bytes: Optional[int] = Field(default=None, gt=0)
    max_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024
    max_artifact_files: int = 2000
    dataset_repo: str = "mercor/apex-agents"
    world_cache_dir: str = "benchmarks/apex_agents/data/world_cache"
    task_files_cache_dir: str = "benchmarks/apex_agents/data/task_files_cache"
    artifact_output_dir: Optional[str] = "results/apex_agents_artifacts"
    # LibreOffice and Poppler open files the model wrote, so they run only inside this Apptainer image, with
    # nothing of the host visible but one scratch directory. Unset, documents are graded on extracted text alone.
    document_converter_image: Optional[str] = None
    # Opt in to the copies of those tools installed on this host (preferred over the image when present). Off by
    # default: a host tool runs model-written documents outside any sandbox.
    allow_host_document_tools: bool = False


class ApexSeedSessionRequest(BaseSeedSessionRequest):
    task_id: str = Field(pattern=_TASK_ID_RE.pattern)
    world_id: str = Field(pattern=r"^world_[0-9a-f]{32}$")
    task_input_files: Optional[str] = Field(default=None, pattern=_TASK_INPUT_FILES_RE.pattern)


class ApexSeedSessionResponse(BaseSeedSessionResponse):
    world_ready: bool = True


class ApexVerifierMetadata(BaseModel):
    rubric: List[Dict[str, Any]]
    # all_pass (APEX-Agents): reward 1 only when every criterion passes. weighted_average: the weighted mean of
    # criterion scores, using each rubric item's optional `weight` (default 1).
    scoring: Literal["all_pass", "weighted_average"] = "all_pass"
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


class ApexFixtureRequest(ApexVerifyRequest):
    """Synthetic deliverables and fixed judge replies, provided only by the verifier fixture runner."""

    # Files of the final snapshot by archive path; None leaves the final snapshot out.
    fixture_files: Optional[Dict[str, str]] = Field(default=None, exclude=True)
    # Raw judge reply for each criterion, keyed by the criterion text.
    fixture_judge_replies: Dict[str, str] = Field(default_factory=dict, exclude=True)


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


class _PolicyOutputRejected(Exception):
    """The final snapshot, which the policy's work produced, breaks a configured limit: a scored 0."""


@dataclass(frozen=True)
class _PreparedSubmission:
    """What the blocking half of a verify produced: the extracted final snapshot and its changes."""

    final_root: Path
    changes: list[ArtifactChange]
    artifact_paths: list[str]
    persisted_output_dir: Optional[Path]


class _VerifyWorkspace:
    """One verify's scratch directory and the worker-thread work that reads it.

    Blocking work runs on the server's bounded pool through ``run``. The directory is removed by ``release``,
    which first waits for every piece of that work to return: a verify that fails or is cancelled while a
    conversion is still running does not pull the files out from under it.
    """

    _releases: set[asyncio.Future] = set()

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="apex-verify-"))
        self._executor = executor
        self._pending: set[asyncio.Future] = set()

    async def run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        call = functools.partial(function, *args, **kwargs)
        # The worker runs with a copy of this task's context, so what it records reaches this verify.
        work = asyncio.get_running_loop().run_in_executor(self._executor, contextvars.copy_context().run, call)
        self._pending.add(work)
        work.add_done_callback(self._settled)
        # Shielded: cancelling the verify cannot stop a thread, so its work stays tracked until it returns.
        return await asyncio.shield(work)

    def _settled(self, work: asyncio.Future) -> None:
        self._pending.discard(work)
        if not work.cancelled():
            work.exception()

    def release(self) -> asyncio.Future:
        """Remove the scratch directory once no worker is using it; the returned future settles when it is gone."""

        async def remove() -> None:
            while self._pending:
                await asyncio.wait(set(self._pending))
            await asyncio.get_running_loop().run_in_executor(
                self._executor, functools.partial(shutil.rmtree, self.root, ignore_errors=True)
            )

        release = asyncio.ensure_future(remove())
        # Held until done, so a cancelled verify does not leave the removal unreferenced.
        self._releases.add(release)
        release.add_done_callback(self._releases.discard)
        return release


def _task_instruction(params_input: Any) -> str:
    """The task instruction the agent was given: the user's text, from string content and from text blocks."""
    if isinstance(params_input, str):
        return params_input
    parts: list[str] = []
    for item in params_input:
        payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if payload.get("role") != "user":
            continue
        content = payload.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            block = block.model_dump() if hasattr(block, "model_dump") else block
            if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
                parts.append(str(block.get("text") or ""))
    return "\n\n".join(part for part in parts if part).strip()


class ApexResourcesServer(SimpleResourcesServer):
    config: ApexResourcesServerConfig
    _semaphore: Any = None
    _executor: Any = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.num_processes)
        # Extraction, document conversion and persistence block; they run here, never on the event loop.
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.config.num_processes), thread_name_prefix="apex-verify"
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/world")(self.world)
        app.get("/task_files")(self.task_files)
        app.router.on_startup.append(self._preflight_world_cache)
        app.router.on_startup.append(self._preflight_document_converter)
        return app

    def _world_cache_dir(self) -> Path:
        path = Path(self.config.world_cache_dir).expanduser()
        if not path.is_absolute():
            path = PARENT_DIR / path
        return path.resolve()

    def _task_files_cache_dir(self) -> Path:
        path = Path(self.config.task_files_cache_dir).expanduser()
        if not path.is_absolute():
            path = PARENT_DIR / path
        return path.resolve()

    def _document_converter_image(self) -> str | None:
        return resolve_converter_image(self.config.document_converter_image, PARENT_DIR)

    async def _preflight_document_converter(self) -> None:
        """Refuse to start with a converter that cannot run, rather than mask every document task at verify time."""
        problem = converter_setup_problem(
            self._document_converter_image(), allow_host_document_tools=self.config.allow_host_document_tools
        )
        if problem:
            raise RuntimeError(
                f"Apex document converter is not usable: {problem}. Either point `document_converter_image` at an "
                "image this host can run with apptainer, or set `allow_host_document_tools: true` to use "
                "LibreOffice and Poppler installed on this host, or set `document_converter_image: null` to grade "
                "Office and PDF files from their extracted text only"
            )

    async def _preflight_world_cache(self) -> None:
        """Require preprocessing before this server accepts rollouts."""
        cache_dir = self._world_cache_dir()
        if not cache_dir.is_dir():
            raise RuntimeError(
                f"Apex world cache does not exist at {cache_dir}; run "
                "`gym eval prepare --benchmark apex_agents` before starting the environment"
            )
        task_files_cache_dir = self._task_files_cache_dir()
        if not task_files_cache_dir.is_dir():
            raise RuntimeError(
                f"Apex task-files cache does not exist at {task_files_cache_dir}; run "
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
                if body.task_input_files:
                    await asyncio.to_thread(
                        self._task_files_archive,
                        body.task_id,
                        body.task_input_files,
                    )
        except Exception as exc:
            input_name = "world and task attachments" if body.task_input_files else "world"
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{input_name} for {body.task_id} are missing from the preprocessed Apex cache; run "
                    "`gym eval prepare --benchmark apex_agents`"
                ),
            ) from exc
        request.session[_SESSION_WORLD_KEY] = body.world_id
        request.session[_SESSION_TASK_KEY] = body.task_id
        if body.task_input_files:
            request.session[_SESSION_TASK_INPUT_FILES_KEY] = body.task_input_files
        else:
            request.session.pop(_SESSION_TASK_INPUT_FILES_KEY, None)
        return ApexSeedSessionResponse()

    async def world(self, request: Request) -> FileResponse:
        has_task_files = bool(request.session.get(_SESSION_TASK_INPUT_FILES_KEY))
        if has_task_files:
            world_id = str(request.session.get(_SESSION_WORLD_KEY, ""))
        else:
            world_id = str(request.session.pop(_SESSION_WORLD_KEY, ""))
            request.session.pop(_SESSION_TASK_KEY, None)
        if not _WORLD_ID_RE.fullmatch(world_id):
            raise HTTPException(status_code=404, detail="Call seed_session before requesting the world")
        async with self._semaphore:
            world_path = Path(await asyncio.to_thread(self._download_world, world_id))
        if world_path is None or not world_path.is_file():
            raise HTTPException(status_code=404, detail="Call seed_session before requesting the world")
        return FileResponse(world_path, media_type="application/zip", filename="world.zip")

    async def task_files(self, request: Request) -> FileResponse:
        task_id = str(request.session.pop(_SESSION_TASK_KEY, ""))
        task_input_files = str(request.session.pop(_SESSION_TASK_INPUT_FILES_KEY, ""))
        request.session.pop(_SESSION_WORLD_KEY, None)
        if not _TASK_ID_RE.fullmatch(task_id) or not _TASK_INPUT_FILES_RE.fullmatch(task_input_files):
            raise HTTPException(status_code=404, detail="Call seed_session before requesting task files")
        async with self._semaphore:
            archive_path = Path(await asyncio.to_thread(self._task_files_archive, task_id, task_input_files))
        if not archive_path.is_file():
            raise HTTPException(status_code=404, detail="Call seed_session before requesting task files")
        return FileResponse(archive_path, media_type="application/zip", filename="task_files.zip")

    def _download_world(self, world_id: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self.config.dataset_repo,
            filename=f"world_files_zipped/{world_id}.zip",
            repo_type="dataset",
            cache_dir=str(self._world_cache_dir()),
            local_files_only=True,
        )

    def _task_files_archive(self, task_id: str, task_input_files: str) -> str:
        """Build a small filesystem overlay from prefetched task-specific files."""
        from huggingface_hub import snapshot_download

        if not _TASK_ID_RE.fullmatch(task_id) or not _TASK_INPUT_FILES_RE.fullmatch(task_input_files):
            raise ValueError("invalid Apex task attachment identity")
        cache_dir = self._task_files_cache_dir()
        overlay_dir = cache_dir / "task_overlays"
        archive_path = overlay_dir / f"{task_id}-{task_input_files}.zip"
        if archive_path.is_file():
            return str(archive_path)
        snapshot_root = Path(
            snapshot_download(
                repo_id=self.config.dataset_repo,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                allow_patterns=[f"task_files/{task_id}/**"],
                local_files_only=True,
            )
        )
        task_root = snapshot_root / "task_files" / task_id
        files = sorted(path for path in task_root.rglob("*") if path.is_file()) if task_root.is_dir() else []
        if not files:
            raise FileNotFoundError(f"no cached Apex task files found for {task_id}")

        overlay_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=overlay_dir, suffix=".zip", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in files:
                    relative = path.relative_to(task_root)
                    archive_name = relative.as_posix()
                    if relative.parts[0] not in {"filesystem", ".apps_data"}:
                        archive_name = f"filesystem/{archive_name}"
                    archive.write(path, archive_name)
            temporary_path.replace(archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return str(archive_path)

    def _failure(
        self,
        body: ApexVerifyRequest,
        error: str,
        *,
        artifact_paths: list[str] | None = None,
        persisted_output_dir: Path | None = None,
        policy_caused: bool = False,
    ) -> ApexVerifyResponse:
        """An ungraded verify. Masked, because the verifier, the dataset or the transfer failed, not the policy;
        unless ``policy_caused``: the final snapshot broke a limit, which is the policy's own result and scores 0."""
        payload = body.model_dump(exclude={"initial_artifact_snapshot_b64", "artifact_snapshot_b64"})
        if not policy_caused:
            payload |= {
                "mask_sample": True,
                "evaluation_completed": False,
                "failure_kind": VERIFIER_ERROR,
                "failure_reason": error,
            }
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

    def _prepare_submission(
        self, body: ApexVerifyRequest, snapshots: dict[str, bytes], root: Path
    ) -> _PreparedSubmission:
        """The blocking half of a verify: unpack both snapshots, diff them and persist the submission."""
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
        try:
            final_files = safe_extract_snapshot(
                final_archive_path,
                final_root,
                max_files=self.config.max_artifact_files,
                max_uncompressed_bytes=self.config.max_uncompressed_bytes,
            )
        except SnapshotLimitError as exc:
            raise _PolicyOutputRejected(f"final artifact snapshot: {exc}") from exc
        changes = snapshot_changes(
            initial_root,
            initial_files,
            final_root,
            final_files,
            document_converter_image=self._document_converter_image(),
            allow_host_document_tools=self.config.allow_host_document_tools,
        )
        artifact_paths = [change.path for change in changes]
        persisted_output_dir = self._persist_submission(
            body,
            initial_archive_path,
            final_archive_path,
            final_root,
            artifact_paths,
        )
        return _PreparedSubmission(final_root, changes, artifact_paths, persisted_output_dir)

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
                return self._failure(
                    body, f"{name} artifact snapshot exceeds configured size limit", policy_caused=name == "final"
                )
            try:
                snapshot = await asyncio.get_running_loop().run_in_executor(
                    self._executor, functools.partial(base64.b64decode, encoded, validate=True)
                )
            except (binascii.Error, ValueError) as exc:
                return self._failure(body, f"invalid {name} artifact snapshot encoding: {exc}")
            if self.config.max_snapshot_bytes is not None and len(snapshot) > self.config.max_snapshot_bytes:
                return self._failure(
                    body, f"{name} artifact snapshot exceeds configured size limit", policy_caused=name == "final"
                )
            snapshots[name] = snapshot

        task_prompt = _task_instruction(body.responses_create_params.input)
        persisted_output_dir: Optional[Path] = None
        artifact_paths: list[str] = []

        async with self._semaphore:
            workspace = _VerifyWorkspace(self._executor)
            start_conversion_record()
            try:
                prepared: _PreparedSubmission = await workspace.run(
                    self._prepare_submission, body, snapshots, workspace.root
                )
                del snapshots
                artifact_paths = prepared.artifact_paths
                persisted_output_dir = prepared.persisted_output_dir
                reward, rubric_scores, judge = await grade_apex_output(
                    server_client=self.server_client,
                    model_server_name=self.config.judge_model_server.name,
                    task_id=body.task_id,
                    world_id=body.world_id,
                    instruction=task_prompt,
                    response=body.response.output_text,
                    rubric=metadata.rubric,
                    expected_output=metadata.expected_output,
                    artifact_changes=prepared.changes,
                    final_root=prepared.final_root,
                    judge_model=self.config.judge_model,
                    judge_create_params_overrides=self.config.judge_create_params_overrides,
                    judge_context_window_size=self.config.judge_context_window_size,
                    document_converter_image=self._document_converter_image(),
                    allow_host_document_tools=self.config.allow_host_document_tools,
                    scoring_method=metadata.scoring,
                    run_blocking=workspace.run,
                    metadata={
                        "task_id": body.task_id,
                        "execution": {
                            "rollout_slot_id": str((body.__pydantic_extra__ or {}).get("_ng_rollout_index", ""))
                        },
                    },
                )
                await workspace.run(
                    self._persist_grading_result,
                    persisted_output_dir,
                    reward=reward,
                    rubric_scores=rubric_scores,
                    judge_response=judge,
                )
            except Exception as exc:
                await workspace.run(self._persist_grading_result, persisted_output_dir, error=str(exc))
                if is_judge_failure(exc) or is_judge_provider_stop(exc):
                    # A judge that could not be reached or gave no usable verdict did not grade the answer: the
                    # verify endpoint's judge failsafe records the rollout as judge_failed, not as a scored zero.
                    raise
                # Anything else, an initial snapshot over its limits or a document converter that could not run
                # included, is not the policy's doing.
                return self._failure(
                    body,
                    str(exc),
                    artifact_paths=artifact_paths,
                    persisted_output_dir=persisted_output_dir,
                    policy_caused=isinstance(exc, _PolicyOutputRejected),
                )
            finally:
                # The slot is held until the scratch directory is gone, so disk use stays bounded by the slots.
                await asyncio.shield(workspace.release())

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


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=ApexFixtureRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
    invoke=invoke_fixture,
    reseed=reseed_fixture,
)


if __name__ == "__main__":
    ApexResourcesServer.run_webserver()
