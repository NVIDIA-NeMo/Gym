# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BeforeValidator, ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JudgeError
from resources_servers.job_bench.dataset import resolve_task_path
from resources_servers.job_bench.task_data import TaskData
from resources_servers.job_bench.vendor import judge
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer


LOG = logging.getLogger(__name__)

# The agent works here: the task folder is unpacked into task/ and deliverables go to output/.
WORKDIR = "/workspace"


class JobBenchConfig(BaseResourcesServerConfig):
    artifact_root: Path | None = None
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    # Upstream asks every judge call for 200000 completion tokens. A judge whose context is not
    # much larger than that (a self-hosted 262k model) rejects every prompt over the remainder.
    judge_max_completion_tokens: int = 200000
    max_judge_workers: int = 10
    sandbox_provider: str
    sandbox_config: dict[str, Any]


class JobBenchRequest(TaskData, BaseSeedSessionRequest):
    """The row an agent's ``/run`` seeds with."""


class JobBenchSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


class JobBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_dir: str
    rubrics_file: str
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class JobBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    task_id: str
    score: float
    max_score: float
    passed_count: int
    total_count: int
    judge_model: str
    judge_max_completion_tokens: int | None = None
    rubrics: list[dict[str, Any]]


class JobBenchResourcesServer(SandboxSessionResourcesServer):
    """Serves Job-Bench tasks in a task sandbox the agent works in, and grades its ``/workspace/output``.

    An Environment Server seeds an episode session and lends the agent the sandbox through ``sandbox_access``;
    ``harness_agent``'s ``/run`` seeds with the row and gets the sandbox descriptor.
    """

    # Its judge calls go through an external SDK client that notes no model request.
    grading_may_use_uninstrumented_model = True

    config: JobBenchConfig

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest | Annotated[JobBenchRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> JobBenchSeedResponse | ResourcesSeedSessionResponse:
        """Start the task sandbox with the task folder in ``/workspace/task`` and an empty ``/workspace/output``."""
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, TaskData)
        session_id = request.session[SESSION_ID_KEY]
        async with self._locked_task_session(session_id) as record:
            if record.seed is not None or self._task_session_fences.get(session_id) is not None:
                raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
            # A repeated agent-run seed replaces the earlier sandbox instead of orphaning it.
            await self._stop_task_sandbox(session_id)
            try:
                await self._start_task_sandbox(session_id, body)
                descriptor = await self._session_id_to_sandbox[session_id].serialize()
            except BaseException:
                await self._stop_failed_seed_sandbox(session_id)
                raise
        return JobBenchSeedResponse(sandbox_handle=descriptor["sandbox_id"], sandbox_descriptor=descriptor)

    async def _start_task_sandbox(self, session_id: str, task: TaskData) -> str:
        task_dir = await asyncio.to_thread(resolve_task_path, task.task_dir)
        if not (task_dir / "task_folder" / "TASK_INSTRUCTIONS.txt").is_file():
            raise ResourcesSeedRejectedError(f"Invalid Job-Bench task directory: {task_dir}")

        global_config = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        sandbox = AsyncSandbox(provider)
        resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        spec = SandboxSpec(
            image=self.config.sandbox_config["image"],
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=WORKDIR,
            env={},
            files={},
            metadata={
                **resolve_provider_metadata(self.config.sandbox_provider, global_config),
                **self.config.sandbox_config.get("metadata", {}),
                "task_id": task.task_id[:63],
            },
            resources=resources,
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        # Own the sandbox before starting it, so a seed that fails or is cancelled from here on still stops it.
        self._session_id_to_sandbox[session_id] = sandbox
        await sandbox.start(spec)
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive = Path(temporary_dir) / "task.tar.gz"

            def build_archive() -> None:
                with tarfile.open(archive, "w:gz", dereference=True) as tar:
                    tar.add(task_dir / "task_folder", arcname="task")

            await asyncio.to_thread(build_archive)
            await sandbox.upload(archive, "/tmp/task.tar.gz")
        result = await sandbox.exec(
            f"mkdir -p {WORKDIR}/output && tar -xzf /tmp/task.tar.gz -C {WORKDIR}",
            cwd="/",
        )
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(f"Failed to seed Job-Bench task ({result.error_type}): {result.stderr}")
        return WORKDIR

    async def _collect_output(self, session_id: str) -> bytes:
        """Archive the task sandbox's ``/workspace/output``, then stop the sandbox.

        An episode session keeps the archive, so a retried verify grades the same output after the sandbox is gone.
        """

        async def collect() -> bytes:
            sandbox = self._session_id_to_sandbox[session_id]
            # The agent may have removed output/; that is graded as no output, not failed as a collection error.
            result = await sandbox.exec(
                f"rm -f /tmp/output.tar.gz && mkdir -p {WORKDIR}/output && tar -czf /tmp/output.tar.gz -C {WORKDIR}/output ."
            )
            # GNU tar exits 1 when a file changed while it was read (a process the agent left running) and still
            # writes the archive; that output is the agent's result. Exit 2 is a fatal tar error.
            if result.return_code == 1 and result.error_type is None:
                LOG.warning(
                    "Job-Bench output of session %s changed while it was archived: %s", session_id, result.stderr
                )
            elif result.return_code != 0 or result.error_type is not None:
                raise RuntimeError(
                    f"Failed to collect Job-Bench output ({result.error_type}): {result.stderr}; "
                    f"output={(result.stdout or '')[-4000:]}"
                )
            with tempfile.TemporaryDirectory() as temporary_dir:
                local = Path(temporary_dir) / "output.tar.gz"
                await sandbox.download("/tmp/output.tar.gz", local)
                return local.read_bytes()

        try:
            archive = await self.capture_task_sandbox_patch(session_id, collect)
        except BaseException:
            # A retry could not grade more than this attempt, and an agent-run session gets no close: stop it now.
            try:
                await self._stop_task_sandbox(session_id)
            except Exception:
                LOG.exception("Failed to stop the Job-Bench task sandbox of session %s", session_id)
            raise
        await self.stop_task_sandbox_after_capture(session_id)
        return archive

    async def _stop_agent_run_sandbox(self, session_id: str) -> None:
        """Stop the task sandbox of a session an agent's ``/run`` seeded, whose verify failed before collection.

        Such a session gets no close, so nothing else would stop the sandbox before shutdown. An episode session
        keeps its sandbox for the retried verify or the close.
        """
        record = self._task_sessions.get(session_id)
        if record is None or record.seed is None:
            try:
                await self._stop_task_sandbox(session_id)
            except Exception:
                LOG.exception("Failed to stop the Job-Bench task sandbox of session %s", session_id)

    def _judge(self, output_dir: Path, rubrics_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rubrics_data = json.loads(rubrics_file.read_text(encoding="utf-8"))
        rubrics = rubrics_data.get("rubrics") or rubrics_data.get("evaluation_rubrics") or []
        if not any(path.is_file() for path in output_dir.rglob("*")):
            results = [
                judge.build_failed_rubric_result(index, rubric, "No output files found in the model output directory.")
                for index, rubric in enumerate(rubrics)
            ]
            return judge.build_scorecard(results), results
        file_contents = judge.extract_all_file_contents(output_dir)
        if not file_contents.strip():
            results = [
                judge.build_failed_rubric_result(
                    index, rubric, "Output files were unreadable or empty after conversion."
                )
                for index, rubric in enumerate(rubrics)
            ]
            return judge.build_scorecard(results), results
        images = judge.collect_image_attachments(output_dir)

        with ThreadPoolExecutor(max_workers=self.config.max_judge_workers) as executor:
            futures = [
                executor.submit(
                    judge.judge_rubric,
                    index,
                    rubric,
                    file_contents,
                    self.config.judge_model,
                    self.config.judge_base_url,
                    self.config.judge_api_key,
                    300,
                    3,
                    images,
                    self.config.judge_max_completion_tokens,
                )
                for index, rubric in enumerate(rubrics)
            ]
            judged = [future.result() for future in futures]
        receipt = [
            {
                "result": result,
                "debug": {
                    key: debug[key]
                    for key in (
                        "parse_status",
                        "api_exit_code",
                        "raw_response",
                        "error",
                        "vision_used",
                        "attached_images",
                    )
                    if key in debug
                },
            }
            for result, debug in judged
        ]
        try:
            (output_dir.parent / "judge-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            LOG.warning("Job-Bench judge receipt retention failed (%s)", type(exc).__name__)
        errors = [debug["error"] for _, debug in judged if debug["api_exit_code"] == 2]
        if errors:
            raise JudgeError("; ".join(errors))
        results = [result for result, _ in judged]
        return judge.build_scorecard(results), results

    async def verify(self, request: Request, body: JobBenchVerifyRequest) -> JobBenchVerifyResponse:
        """Grade the task's output; an episode session replays its first verdict to an identical retry.

        A verify for an episode session this server closed, or whose sandbox expired, is masked. Passing a saved
        ``artifact_id`` regrades that trial from its artifacts, without a session.
        """
        if body.artifact_id is not None:
            return await self._grade(request, body)
        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._grade(request, body),
            lost_response=lambda reason, failure_kind: self._lost_session_response(body, reason, failure_kind),
        )

    def _lost_session_response(
        self, body: JobBenchVerifyRequest, reason: str, failure_kind: str
    ) -> JobBenchVerifyResponse:
        return JobBenchVerifyResponse.model_validate(
            body.model_dump()
            | lost_session_verification(reason, failure_kind=failure_kind)
            | {
                "score": 0.0,
                "max_score": 0.0,
                "passed_count": 0,
                "total_count": 0,
                "judge_model": self.config.judge_model,
                "judge_max_completion_tokens": self.config.judge_max_completion_tokens,
                "rubrics": [],
            }
        )

    async def _grade(self, request: Request, body: JobBenchVerifyRequest) -> JobBenchVerifyResponse:
        session_id = str(request.session[SESSION_ID_KEY])
        replay = body.artifact_id is not None
        saved = None
        if self.config.artifact_root is not None:
            body.artifact_id = body.artifact_id or hashlib.sha256(session_id.encode()).hexdigest()
            saved = self.config.artifact_root / body.artifact_id
            if replay:
                body = JobBenchVerifyRequest.model_validate_json((saved / "request.json").read_text())
                if (saved / "result.json").exists():
                    return JobBenchVerifyResponse.model_validate_json((saved / "result.json").read_text())
            else:
                # A retried verify of an episode session reuses the trial directory of its first attempt.
                record = self._task_sessions.get(session_id)
                saved.mkdir(parents=True, exist_ok=record is not None and record.seed is not None)
        elif replay:
            raise HTTPException(status_code=400, detail="Artifact replay requires artifact_root")
        with tempfile.TemporaryDirectory() as temporary_dir:
            local_dir = Path(temporary_dir)
            archive = (saved or local_dir) / "output.tar.gz"
            if not replay:
                if saved is not None:
                    try:
                        rubrics_source = await asyncio.to_thread(resolve_task_path, body.rubrics_file)
                        await asyncio.to_thread(shutil.copyfile, rubrics_source, saved / "rubrics.json")
                        pending = saved / "request.partial"
                        pending.write_text(body.model_dump_json())
                        pending.replace(saved / "request.json")
                    except BaseException:
                        await self._stop_agent_run_sandbox(session_id)
                        raise
                output = await self._collect_output(session_id)

                def write_archive() -> None:
                    partial = archive.with_suffix(".partial")
                    with partial.open("wb") as stream:
                        stream.write(output)
                        stream.flush()
                        os.fsync(stream.fileno())
                    partial.replace(archive)

                await asyncio.to_thread(write_archive)
            output_dir = local_dir / "output"
            output_dir.mkdir()

            def extract_output() -> None:
                with tarfile.open(archive, "r:gz") as tar:
                    # Agent-created environments may contain links outside the output tree.
                    runtime_dirs = {"venv", ".venv", "node_modules", ".git", ".cache", "__pycache__"}
                    members = (
                        member
                        for member in tar
                        if (member.isfile() or member.isdir())
                        and not runtime_dirs.intersection(
                            Path(member.name).parts if member.isdir() else Path(member.name).parts[:-1]
                        )
                    )
                    tar.extractall(output_dir, members=members, filter="data")

            await asyncio.to_thread(extract_output)
            rubrics_file = (
                saved / "rubrics.json"
                if saved is not None
                else await asyncio.to_thread(resolve_task_path, body.rubrics_file)
            )
            try:
                scorecard, rubrics = await asyncio.to_thread(self._judge, output_dir, rubrics_file)
            finally:
                receipt = local_dir / "judge-receipt.json"
                if saved is not None and receipt.is_file():
                    try:
                        with (saved / f"judge-receipt-{uuid4().hex}.json").open("xb") as stream:
                            stream.write(receipt.read_bytes())
                    except OSError as exc:
                        LOG.warning("Job-Bench judge receipt retention failed (%s)", type(exc).__name__)

        response = JobBenchVerifyResponse(
            **body.model_dump(),
            reward=float(scorecard["normalized_score"]),
            score=float(scorecard["total_score"]),
            max_score=float(scorecard["max_score"]),
            passed_count=int(scorecard["passed_count"]),
            total_count=int(scorecard["total_count"]),
            judge_model=self.config.judge_model,
            judge_max_completion_tokens=self.config.judge_max_completion_tokens,
            grading_protocol=judge.INPUT_PROTOCOL,
            rubrics=rubrics,
        )
        if saved is not None:
            pending = saved / "result.partial"
            pending.write_text(response.model_dump_json())
            pending.replace(saved / "result.json")
        return response


if __name__ == "__main__":
    JobBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = JobBenchResourcesServer.run_webserver()  # noqa: F401
