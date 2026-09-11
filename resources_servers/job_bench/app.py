# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JobBench resources server.

An OpenCode agent works in a sandbox seeded with one JobBench ``task_folder`` and
writes deliverables to an output directory. At verify time the server pulls that
directory back, renders every deliverable as text, and grades it rubric-by-rubric
with an LLM judge. The reward is JobBench's weighted normalized score.

Rubrics never enter the sandbox and never appear in the model-visible JSONL; they
are read from the prepared control-plane cache keyed by task ID.
"""

from __future__ import annotations

import asyncio
import sys
import tarfile
import tempfile
from asyncio import Semaphore
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from traceback import format_exc
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

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
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY, get_first_server_config_dict, is_nemo_gym_fastapi_entrypoint
from resources_servers.job_bench.judge import (
    JUDGE_SYSTEM_PROMPT,
    MAX_CHARS_PER_FILE,
    MAX_VISION_IMAGES,
    build_failed_rubric_result,
    build_rubric_prompt,
    build_rubric_result,
    build_scorecard,
    build_user_content,
    collect_image_attachments,
    extract_all_file_contents,
    parse_judge_json,
    rubric_needs_vision,
)
from resources_servers.job_bench.task_store import (
    OUTPUT_DIR,
    SEARCH_FILES_NAME,
    TASK_FOLDER_DIR,
    TASK_FOLDER_NAME,
    WORKSPACE_DIR,
    JobBenchTask,
    JobBenchTaskStore,
)


PACKAGE_DIR = Path(__file__).resolve().parent
NEMO_GYM_ROOT = PACKAGE_DIR.parents[1]

REMOTE_TASK_ARCHIVE = "/tmp/job_bench_task.tgz"
REMOTE_OUTPUT_ARCHIVE = "/tmp/job_bench_output.tgz"


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (NEMO_GYM_ROOT / expanded).resolve()


class JobBenchResourcesServerConfig(BaseResourcesServerConfig):
    # Verification consumes the agent's live sandbox, so a stored rollout cannot be replayed.
    REVERIFY_MODE = ReverifyMode.UNSUPPORTED

    cache_dir: Path = Path("resources_servers/job_bench/data/cache/tasks")
    archives_dir: Path = Path("resources_servers/job_bench/data/cache/archives")
    split: str = "main"
    expected_task_count: int | None = None

    # JobBench's own runner withholds files_required_to_search/ so the agent has to
    # rediscover those references online. Enabling this mounts them in the sandbox,
    # which makes runs hermetic but no longer comparable to the public leaderboard.
    include_search_files: bool = False

    sandbox_provider: str
    sandbox_image: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    # JobBench tasks legitimately need the open internet; deny-all costs the
    # search-dependent tasks. Kept as a knob for hermetic runs.
    enforce_agent_no_network: bool = False
    sandbox_model_server: ModelServerRef | None = None
    max_output_mib: int = 512

    judge_model_server: ModelServerRef
    judge_concurrency: int = Field(default=10, ge=1)
    judge_max_tokens: int = 200_000
    judge_temperature: float = 0.0
    max_chars_per_file: int = Field(default=MAX_CHARS_PER_FILE, ge=1)
    max_vision_images: int = Field(default=MAX_VISION_IMAGES, ge=0)

    include_rubric_details_in_response: bool = True


class JobBenchInstanceRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    verifier_metadata: dict[str, Any] | None = None


class JobBenchSeedSessionRequest(JobBenchInstanceRequest, BaseSeedSessionRequest):
    pass


class JobBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    sandbox_descriptor: dict[str, Any]


class JobBenchVerifyRequest(JobBenchInstanceRequest, BaseVerifyRequest):
    sandbox_handle: str | None = None


class JobBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    evaluation_completed: bool

    total_score: float = 0.0
    max_score: float = 0.0
    normalized_score: float = 0.0
    pass_rate: float = 0.0
    passed_count: int = 0
    total_count: int = 0

    num_output_files: int = 0
    num_vision_images: int = 0
    rubrics: list[dict] | None = None
    collection_error: str | None = None

    output_collection_time_s: float = 0.0
    judging_time_s: float = 0.0


@dataclass
class AgentSandboxSession:
    task_id: str
    sandbox: AsyncSandbox
    sandbox_handle: str


def _resolve_task_id(body: JobBenchInstanceRequest) -> str:
    metadata_task_id = (body.verifier_metadata or {}).get("task_id")
    if body.task_id and metadata_task_id and body.task_id != metadata_task_id:
        raise ValueError(
            f"Conflicting JobBench task IDs: task_id={body.task_id!r}, verifier_metadata.task_id={metadata_task_id!r}"
        )
    task_id = body.task_id or metadata_task_id
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("JobBench requests must provide verifier_metadata.task_id or task_id")
    return task_id


def _build_archive(source_dirs: dict[str, Path], archive_path: Path) -> None:
    """Write a gzipped tar containing each ``source_dir`` under its archive name."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    with tarfile.open(temporary_path, "w:gz") as archive:
        for arcname, source in source_dirs.items():
            archive.add(source, arcname=arcname)
    # Rename last so a concurrent reader never sees a half-written archive.
    temporary_path.replace(archive_path)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        # filter="data" drops absolute paths, symlink escapes and device nodes:
        # the archive is built from untrusted agent output.
        archive.extractall(destination, filter="data")


class JobBenchResourcesServer(SimpleResourcesServer):
    config: JobBenchResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._task_store = JobBenchTaskStore(
            _resolve_repo_path(self.config.cache_dir),
            self.config.split,
            expected_task_count=self.config.expected_task_count,
        )
        self._agent_sessions: dict[str, AgentSandboxSession] = {}
        self._judge_semaphore: Semaphore = Semaphore(self.config.judge_concurrency)
        self._archive_locks: dict[str, asyncio.Lock] = {}

    # ---------------------------------------------------------------- sandbox

    def _model_egress_target(self) -> str | None:
        if not self.config.sandbox_model_server:
            return None
        model_config = get_first_server_config_dict(get_global_config_dict(), self.config.sandbox_model_server.name)
        target = str(model_config.get("host") or "")
        if not target:
            raise ValueError(f"Model server {self.config.sandbox_model_server.name!r} does not have a host")
        if target in {"0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}:
            raise ValueError(
                f"JobBench task sandboxes cannot reach loopback model host {target!r}; "
                "set NEMO_GYM_SANDBOX_MODEL_BASE_URL or launch Gym with use_absolute_ip=true"
            )
        return target

    def _provider_options(self) -> dict[str, Any]:
        options = deepcopy(self.config.sandbox_config.get("provider_options", {}))
        if not self.config.enforce_agent_no_network:
            # JobBench's default posture is the open internet, as upstream runs it.
            return options

        network_policy = options.setdefault("network_policy", {"defaultAction": "deny", "egress": []})
        if not isinstance(network_policy, dict):
            raise TypeError("JobBench sandbox network_policy must be a mapping")
        egress = network_policy.setdefault("egress", [])
        if not isinstance(egress, list):
            raise TypeError("JobBench sandbox network_policy.egress must be a list")
        model_egress_target = self._model_egress_target()
        if model_egress_target is not None:
            model_rule = {"action": "allow", "target": model_egress_target}
            if model_rule not in egress:
                egress.append(model_rule)
        return options

    async def _create_sandbox(self, task: JobBenchTask) -> AsyncSandbox:
        global_config = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)

        spec = SandboxSpec(
            image=self.config.sandbox_image,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=WORKSPACE_DIR,
            env=dict(self.config.sandbox_config.get("env", {})),
            files={},
            metadata=provider_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {
                "benchmark": "job-bench",
                "job-bench-split": task.split,
                "job-bench-task": task.label,
                "nemo_gym_agent": self.config.name or "job_bench",
            },
            resources=SandboxResources.from_mapping(dict(self.config.sandbox_config.get("resources", {}))),
            provider_options=self._provider_options(),
        )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(spec)
        return sandbox

    async def _stop_sandbox(self, sandbox: AsyncSandbox, *, task_id: str) -> None:
        try:
            await sandbox.stop()
        except Exception:
            print(f"Failed to stop JobBench sandbox for {task_id}: {format_exc()}", file=sys.stderr)

    async def _task_archive(self, task: JobBenchTask) -> Path:
        """Return a cached tarball of the task's sandbox-visible materials, building it once."""
        suffix = "with-search" if self.config.include_search_files else "task-only"
        archive_path = (
            _resolve_repo_path(self.config.archives_dir)
            / task.split
            / task.profession
            / f"{task.task_name}.{suffix}.tgz"
        )
        lock = self._archive_locks.setdefault(str(archive_path), asyncio.Lock())
        async with lock:
            if not archive_path.is_file():
                sources = {TASK_FOLDER_NAME: task.task_folder}
                if self.config.include_search_files and task.search_files_dir.is_dir():
                    sources[SEARCH_FILES_NAME] = task.search_files_dir
                await asyncio.to_thread(_build_archive, sources, archive_path)
        return archive_path

    async def _seed_workspace(self, sandbox: AsyncSandbox, task: JobBenchTask) -> None:
        """Lay out /workspace with the task materials and an empty output directory."""
        archive_path = await self._task_archive(task)
        await sandbox.upload(archive_path, REMOTE_TASK_ARCHIVE)

        setup = await sandbox.exec(
            f"mkdir -p {WORKSPACE_DIR} {OUTPUT_DIR}"
            f" && tar -xzf {REMOTE_TASK_ARCHIVE} -C {WORKSPACE_DIR}"
            f" && rm -f {REMOTE_TASK_ARCHIVE}"
            f" && test -f {TASK_FOLDER_DIR}/TASK_INSTRUCTIONS.txt",
            timeout_s=600,
        )
        if setup.return_code != 0:
            details = ((setup.stderr or "") + (setup.stdout or "")).strip()
            raise RuntimeError(f"Failed to seed JobBench workspace for {task.task_id}: {details[-4000:]}")

    async def _collect_output(self, sandbox: AsyncSandbox, task: JobBenchTask, destination: Path) -> None:
        """Download the agent's output directory into ``destination``."""
        probe = await sandbox.exec(f"test -d {OUTPUT_DIR} && du -sm {OUTPUT_DIR} | cut -f1", timeout_s=120)
        if probe.return_code != 0:
            raise RuntimeError(f"JobBench output directory {OUTPUT_DIR} is missing in the agent sandbox")
        try:
            output_mib = int((probe.stdout or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            output_mib = 0
        if output_mib > self.config.max_output_mib:
            raise RuntimeError(
                f"JobBench output directory is {output_mib} MiB, above the {self.config.max_output_mib} MiB cap"
            )

        archive = await sandbox.exec(
            f"tar -czf {REMOTE_OUTPUT_ARCHIVE} -C {WORKSPACE_DIR} output",
            timeout_s=900,
        )
        if archive.return_code != 0:
            details = ((archive.stderr or "") + (archive.stdout or "")).strip()
            raise RuntimeError(f"Failed to archive JobBench output for {task.task_id}: {details[-4000:]}")

        with tempfile.TemporaryDirectory(prefix="nemo-gym-job-bench-output-") as temporary_dir:
            local_archive = Path(temporary_dir) / "output.tgz"
            await sandbox.download(REMOTE_OUTPUT_ARCHIVE, local_archive)
            await asyncio.to_thread(_extract_archive, local_archive, destination)

    # ------------------------------------------------------------------ judge

    async def _judge_rubric(self, rubric_index: int, rubric: dict, file_contents: str, attachments: list) -> dict:
        """Grade one rubric; every criterion must pass for the rubric to score."""
        rubric_attachments = attachments if (attachments and rubric_needs_vision(rubric)) else []
        prompt = build_rubric_prompt(rubric, file_contents, vision_used=bool(rubric_attachments))

        chat_params = NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(prompt, rubric_attachments)},
            ],
            max_tokens=self.config.judge_max_tokens,
            temperature=self.config.judge_temperature,
        )
        async with self._judge_semaphore:
            chat_response = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=chat_params,
                response_model=NeMoGymChatCompletion,
            )

        content = chat_response.choices[0].message.content if chat_response.choices else None
        if not content or not content.strip():
            return build_failed_rubric_result(rubric_index, rubric, "Judge returned an empty response")
        try:
            parsed, _ = parse_judge_json(content.strip())
        except ValueError as error:
            # A reply that arrived but cannot be parsed is a failed grade, not a
            # failed call: score it zero rather than routing the row to the sidecar.
            return build_failed_rubric_result(rubric_index, rubric, f"Unparseable judge response: {error}")
        return build_rubric_result(rubric_index, rubric, parsed)

    async def _judge_output(self, task: JobBenchTask, output_dir: Path) -> tuple[list[dict], int, int]:
        file_contents = await asyncio.to_thread(
            extract_all_file_contents, output_dir, max_chars_per_file=self.config.max_chars_per_file
        )
        attachments = await asyncio.to_thread(collect_image_attachments, output_dir, self.config.max_vision_images)
        num_output_files = sum(1 for path in output_dir.rglob("*") if path.is_file())

        results = await asyncio.gather(
            *(
                self._judge_rubric(index, rubric, file_contents, attachments)
                for index, rubric in enumerate(task.rubrics)
            )
        )
        return list(results), num_output_files, len(attachments)

    # -------------------------------------------------------------- endpoints

    async def seed_session(self, request: Request, body: JobBenchSeedSessionRequest) -> JobBenchSeedSessionResponse:
        task = self._task_store.get(_resolve_task_id(body))
        session_id = str(request.session[SESSION_ID_KEY])

        previous_session = self._agent_sessions.pop(session_id, None)
        if previous_session is not None:
            await self._stop_sandbox(previous_session.sandbox, task_id=previous_session.task_id)

        sandbox: AsyncSandbox | None = None
        try:
            sandbox = await self._create_sandbox(task)
            descriptor = await sandbox.serialize()
            sandbox_handle = descriptor.get("sandbox_id") if isinstance(descriptor, dict) else None
            if not isinstance(sandbox_handle, str) or not sandbox_handle:
                raise RuntimeError("JobBench sandbox provider did not return a sandbox_id")

            await self._seed_workspace(sandbox, task)
            self._agent_sessions[session_id] = AgentSandboxSession(
                task_id=task.task_id,
                sandbox=sandbox,
                sandbox_handle=sandbox_handle,
            )
            return JobBenchSeedSessionResponse(
                sandbox_handle=sandbox_handle,
                sandbox_descriptor=dict(descriptor),
            )
        except Exception:
            if sandbox is not None:
                await self._stop_sandbox(sandbox, task_id=task.task_id)
            raise

    async def verify(self, request: Request, body: JobBenchVerifyRequest) -> JobBenchVerifyResponse:
        task = self._task_store.get(_resolve_task_id(body))
        session_id = str(request.session.get(SESSION_ID_KEY, "unknown"))

        agent_session = self._agent_sessions.pop(session_id, None)
        sandbox_handle = body.sandbox_handle
        collection_error: str | None = None
        output_collection_time_s = 0.0

        def empty_response(error: str) -> JobBenchVerifyResponse:
            return JobBenchVerifyResponse.model_validate(
                body.model_dump()
                | {
                    "task_id": task.task_id,
                    "sandbox_handle": sandbox_handle,
                    "reward": 0.0,
                    "evaluation_completed": False,
                    "max_score": float(task.max_score),
                    "total_count": len(task.rubrics),
                    "collection_error": error,
                    "failure_reason": error,
                    "output_collection_time_s": output_collection_time_s,
                }
            )

        if agent_session is None:
            return empty_response(f"No JobBench agent sandbox exists for session {session_id!r}")
        sandbox_handle = agent_session.sandbox_handle

        with tempfile.TemporaryDirectory(prefix="nemo-gym-job-bench-verify-") as temporary_dir:
            output_dir = Path(temporary_dir) / "output"
            started = monotonic()
            try:
                if agent_session.task_id != task.task_id:
                    raise RuntimeError(
                        f"JobBench session task {agent_session.task_id!r} does not match verify task {task.task_id!r}"
                    )
                await self._collect_output(agent_session.sandbox, task, Path(temporary_dir))
            except Exception as error:
                print(f"Failed to collect JobBench output for {task.task_id}: {format_exc()}", file=sys.stderr)
                collection_error = f"{type(error).__name__}: {error}"
            finally:
                output_collection_time_s = monotonic() - started
                await self._stop_sandbox(agent_session.sandbox, task_id=task.task_id)

            if collection_error is not None:
                return empty_response(collection_error)

            started = monotonic()
            results, num_output_files, num_vision_images = await self._judge_output(task, output_dir)
            judging_time_s = monotonic() - started

        scorecard = build_scorecard(results)
        return JobBenchVerifyResponse.model_validate(
            body.model_dump()
            | {
                "task_id": task.task_id,
                "sandbox_handle": sandbox_handle,
                "reward": float(scorecard["normalized_score"]),
                "evaluation_completed": True,
                "total_score": float(scorecard["total_score"]),
                "max_score": float(scorecard["max_score"]),
                "normalized_score": float(scorecard["normalized_score"]),
                "pass_rate": float(scorecard["pass_rate"]),
                "passed_count": int(scorecard["passed_count"]),
                "total_count": int(scorecard["total_count"]),
                "num_output_files": num_output_files,
                "num_vision_images": num_vision_images,
                "rubrics": results if self.config.include_rubric_details_in_response else None,
                "output_collection_time_s": output_collection_time_s,
                "judging_time_s": judging_time_s,
            }
        )


if __name__ == "__main__":
    JobBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = JobBenchResourcesServer.run_webserver()  # noqa: F401
