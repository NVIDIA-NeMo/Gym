# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CVDP resources server.

This module owns the *policy* side of CVDP verification: the HTTP ``verify``
contract, the request/response schemas, and how a model/agent answer is turned
into a reward. ``verify`` routes by category — code-comprehension tasks are
scored with BLEU/ROUGE against a reference answer (``_verify_subjective``),
while code-generation tasks are graded by actually running the task's test
harness (``_verify_objective``). The objective path either grades the files an
agent already wrote (``rtl_files``) or parses RTL out of the model's text, then
delegates execution to :class:`resources_servers.cvdp.testbench_runner.TestbenchRunner`,
which owns the *mechanism* (Compose-manifest translation and the sandbox
provider). Keeping execution in ``testbench_runner.py``
lets this file stay focused on the contract and scoring.

Agent sessions: with ``agent_sandbox_provider`` set, a typed ``seed_session`` starts a task
sandbox from the simulation image, stages the task's context files under ``agent_workdir``
(never the hidden harness files), and lends it to the agent through ``SandboxAccess``. A
harness agent (``hermes_agent``, ``opencode_agent``, ``claude_code_agent``) edits and
simulates there. ``verify`` then copies the RTL and verification files it wrote out of the
sandbox, stops the sandbox, and grades them like ``rtl_files``. ``close_session`` stops a
sandbox that verify has not. The legacy paths are unchanged: ``cvdp_agent``'s simple flavor
seeds and verifies without a session, and its agentic flavor sends ``rtl_files``.
"""

import asyncio
import logging
import re
import shlex
import tarfile
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from cvdp_lib.cvdp_constants import (
    BLEU_THRESHOLD,
    CODE_COMPREHENSION_CATEGORIES,
    N_GRAM_DEFAULT,
    ROUGE_THRESHOLD,
    VERIF_EDA_CATEGORIES,
    is_score_based_category,
)
from cvdp_lib.model_helpers import ModelHelpers
from cvdp_lib.subjective import calculate_BLEU, calculate_ROUGE
from fastapi import FastAPI, Request
from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import PROVIDER_REQUEST_FAILED, VERIFIER_ERROR, EpisodeId, NgFailureEvent, TaskId
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxConfig,
    resolve_provider_config,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    SimpleResourcesServer,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.cvdp.testbench_runner import TestbenchRunner


_helpers = ModelHelpers()

# The files an agent session harvests by default: the RTL and testbench sources it may write.
DEFAULT_HARVEST_GLOBS = ["rtl/**/*.sv", "rtl/**/*.v", "rtl/**/*.vhd", "verif/**/*.sv", "verif/**/*.v"]
# The harvest command's exit code when the workspace holds none of the harvested paths. GNU tar exits 1 when a
# file changed while it was read (the archive is complete) and 2 on a fatal error (a full disk), after which the
# archive may be missing files, so this must differ from both.
_NOTHING_TO_HARVEST_EXIT = 3
_CATEGORY_RE = re.compile(r"cid\d+")


# ----------------------------
# Config
# ----------------------------


class CVDPAgentSandboxConfig(SandboxConfig):
    # The image of the sandbox lent to the agent; None uses oss_sim_image, which has the simulators the
    # agentic prompts tell the agent to use.
    image: Optional[str] = None


class CVDPResourcesServerConfig(BaseResourcesServerConfig):
    oss_sim_image: str = "ghcr.io/hdl/sim/osvb"
    oss_pnr_image: str = "ghcr.io/hdl/impl/pnr"
    eda_sim_image: str = ""  # Set to a commercial EDA image (e.g. Cadence Xcelium)
    container_timeout: int = 600
    num_processes: int = 4  # Max concurrent verifier sandboxes
    harness_workspace_dir: str = ""  # Optional host directory for per-rollout temp workspaces
    container_workspace: str = "/code"
    container_transfer_dir: str = "/sandbox"
    container_tmp_bind_path: str = ""  # If set, redirect temp into the uploaded per-rollout workspace
    sandbox_provider: str | Dict[str, Any] = Field(default_factory=lambda: {"apptainer": {}})
    sandbox_spec: Dict[str, Any] = Field(default_factory=dict)
    prepared_images: Dict[str, str] = Field(default_factory=dict)
    prepared_image_manifest: str = ""

    # Agent sessions only (configs/cvdp_agentic_*.yaml). Names a top-level provider block; the agent borrows the
    # task sandbox through it, so the provider must support reattaching (serialize/connect). When it is set, a
    # typed seed lends a sandbox and a verify without a live session is masked as lost. Unset, typed seeds are
    # stateless and verify grades the request alone, as before.
    agent_sandbox_provider: Optional[str] = None
    agent_sandbox_config: CVDPAgentSandboxConfig = Field(default_factory=CVDPAgentSandboxConfig)
    # Where the task's context files are staged and the agent works. The agentic prompts name /code paths.
    agent_workdir: str = "/code"
    # Files under agent_workdir that verify grades when they differ from what seed staged; the task's target files
    # are graded whenever they exist.
    harvest_globs: List[str] = Field(default_factory=lambda: list(DEFAULT_HARVEST_GLOBS))
    # Ceiling on archiving the workspace inside the sandbox.
    harvest_timeout_s: int = 300


# ----------------------------
# Schemas
# ----------------------------


class CVDPVerifierMetadata(BaseModel):
    task_id: str
    categories: List[str] = []
    difficulty: str = ""
    target_files: List[str] = []  # Empty for code-comprehension categories
    harness_files: Dict[str, Optional[str]] = {}  # Empty for code-comprehension categories
    context_files: Dict[str, str] = {}  # Companion RTL from input.context (non-target files needed for compilation)
    subjective_reference: Optional[str] = None  # Reference answer for code-comprehension categories (6,8,9,10)


class CVDPRunRequest(BaseRunRequest):
    pass


class CVDPVerifyRequest(CVDPRunRequest, BaseVerifyRequest):
    verifier_metadata: Dict[str, Any]
    rtl_files: Optional[Dict[str, str]] = (
        None  # files the agent already wrote to disk in the sandbox (agentic flow). When present, these are graded directly instead of re-parsing RTL out of the model's chat text.
    )


class CVDPVerifyResponse(BaseVerifyResponse):
    task_id: Optional[str] = None
    category: Optional[str] = None  # e.g. "cid003" — for CVDP report
    difficulty: Optional[str] = None  # e.g. "easy" — for CVDP report
    extracted_rtl: Optional[Dict[str, str]] = None
    container_exit_code: Optional[int] = None
    container_stderr: Optional[str] = None
    container_services: Optional[List[Dict]] = None  # per-service results: [{"service", "exit_code", "stderr"}]
    execution_time: Optional[float] = None  # total harness wall time in seconds
    parse_failed: bool = False  # True when model produced output but RTL extraction failed
    bleu_score: Optional[float] = None  # BLEU score for code-comprehension categories
    rouge_score: Optional[float] = None  # ROUGE score for code-comprehension categories


# ----------------------------
# Code extraction helpers
# ----------------------------


def _parse_model_response(res: str, target_files: List[str]) -> Optional[Dict[str, str]]:
    """
    Parse model output using ModelHelpers.parse_model_response().
    Returns {filename: code} or None on failure.
    """
    if not target_files:
        return None

    no_schema = len(target_files) == 1

    # Match CVDP's openai_llm.py: strip response before parsing
    res = res.strip()

    # Match CVDP's openai_llm.py: fix JSON formatting for multi-file responses
    if not no_schema and res.startswith("{") and res.endswith("}"):
        res = _helpers.fix_json_formatting(res)

    output, success = _helpers.parse_model_response(res, files=target_files, no_schema=no_schema)

    if not success:
        return None

    if no_schema:  # schema is one first or multiple
        code = output.get("direct_text") or output.get("response")
        return {target_files[0]: code} if code else None

    result: Dict[str, str] = {}
    if "code" in output and isinstance(output["code"], list):
        for item in output["code"]:
            if isinstance(item, dict):
                result.update(item)

    return result if result else None


# ----------------------------
# Agent session workspace
# ----------------------------


class CVDPSeedSessionRequest(BaseSeedSessionRequest):
    """The legacy seed: cvdp_agent's simple flavor sends its row, and nothing is kept."""


def _is_harness_path(rel: str) -> bool:
    """Whether a path belongs to the hidden grading harness: the test scripts in ``src/`` and the compose file."""
    norm = rel.replace("\\", "/").strip("/")
    return (
        norm == "src"
        or norm.startswith("src/")
        or norm == "docker-compose.yml"
        or norm.endswith("/docker-compose.yml")
    )


def _safe_rel(rel: str) -> bool:
    """Whether a path stays inside the workspace: relative and without ``..``."""
    return bool(rel) and not rel.startswith("/") and ".." not in Path(rel).parts


def _seeded_files(meta: CVDPVerifierMetadata) -> Dict[str, str]:
    """The task's context files the agent may see, by workspace path: no harness file, nothing outside it."""
    return {
        rel: content
        for rel, content in meta.context_files.items()
        if rel not in meta.harness_files and not _is_harness_path(rel) and _safe_rel(rel)
    }


def _harvest_roots(globs: List[str], target_files: List[str]) -> List[str]:
    """The workspace paths to archive: each glob's top directory, plus target files outside them."""
    roots = set()
    for pattern in globs:
        top = pattern.split("/")[0]
        if "/" not in pattern or any(char in top for char in "*?["):
            return ["."]
        roots.add(top)
    return sorted(roots) + sorted({t for t in target_files if _safe_rel(t) and t.split("/")[0] not in roots})


def _harvest_command(workdir: str, roots: List[str], archive: str) -> str:
    """Archive whichever roots exist; exit ``_NOTHING_TO_HARVEST_EXIT`` when none do.

    The agent decides which files are readable, so an unreadable file must not fail the archive: tar exit 2 masks
    the sample. The command first makes the agent's files readable again (it runs as their owner, or as root), and
    GNU tar then leaves out whatever still is not with a warning. A target left out is graded as missing, a failure
    of the model; write failures, such as a full disk, still exit 2.
    """
    candidates = " ".join(shlex.quote(root) for root in roots)
    return (
        f"cd {shlex.quote(workdir)} || exit {_NOTHING_TO_HARVEST_EXIT}; set --; "
        f'for p in {candidates}; do if [ -e "$p" ]; then set -- "$@" "$p"; fi; done; '
        f'if [ "$#" -eq 0 ]; then exit {_NOTHING_TO_HARVEST_EXIT}; fi; '
        'chmod -R u+rX -- "$@" 2>/dev/null; '
        # Only GNU tar has the flag; others exit 1 on an unreadable file, which grades too.
        'if tar --version 2>/dev/null | grep -q GNU; then set -- --ignore-failed-read -- "$@"; else set -- -- "$@"; fi; '
        f'tar -cf {shlex.quote(archive)} "$@"'
    )


def _collect_harvest(
    archive: Path, dest: Path, globs: List[str], seeded: Dict[str, str], target_files: List[str]
) -> Dict[str, str]:
    """Unpack the regular files of the workspace archive and return the ones to grade.

    Glob matches count only when they differ from what seed staged, so unchanged context files are not graded as
    the agent's work; a declared target file counts whenever it exists. Files that are not UTF-8 text are skipped.
    """
    with tarfile.open(archive) as tar:
        regular = {m.name for m in tar.getmembers() if m.isreg() and _safe_rel(m.name)}
        # tar stores a second path to an archived file as a hard link to the first; keep it when that one is kept.
        members = [
            m
            for m in tar.getmembers()
            if m.name in regular or (m.islnk() and _safe_rel(m.name) and m.linkname in regular)
        ]
        tar.extractall(dest, members=members, filter="data")

    def read(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8") if path.is_file() else None
        except (OSError, UnicodeDecodeError):
            return None

    produced: Dict[str, str] = {}
    for pattern in globs:
        for path in sorted(dest.glob(pattern)):
            rel = path.relative_to(dest).as_posix()
            content = read(path)
            if rel in produced or content is None or seeded.get(rel) == content:
                continue
            produced[rel] = content
    for rel in target_files:
        if rel not in produced and _safe_rel(rel) and (content := read(dest / rel)) is not None:
            produced[rel] = content
    return produced


class _HarvestUnavailable(RuntimeError):
    """The sandbox provider failed to archive or transfer the workspace: the environment failed, not the model."""

    def __init__(self, failure_kind: str, reason: str) -> None:
        super().__init__(reason)
        self.failure_kind = failure_kind


def _harvest_unavailable(operation: str, error: BaseException) -> _HarvestUnavailable:
    failure_kind = error.failure_kind if isinstance(error, NgFailureEvent) else PROVIDER_REQUEST_FAILED
    return _HarvestUnavailable(
        failure_kind, f"The task sandbox failed to {operation}: {type(error).__name__}: {error}"
    )


@dataclass
class _AgentSession:
    episode_id: EpisodeId
    task_id: TaskId
    meta: CVDPVerifierMetadata
    # The context files staged into the workspace, by workspace path.
    seeded: Dict[str, str]
    # Released (stopped and set to None) by the first verify that harvests, or by close.
    sandbox: Optional[AsyncSandbox] = None
    # Only a fully staged session can be lent or verified; a failed setup stays owned for cleanup.
    ready: bool = False
    # The harvested files, set once a verify has harvested. A provider failure leaves it unset and the sandbox
    # running, so a resent verify can harvest again.
    harvested: Optional[Dict[str, str]] = None


# ----------------------------
# Server
# ----------------------------


class CVDPResourcesServer(SimpleResourcesServer):
    config: CVDPResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)
        self._harness = TestbenchRunner(
            self.config,
            named_sandbox_configs=getattr(self.server_client, "global_config_dict", None),
        )

        # Warn if commercial EDA image is not configured.
        # Categories 12, 13, 14 require a commercial EDA image (e.g. Cadence Xcelium).
        # Apptainer uses host networking so no license network setup is needed
        # (unlike Docker which requires a dedicated license network).
        # This mirrors CVDP's validate_commercial_eda_setup() — warn but don't block.
        if not self.config.eda_sim_image:
            logging.warning(
                "eda_sim_image is not configured. "
                "Categories %s (commercial EDA) will fail if __VERIF_EDA_IMAGE__ "
                "is referenced in harness files.",
                VERIF_EDA_CATEGORIES,
            )

        if self.config.agent_sandbox_provider and self.config.num_workers not in (None, 1):
            raise ValueError("CVDP agent sessions are process-local and require num_workers=1")
        self._sessions: Dict[str, _AgentSession] = {}
        self._locks = ResourcesSessionLocks()
        self._fences: ResourcesSessionFences[EpisodeId] = ResourcesSessionFences(
            self.config.session_close_retry_window_seconds
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    # ---------------------------------------------------------------- agent sessions

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[CVDPSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Start the task sandbox with the context files staged, and lend it to the agent.

        Without ``agent_sandbox_provider``, and for the legacy seed, nothing is kept, as before sessions.
        """
        if not isinstance(body, ResourcesSeedSessionRequest) or not self.config.agent_sandbox_provider:
            return await super().seed_session(body, request=request)

        session_id = body.resources_session_id
        # The cookie is set only on success: a refusal's reply passes through the session middleware too, and must
        # not hand the caller a cookie for a session that another episode holds.
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            session = self._sessions.get(session_id)
            if session is not None:
                if (session.episode_id, session.task_id) != (body.episode_id, body.task_id):
                    raise ResourcesSessionConflictError(
                        "resources_session_id is already bound to another episode or task"
                    )
                if session.ready:
                    if session.harvested is not None or session.sandbox is None:
                        raise ResourcesSessionConflictError(
                            "The task sandbox is not available for seeding (already harvested)"
                        )
                    request.session[SESSION_ID_KEY] = session_id
                    return await self._seed_response(session_id, session)
                # A failed setup whose cleanup failed too: stop that sandbox before starting another. A stop that
                # fails again raises a retryable error and keeps the session owned.
                await self._release_sandbox(session)
                del self._sessions[session_id]
            try:
                meta = CVDPVerifierMetadata.model_validate(body.task_data.get("verifier_metadata"))
                if len(meta.categories) < 2 or not _CATEGORY_RE.fullmatch(meta.categories[0]):
                    raise ValueError(f"categories must be [cidNNN, difficulty], got {meta.categories!r}")
            except (ValidationError, ValueError) as error:
                # A row that names no CVDP task: no retry fixes it.
                raise ResourcesSeedRejectedError(f"task_data is not a CVDP task: {error}") from error

            session = _AgentSession(
                episode_id=body.episode_id, task_id=body.task_id, meta=meta, seeded=_seeded_files(meta)
            )
            self._sessions[session_id] = session
            try:
                await self._start_task_sandbox(session)
                response = await self._seed_response(session_id, session)
                session.ready = True
            except BaseException:
                # Keep ownership if the stop fails; close and retirement retry it, but seed never lends partial setup.
                await self._release_sandbox(session)
                del self._sessions[session_id]
                raise
            request.session[SESSION_ID_KEY] = session_id
            return response

    async def _start_task_sandbox(self, session: _AgentSession) -> None:
        provider = self.config.agent_sandbox_provider
        workdir = self.config.agent_workdir.rstrip("/")
        global_config_dict = get_global_config_dict()
        spec = self.config.agent_sandbox_config.spec(
            provider,
            image=self.config.agent_sandbox_config.image or self.config.oss_sim_image,
            workdir=self.config.agent_workdir,
            metadata={"benchmark": "cvdp"},
            files={f"{workdir}/{rel}": content for rel, content in session.seeded.items()},
            named_configs=global_config_dict,
        )
        sandbox = AsyncSandbox(resolve_provider_config(provider, global_config_dict))
        # Own the handle before start: initial file delivery can fail after the provider created the sandbox.
        session.sandbox = sandbox
        await sandbox.start(spec)

    async def _seed_response(self, session_id: str, session: _AgentSession) -> ResourcesSeedSessionResponse:
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.agent_sandbox_provider,
                    descriptor=await session.sandbox.serialize(),
                ),
                workdir=self.config.agent_workdir,
            ),
        )

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            fenced_episode = self._fences.get(session_id)
            if fenced_episode is None:
                session = self._sessions.get(session_id)
                if session is not None:
                    if session.episode_id != body.episode_id:
                        raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
                    # A failed stop raises and keeps the session, so the caller can retry the close.
                    await self._release_sandbox(session)
                    del self._sessions[session_id]
                self._fences.fence(session_id, body.episode_id)
            elif fenced_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
        request.session.pop(SESSION_ID_KEY, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def retire_session_state(self, session_id: str) -> None:
        """Release resources when checkpoint retirement ends an attempt."""
        async with self._locks.locked(session_id):
            session = self._sessions.get(session_id)
            if session is not None:
                await self._release_sandbox(session)
                del self._sessions[session_id]
                self._fences.fence(session_id, session.episode_id)

    def session_release_pending(self, session_id: str) -> bool:
        """Whether the session still holds a sandbox to stop.

        Verify keeps it when the harvest fails or the stop after it does. Checkpoint and worker ownership then
        outlive the verify, so a lost close still leaves retirement a session to release.
        """
        session = self._sessions.get(session_id)
        return session is not None and session.sandbox is not None

    @staticmethod
    async def _release_sandbox(session: _AgentSession) -> None:
        if session.sandbox is not None:
            await session.sandbox.stop()
            session.sandbox = None

    async def shutdown(self) -> None:
        for session_id, session in list(self._sessions.items()):
            try:
                await self._release_sandbox(session)
            except Exception:
                logging.exception("Failed to stop the task sandbox of CVDP session %s", session_id)
            else:
                del self._sessions[session_id]
        self._fences.clear()

    async def _harvest(self, session: _AgentSession) -> Dict[str, str]:
        """Copy the files the agent wrote out of the task sandbox, then stop the sandbox.

        The first verify that harvests stops the sandbox; later ones grade the same files. A workspace with none of
        the harvested paths yields no files, and grading falls back to the response text as the legacy agentic path
        did. Raises ``_HarvestUnavailable`` when the provider fails to run the command or transfer the archive, or
        when tar fails (exit 2): its archive may be missing files, and a full disk is not the agent's failure. Files
        the agent made unreadable do not fail it (``_harvest_command``). The sandbox is then kept, so a resent verify
        can try again and close still stops it.
        """
        if session.harvested is None:
            workdir = self.config.agent_workdir.rstrip("/")
            # In the workspace, not /tmp: Apptainer binds the host's /tmp into every sandbox.
            remote_archive = f"{workdir}/.cvdp-harvest-{uuid.uuid4().hex}.tar"
            roots = _harvest_roots(self.config.harvest_globs, session.meta.target_files)
            try:
                result = await session.sandbox.exec(
                    _harvest_command(workdir, roots, remote_archive), cwd="/", timeout_s=self.config.harvest_timeout_s
                )
            except Exception as error:
                raise _harvest_unavailable("archive the workspace", error) from error
            output = (result.stderr or result.stdout or "").strip()
            if result.error_type is not None:
                raise _HarvestUnavailable(
                    PROVIDER_REQUEST_FAILED, f"The task sandbox did not run the harvest command: {result.error_type}"
                )
            if result.return_code in (0, 1):
                if result.return_code == 1:
                    # A process the agent left running wrote to a file while tar read it, or a non-GNU tar skipped
                    # a file it could not read. The archive is complete.
                    logging.warning("tar reported files that changed while it read them: %s", output)
                with tempfile.TemporaryDirectory(prefix="cvdp-harvest-") as scratch:
                    local = Path(scratch) / "workspace.tar"
                    try:
                        await session.sandbox.download(remote_archive, local)
                    except Exception as error:
                        raise _harvest_unavailable("transfer the workspace archive", error) from error
                    try:
                        files = await asyncio.to_thread(
                            _collect_harvest,
                            local,
                            Path(scratch) / "workspace",
                            self.config.harvest_globs,
                            session.seeded,
                            session.meta.target_files,
                        )
                    except tarfile.TarError as error:
                        raise _HarvestUnavailable(
                            VERIFIER_ERROR, f"The workspace archive could not be read: {error}"
                        ) from error
            elif result.return_code == _NOTHING_TO_HARVEST_EXIT:
                logging.warning("The agent left none of the harvested paths: %s", output)
                files = {}
            elif result.return_code == 2:
                # Not an unreadable file (the command tolerates those) but a failed write. Python's tarfile reads an
                # archive cut short at a header as complete, so grading it could silently drop files. Mask instead.
                raise _HarvestUnavailable(VERIFIER_ERROR, f"tar failed to archive the workspace: {output[:2000]}")
            else:
                # Not tar's own outcome: the image lacks tar, or the shell itself failed.
                raise _HarvestUnavailable(
                    VERIFIER_ERROR, f"The harvest command exited {result.return_code}: {output[:2000]}"
                )
            session.harvested = files
            try:
                await self._release_sandbox(session)
            except Exception:
                # The files are already out, so a failed stop must not cost their grade. The sandbox stays on the
                # session, and close stops it, raising if that fails too so the caller can retry.
                logging.exception("Failed to stop the task sandbox after the harvest")
        return session.harvested

    @staticmethod
    def _ungraded(body: CVDPVerifyRequest, masked: Dict[str, Any]) -> CVDPVerifyResponse:
        """A masked result: the environment, not the model, kept the workspace from being graded."""
        meta = body.verifier_metadata
        categories = meta.get("categories") or []
        return CVDPVerifyResponse(
            **(body.model_dump() | masked),
            task_id=meta.get("task_id"),
            category=categories[0] if categories else None,
            difficulty=categories[1] if len(categories) > 1 else None,
        )

    @classmethod
    def _lost(cls, body: CVDPVerifyRequest) -> CVDPVerifyResponse:
        # Called only when masking: lost_session_verification logs a warning on every call.
        return cls._ungraded(body, lost_session_verification("No live CVDP session: there is no workspace to grade"))

    async def verify(self, request: Request, body: CVDPVerifyRequest) -> CVDPVerifyResponse:
        if not self.config.agent_sandbox_provider or "rtl_files" in body.model_fields_set:
            # Without sessions, or from cvdp_agent's agentic flavor, which harvested its own sandbox.
            return await self._grade(body)
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id not in self._sessions:
            # A session that is gone (a restart, a checkpoint restore) took its sandbox with it; a 0 would read as
            # an agent that wrote nothing.
            return self._lost(body)
        # Only the harvest holds the session's lock, so a close waits for it; grading runs outside the lock.
        async with self._locks.locked(session_id):
            session = self._sessions.get(session_id)
            if session is None or not session.ready:
                # Closed while this verify waited for the lock.
                return self._lost(body)
            try:
                files = await self._harvest(session)
            except _HarvestUnavailable as error:
                logging.warning("Masking the verify of %s: %s", session.meta.task_id, error)
                return self._ungraded(
                    body,
                    {
                        "reward": 0.0,
                        "mask_sample": True,
                        "failure_kind": error.failure_kind,
                        "failure_reason": str(error),
                    },
                )
        # No files falls back to parsing RTL out of the response text, as the legacy agentic path did.
        return await self._grade(body.model_copy(update={"rtl_files": dict(files) or None}))

    # ---------------------------------------------------------------- grading

    async def _grade(self, body: CVDPVerifyRequest) -> CVDPVerifyResponse:
        meta = CVDPVerifierMetadata.model_validate(body.verifier_metadata)

        category, difficulty = (
            meta.categories[0],
            meta.categories[1],
        )  # categories is [category_id, difficulty], e.g. ["cid003", "medium"]
        category_num = int(category[3:])  # "cid003" -> 3

        model_out = body.response.output_text

        # Fallback: if output_text is empty, try extracting RTL from reasoning content.
        # Mirrors cvdp's logic: if message.content is None, fall back to message.reasoning_content.
        if not model_out or not model_out.strip():
            for item in body.response.output:
                if getattr(item, "type", None) == "reasoning":
                    reasoning_texts = [s.text for s in (item.summary or []) if s.text]
                    if reasoning_texts:
                        model_out = "\n".join(reasoning_texts)
                        break

        has_model_output = bool(model_out and model_out.strip())
        if not has_model_output:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                extracted_rtl=None,
                container_exit_code=None,
                container_stderr=None,
                container_services=None,
                execution_time=0.0,
                parse_failed=False,
            )

        # Route: code-comprehension categories use subjective scoring,
        # code-generation categories use the docker-compose harness.
        if category_num in CODE_COMPREHENSION_CATEGORIES:
            return self._verify_subjective(body, meta, category, difficulty, category_num, model_out)

        return await self._verify_objective(body, meta, category, difficulty, model_out)

    def _verify_subjective(
        self,
        body: CVDPVerifyRequest,
        meta: CVDPVerifierMetadata,
        category: str,
        difficulty: str,
        category_num: int,
        model_out: str,
    ) -> CVDPVerifyResponse:
        """
        Subjective scoring for code-comprehension categories (6, 8, 9, 10).

        Mirrors repository.sbj() + dataset_processor.run_subjective_scoring():
        - Categories 6, 8: BLEU/ROUGE n-gram scoring
        - Categories 9, 10: Also BLEU/ROUGE (LLM subjective scoring requires a
          separate judge model endpoint which is not wired up here; BLEU/ROUGE
          serves as the default fallback, matching CVDP's behavior when no
          sbj_llm_model is configured)
        """
        reference = meta.subjective_reference or ""
        if not reference:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                container_stderr="No subjective_reference provided for code-comprehension category",
            )

        n_gram = N_GRAM_DEFAULT
        bleu_val = calculate_BLEU(model_out.strip(), reference, n_gram)
        rouge_val = calculate_ROUGE(model_out.strip(), reference, n_gram)

        # Score-based categories (6, 8, 9, 10) return the BLEU score directly
        # as a fractional reward, matching CVDP's SCORING_MODE_SCORE behavior.
        if is_score_based_category(category_num):
            reward = bleu_val
        else:
            # Threshold-based: both ROUGE and BLEU must pass
            rouge_pass = rouge_val > ROUGE_THRESHOLD
            bleu_pass = bleu_val > BLEU_THRESHOLD
            reward = 1.0 if (rouge_pass and bleu_pass) else 0.0

        return CVDPVerifyResponse(
            **body.model_dump(),
            reward=reward,
            task_id=meta.task_id,
            category=category,
            difficulty=difficulty,
            bleu_score=bleu_val,
            rouge_score=rouge_val,
        )

    async def _verify_objective(
        self,
        body: CVDPVerifyRequest,
        meta: CVDPVerifierMetadata,
        category: str,
        difficulty: str,
        model_out: str,
    ) -> CVDPVerifyResponse:
        """
        Objective scoring for code-generation categories via docker-compose harness.
        """
        # Agentic flow: the agent ran in its own sandbox and reports the files it
        # wrote on disk. Grade those directly. Model-only flow: fall back to
        # parsing RTL out of the model's text response.
        if body.rtl_files:
            rtl_files = dict(body.rtl_files)
        else:
            rtl_files = _parse_model_response(model_out, meta.target_files)

        # If model produced output but parsing failed, signal parse_failed so the
        # agent can retry with a fresh model completion — mirrors CVDP's
        # LLM_RETRY_COUNT loop in dataset_processor.py.
        if rtl_files is None:
            return CVDPVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                task_id=meta.task_id,
                category=category,
                difficulty=difficulty,
                extracted_rtl=None,
                container_exit_code=None,
                container_stderr="parse_failed: could not extract RTL from model output",
                container_services=[],
                execution_time=0.0,
                parse_failed=True,
            )

        async with self._semaphore:
            t0 = time.time()
            exit_code, stderr, service_results = await self._harness.run(
                rtl_files=rtl_files or {},
                harness_files=meta.harness_files,
                task_id=meta.task_id,
                context_files=meta.context_files,
            )
            execution_time = time.time() - t0

        return CVDPVerifyResponse(
            **body.model_dump(),
            reward=1.0 if exit_code == 0 else 0.0,
            task_id=meta.task_id,
            category=category,
            difficulty=difficulty,
            extracted_rtl=rtl_files,
            container_exit_code=exit_code,
            container_stderr=stderr,
            container_services=service_results,
            execution_time=execution_time,
        )


if __name__ == "__main__":
    CVDPResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = CVDPResourcesServer.run_webserver()  # noqa: F401
