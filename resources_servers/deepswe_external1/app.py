# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE-style tasks with committed-patch transfer into a fresh verifier.

The server serves two kinds of seed. An Environment Server such as ``single_agent_turn`` seeds an episode session
with a ``ResourcesSeedSessionRequest`` and hands the agent sandbox to the agent through ``sandbox_access``; the
session lifecycle (verify once, close, expiry, shutdown) is ``SandboxSessionResourcesServer``'s. An agent that runs
the episode itself through ``/run`` seeds with the flat task row and gets the sandbox handle back.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from shutil import rmtree
from time import monotonic
from typing import Annotated, Any, Literal
from uuid import uuid4

import anyio
from fastapi import Request
from pydantic import BaseModel, BeforeValidator, Field, model_validator

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import ENVIRONMENT_SERVER_FAILED, SESSION_LOST, VERIFIER_ERROR
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ReverifyMode,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.deepswe.app import (
    DeepSWEResourcesServer,
    DeepSWESeedSessionResponse,
    DeepSWEVerifyResponse,
    VerifierResult,
    _resolve_repo_path,
)
from resources_servers.deepswe.task_store import task_sandbox_resources
from resources_servers.deepswe_external1.inline_task import InlineTask
from resources_servers.deepswe_external1.task_data import TaskData
from resources_servers.swebench.sandbox_sessions import STOP_TIMEOUT_S, SandboxSessionResourcesServer


logger = logging.getLogger(__name__)


# Applies /logs/artifacts/model.patch in the verifier the way the DeepSWE v1.0 graders did it: first reset every
# file the patch touches to the base commit (or delete it when it does not exist there), then `git apply`. Image
# builds modify tracked files in place (lockfiles, go.sum, generated sources); the agent's commit captures those
# edits together with its own, so applying the diff onto the pristine verifier tree without the reset fails.
APPLY_CANDIDATE_PATCH = r"""
import pathlib, re, subprocess, sys
base = sys.argv[1]
patch = pathlib.Path("/logs/artifacts/model.patch")
paths = set()
header = re.compile(r'^diff --git (?:"a/((?:[^"\\]|\\.)*)"|a/(\S+)) (?:"b/((?:[^"\\]|\\.)*)"|b/(\S+))$')
for line in patch.read_bytes().decode("utf-8", "replace").splitlines():
    if line.startswith("diff --git "):
        match = header.match(line)
        if match:
            paths.update(part for part in match.groups() if part)
for path in sorted(paths):
    if subprocess.run(["git", "cat-file", "-e", f"{base}:{path}"], capture_output=True).returncode == 0:
        subprocess.run(["git", "checkout", base, "--", path], capture_output=True)
    else:
        pathlib.Path(path).unlink(missing_ok=True)
result = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], capture_output=True, text=True)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
"""


class InvalidSubmissionError(RuntimeError):
    """The seeded repository no longer contains a collectable submission."""


class SessionMismatchError(RuntimeError):
    """The seeded agent sandbox is missing or does not match the task being verified."""


VERIFIER_PYTHON_SETUP = """\
set -eu
if ! command -v python3 >/dev/null 2>&1; then
    if [ "$ALLOW_PYTHON_INSTALL" != 1 ]; then
        echo "Verifier Python is missing; a network-disabled verifier needs Python preinstalled in its image." >&2
        exit 1
    fi
    if [ "$(id -u)" != 0 ]; then
        echo "Installing verifier Python requires root; use a verifier image with Python preinstalled." >&2
        exit 1
    fi
    echo "Installing Python in the disposable verifier sandbox."
    if command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
        apt-get -o DPkg::Lock::Timeout=60 -o Acquire::Retries=2 -y --no-install-recommends install python3
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache python3
    elif command -v microdnf >/dev/null 2>&1; then
        microdnf -y install python3
    elif command -v dnf >/dev/null 2>&1; then
        dnf -y install python3
    elif command -v yum >/dev/null 2>&1; then
        yum -y install python3
    else
        echo "No supported package manager for verifier Python; use a verifier image with Python preinstalled." >&2
        exit 1
    fi
fi
python3 --version
"""


class DeepsweExternal1ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.UNSUPPORTED

    is_verifying_golden_patch: bool = False
    task_cpu_multiplier: float = Field(default=2.0, gt=0)
    task_memory_multiplier: float = Field(default=2.0, gt=0)
    sandbox_provider: str
    sandbox_config: dict[str, Any]
    enforce_agent_no_network: bool = True
    sandbox_model_server: ModelServerRef | None = None
    logs_dir: Path = Path("resources_servers/deepswe_external1/logs")
    clear_verifier_logs: bool = False
    include_model_patch_in_response: bool = True
    is_verifying_null_patch: bool = False
    enforce_verifier_no_network: bool = False

    @model_validator(mode="after")
    def one_validation_mode(self) -> DeepsweExternal1ResourcesServerConfig:
        if self.is_verifying_golden_patch and self.is_verifying_null_patch:
            raise ValueError("Golden and null validation modes are mutually exclusive")
        return self


class DeepsweExternal1SeedSessionRequest(TaskData, BaseSeedSessionRequest):
    pass


class DeepsweExternal1VerifyRequest(TaskData, BaseVerifyRequest):
    sandbox_handle: str | None = None


class DeepsweExternal1VerifyResponse(DeepSWEVerifyResponse):
    validation_mode: Literal["agent", "golden", "null"]
    agent_sandbox_id: str | None = None
    verifier_sandbox_id: str | None = None
    golden_execution_time_s: float = 0.0
    failure_stage: str | None = None
    cleanup_errors: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SeededAgentSandbox:
    """What a seed recorded about the agent sandbox it started, for checks at verify."""

    task_id: str
    image: str
    sandbox_id: str
    descriptor: dict[str, Any]


class DeepsweExternal1ResourcesServer(DeepSWEResourcesServer, SandboxSessionResourcesServer):
    """Reuse DeepSWE collection/grading with self-contained task rows.

    Agent sandboxes of both kinds of seed live in ``_session_id_to_sandbox``, so verify, close and shutdown stop
    them. Only an episode session also expires after the sandbox's ``ttl_s``; an agent-run seed that never reaches
    verify keeps its sandbox until a re-seed or shutdown, and the provider reclaims it after ``ttl_s``. Golden- and
    null-patch validation start and stop their own sandboxes and hold no session.
    """

    config: DeepsweExternal1ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        # Skip DeepSWE's initialization, which loads its filesystem task store: these rows carry their whole task.
        SandboxSessionResourcesServer.model_post_init(self, context)
        self._seeded_agent_sandboxes: dict[str, SeededAgentSandbox] = {}

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Stop the session's sandboxes unless verify already did, and fence later seeds.

        The close is the shared base's, whatever DeepSWE defines: both kinds of seed register their sandbox there.
        """
        return await SandboxSessionResourcesServer.close_resources_session(self, request, body)

    def _provider_options(self, *, phase: str) -> dict[str, Any]:
        options = super()._provider_options(phase=phase)
        if phase != "agent" and self.config.enforce_verifier_no_network:
            options["network_policy"] = {"defaultAction": "deny", "egress": []}
        return options

    async def _create_sandbox(
        self,
        task: InlineTask,
        *,
        phase: str,
        files: dict[str, str] | None = None,
        session_id: str | None = None,
        verifier_session_id: str | None = None,
    ) -> AsyncSandbox:
        global_config = get_global_config_dict()
        definition = task.data
        agent = phase == "agent"
        phase_limits = definition.agent if agent else definition.verifier
        resources = task_sandbox_resources(task, phase=phase)
        resources["cpu"] *= self.config.task_cpu_multiplier
        resources["memory_mib"] = ceil(resources["memory_mib"] * self.config.task_memory_multiplier)
        resources.update(self.config.sandbox_config.get("resources", {}))
        spec = SandboxSpec(
            image=definition.image if agent else definition.verifier_image,
            workdir=definition.workdir,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            env=phase_limits.env | dict(self.config.sandbox_config.get("env", {})),
            files=files or {},
            metadata=resolve_provider_metadata(self.config.sandbox_provider, global_config)
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {"task": definition.task_id[:63].rstrip("._-"), "phase": phase, "nemo_gym_agent": self.config.name},
            resources=SandboxResources.from_mapping(resources),
            provider_options=self._provider_options(phase=phase),
        )
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config))
        # Register before startup awaits so cancelled setup remains reachable by session cleanup.
        if session_id is not None:
            self._session_id_to_sandbox[session_id] = sandbox
        if verifier_session_id is not None:
            self._verifier_sandboxes[verifier_session_id] = sandbox

        async def setup(started: AsyncSandbox) -> None:
            command = (
                # The workdir may be a symlink to the real clone (e.g. /app -> /workspace/repo), so compare real paths.
                f"set -eu; cd {definition.workdir}; git config --global --add safe.directory {definition.workdir}; "
                f'git config --global --add safe.directory "$(readlink -f {definition.workdir})"; '
                f'test "$(readlink -f "$(git rev-parse --show-toplevel)")" = "$(readlink -f {definition.workdir})"; '
                f"git cat-file -e {definition.base_commit}^{{commit}}; "
            )
            if agent:
                command += (
                    "git cat-file -e HEAD^{commit}; "
                    "git config --global user.email agent@nemo-gym.local; "
                    "git config --global user.name 'NeMo Gym Agent'"
                )
            else:
                command += "mkdir -p /tests /logs/artifacts /logs/verifier"
            result = await started.exec(command, timeout_s=60)
            if result.return_code != 0:
                raise RuntimeError(f"{phase} image setup failed: {(result.stderr or '')[-2000:]}")
            if not agent:
                await self._ensure_verifier_python(started)

        await sandbox.start_with_setup(spec, setup)
        return sandbox

    async def _ensure_verifier_python(self, sandbox: AsyncSandbox) -> None:
        # Run only in fresh B, before executing tests or applying the candidate patch.
        # Never relax an explicitly requested no-network policy to install packages.
        allow_install = int(not self.config.enforce_verifier_no_network)
        result = await sandbox.exec(
            f"ALLOW_PYTHON_INSTALL={allow_install}\n" + VERIFIER_PYTHON_SETUP,
            timeout_s=300,
        )
        details = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.return_code != 0:
            raise RuntimeError(f"Verifier Python setup failed (exit {result.return_code}): {details[-4000:]}")
        logger.info("Verifier Python setup: %s", details)

    async def _release_sandbox(self, sandbox: AsyncSandbox, *, task_id: str, phase: str) -> bool:
        """Stop a golden- or null-validation sandbox, which no session owns, and say whether it stopped.

        The stop is shielded and bounded, so a cancelled request still stops it instead of leaving it to ``ttl_s``.
        """
        try:
            with anyio.CancelScope(shield=True), anyio.fail_after(STOP_TIMEOUT_S):
                await sandbox.stop()
            return True
        except Exception:
            logger.exception("Could not release %s sandbox for %s", phase, task_id)
            return False

    async def _stop_session_verifier(self, session_id: str, *, task_id: str) -> bool:
        """Stop the verifier registered with the session, trying twice, and say whether it stopped.

        Both tries run before the verdict is made, so ``cleanup_errors`` names a verifier only when it is still
        running after them. The session keeps it then, and the verify wrapper, an episode's close or shutdown stops it.
        """
        for _ in range(2):
            try:
                await self._stop_verifier_sandbox(session_id)
                return True
            except Exception:
                logger.exception("Could not release verifier sandbox for %s", task_id)
        return False

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[DeepsweExternal1SeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> DeepSWESeedSessionResponse | ResourcesSeedSessionResponse:
        """Start the task's agent sandbox.

        An Environment Server seeds an episode session and gets the sandbox back as ``sandbox_access``. An agent's
        ``/run`` seeds with the task row and gets the sandbox handle; repeating that seed replaces the sandbox. A body
        that names a ``resources_session_id`` is an episode seed and must validate as one (422 otherwise).
        """
        if self.config.is_verifying_golden_patch or self.config.is_verifying_null_patch:
            raise ResourcesSeedRejectedError("seed_session is unavailable in golden/null-validation mode")
        if isinstance(body, ResourcesSeedSessionRequest):
            return await self.seed_task_sandbox_session(request, body, TaskData)
        session_id = str(request.session[SESSION_ID_KEY])
        async with self._locked_task_session(session_id) as record:
            if record.seed is not None or self._task_session_fences.get(session_id) is not None:
                raise ResourcesSessionConflictError("An agent-run seed cannot replace an episode session")
            await self._stop_task_sandbox(session_id)
            try:
                await self._start_task_sandbox(session_id, body)
            except BaseException:
                await self._stop_failed_seed_sandbox(session_id)
                raise
            seeded = self._seeded_agent_sandboxes[session_id]
            return DeepSWESeedSessionResponse(sandbox_handle=seeded.sandbox_id, sandbox_descriptor=seeded.descriptor)

    async def _start_task_sandbox(self, session_id: str, task: BaseModel) -> str:
        """Start and set up the agent sandbox for a task row, and return the repository the agent works in."""
        data = TaskData.model_validate(task)
        self._forget_task_sandbox_state(session_id)
        sandbox = await self._create_sandbox(InlineTask(data), phase="agent", session_id=session_id)
        descriptor = dict(await sandbox.serialize())
        self._seeded_agent_sandboxes[session_id] = SeededAgentSandbox(
            data.task_id, data.image, str(descriptor["sandbox_id"]), descriptor
        )
        return data.workdir

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._seeded_agent_sandboxes.pop(session_id, None)

    async def _execute_golden(self, sandbox: AsyncSandbox, task: InlineTask, log_dir: Path) -> None:
        result = await sandbox.exec(
            "bash /solution/solve.sh", cwd=task.data.workdir, timeout_s=task.data.solution_timeout_sec
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "golden.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
        if result.return_code != 0:
            raise RuntimeError(f"Golden solution failed with exit code {result.return_code}")

    async def verify(self, request: Request, body: DeepsweExternal1VerifyRequest) -> DeepsweExternal1VerifyResponse:
        """Grade the committed patch of the session's agent sandbox, or of a fresh one in a validation mode.

        An episode session is verified once: an identical retry replays the verdict, and a verify after close is
        masked as a lost session.
        """
        task = InlineTask(body)
        if self.config.is_verifying_golden_patch or self.config.is_verifying_null_patch:
            return await self._verify_task(request, body, task)
        response = await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._verify_task(request, body, task),
            lost_response=lambda reason, failure_kind: self._lost_session_response(body, task, reason, failure_kind),
        )
        # The wrapper makes a final cleanup attempt after _verify_task builds and persists the verdict.
        # Its saved replay references this same response, so reconcile both copies before returning to HTTP.
        session_id = request.session[SESSION_ID_KEY]
        if "verifier" in response.cleanup_errors and session_id not in self._verifier_sandboxes:
            response.cleanup_errors.remove("verifier")
            if response.log_dir and not self.config.clear_verifier_logs:
                (Path(response.log_dir) / "result.json").write_text(
                    response.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
        return response

    def _check_seeded_agent_sandbox(
        self, session_id: str, body: DeepsweExternal1VerifyRequest, task: InlineTask
    ) -> str | None:
        """Return the ID of the agent sandbox the session seeded, or raise when it does not match the verify task.

        The shared base already matched an episode session's verify task against its seed. Its retry may find the
        sandbox stopped and return ``None``; the saved capture then holds the sandbox's ID.
        """
        seeded = self._seeded_agent_sandboxes.get(session_id)
        record = self._task_sessions.get(session_id)
        if record is not None and record.seed is not None:
            return seeded.sandbox_id if seeded is not None else None
        if seeded is None or session_id not in self._session_id_to_sandbox:
            raise SessionMismatchError("No seeded agent sandbox exists for this session")
        if seeded.task_id != task.data.task_id or seeded.image != task.data.image:
            raise SessionMismatchError("Seeded session task/image does not match verification")
        if body.sandbox_handle is not None and body.sandbox_handle != seeded.sandbox_id:
            raise SessionMismatchError("Sandbox handle does not match the seeded session")
        return seeded.sandbox_id

    def _lost_session_response(
        self, body: DeepsweExternal1VerifyRequest, task: InlineTask, reason: str, failure_kind: str
    ) -> DeepsweExternal1VerifyResponse:
        return DeepsweExternal1VerifyResponse.model_validate(
            body.model_dump(include={"responses_create_params", "response", "image", "sandbox_handle"})
            | {
                "task_id": task.data.task_id,
                "validation_mode": "agent",
                "evaluation_completed": False,
                "verifier_error": reason,
                "model_patch_sha256": hashlib.sha256(b"").hexdigest(),
                "model_patch_bytes": 0,
                "log_dir": "",
                "patch_collection_time_s": 0.0,
                "sandbox_start_time_s": 0.0,
                "verification_time_s": 0.0,
                "failure_stage": "agent_setup",
            }
            | lost_session_verification(reason, failure_kind=failure_kind)
        )

    async def _collect_model_patch(self, sandbox: AsyncSandbox, task: InlineTask) -> bytes:
        try:
            return await super()._collect_model_patch(sandbox, task)
        except RuntimeError as error:
            # Setup already validated this repository/base. Confirm submission damage
            # rather than blaming the agent for a transport failure or missing artifact.
            integrity = await sandbox.exec(
                f'test "$(readlink -f "$(git -C {task.data.workdir} rev-parse --show-toplevel)")" = "$(readlink -f {task.data.workdir})" && '
                f"git -C {task.data.workdir} cat-file -e {task.data.base_commit}^{{commit}} && "
                f"git -C {task.data.workdir} cat-file -e HEAD^{{commit}}",
                timeout_s=30,
            )
            if integrity.return_code != 0 and not integrity.error_type:
                raise InvalidSubmissionError("Submission repository, base commit or HEAD is unavailable") from error
            raise

    def _verifier_workdir(self, task: InlineTask) -> str:  # type: ignore[override]
        # Rows declare /app or /workspace/repo; test.sh and grader.py of each task already target that path.
        return task.data.workdir

    async def _stage_verifier(self, sandbox: AsyncSandbox, task: InlineTask, model_patch: bytes) -> None:
        # Files were supplied through SandboxSpec.files at B's creation, like Swemer.
        result = await sandbox.exec(
            "python3 -I -c 'import base64; from pathlib import Path; "
            'Path("/logs/artifacts/model.patch").write_bytes(base64.b64decode('
            'Path("/logs/artifacts/model.patch.b64").read_bytes(), validate=True))'
            "' && "
            "chmod 0755 /tests/test.sh /tests/grader.py",
            timeout_s=60,
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to prepare DeepSWE verifier: {result.stderr or ''}")
        if model_patch and not task.data.grader_applies_model_patch:
            # v1.1 graders grade the working tree as-is, so the candidate patch has to be in place before test.sh.
            # A patch that does not apply at the base commit is graded as such (the base tree fails FAIL_TO_PASS),
            # the same outcome the v1.0 graders give a non-applying model.patch.
            applied = await sandbox.exec(
                f"cd {task.data.workdir} && python3 -I - {task.data.base_commit} <<'NEMO_GYM_APPLY'\n"
                f"{APPLY_CANDIDATE_PATCH}\nNEMO_GYM_APPLY",
                timeout_s=300,
            )
            if applied.return_code != 0:
                logger.warning(
                    "Task %s: candidate patch did not apply in the verifier; grading the base tree: %s",
                    task.data.task_id,
                    ((applied.stderr or "") + (applied.stdout or ""))[-2000:],
                )

    async def _verify_task(
        self, request: Request, body: DeepsweExternal1VerifyRequest, task: InlineTask
    ) -> DeepsweExternal1VerifyResponse:
        mode: Literal["agent", "golden", "null"] = "agent"
        if self.config.is_verifying_golden_patch:
            mode = "golden"
        elif self.config.is_verifying_null_patch:
            mode = "null"
        task_id = task.data.task_id
        session_id = str(request.session.get(SESSION_ID_KEY, "validation"))
        # Session IDs are untrusted path components; each attempt gets its own generated log directory.
        log_dir = _resolve_repo_path(self.config.logs_dir) / task_id / uuid4().hex
        model_patch = b""
        agent_sandbox: AsyncSandbox | None = None
        verifier_sandbox: AsyncSandbox | None = None
        agent_id = verifier_id = None
        collect_time = start_time = verify_time = golden_time = 0.0
        result = VerifierResult(evaluation_completed=False, reward=0.0)
        cleanup_errors: list[str] = []
        invalid_submission = False
        failure_kind = VERIFIER_ERROR
        failure_stage: str | None = "agent_setup"
        try:
            if mode == "agent":
                try:
                    agent_id = self._check_seeded_agent_sandbox(session_id, body, task)
                    failure_stage = "patch_collection"
                    started = monotonic()
                    seeded_id = agent_id

                    async def capture() -> tuple[str | None, bytes]:
                        sandbox = self._session_id_to_sandbox[session_id]
                        return seeded_id, await self._collect_model_patch(sandbox, task)

                    # An episode session keeps the capture with its sandbox's ID, so a verify cancelled after it
                    # reports the same sandbox and patch on retry. A failed stop below is masked instead, and that
                    # verdict is the one a retry replays.
                    agent_id, model_patch = await self.capture_task_sandbox_patch(session_id, capture)
                    collect_time = monotonic() - started
                finally:
                    for attempt in range(2):
                        try:
                            await self._stop_task_sandbox(session_id)
                            break
                        except Exception:
                            logger.exception("Could not release agent sandbox for %s", task_id)
                            if attempt == 1:
                                # Keep ownership so close, expiry or shutdown can retry.
                                cleanup_errors.append("agent")
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "model.patch").write_bytes(model_patch)
            else:
                agent_sandbox = await self._create_sandbox(
                    task, phase="agent", files=task.data.files.solution_files() if mode == "golden" else {}
                )
                agent_id = str((await agent_sandbox.serialize())["sandbox_id"])

                try:
                    if mode == "golden":
                        failure_stage = "golden_execution"
                        started = monotonic()
                        await self._execute_golden(agent_sandbox, task, log_dir)
                        golden_time = monotonic() - started
                    failure_stage = "patch_collection"
                    started = monotonic()
                    model_patch = await self._collect_model_patch(agent_sandbox, task)
                    collect_time = monotonic() - started
                    if mode == "golden" and task.data.files.solution_patch and not model_patch:
                        raise RuntimeError("Golden solution produced no committed patch")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    (log_dir / "model.patch").write_bytes(model_patch)
                finally:
                    released = await self._release_sandbox(agent_sandbox, task_id=task_id, phase="agent")
                    agent_sandbox = None
                    if not released:
                        cleanup_errors.append("agent")

            if cleanup_errors:
                failure_stage = "agent_cleanup"
                raise RuntimeError("Agent sandbox cleanup failed; verifier was not started")

            failure_stage = "verifier_setup"
            started = monotonic()
            verifier_sandbox = await self._create_sandbox(
                task,
                phase="verifier",
                files=task.data.files.verification_files(model_patch),
                verifier_session_id=session_id if mode == "agent" else None,
            )
            verifier_id = str((await verifier_sandbox.serialize())["sandbox_id"])
            if verifier_id == agent_id:
                raise RuntimeError("Provider reused the agent sandbox as the verifier")
            start_time = monotonic() - started
            failure_stage = "native_verifier"
            started = monotonic()
            result = await self._run_verifier(verifier_sandbox, task, model_patch, log_dir)
            verify_time = monotonic() - started
            if result.evaluation_completed:
                failure_stage = None
        except Exception as error:
            invalid_submission = mode == "agent" and isinstance(error, InvalidSubmissionError)
            # Name what the evidence shows: a lost or mismatched session, or a submission that could not be
            # collected from the agent sandbox. Anything else stays a verifier error. Collecting the golden
            # solution is the verifier's own work, so only agent mode counts as a collection failure.
            if isinstance(error, SessionMismatchError):
                failure_kind = SESSION_LOST
            elif mode == "agent" and failure_stage == "patch_collection":
                failure_kind = ENVIRONMENT_SERVER_FAILED
            logger.exception("Task %s failed during %s", task_id, failure_stage)
            result = VerifierResult(
                evaluation_completed=False, reward=0.0, verifier_error=f"{type(error).__name__}: {error}"
            )
        finally:
            if agent_sandbox is not None:
                if not await self._release_sandbox(agent_sandbox, task_id=task_id, phase="agent"):
                    cleanup_errors.append("agent")
            if mode == "agent":
                # Setup may fail before _create_sandbox returns its already registered sandbox.
                if not await self._stop_session_verifier(session_id, task_id=task_id):
                    cleanup_errors.append("verifier")
            elif verifier_sandbox is not None:
                if not await self._release_sandbox(verifier_sandbox, task_id=task_id, phase="verifier"):
                    cleanup_errors.append("verifier")

        response = DeepsweExternal1VerifyResponse.model_validate(
            body.model_dump(include={"responses_create_params", "response", "image", "sandbox_handle"})
            | result.model_dump()
            | {
                "task_id": task_id,
                "validation_mode": mode,
                "agent_sandbox_id": agent_id,
                "verifier_sandbox_id": verifier_id,
                "golden_execution_time_s": golden_time,
                "model_patch": model_patch.decode("utf-8", errors="replace")
                if self.config.include_model_patch_in_response
                else None,
                "model_patch_sha256": hashlib.sha256(model_patch).hexdigest(),
                "model_patch_bytes": len(model_patch),
                "log_dir": str(log_dir),
                "patch_collection_time_s": collect_time,
                "sandbox_start_time_s": start_time,
                "verification_time_s": verify_time,
                "mask_sample": not result.evaluation_completed and not invalid_submission,
                "failure_kind": "deepswe_external1:invalid_submission"
                if invalid_submission
                else (failure_kind if not result.evaluation_completed else None),
                "failure_reason": result.verifier_error,
                "failure_stage": failure_stage,
                "cleanup_errors": cleanup_errors,
            }
        )
        if self.config.clear_verifier_logs:
            try:
                rmtree(log_dir)
            except FileNotFoundError:
                response.log_dir = ""
            except OSError:
                logger.exception("Could not remove verifier logs for %s", task_id)
                response.cleanup_errors.append("verifier_logs")
            else:
                response.log_dir = ""
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "result.json").write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return response


if __name__ == "__main__":
    DeepsweExternal1ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = DeepsweExternal1ResourcesServer.run_webserver()  # noqa: F401
