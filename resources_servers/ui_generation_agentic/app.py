# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Agentic UI generation: a coding agent writes one UI entry file from scratch in a per-stack sandbox.

The server lends one sandbox per episode (``LentSandboxes``, one per stack image) to a harness such as OpenCode
through ``single_agent_turn``, and grades in that same sandbox once the agent session has closed:

- G1 stops every process the agent left (detached dev servers, watchers). The sandbox runtime is recognized by the
  process identities (PID and start time) recorded at seed, before the agent ran, never by a name the agent could copy;
- G2 snapshots the entry file once (a regular file within the size cap); only the snapshot is built and judged;
- G3 checks the image's root-owned, read-only template and toolchain against their manifest (the manifest's own
  pinned SHA-256 is checked at seed);
- G4 builds the snapshot in a fresh copy of that template with a build script uploaded at verify time. The build
  links the template's installed packages, has private scratch directories, and fails when the entry reads any file
  outside the build directory and those packages, so nothing the agent wrote elsewhere reaches the build;
- G5 (``guard_mode: full`` only) runs the build as a separate unprivileged user under a root-owned directory;
- G6 takes the verdict from the build's exit status and the judge only.

``guard_mode: detection_only`` is for sandboxes whose commands cannot switch user: G1 then targets the command user's
own processes, and G5 is skipped. The reward is ``build × score / 4``, where ``reward_score`` picks the judge score.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shlex
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Annotated, Any, ClassVar, Literal, Optional

import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from nemo_gym.agents import restricted_network_policy
from nemo_gym.client import SESSION_ID_KEY, get_server_url
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    PROVIDER_REQUEST_FAILED,
    SESSION_LOST,
    VERIFIER_ERROR,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxAccess, SandboxConfig, resolve_provider_config
from nemo_gym.server_runtime import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    LentSandboxes,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
    observe_llm_judge_grading,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import call_judge, record_parsed_judge_verdict
from resources_servers.ui_generation_agentic.judge import MAX_SCORE, judge_case, parse_judge_completion
from resources_servers.ui_generation_agentic.task_data import UiGenTaskFields


LOG = logging.getLogger(__name__)
_HERE = Path(__file__).parent
VERIFY_SCRIPT = _HERE / "scripts" / "ui_gen_verify.sh"
TEMPLATE_DIR = "/opt/ui-gen/template"
MANIFEST = "/opt/ui-gen/MANIFEST.sha256"
BUILT_STAMP = "/opt/ui-gen/.built"
# Exit code of ui_gen_verify.sh when the verifier itself could not set up the build (not the model's outcome). 126 and
# 127 (the shell could not run the script or the bundler) are setup failures too.
VERIFIER_SETUP_EXITS = frozenset({90, 126, 127})
LOG_TAIL_CHARS = 8000
G1_ROUNDS = 10

# One line per process: "<pid> <uid> <ppid> <pgid> <start time> <state>", read from /proc (POSIX sh, no ps needed).
# The fields after the command name's closing parenthesis start at the state; the start time is field 22 of stat.
_PROCS = r"""procs() {
  for d in /proc/[0-9]*; do
    s=$(cat "$d/stat" 2>/dev/null) || continue
    u=$(stat -c %u "$d" 2>/dev/null) || continue
    set -- ${s##*) }
    echo "${d#/proc/} $u $2 $3 ${20} $1"
  done
}"""

# Seed: the identities ("<pid>:<start time>") of every process that runs before the agent does (the sandbox runtime:
# its exec daemon, bootstrap and keep-alive), and the manifest's SHA-256.
_BASELINE = (
    _PROCS
    + '\nprocs | while read -r pid u ppid pg start state; do echo "proc=$pid:$start"; done\n'
    + f"echo \"manifest_sha=$(sha256sum {MANIFEST} 2>/dev/null | cut -d' ' -f1)\""
)

# G1: kill every live process of the target user that is not a recorded runtime identity, not PID 1 and not this
# command (its ancestors, and the processes of its own group started after it), in up to G1_ROUNDS rounds so that a
# process forking in between is caught too. __UID__ sets $uid; __KEEP__ is the recorded identities.
_G1 = (
    "__UID__\n"
    'keep=" __KEEP__ "\n'
    + _PROCS
    + r"""
s=$(cat /proc/$$/stat); set -- ${s##*) }; mypg=$3; mystart=${20}
ancestors=" "; p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  ancestors="$ancestors$p "
  s=$(cat /proc/$p/stat 2>/dev/null) || break
  set -- ${s##*) }; p=$2
done
victims() {
  procs | while read -r pid u ppid pg start state; do
    [ "$u" = "$uid" ] || continue
    [ "$pid" = 1 ] && continue
    [ "$state" = Z ] && continue
    case "$keep" in *" $pid:$start "*) continue ;; esac
    case "$ancestors" in *" $pid "*) continue ;; esac
    if [ "$pg" = "$mypg" ] && [ "$start" -ge "$mystart" ]; then continue; fi
    echo "$pid"
  done
}
killed=0
for round in __ROUNDS__; do
  v=$(victims)
  [ -z "$v" ] && break
  killed=$((killed + $(echo $v | wc -w)))
  kill -9 $v 2>/dev/null
  sleep 0.3
done
left=$(victims)
for pid in $left; do echo "survivor=$pid $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-200)"; done
echo "killed=$killed left=$(echo $left | wc -w)"
"""
)
_IDENTITY = re.compile(r"[0-9]+:[0-9]+")
_WORK_DIR = re.compile(r"/(tmp/ui-gen-verify|opt/ui-gen-verify/run)\.[A-Za-z0-9]+")


class StackConfig(BaseModel):
    """One task stack: its agent image, where the agent works and which file it must write."""

    model_config = ConfigDict(extra="forbid")

    image: Optional[str] = Field(description="The stack's agent image; a seed of a stack without one is refused.")
    workdir: str
    entry_path: str = Field(description="The deliverable, relative to workdir; it does not exist at seed.")
    language: str = Field(description="Fence language of the entry file for the judge, e.g. tsx or html.")
    description: str = Field(description="One-line description of the stack, shown to the judge.")
    agent_user: str = Field(description="The image user the agent runs as; G1 targets it in guard_mode full.")
    manifest_sha256: Optional[str] = Field(default=None, description="Pinned SHA-256 of the image's manifest file.")


class UiGenerationAgenticConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    sandbox_provider: str
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    stacks: dict[str, StackConfig]
    agent_network_mode: Literal["restricted", "open"] = Field(
        default="restricted",
        description="restricted: deny all egress except the model server and extra_allowed_hosts (OpenSandbox only).",
    )
    model_server: ModelServerRef = Field(description="The policy model server the harness calls from the sandbox.")
    extra_allowed_hosts: list[str] = Field(default_factory=lambda: ["fonts.googleapis.com", "fonts.gstatic.com"])
    egress_enforcement: Literal["dns+nft", "dns", "unknown"] = Field(
        default="unknown", description="What the sandbox deployment enforces, from the probe; stamped on rollouts."
    )
    guard_mode: Literal["full", "detection_only"] = Field(
        default="detection_only",
        description="full needs commands to switch user (root, verifier_user); detection_only does not.",
    )
    verifier_user: str = "ui-verify"
    max_deliverable_bytes: int = Field(
        default=262_144,
        gt=0,
        description="Larger entry files score 0 (invalid_deliverable); the judge always sees the whole file.",
    )
    build_timeout_s: int = Field(default=600, gt=0)
    inconclusive_verification_retries: int = Field(default=2, ge=0)
    reward_score: Literal["quality", "prompt_adherence", "mean", "min"] = Field(
        default="quality",
        description="The judge score a built file earns: quality, prompt_adherence, their mean or their minimum.",
    )
    judge_model_server: ModelServerRef
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_prompt_path: str = str(_HERE / "prompts" / "judge.yaml")
    judge_max_concurrency: int = Field(default=8, ge=1)
    judge_max_attempts: int = Field(default=4, ge=1)


class UiGenTask(UiGenTaskFields, BaseSeedSessionRequest):
    """The task fields this server reads from ``task_data`` at seed (other fields are ignored)."""

    model_config = ConfigDict(extra="ignore")


class UiGenVerifyRequest(UiGenTaskFields, BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class UiGenVerifyResponse(BaseMultiRewardVerifyResponse):
    model_config = ConfigDict(extra="allow")

    build_passed: bool = False
    build_exit_code: Optional[int] = None
    build_seconds: float = 0.0
    build_log_tail: str = ""
    final_code: Optional[str] = None
    final_code_sha256: Optional[str] = None
    tamper_detected: bool = False
    tamper_details: list[str] = Field(default_factory=list)
    judge_quality: Optional[int] = None
    judge_prompt_adherence: Optional[int] = None
    judge_verdict: Optional[dict[str, Any]] = None
    judge_reasoning: Optional[str] = None
    guard_mode: str = ""
    agent_network_mode: str = ""
    egress_enforcement: str = ""
    failure_reason: Optional[str] = None
    agent_stop_reason: Optional[str] = None


def agent_stop_reason(response: NeMoGymResponse) -> Optional[str]:
    """The harness's run status when the run did not complete, with the cause when the harness reports one.

    OpenCode stops a run whose model call was cut off at ``max_output_tokens`` as ``incomplete``. It names the cause
    (``incomplete:max_output_tokens``) only for continuation sessions, so for a single-activation run ``incomplete``
    covers both an output-cap cut-off and the agent timeout.
    """
    if response.status in (None, "completed"):
        return None
    reason = response.incomplete_details.reason if response.incomplete_details else None
    return f"{response.status}:{reason}" if reason else str(response.status)


@dataclass
class _Session:
    task: UiGenTask
    # Identities ("<pid>:<start time>") of the processes that ran in the sandbox at seed, before the agent.
    runtime: tuple[str, ...]


class _Inconclusive(Exception):
    """The sandbox could not answer (provider or exec failure): mask the sample, never score it."""

    def __init__(self, failure_kind: str, reason: str) -> None:
        super().__init__(reason)
        self.failure_kind = failure_kind


@dataclass
class _Grade:
    build_passed: bool = False
    build_exit_code: Optional[int] = None
    build_seconds: float = 0.0
    build_log_tail: str = ""
    final_code: Optional[str] = None
    final_code_sha256: Optional[str] = None
    tamper_details: list[str] = field(default_factory=list)
    failure_reason: Optional[str] = None


class UiGenerationAgenticResourcesServer(SimpleResourcesServer):
    """Lends a stack sandbox per episode through the base session-state hooks and grades the entry file in it."""

    config: UiGenerationAgenticConfig
    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = UiGenTask

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        return "lent sandboxes and their sessions live in one worker's memory"

    def model_post_init(self, context: Any) -> None:
        if "network_policy" in self.config.sandbox_config.provider_options:
            raise ValueError("Set agent_network_mode, not sandbox_config.provider_options.network_policy")
        if self.config.agent_network_mode == "restricted":
            provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            provider_name = next(iter(provider_config))
            if provider_name != "opensandbox":
                raise ValueError(
                    f"agent_network_mode restricted needs the opensandbox provider (got {provider_name}); "
                    "use agent_network_mode open with other providers"
                )
        self._judge_rubric: str = yaml.safe_load(Path(self.config.judge_prompt_path).read_text())["system"]
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        self._lenders: dict[str, LentSandboxes] = {}
        self._sessions: dict[str, _Session] = {}
        self._sandbox_access: dict[str, SandboxAccess] = {}
        super().model_post_init(context)

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

    async def shutdown(self) -> None:
        """Stop every sandbox still lent, all stacks at once (the supervisor gives the server little time to exit)."""
        await asyncio.gather(*(lender.release_all() for lender in self._lenders.values()))
        self._sessions.clear()
        self._sandbox_access.clear()

    # ------------------------------------------------------------------ sessions

    def _lender(self, stack_name: str) -> LentSandboxes:
        """The stack's sandbox lender, built on first use (the model server's address is known only at run time)."""
        if stack_name in self._lenders:
            return self._lenders[stack_name]
        stack = self.config.stacks[stack_name]
        sandbox_config = self.config.sandbox_config
        if self.config.agent_network_mode == "restricted":
            # LentSandboxes(network_access="model_only") allows only the model server; the pages also load fonts.
            urls = [get_server_url(self.config.model_server.name)]
            urls += [f"https://{host}" for host in self.config.extra_allowed_hosts]
            policy = restricted_network_policy("opensandbox", urls)
            options = {**sandbox_config.provider_options, "network_policy": policy}
            sandbox_config = sandbox_config.model_copy(update={"provider_options": options})
        lender = LentSandboxes(self.config.sandbox_provider, sandbox_config, image=stack.image, workdir=stack.workdir)
        self._lenders[stack_name] = lender
        return lender

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Start the task's stack sandbox (no entry file yet), record its runtime processes and check its manifest."""
        assert isinstance(body, UiGenTask)
        identity = self.session_identity(session_id)
        if identity is None:
            # A cookie seed comes from an agent's own /run: no harness would receive the sandbox.
            raise ResourcesSeedRejectedError(
                "This server lends a sandbox only to an Environment Server's episode seed"
            )
        stack = self.config.stacks.get(body.stack)
        if stack is None or body.entry_path != stack.entry_path:
            raise ResourcesSeedRejectedError(f"Unknown stack or entry path: {body.stack} / {body.entry_path}")
        if not stack.image:
            raise ResourcesSeedRejectedError(
                f"Stack {body.stack} has no image: set UI_GEN_{body.stack.removeprefix('ui-').upper()}_IMAGE"
            )
        lender = self._lender(body.stack)
        # On failure lend() stops the sandbox it started, and the base calls release_session_state.
        self._sandbox_access[session_id] = await lender.lend(session_id, identity[1])
        sandbox = lender.get(session_id)
        assert sandbox is not None
        result = await sandbox.exec(_BASELINE, cwd="/", timeout_s=120)
        lines = (result.stdout or "").splitlines()
        runtime = tuple(line[5:] for line in lines if line.startswith("proc=") and _IDENTITY.fullmatch(line[5:]))
        if result.error_type is not None or result.return_code != 0 or not runtime:
            raise RuntimeError(
                f"Could not record the sandbox's runtime processes: {result.error_type or result.stderr}"
            )
        manifest_sha = next((line[13:] for line in lines if line.startswith("manifest_sha=")), "")
        if stack.manifest_sha256 and manifest_sha != stack.manifest_sha256:
            # The template is root-owned and read-only, so a mismatch means a wrong pin or another image build.
            raise ResourcesSeedRejectedError(
                f"The {body.stack} image's manifest SHA-256 {manifest_sha or '(missing)'} is not the pinned "
                f"{stack.manifest_sha256}: check the image and its UI_GEN_*_MANIFEST_SHA256 pin"
            )
        self._sessions[session_id] = _Session(task=body, runtime=runtime)

    async def release_session_state(self, session_id: str) -> None:
        # A failed stop raises and keeps the sandbox, so a later verify, close, retirement or the shutdown retries it.
        # Releasing through every lender also stops a sandbox that a failed seed left behind.
        for lender in self._lenders.values():
            await lender.release(session_id)
        self._sessions.pop(session_id, None)
        self._sandbox_access.pop(session_id, None)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """The base seed, plus the lent sandbox's access in the reply of an episode seed."""
        response = await super().seed_session(body, request=request)
        if isinstance(response, ResourcesSeedSessionResponse):
            response.sandbox_access = self._sandbox_access.get(response.resources_session_id)
        return response

    # ------------------------------------------------------------------ verify

    def _stamps(self) -> dict[str, str]:
        return {
            "guard_mode": self.config.guard_mode,
            "agent_network_mode": self.config.agent_network_mode,
            "egress_enforcement": self.config.egress_enforcement
            if self.config.agent_network_mode == "restricted"
            else "none",
        }

    def _response(self, body: UiGenVerifyRequest, fields: dict[str, Any]) -> UiGenVerifyResponse:
        # The computed fields win over request fields of the same name (the request allows extra fields).
        return UiGenVerifyResponse(**(body.model_dump() | fields | self._stamps()))

    def _masked(self, body: UiGenVerifyRequest, failure_kind: str, reason: str) -> UiGenVerifyResponse:
        if failure_kind == SESSION_LOST:
            fields: dict[str, Any] = dict(lost_session_verification(reason))
        else:
            fields = {"reward": 0.0, "mask_sample": True, "failure_kind": failure_kind, "failure_reason": reason}
        zeros = {"build": 0.0, "judge_quality": 0.0, "judge_prompt_adherence": 0.0}
        return self._response(body, {**fields, "reward_components": zeros})

    async def verify(self, request: Request, body: UiGenVerifyRequest) -> UiGenVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            return self._masked(body, SESSION_LOST, "verify without a seeded resources session")
        async with self.verifying(session_id) as verification:
            if verification.is_lost:
                return self._masked(body, SESSION_LOST, "the resources session is gone or closed")
            if verification.is_live:
                session = self._sessions[session_id]
                task = session.task
                sandbox = self._lenders[task.stack].get(session_id)
                if sandbox is None:
                    return self._masked(body, SESSION_LOST, "the lent sandbox is gone")
                try:
                    grade = await self._grade_in_sandbox(sandbox, session)
                except _Inconclusive as error:
                    LOG.warning("Masking verify of session %s: %s", session_id, error)
                    return self._masked(body, error.failure_kind, str(error))
                # A retry after a judge failure grades this evidence again without the sandbox.
                verification.retain({"task": task.model_dump(), "grade": asdict(grade)})
                try:
                    # Stop the sandbox before waiting on the judge; verifying() retries a failed stop on exit.
                    await self.release_session_state(session_id)
                except Exception:
                    LOG.warning("Failed to stop the sandbox of session %s before judging", session_id, exc_info=True)
            else:
                retained = verification.retained
                assert isinstance(retained, dict)
                if "response" in retained:
                    return UiGenVerifyResponse.model_validate(retained["response"])
                task = UiGenTask.model_validate(retained["task"])
                grade = _Grade(**retained["grade"])
            response = await self._score(body, task, grade)
            verification.retain({"response": response.model_dump(mode="json")})
            return response

    async def _exec(self, sandbox: AsyncSandbox, command: str, *, timeout_s: float = 120, user: str | None = None):
        try:
            result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s, user=user)
        except Exception as error:  # the provider could not run the command: no verdict
            raise _Inconclusive(PROVIDER_REQUEST_FAILED, f"exec failed: {type(error).__name__}: {error}") from error
        return result

    async def _grade_in_sandbox(self, sandbox: AsyncSandbox, session: _Session) -> _Grade:
        stack_name = session.task.stack
        stack = self.config.stacks[stack_name]
        full = self.config.guard_mode == "full"
        root = "root" if full else None
        grade = _Grade()

        # G1: stop what the agent left running.
        if full:
            target = (
                f"uid=$(id -u -- {shlex.quote(stack.agent_user)}) || {{ echo 'g1_error=unknown agent user'; exit 0; }}"
            )
        else:
            target = "uid=$(id -u)"
        command = (
            _G1.replace("__UID__", target)
            .replace("__KEEP__", " ".join(session.runtime))
            .replace("__ROUNDS__", " ".join(str(n) for n in range(G1_ROUNDS)))
        )
        result = await self._exec(sandbox, command, user=root)
        stdout = result.stdout or ""
        summary = re.search(r"^killed=(\d+) left=(\d+)$", stdout, re.MULTILINE)
        if result.error_type is not None or summary is None:
            raise _Inconclusive(VERIFIER_ERROR, f"could not stop the agent's processes: {result.stderr or stdout}")
        if int(summary.group(2)) != 0:
            survivors = "; ".join(line[9:] for line in stdout.splitlines() if line.startswith("survivor="))
            raise _Inconclusive(
                VERIFIER_ERROR,
                f"harness_termination_unconfirmed: {summary.group(2)} agent processes survived: {survivors}",
            )

        # G2: snapshot the entry file once.
        entry = f"{stack.workdir.rstrip('/')}/{stack.entry_path}"
        result = await self._exec(sandbox, f"stat -c '%F|%s' {shlex.quote(entry)}", user=root)
        if result.error_type is not None:
            raise _Inconclusive(VERIFIER_ERROR, f"stat failed: {result.error_type}")
        if result.return_code != 0:
            grade.failure_reason = "no_deliverable"
            return grade
        kind, _, size = (result.stdout or "").strip().partition("|")
        if kind != "regular file" or not size.isdigit() or int(size) > self.config.max_deliverable_bytes:
            grade.failure_reason = f"invalid_deliverable: {kind}, {size} bytes"
            return grade
        with tempfile.TemporaryDirectory(prefix="ui-gen-agentic-") as tmp:
            local = Path(tmp) / "entry"
            try:
                await sandbox.download(entry, local)
            except Exception as error:
                raise _Inconclusive(PROVIDER_REQUEST_FAILED, f"download failed: {error}") from error
            data = local.read_bytes()
        if len(data) > self.config.max_deliverable_bytes:
            grade.failure_reason = f"invalid_deliverable: regular file, {len(data)} bytes"
            return grade
        grade.final_code = data.decode("utf-8", errors="replace")
        grade.final_code_sha256 = hashlib.sha256(data).hexdigest()

        # G3: the template and toolchain are unchanged since the image was built.
        check = (
            f"cd / && if out=$(sha256sum -c --quiet {MANIFEST} 2>&1); then echo manifest=ok; "
            "else echo manifest=bad; printf '%s\\n' \"$out\" | head -3 | sed 's/^/bad=/'; fi; "
            f"find {TEMPLATE_DIR} -newer {BUILT_STAMP} -not -path '*/.vite-temp*' -print 2>/dev/null "
            "| head -3 | sed 's/^/newer=/'; echo integrity=done"
        )
        result = await self._exec(sandbox, check, user=root)
        lines = (result.stdout or "").splitlines()
        if (
            result.error_type is not None
            or "integrity=done" not in lines
            or not {"manifest=ok", "manifest=bad"} & set(lines)
        ):
            raise _Inconclusive(
                VERIFIER_ERROR, f"integrity check gave no answer: {result.error_type or result.stderr}"
            )
        if "manifest=bad" in lines:
            failing = [line[4:] for line in lines if line.startswith("bad=")]
            grade.tamper_details.append(f"template or toolchain differs from its manifest: {failing}")
        newer = [line[6:] for line in lines if line.startswith("newer=")]
        if newer:
            grade.tamper_details.append(f"files in the template are newer than the image build: {newer}")
        if grade.tamper_details:
            grade.failure_reason = "tamper_detected"
            return grade

        # G4 + G5: build the snapshot in a fresh copy of the template with a freshly uploaded script.
        last_error = ""
        for _attempt in range(self.config.inconclusive_verification_retries + 1):
            try:
                exit_code, seconds, log_tail = await self._build(sandbox, stack_name, data)
            except _Inconclusive as error:
                last_error = str(error)
                continue
            grade.build_exit_code, grade.build_seconds, grade.build_log_tail = exit_code, seconds, log_tail
            grade.build_passed = exit_code == 0
            if not grade.build_passed:
                grade.failure_reason = "build_timeout" if exit_code == 124 else "build_failed"
            return grade
        raise _Inconclusive(VERIFIER_ERROR, f"inconclusive build after retries: {last_error}")

    async def _build(self, sandbox: AsyncSandbox, stack_name: str, data: bytes) -> tuple[int, float, str]:
        """Run one build of ``data``; return (exit code, seconds, log tail) or raise ``_Inconclusive``."""
        full = self.config.guard_mode == "full"
        root = "root" if full else None
        started = time.monotonic()
        if full:
            # Under a root-owned parent, so no directory the agent could write lies above the build.
            make = "install -d -m 755 -o root -g root /opt/ui-gen-verify && mktemp -d /opt/ui-gen-verify/run.XXXXXX"
        else:
            make = "mktemp -d /tmp/ui-gen-verify.XXXXXX"
        result = await self._exec(sandbox, make, user=root)
        work = (result.stdout or "").strip()
        if result.error_type is not None or result.return_code != 0 or not _WORK_DIR.fullmatch(work):
            raise _Inconclusive(VERIFIER_ERROR, f"mktemp failed: {result.error_type or result.stderr}")
        with tempfile.TemporaryDirectory(prefix="ui-gen-agentic-") as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.write_bytes(data)
            try:
                await sandbox.upload(snapshot, f"{work}/snapshot")
                await sandbox.upload(VERIFY_SCRIPT, f"{work}/verify.sh")
            except Exception as error:
                raise _Inconclusive(PROVIDER_REQUEST_FAILED, f"upload failed: {error}") from error
        quoted = shlex.quote(work)
        prepare = f"mkdir {quoted}/build"
        if full:
            prepare += f" && chown -R {shlex.quote(self.config.verifier_user)} {quoted}"
        result = await self._exec(sandbox, prepare, user=root)
        if result.error_type is not None or result.return_code != 0:
            raise _Inconclusive(VERIFIER_ERROR, f"build directory setup failed: {result.error_type or result.stderr}")
        command = (
            f"bash {quoted}/verify.sh {shlex.quote(stack_name)} {quoted}/snapshot {quoted}/build "
            f"{self.config.build_timeout_s} 2>&1"
        )
        result = await self._exec(
            sandbox,
            command,
            timeout_s=self.config.build_timeout_s + 120,
            user=self.config.verifier_user if full else None,
        )
        # The script's own timeout (with a SIGKILL 10 s after SIGTERM) scores a hanging build as 124. A provider
        # timeout or another exec error means the outcome is unknown.
        if result.error_type is not None:
            raise _Inconclusive(VERIFIER_ERROR, f"build could not run: {result.error_type}")
        if result.return_code is None or result.return_code in VERIFIER_SETUP_EXITS:
            raise _Inconclusive(VERIFIER_ERROR, f"verifier setup failed (exit {result.return_code})")
        return result.return_code, round(time.monotonic() - started, 1), (result.stdout or "")[-LOG_TAIL_CHARS:]

    def _judged_score(self, quality: int, adherence: int) -> float:
        score = {
            "quality": quality,
            "prompt_adherence": adherence,
            "mean": (quality + adherence) / 2,
            "min": min(quality, adherence),
        }[self.config.reward_score]
        return score / MAX_SCORE

    async def _score(self, body: UiGenVerifyRequest, task: UiGenTask, grade: _Grade) -> UiGenVerifyResponse:
        """The verify response for ``grade``: 0 unless the file built, else ``reward_score`` of the judge's verdict."""
        stack = self.config.stacks[task.stack]
        fields: dict[str, Any] = {
            **task.model_dump(),
            "build_passed": grade.build_passed,
            "build_exit_code": grade.build_exit_code,
            "build_seconds": grade.build_seconds,
            "build_log_tail": grade.build_log_tail,
            "final_code": grade.final_code,
            "final_code_sha256": grade.final_code_sha256,
            "tamper_detected": bool(grade.tamper_details),
            "tamper_details": grade.tamper_details,
            "failure_reason": grade.failure_reason,
            "agent_stop_reason": agent_stop_reason(body.response),
            "mask_sample": False,
        }
        if not grade.build_passed:
            components = {"build": 0.0, "judge_quality": 0.0, "judge_prompt_adherence": 0.0}
            return self._response(body, {**fields, "reward": 0.0, "reward_components": components})

        params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
        params.messages = [
            {"role": "system", "content": self._judge_rubric},
            {
                "role": "user",
                "content": judge_case(
                    task.user_prompt, stack.description, stack.entry_path, grade.final_code or "", stack.language
                ),
            },
        ]

        async def judge_and_score() -> UiGenVerifyResponse:
            async with self._judge_semaphore:
                completion = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/chat/completions",
                    json=params,
                    response_model=NeMoGymChatCompletion,
                    max_attempts=self.config.judge_max_attempts,
                )
            verdict, reasoning = parse_judge_completion(completion)
            record_parsed_judge_verdict()
            quality, adherence = verdict["quality"]["score"], verdict["prompt_adherence"]["score"]
            return self._response(
                body,
                {
                    **fields,
                    "reward": self._judged_score(quality, adherence),
                    "reward_components": {
                        "build": 1.0,
                        "judge_quality": quality / MAX_SCORE,
                        "judge_prompt_adherence": adherence / MAX_SCORE,
                    },
                    "judge_quality": quality,
                    "judge_prompt_adherence": adherence,
                    "judge_verdict": verdict,
                    "judge_reasoning": reasoning,
                },
            )

        # A judge failure raises JudgeError, which the server runtime turns into a masked sample. The observed result
        # is the returned response, so the grading record binds to the reward it produced.
        return await observe_llm_judge_grading(judge_and_score)

    # ------------------------------------------------------------------ metrics

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Per-stack means over scored rollouts: reward, build pass rate, judge quality, tamper and agent-incomplete
        rates. The shared aggregation passes only scored (unmasked) rollouts here; see ``aggregate_metrics``."""
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for rollout in (rollout for task in tasks for rollout in task):
            if rollout.get("mask_sample"):
                continue
            stack = rollout.get("stack") or "unknown"
            values[stack, "reward"].append(float(rollout.get("reward") or 0.0))
            values[stack, "build_passed"].append(float(bool(rollout.get("build_passed"))))
            values[stack, "tamper_detected"].append(float(bool(rollout.get("tamper_detected"))))
            stop = rollout.get("agent_stop_reason") or ""
            values[stack, "agent_incomplete"].append(float(stop.startswith("incomplete")))
            if isinstance(rollout.get("judge_quality"), int):
                values[stack, "judge_quality"].append(float(rollout["judge_quality"]))
        return {f"{stack}/mean/{key}": fmean(xs) for (stack, key), xs in sorted(values.items()) if xs}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """The shared aggregation plus each stack's masked rate, counted over every rollout (masked ones included)."""
        result = await super().aggregate_metrics(body)
        masked: dict[str, list[float]] = defaultdict(list)
        for rollout in body.verify_responses:
            masked[rollout.get("stack") or "unknown"].append(float(bool(rollout.get("mask_sample"))))
        result.agent_metrics.update({f"{stack}/mean/masked": fmean(xs) for stack, xs in sorted(masked.items())})
        return result


if __name__ == "__main__":
    UiGenerationAgenticResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = UiGenerationAgenticResourcesServer.run_webserver()  # noqa: F401
