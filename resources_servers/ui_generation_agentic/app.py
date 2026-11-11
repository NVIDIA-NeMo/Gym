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

- G1 stops every process the agent left (detached dev servers, watchers), keeping the sandbox runtime;
- G2 snapshots the entry file once (a regular file within the size cap); only the snapshot is built and judged;
- G3 checks the image's root-owned, read-only template and toolchain against their manifest;
- G4 builds the snapshot in a fresh copy of that template with a build script uploaded at verify time, so nothing
  the agent changed in its workspace (configs, package.json, node_modules) reaches the build;
- G5 (``guard_mode: full`` only) runs the build as a separate unprivileged user;
- G6 takes the verdict from the build's exit status and the judge only.

``guard_mode: detection_only`` is for sandboxes whose commands cannot switch user: G1 then spares the runtime and
the verifier's own process group, and G5 is skipped. The reward is ``build × quality / 4`` (``reward_mode: gated``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shlex
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, ClassVar, Literal, Optional

import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.agents import restricted_network_policy
from nemo_gym.client import SESSION_ID_KEY, get_server_url
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.core import (
    PROVIDER_REQUEST_FAILED,
    SESSION_LOST,
    VERIFIER_ERROR,
    EpisodeId,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxConfig, resolve_provider_config
from nemo_gym.server_runtime import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    LentSandboxes,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    ReverifyMode,
    SimpleResourcesServer,
    observe_llm_judge_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import call_judge, record_parsed_judge_verdict
from resources_servers.ui_generation_agentic.judge import MAX_SCORE, judge_case, parse_judge_completion


LOG = logging.getLogger(__name__)
_HERE = Path(__file__).parent
VERIFY_SCRIPT = _HERE / "scripts" / "ui_gen_verify.sh"
TEMPLATE_DIR = "/opt/ui-gen/template"
MANIFEST = "/opt/ui-gen/MANIFEST.sha256"
BUILT_STAMP = "/opt/ui-gen/.built"
# Exit code of ui_gen_verify.sh when the verifier itself could not set up the build (not the model's outcome).
VERIFIER_SETUP_EXIT = 90
LOG_TAIL_CHARS = 8000

# G1: list the processes to stop. Kept: PID 1, the sandbox's exec daemon and keep-alive, and this command's own
# process group. Prints one PID per line.
_G1_LIST = r"""me=$(ps -o pgid= -p $$ | tr -d ' ')
ps -u "$(id -u)" -o pid=,ppid=,pgid=,comm=,args= | awk -v pg="$me" '{
  pid=$1; ppid=$2; pgid=$3; comm=$4; args=""; for (i=5;i<=NF;i++) args=args (i>5?" ":"") $i;
  if (pid==1 || pgid==pg || comm=="execd" || comm=="bootstrap.sh" || (ppid==1 && args=="tail -f /dev/null")) next;
  print pid }'"""
_G1_KILL = (
    f"victims=$({_G1_LIST})\n"
    '[ -n "$victims" ] && kill -9 $victims 2>/dev/null; sleep 1\n'
    f"left=$({_G1_LIST} | wc -l)\n"
    'echo "killed=$(echo $victims | wc -w) left=$left"'
)


class StackConfig(BaseModel):
    """One task stack: its agent image, where the agent works and which file it must write."""

    model_config = ConfigDict(extra="forbid")

    image: str
    workdir: str
    entry_path: str = Field(description="The deliverable, relative to workdir; it does not exist at seed.")
    language: str = Field(description="Fence language of the entry file for the judge, e.g. tsx or html.")
    description: str = Field(description="One-line description of the stack, shown to the judge.")
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
    agent_user: str = "node"
    verifier_user: str = "ui-verify"
    max_deliverable_bytes: int = Field(default=1_048_576, gt=0)
    build_timeout_s: int = Field(default=600, gt=0)
    inconclusive_verification_retries: int = Field(default=2, ge=0)
    reward_mode: Literal["gated", "build", "judge"] = "gated"
    judge_model_server: ModelServerRef
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_prompt_path: str = str(_HERE / "prompts" / "judge.yaml")
    judge_max_concurrency: int = Field(default=8, ge=1)
    judge_max_attempts: int = Field(default=4, ge=1)


class UiGenTask(BaseModel):
    """The task fields this server reads from ``task_data`` (other fields are kept but ignored)."""

    model_config = ConfigDict(extra="ignore")

    stack: str
    entry_path: str
    user_prompt: str


class UiGenVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    stack: str
    entry_path: str
    user_prompt: str


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
    episode_id: EpisodeId
    stack: str
    verdict: Optional[UiGenVerifyResponse] = None


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
    config: UiGenerationAgenticConfig

    def model_post_init(self, context: Any) -> None:
        if self.config.num_workers not in (None, 1):
            raise ValueError("ui_generation_agentic sessions are process-local and require num_workers=1")
        self._judge_rubric: str = yaml.safe_load(Path(self.config.judge_prompt_path).read_text())["system"]
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        self._lenders: dict[str, LentSandboxes] = {}
        self._sessions: dict[str, _Session] = {}
        self._locks = ResourcesSessionLocks()
        self._fences: ResourcesSessionFences[EpisodeId] = ResourcesSessionFences(
            self.config.session_close_retry_window_seconds
        )
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
                for lender in self._lenders.values():
                    await lender.release_all()

        app.router.lifespan_context = lifespan
        return app

    # ------------------------------------------------------------------ sessions

    def _lender(self, stack_name: str) -> LentSandboxes:
        """The stack's sandbox lender, built on first use (the model server's address is known only at run time)."""
        if stack_name in self._lenders:
            return self._lenders[stack_name]
        stack = self.config.stacks[stack_name]
        sandbox_config = self.config.sandbox_config
        if self.config.agent_network_mode == "restricted":
            provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            urls = [get_server_url(self.config.model_server.name)]
            urls += [f"https://{host}" for host in self.config.extra_allowed_hosts]
            try:
                policy = restricted_network_policy(next(iter(provider_config)), urls)
            except ValueError as error:
                raise ResourcesSeedRejectedError(f"Cannot restrict the sandbox network: {error}") from error
            options = {**sandbox_config.provider_options, "network_policy": policy}
            sandbox_config = sandbox_config.model_copy(update={"provider_options": options})
        lender = LentSandboxes(self.config.sandbox_provider, sandbox_config, image=stack.image, workdir=stack.workdir)
        self._lenders[stack_name] = lender
        return lender

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        """Start the task's stack sandbox (no entry file yet) and lend it to the agent."""
        session_id = body.resources_session_id
        async with self._locks.locked(session_id):
            self._fences.raise_if_fenced(session_id)
            session = self._sessions.get(session_id)
            if session is not None and session.episode_id != body.episode_id:
                raise ResourcesSessionConflictError("resources_session_id is bound to another episode")
            try:
                task = UiGenTask.model_validate(body.task_data)
            except ValidationError as error:
                raise ResourcesSeedRejectedError(f"task_data is not a ui_generation_agentic task: {error}") from error
            stack = self.config.stacks.get(task.stack)
            if stack is None or task.entry_path != stack.entry_path:
                raise ResourcesSeedRejectedError(f"Unknown stack or entry path: {task.stack} / {task.entry_path}")
            access = await self._lender(task.stack).lend(session_id, body.task_id)
            if session is None:
                self._sessions[session_id] = _Session(episode_id=body.episode_id, stack=task.stack)
        request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=access)

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
                    # A failed stop raises and keeps the sandbox, so the caller can retry the close.
                    await self._lenders[session.stack].release(session_id)
                    del self._sessions[session_id]
                self._fences.fence(session_id, body.episode_id)
            elif fenced_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
        request.session.pop(SESSION_ID_KEY, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    # ------------------------------------------------------------------ verify

    def _stamps(self) -> dict[str, str]:
        return {
            "guard_mode": self.config.guard_mode,
            "agent_network_mode": self.config.agent_network_mode,
            "egress_enforcement": self.config.egress_enforcement
            if self.config.agent_network_mode == "restricted"
            else "none",
        }

    def _masked(self, body: UiGenVerifyRequest, failure_kind: str, reason: str) -> UiGenVerifyResponse:
        return UiGenVerifyResponse(
            **body.model_dump(),
            reward=0.0,
            reward_components={"build": 0.0, "judge_quality": 0.0, "judge_prompt_adherence": 0.0},
            mask_sample=True,
            failure_kind=failure_kind,
            failure_reason=reason,
            **self._stamps(),
        )

    async def verify(self, request: Request, body: UiGenVerifyRequest) -> UiGenVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            return self._masked(body, SESSION_LOST, "verify without a seeded resources session")
        async with self._locks.locked(session_id):
            session = self._sessions.get(session_id)
            if session is None:
                return self._masked(body, SESSION_LOST, "the resources session is gone or closed")
            if session.verdict is not None:
                return session.verdict
            sandbox = self._lenders[session.stack].get(session_id)
            if sandbox is None:
                return self._masked(body, SESSION_LOST, "the lent sandbox is gone")
            try:
                grade = await self._grade_in_sandbox(sandbox, session.stack)
            except _Inconclusive as error:
                LOG.warning("Masking verify of session %s: %s", session_id, error)
                return self._masked(body, error.failure_kind, str(error))
            response = await self._score(body, grade)
            session.verdict = response
            return response

    async def _exec(self, sandbox: AsyncSandbox, command: str, *, timeout_s: float = 120, user: str | None = None):
        try:
            result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s, user=user)
        except Exception as error:  # the provider could not run the command: no verdict
            raise _Inconclusive(PROVIDER_REQUEST_FAILED, f"exec failed: {type(error).__name__}: {error}") from error
        return result

    async def _grade_in_sandbox(self, sandbox: AsyncSandbox, stack_name: str) -> _Grade:
        stack = self.config.stacks[stack_name]
        full = self.config.guard_mode == "full"
        root = "root" if full else None
        grade = _Grade()

        # G1: stop what the agent left running.
        if full:
            quiesce = f"pkill -KILL -u {shlex.quote(self.config.agent_user)}; sleep 1; pgrep -u {shlex.quote(self.config.agent_user)} | wc -l | sed 's/^/left=/'"
        else:
            quiesce = _G1_KILL
        result = await self._exec(sandbox, quiesce, user=root)
        left = (result.stdout or "").strip().rsplit("left=", 1)[-1].strip()
        if result.error_type is not None or not left.isdigit():
            raise _Inconclusive(
                VERIFIER_ERROR, f"could not stop the agent's processes: {result.stderr or result.stdout}"
            )
        if int(left) != 0:
            raise _Inconclusive(VERIFIER_ERROR, f"harness_termination_unconfirmed: {left} agent processes survived")

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
        grade.final_code = data.decode("utf-8", errors="replace")
        grade.final_code_sha256 = hashlib.sha256(data).hexdigest()

        # G3: the template and toolchain are unchanged since the image was built.
        check = (
            f"cd / && sha256sum -c --quiet {MANIFEST} >/dev/null 2>&1 && echo manifest=ok || echo manifest=bad; "
            f"echo manifest_sha=$(sha256sum {MANIFEST} | cut -d' ' -f1); "
            f"echo newer=$(find {TEMPLATE_DIR} -newer {BUILT_STAMP} -not -path '*/.vite-temp*' -print 2>/dev/null | head -5 | wc -l)"
        )
        result = await self._exec(sandbox, check, user=root)
        if result.error_type is not None:
            raise _Inconclusive(VERIFIER_ERROR, f"integrity check failed to run: {result.error_type}")
        facts = dict(line.split("=", 1) for line in (result.stdout or "").split() if "=" in line)
        if facts.get("manifest") != "ok":
            grade.tamper_details.append("template or toolchain differs from its manifest")
        if stack.manifest_sha256 and facts.get("manifest_sha") != stack.manifest_sha256:
            grade.tamper_details.append("manifest file differs from the pinned SHA-256")
        if facts.get("newer", "0") != "0":
            grade.tamper_details.append("files in the template are newer than the image build")
        if grade.tamper_details:
            grade.failure_reason = "tamper_detected"
            return grade

        # G4 + G5: build the snapshot in a fresh copy of the template with a freshly uploaded script.
        last_error = ""
        for attempt in range(self.config.inconclusive_verification_retries + 1):
            started = time.monotonic()
            result = await self._exec(sandbox, "mktemp -d /tmp/ui-gen-verify.XXXXXX", user=root)
            work = (result.stdout or "").strip()
            if result.return_code != 0 or not work.startswith("/tmp/ui-gen-verify."):
                last_error = f"mktemp failed: {result.stderr}"
                continue
            with tempfile.TemporaryDirectory(prefix="ui-gen-agentic-") as tmp:
                snap = Path(tmp) / "snapshot"
                snap.write_bytes(data)
                try:
                    await sandbox.upload(snap, f"{work}/snapshot")
                    await sandbox.upload(VERIFY_SCRIPT, f"{work}/verify.sh")
                except Exception as error:
                    last_error = f"upload failed: {error}"
                    continue
            build_user = self.config.verifier_user if full else None
            if full:
                await self._exec(
                    sandbox, f"mkdir {work}/build && chown -R {self.config.verifier_user} {work}", user=root
                )
            else:
                await self._exec(sandbox, f"mkdir {work}/build")
            command = (
                f"bash {work}/verify.sh {shlex.quote(stack_name)} {work}/snapshot {work}/build "
                f"{self.config.build_timeout_s} 2>&1"
            )
            result = await self._exec(sandbox, command, timeout_s=self.config.build_timeout_s + 120, user=build_user)
            grade.build_seconds = round(time.monotonic() - started, 1)
            grade.build_log_tail = (result.stdout or "")[-LOG_TAIL_CHARS:]
            if result.error_type not in (None, "timeout") or result.return_code == VERIFIER_SETUP_EXIT:
                last_error = f"build could not run: {result.error_type or 'verifier setup failed'}"
                continue
            grade.build_exit_code = 124 if result.error_type == "timeout" else result.return_code
            grade.build_passed = grade.build_exit_code == 0
            if not grade.build_passed:
                grade.failure_reason = "build_timeout" if grade.build_exit_code == 124 else "build_failed"
            return grade
        raise _Inconclusive(VERIFIER_ERROR, f"inconclusive build after retries: {last_error}")

    async def _score(self, body: UiGenVerifyRequest, grade: _Grade) -> UiGenVerifyResponse:
        stack = self.config.stacks[body.stack]
        quality = adherence = None
        verdict = reasoning = None
        run_judge = grade.final_code is not None and not grade.tamper_details
        if self.config.reward_mode == "gated":
            run_judge = run_judge and grade.build_passed
        elif self.config.reward_mode == "build":
            run_judge = False
        if run_judge:
            params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
            params.messages = [
                {"role": "system", "content": self._judge_rubric},
                {
                    "role": "user",
                    "content": judge_case(
                        body.user_prompt, stack.description, stack.entry_path, grade.final_code or "", stack.language
                    ),
                },
            ]

            async def judge() -> tuple[dict[str, Any], Optional[str]]:
                async with self._judge_semaphore:
                    completion = await call_judge(
                        self.server_client,
                        server_name=self.config.judge_model_server.name,
                        url_path="/v1/chat/completions",
                        json=params,
                        response_model=NeMoGymChatCompletion,
                        max_attempts=self.config.judge_max_attempts,
                    )
                parsed = parse_judge_completion(completion)
                record_parsed_judge_verdict()
                return parsed

            # A judge failure raises JudgeError, which the server runtime turns into a masked sample.
            verdict, reasoning = await observe_llm_judge_grading(judge)
            quality, adherence = verdict["quality"]["score"], verdict["prompt_adherence"]["score"]
        build = 1.0 if grade.build_passed else 0.0
        quality_frac = (quality or 0) / MAX_SCORE
        reward = {"gated": build * quality_frac, "build": build, "judge": quality_frac}[self.config.reward_mode]
        return UiGenVerifyResponse(
            **body.model_dump(),
            reward=reward,
            reward_components={
                "build": build,
                "judge_quality": quality_frac,
                "judge_prompt_adherence": (adherence or 0) / MAX_SCORE,
            },
            build_passed=grade.build_passed,
            build_exit_code=grade.build_exit_code,
            build_seconds=grade.build_seconds,
            build_log_tail=grade.build_log_tail,
            final_code=grade.final_code,
            final_code_sha256=grade.final_code_sha256,
            tamper_detected=bool(grade.tamper_details),
            tamper_details=grade.tamper_details,
            judge_quality=quality,
            judge_prompt_adherence=adherence,
            judge_verdict=verdict,
            judge_reasoning=reasoning,
            failure_reason=grade.failure_reason,
            agent_stop_reason=agent_stop_reason(body.response),
            **self._stamps(),
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Per-stack means of reward, build pass rate and judge scores, plus mask, tamper and agent-incomplete rates."""
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for rollout in (rollout for task in tasks for rollout in task):
            stack = rollout.get("stack") or "unknown"
            masked = bool(rollout.get("mask_sample"))
            values[stack, "masked"].append(float(masked))
            if masked:
                continue
            values[stack, "reward"].append(float(rollout.get("reward") or 0.0))
            values[stack, "build_passed"].append(float(bool(rollout.get("build_passed"))))
            values[stack, "tamper_detected"].append(float(bool(rollout.get("tamper_detected"))))
            stop = rollout.get("agent_stop_reason") or ""
            values[stack, "agent_incomplete"].append(float(stop.startswith("incomplete")))
            if isinstance(rollout.get("judge_quality"), int):
                values[stack, "judge_quality"].append(float(rollout["judge_quality"]))
        return {f"{stack}/mean/{key}": fmean(xs) for (stack, key), xs in sorted(values.items()) if xs}


if __name__ == "__main__":
    UiGenerationAgenticResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = UiGenerationAgenticResourcesServer.run_webserver()  # noqa: F401
