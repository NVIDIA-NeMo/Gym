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
"""ViBench resources server (P0).

This server owns the build sandbox and lends it to the agent through ``SandboxAccess``, the way
``resources_servers/gdpval`` does. The agent never sees the test plans; grading happens on this
host in a fresh compose stack, not in the box the model controlled.

  * ``seed_session`` starts the build sandbox from ViBench's base image with the task's PRD at
    ``<app_workdir>/prd.txt`` and its PRD assets under ``<app_workdir>/assets/``, and lends it.
    An agent session (e.g. ``opencode_agent``) borrows it and builds the app there.
  * ``verify`` copies the built app out of that sandbox, stops the sandbox, and grades the copy
    with ViBench's ``run-seed.py`` then ``run-evaluate-post-seeding.py``, once per test plan.
    Each run stands up the app + postgres + code-browse in a *fresh* compose project, seeds it
    with the seeding agent, then scores it with the evaluation agent. (The single-call
    run-seed-then-evaluate.py is not used; see the comment above SEED_SCRIPT.)
  * ``close_session`` stops the sandbox if verify has not, and removes the copied app.

The deprecated legacy relay path (``responses_api_agents/vibench_agent``, ``configs/vibench_legacy_agent.yaml``)
still works: its ``seed_session`` gets the PRD text and asset directories and starts no sandbox, its agent builds in a
sandbox of its own, and its ``verify`` names a tarball in the shared ``artifact_dir``, which this server unpacks and
grades the same way.

Sessions are process-local, so the server runs with one worker. Grading shells into a local
Docker daemon, so this server needs Docker and ``docker-compose`` on its host.

The verifier is itself an LLM agent driving a browser, so reward is stochastic -- profile
its variance before using this for training. ``REVERIFY_MODE`` is ``UNSUPPORTED`` because
grading depends on live app and database state.
"""

import asyncio
import json
import os
import re
import shutil
import signal
import sys
import tarfile
import tempfile
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
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
    lost_session_verification,
)


# ViBench's two-phase grading path. run-seed-then-evaluate.py looks like a convenient
# single call, but its build context omits the `seeding/` directory that
# Dockerfile.completed-app requires, so it always dies at `COPY seeding /seeding` -- every
# sibling script creates that directory (run-seed.py:95 makes an empty one). The two-phase
# path is also what ViBench's own results tree drives, and it is the only one that accepts
# --test-assets, which the evaluation agent needs and the builder must never see.
SEED_SCRIPT = Path("_harness/runner/scripts/run-seed.py")
EVALUATE_SCRIPT = Path("_harness/runner/scripts/run-evaluate-post-seeding.py")

# ViBench scores a test plan by having the evaluation agent fill in <pass>/<comment> tags
# after every <skippable> block. populate_results_folder.py does this when it materializes
# the results tree; we do it here because we feed test plans straight from the PRD tree.
_SKIPPABLE_RE = re.compile(r"(<skippable>[^<]*</skippable>)")


# The builder reads its brief from this path inside the sandbox; the seeding and evaluation
# agents are handed the same PRD text separately at grade time.
PRD_FILENAME = "prd.txt"

# Dependency and VCS trees never travel with the app. Patterns are deliberately NOT
# ``./``-anchored: GNU tar applies an anchored pattern only at the top level, so
# ``--exclude=./node_modules`` misses ``sub/node_modules`` -- and a nested ``.venv/bin/python``
# symlinks outside the app dir, which makes the unpacker reject the entire artifact.
_EXCLUDED_NAMES = (
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    "dist-cache",
)
HARVEST_EXCLUDES = " ".join(f"--exclude={name}" for name in _EXCLUDED_NAMES)
# Where the harvest tarball is written inside the sandbox before it is downloaded.
REMOTE_ARCHIVE = "/tmp/vibench-app.tar"
# The harvest command's exit code when the app directory is gone. GNU tar exits 1 when a file changed while it
# was read (the archive is complete) and 2 on a fatal error, so this must differ from both.
_NO_APP_DIR_EXIT = 3
_UNSAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _label(*parts: str) -> str:
    """A temp-dir prefix naming the task; row fields are untrusted and may contain path separators."""
    return "-".join(_UNSAFE_LABEL_RE.sub("_", part) for part in parts)


def add_evaluation_tags(test_plan_text: str) -> str:
    """Insert the ``<pass>``/``<comment>`` scaffolding the evaluation agent fills in.

    Raises when a plan contains ``<skippable>`` blocks but none matched: a silent no-op here
    produces a plan the evaluation agent cannot score, which surfaces much later as an
    unexplained zero rather than as a bad test plan.
    """
    tagged, count = _SKIPPABLE_RE.subn(
        lambda m: m.group(1) + "\n<pass>Y/N</pass>\n<comment></comment>", test_plan_text
    )
    if count == 0 and "<skippable" in test_plan_text:
        raise ValueError("test plan has <skippable> blocks but none matched the tagging pattern")
    return tagged


class GraderConfigError(RuntimeError):
    """The grading environment itself is misconfigured.

    Deliberately distinct: per-plan failures are zeroed and reported, but this one applies to
    every plan in every rollout, so zeroing it would produce a full dataset of silent zeros
    that looks like a very bad model.
    """


class _HarvestUnavailable(RuntimeError):
    """The sandbox provider failed to archive or transfer the app: the environment failed, not the model."""

    def __init__(self, failure_kind: str, reason: str) -> None:
        super().__init__(reason)
        self.failure_kind = failure_kind


def _harvest_unavailable(operation: str, error: BaseException) -> _HarvestUnavailable:
    failure_kind = error.failure_kind if isinstance(error, NgFailureEvent) else PROVIDER_REQUEST_FAILED
    return _HarvestUnavailable(
        failure_kind, f"The build sandbox failed to {operation}: {type(error).__name__}: {error}"
    )


class VibenchSandboxConfig(SandboxConfig):
    # ViBench's base image (its own pipeline builds and reuses it), so the build box and the
    # grading stack share one image. Its WORKDIR is /app.
    image: Optional[str] = None


class VibenchResourcesServerConfig(BaseResourcesServerConfig):
    # Absolute path to a ViBench checkout on this host. Grading shells into it.
    vibench_repo_root: str

    # Names a top-level provider block, such as the one
    # nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml defines. The provider must
    # support reattaching (serialize/connect), since the agent borrows the sandbox. Only sessions
    # need it; the legacy relay path's agent starts its own sandbox.
    sandbox_provider: Optional[str] = None
    sandbox_config: VibenchSandboxConfig = Field(default_factory=VibenchSandboxConfig)
    # Where the builder works and the PRD is staged; the image's WORKDIR.
    app_workdir: str = "/app"
    # Ceiling on taring the finished app inside the sandbox.
    harvest_timeout_s: int = 900

    # Wall-clock ceiling for one test plan, shared across its seed and evaluate phases --
    # not per phase, or a slow-but-successful seed would still hand evaluate a full window.
    evaluation_timeout_s: int = 5400
    # Window a timed-out grading script gets to run `docker-compose down` after SIGINT.
    cleanup_grace_s: float = 120.0
    # ViBench grading is compose-heavy (postgres + app + playwright per test plan). Cap how
    # many run at once *within one rollout*; Gym's own concurrency multiplies on top of this.
    max_concurrent_test_plans: int = 2

    # .env holding the raw provider keys ViBench's env_creator maps onto the grader
    # agents' AGENT_SEEDING_LLM_* / AGENT_EVALUATION_LLM_* variables. These are the
    # *verifier's* models and are deliberately not the policy model under test.
    vibench_env_file: Optional[str] = None
    # ViBench model key passed to env_creator.get_env_dict when deriving that env.
    grader_model_name: str = "Sonnet_4.5"

    # Keep per-test-plan output dirs (traces, screenshots, DB dumps) after grading.
    keep_evaluation_artifacts: bool = False

    # Legacy relay path only: the directory vibench_agent drops built-app tarballs into, shared
    # with this server. A verify that names a tarball outside it is a build failure.
    artifact_dir: Optional[str] = None
    # Delete the agent's tarball once it has been graded.
    remove_artifact_after_grading: bool = True

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED


class VibenchTaskRequest(BaseModel):
    """One task = one (app, artifact) pair. See prepare.py for how rows are generated."""

    app: str
    artifact: str = "mvp"
    # All paths are relative to vibench_repo_root so datasets stay checkout-independent.
    prd_files: List[str]
    test_plans: List[str]
    asset_dirs: List[str] = []
    # Fixtures the evaluation agent uploads while driving the app. Grader-only: never
    # staged into the build sandbox.
    test_assets_dir: Optional[str] = None


class VibenchSeedSessionRequest(VibenchTaskRequest, BaseSeedSessionRequest):
    """The legacy relay path's seed: the task row itself, from vibench_agent's /run."""


class VibenchSeedSessionResponse(BaseSeedSessionResponse):
    """The legacy relay path's brief. No sandbox: vibench_agent starts its own."""

    prd_text: str
    asset_paths: List[str]


class VibenchVerifyRequest(VibenchTaskRequest, BaseVerifyRequest):
    # Legacy relay path only: the tarball vibench_agent wrote into artifact_dir, or None when it
    # harvested nothing. Sessions harvest the lent sandbox instead and never send it.
    artifact_path: Optional[str] = None


class PlanResult(BaseModel):
    test_plan: str
    score: float
    full_points: float
    normalized_score: float
    steps_total: int
    steps_passed: int
    seeding_failed: bool
    error: Optional[str] = None
    duration_s: float


class VibenchVerifyResponse(BaseMultiRewardVerifyResponse):
    # The **body.model_dump() splat carries task fields (prd_files, test_plans,
    # test_assets_dir, artifact_path) that this response does not redeclare; without this
    # they are silently dropped and reverify cannot reconstruct the task.
    model_config = ConfigDict(extra="allow")

    app: str
    artifact: str

    # Top-level scalars so aggregate_metrics can see them (it does not descend into
    # reward_components).
    build_failed: bool
    seeding_failure_rate: float
    test_plans_graded: int
    test_plans_total: int

    results: List[PlanResult]
    artifact_extraction_time_s: float
    grading_time_s: float


@dataclass
class _BuildSession:
    episode_id: EpisodeId
    task_id: TaskId
    # Released (stopped and set to None) by the first verify that harvests, or by close.
    sandbox: Optional[AsyncSandbox]
    # Holds the harvested tarball; removed at close.
    scratch_dir: Path
    # Only fully staged sessions can be lent or verified. Failed setup remains owned for cleanup.
    ready: bool = False
    # Set once a verify has harvested: the tarball, or None when the model left nothing to archive. A provider
    # failure leaves it unset and the sandbox running, so a resent verify can harvest again.
    harvested: bool = False
    archive: Optional[Path] = None


class VibenchResourcesServer(SimpleResourcesServer):
    config: VibenchResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._sessions: Dict[str, _BuildSession] = {}
        self._locks = ResourcesSessionLocks()
        self._fences: ResourcesSessionFences[EpisodeId] = ResourcesSessionFences(
            self.config.session_close_retry_window_seconds
        )
        super().model_post_init(context)

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        return "ViBench sessions are process-local"

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

    # Derived once per server: env_creator is a subprocess and the result is identical for
    # every plan, so recomputing it per grading call is pure overhead.
    _cached_grader_env: Optional[Dict[str, str]] = None

    # ---------------------------------------------------------------- helpers

    @property
    def _repo_root(self) -> Path:
        return Path(self.config.vibench_repo_root).expanduser().resolve()

    @property
    def _vibench_python(self) -> str:
        """ViBench's own interpreter when the checkout has one, else the Gym component's.

        ViBench 1.0's venv is in v1/ since vibench-public 2.0 and at the root before it;
        prepare.py resolves the same checkout this way.
        """
        for candidate in (
            self._repo_root / "v1" / ".venv" / "bin" / "python",
            self._repo_root / ".venv" / "bin" / "python",
        ):
            if candidate.exists():
                return str(candidate)
        return sys.executable

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a dataset path against the ViBench checkout, refusing escapes."""
        candidate = (self._repo_root / rel_path).resolve()
        if not candidate.is_relative_to(self._repo_root):
            raise ValueError(f"Path {rel_path!r} escapes vibench_repo_root")
        return candidate

    def _resolve_artifact(self, artifact_path: str) -> Path:
        """Resolve an agent-supplied tarball path, refusing anything outside artifact_dir."""
        if not self.config.artifact_dir:
            raise ValueError("artifact_path needs artifact_dir, which only the legacy relay path sets")
        candidate = Path(artifact_path).expanduser().resolve()
        if not candidate.is_relative_to(Path(self.config.artifact_dir).expanduser().resolve()):
            raise ValueError(f"Artifact {artifact_path!r} escapes artifact_dir")
        return candidate

    def _unpack_artifact(self, artifact: Path, dest: Path) -> None:
        """Unpack the harvested app tarball, refusing members that escape ``dest``.

        The tarball comes out of a sandbox the model controlled, so its members are
        untrusted: a path like ``../../etc`` would otherwise write outside the
        grading directory.
        """
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(artifact) as tar:
            for member in tar.getmembers():
                target = (dest / member.name).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError(f"Refusing tar member escaping the app dir: {member.name!r}")
                if member.issym() or member.islnk():
                    link_target = (target.parent / member.linkname).resolve()
                    if not link_target.is_relative_to(dest.resolve()):
                        raise ValueError(f"Refusing link escaping the app dir: {member.name!r}")
            # filter="data" is the stdlib's own guard (and the 3.14 default); the explicit
            # checks above stay because they give a named error instead of a generic one.
            tar.extractall(dest, filter="data")

    async def _grader_env(self) -> Dict[str, str]:
        """Environment for ViBench's grading subprocesses.

        The compose template reads AGENT_SEEDING_LLM_* / AGENT_EVALUATION_LLM_* straight out
        of the environment (``${AGENT_LLM_API_KEY:-}`` and friends), and run-seed.py does not
        populate them. Loading the .env alone is not enough: raw provider keys have to be
        mapped onto those variables by ViBench's own env_creator, which is also what supplies
        each grader agent's tool list. Skip it and the seeding agent starts with no model and
        no tools, exits immediately, and the run reports a fully failed seeding rate.

        env_creator is invoked in a subprocess so ViBench's module never has to import into
        this process. Values are secrets and are never logged.
        """
        if self._cached_grader_env is not None:
            return dict(self._cached_grader_env)

        env = dict(os.environ)
        env_file = self.config.vibench_env_file
        if env_file:
            for line in Path(env_file).expanduser().read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")

        scripts_dir = self._repo_root / "_harness" / "runner" / "scripts"
        probe = (
            "import json, sys; sys.path.insert(0, %r); import env_creator; "
            "print(json.dumps(env_creator.get_env_dict(%r)))" % (str(scripts_dir), self.config.grader_model_name)
        )
        proc = await asyncio.create_subprocess_exec(
            self._vibench_python,
            "-c",
            probe,
            cwd=str(self._repo_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            # Without this the process survives and is re-spawned for every later plan.
            await self._terminate_group(proc)
            raise GraderConfigError("env_creator timed out after 120s") from None
        if proc.returncode != 0:
            # Failing loudly matters: degrading to an empty grader env makes every plan die
            # inside the container with "LLM API KEYS is not set", which points debugging at
            # credentials rather than at this call.
            # Redact before the message is built: this text reaches PlanResult.error and
            # therefore the committed rollout JSONL, and env_creator's stderr can echo the
            # very environment it was handed.
            detail = self._redact(stderr.decode(errors="replace")[-500:], env)
            raise GraderConfigError(
                f"env_creator failed (rc={proc.returncode}) for grader_model_name="
                f"{self.config.grader_model_name!r}: {detail}"
            )
        env.update({k: str(v) for k, v in json.loads(stdout).items() if v is not None})

        # ViBench's in-container agent validates AGENT_LLM_*, AGENT_SEEDING_LLM_* and
        # AGENT_EVALUATION_LLM_* together and refuses to start if any is unset
        # (_harness/runner/agent/environment.py: "LLM API KEYS is not set"). AGENT_LLM_* is
        # the *builder's* slot, which here is Gym's policy model rather than anything
        # env_creator knows about, so it comes back empty and seeding dies before it runs --
        # despite the seeding agent never using that key. Fill the unused slot from the
        # seeding values to satisfy the check without inventing credentials.
        for suffix in ("API_KEY", "MODEL", "ENDPOINT"):
            if not env.get(f"AGENT_LLM_{suffix}"):
                seeded = env.get(f"AGENT_SEEDING_LLM_{suffix}")
                if seeded:
                    env[f"AGENT_LLM_{suffix}"] = seeded

        self._cached_grader_env = dict(env)
        return env

    @staticmethod
    def _looks_buildable(app_dir: Path) -> bool:
        """Whether the agent produced something the grading stack could even start.

        ViBench's prompt requires setup-environment.sh and start-server.sh, and the seeding
        script invokes them; without either, grading cannot begin whatever the language.
        """
        if not app_dir.is_dir() or not any(app_dir.iterdir()):
            return False
        return (app_dir / "setup-environment.sh").exists() and (app_dir / "start-server.sh").exists()

    def _redact(self, text: str, env: Dict[str, str]) -> str:
        """Strip grader credentials out of captured output.

        Captured stdout is stored in PlanResult.error and ships in the rollout JSONL, which
        gets committed. The env these scripts run with holds AGENT_*_API_KEY, so one `set -x`
        upstream would otherwise put a live key in a public file. Scrubbing at the capture
        point means that cannot happen regardless of what ViBench prints.
        """
        for key, value in env.items():
            if not value or len(value) < 8:
                continue
            if any(marker in key for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
                text = text.replace(value, f"<redacted:{key}>")
        return text

    async def _run_vibench_script(self, cmd: List[str], timeout_s: float) -> tuple[int, str]:
        """Run one ViBench grading script, returning (return_code, merged output).

        A timeout is a failed grade, not an exception: one wedged compose stack should score
        zero rather than take the whole rollout down.

        On timeout the process *group* is interrupted rather than terminated; see
        ``_terminate_group``. The wedged-stack case this timeout exists for is exactly when
        postgres, the app, Playwright, the image tag and a port in the 50000-60000 range
        would otherwise leak.
        """
        env = await self._grader_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._repo_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            return 1, format_exc()

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            # communicate() was cancelled, so nothing is draining stdout. A grading script
            # that fills the pipe would block on write and never reach its compose cleanup,
            # which is the whole point of interrupting it rather than killing it.
            drain = asyncio.create_task(self._drain(proc))
            await self._terminate_group(proc)
            drain.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await drain
            return 1, f"timed out after {timeout_s:g}s: {' '.join(cmd)}"

        code = proc.returncode if proc.returncode is not None else 1
        return code, self._redact((stdout or b"").decode(errors="replace"), env)

    @staticmethod
    async def _drain(proc: Any) -> None:
        """Keep reading stdout so an interrupted script never blocks on a full pipe."""
        with suppress(Exception):
            while await proc.stdout.read(65536):
                pass

    async def _terminate_group(self, proc: Any) -> None:
        """Escalate SIGINT -> SIGTERM -> SIGKILL across the process group.

        SIGINT first, and this ordering is load-bearing. ViBench's grading scripts tear their
        compose project down in a ``finally`` block, and CPython does not unwind ``finally``
        on default SIGTERM -- it dies immediately, leaving postgres, the app, Playwright and
        the network running. SIGINT raises KeyboardInterrupt, which *does* unwind, so the
        script gets to run ``docker-compose down``. A forced-timeout run leaked two
        containers and a network on SIGTERM and cleaned up fully on SIGINT.

        SIGTERM and SIGKILL remain as escalation for anything that ignores the interrupt.
        """
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            with suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(os.getpgid(proc.pid), sig)
            # SIGINT needs a real window: cleanup shells out to docker-compose down.
            grace = self.config.cleanup_grace_s if sig is signal.SIGINT else self.config.cleanup_grace_s / 2
            with suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(proc.wait(), timeout=grace)
                if proc.returncode is not None:
                    return

    async def _grade_one_test_plan(
        self,
        app_dir: Path,
        test_plan_rel: str,
        work_dir: Path,
        test_assets_dir: Optional[str],
    ) -> PlanResult:
        """Seed, then evaluate, a single test plan and parse its score.

        Every failure here becomes a zeroed plan rather than an exception: one unreadable
        plan file or one truncated evaluation-finished.json would otherwise 500 /verify and
        lose the whole rollout, including the plans that graded fine.
        """
        started = time.time()
        name = Path(test_plan_rel).stem
        out_dir = work_dir / name

        def fail(seeding_failed: bool, error: str) -> PlanResult:
            return PlanResult(
                test_plan=name,
                score=0.0,
                full_points=0.0,
                normalized_score=0.0,
                steps_total=0,
                steps_passed=0,
                seeding_failed=seeding_failed,
                error=error,
                duration_s=time.time() - started,
            )

        deadline = started + self.config.evaluation_timeout_s

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            plan_path = out_dir / "test-plan.txt"
            plan_path.write_text(add_evaluation_tags(self._resolve(test_plan_rel).read_text()))
        except Exception:
            return fail(seeding_failed=False, error=format_exc())

        seed_dir = out_dir / "seed"
        seed_dir.mkdir(parents=True, exist_ok=True)
        code, log = await self._run_vibench_script(
            [
                self._vibench_python,
                str(self._repo_root / SEED_SCRIPT),
                "--app-dir",
                str(app_dir),
                "--test-plan",
                str(plan_path),
                "--output-dir",
                str(seed_dir),
            ],
            timeout_s=max(1.0, deadline - time.time()),
        )
        # ViBench refuses to evaluate an app it could not seed; that is a real zero, tracked
        # separately from a bad build so aggregate metrics can tell them apart.
        if code != 0 or not (seed_dir / "seeding").is_dir():
            return fail(seeding_failed=True, error=log[-2000:])

        evaluate_cmd = [
            self._vibench_python,
            str(self._repo_root / EVALUATE_SCRIPT),
            "--app-dir",
            str(app_dir),
            "--seeding",
            str(seed_dir / "seeding"),
            "--test-plan",
            str(plan_path),
            "--output-dir",
            str(out_dir),
        ]
        if test_assets_dir:
            # The evaluation agent uploads these during the run. They are deliberately never
            # staged into the build sandbox.
            evaluate_cmd += ["--test-assets", str(self._resolve(test_assets_dir))]

        code, log = await self._run_vibench_script(evaluate_cmd, timeout_s=max(1.0, deadline - time.time()))

        report_path = out_dir / "evaluation-finished.json"
        if not report_path.exists():
            # Seeding already succeeded to get here, so this is an evaluation failure.
            # Reporting it as a seeding failure would point debugging at the wrong stage --
            # which is exactly what it did the first time this fired.
            return fail(seeding_failed=False, error=log[-4000:])

        try:
            data = json.loads(report_path.read_text())
            score = float(data.get("score", 0) or 0)
            full_points = float(data.get("full_points", 0) or 0)
            steps = data.get("steps", []) or []
        except Exception:
            # A truncated or malformed scorecard is a failed grade, not a crashed rollout.
            return fail(seeding_failed=False, error=format_exc())

        if not self.config.keep_evaluation_artifacts:
            shutil.rmtree(out_dir, ignore_errors=True)

        return PlanResult(
            test_plan=name,
            score=score,
            full_points=full_points,
            normalized_score=(score / full_points) if full_points > 0 else 0.0,
            steps_total=len(steps),
            steps_passed=sum(1 for s in steps if (s.get("points", 0) or 0) > 0),
            seeding_failed=False,
            duration_s=time.time() - started,
        )

    # ---------------------------------------------------------------- routes

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | VibenchSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | VibenchSeedSessionResponse:
        """Start the build sandbox with the task's brief staged, and lend it to the agent.

        A legacy relay seed (vibench_agent's /run) gets the brief only, since that agent starts its own sandbox.
        Feature artifacts concatenate their base PRDs in order, matching how ViBench's
        build-feature path presents prior context to the coding agent.
        """
        if not isinstance(body, ResourcesSeedSessionRequest):
            prd_text = "\n\n".join(self._resolve(prd).read_text() for prd in body.prd_files)
            asset_paths = [str(self._resolve(d)) for d in body.asset_dirs if self._resolve(d).is_dir()]
            return VibenchSeedSessionResponse(prd_text=prd_text, asset_paths=asset_paths)

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
                if not session.ready or session.harvested or session.sandbox is None:
                    raise ResourcesSessionConflictError(
                        "The build sandbox is not available for seeding (setup incomplete or already harvested)"
                    )
                request.session[SESSION_ID_KEY] = session_id
                return await self._seed_response(session_id, session)
            try:
                task = VibenchTaskRequest.model_validate(body.task_data)
                prd_text = "\n\n".join(self._resolve(prd).read_text() for prd in task.prd_files)
                # Only static fixtures the PRD refers to. test_assets/ belongs to the grader and is
                # deliberately never offered to the builder.
                asset_dirs = [self._resolve(d) for d in task.asset_dirs]
            except (ValidationError, ValueError, OSError) as error:
                # A row that names no task, or files outside or missing from the checkout: no retry fixes it.
                raise ResourcesSeedRejectedError(
                    f"task_data is not a ViBench task in this checkout: {error}"
                ) from error

            # Made before the sandbox starts, so that nothing can fail between the start and the session holding it.
            scratch_dir = Path(tempfile.mkdtemp(prefix=f"vibench-build-{_label(task.app)}-"))
            session = _BuildSession(
                episode_id=body.episode_id,
                task_id=body.task_id,
                sandbox=None,
                scratch_dir=scratch_dir,
            )
            self._sessions[session_id] = session
            try:
                await self._start_build_sandbox(session, prd_text, [d for d in asset_dirs if d.is_dir()])
                response = await self._seed_response(session_id, session)
                session.ready = True
            except BaseException:
                # Keep ownership if teardown fails; close/retire can retry, but seed cannot lend partial setup.
                await self._release(session)
                del self._sessions[session_id]
                raise
            request.session[SESSION_ID_KEY] = session_id
            return response

    async def _start_build_sandbox(self, session: _BuildSession, prd_text: str, asset_dirs: List[Path]) -> None:
        """Start a build box with the PRD staged before anything runs, then upload the PRD assets."""
        if not self.config.sandbox_provider:
            raise ValueError("Set sandbox_provider to run ViBench sessions; see resources_servers/vibench/README.md")
        if not self.config.sandbox_config.image:
            raise ValueError(
                "Set sandbox_config.image (or VIBENCH_BUILD_IMAGE) to ViBench's base image; see "
                "resources_servers/vibench/README.md"
            )
        global_config_dict = get_global_config_dict()
        workdir = self.config.app_workdir
        spec = self.config.sandbox_config.spec(
            self.config.sandbox_provider,
            image=self.config.sandbox_config.image,
            workdir=workdir,
            metadata={"benchmark": "vibench"},
            files={f"{workdir}/{PRD_FILENAME}": prd_text},
            named_configs=global_config_dict,
        )
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config_dict))
        # Own the handle before start: initial file delivery can fail after provider creation.
        session.sandbox = sandbox
        await sandbox.start(spec)
        for asset_dir in asset_dirs:
            for item in sorted(asset_dir.rglob("*")):
                if item.is_file():
                    await sandbox.upload(item, f"{workdir}/assets/{item.relative_to(asset_dir).as_posix()}")

    async def _seed_response(self, session_id: str, session: _BuildSession) -> ResourcesSeedSessionResponse:
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider,
                    descriptor=await session.sandbox.serialize(),
                ),
                workdir=self.config.app_workdir,
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
                    await self._release(session)
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
                await self._release(session)
                del self._sessions[session_id]
                self._fences.fence(session_id, session.episode_id)

    async def _release_sandbox(self, session: _BuildSession) -> None:
        if session.sandbox is not None:
            await session.sandbox.stop()
            session.sandbox = None

    async def _release(self, session: _BuildSession) -> None:
        await self._release_sandbox(session)
        await asyncio.to_thread(shutil.rmtree, session.scratch_dir, ignore_errors=True)

    async def shutdown(self) -> None:
        for session_id, session in list(self._sessions.items()):
            try:
                await self._release(session)
            except Exception:
                print(
                    f"Failed to stop the build sandbox of ViBench session {session_id}", format_exc(), file=sys.stderr
                )
            else:
                del self._sessions[session_id]
        self._fences.clear()

    async def _harvest_app(self, session: _BuildSession) -> Optional[Path]:
        """Tar the built app out of the sandbox, then stop the sandbox.

        Dependency trees are excluded because they are huge and machine-specific, and
        setup-environment.sh reinstalls them anyway. That is not only a size concern: a
        virtualenv's bin/python symlinks to the system interpreter *outside* the app, so
        harvesting one makes the unpacker refuse the whole tarball as an escaping link and
        score a working app as a build failure. prd.txt is dropped because the grader
        supplies its own copy.

        The first verify that harvests stops the sandbox; later ones regrade the same copy. Returns None when the
        model left nothing to archive (no app directory, or one tar cannot read), which verify scores as a build
        failure. Raises ``_HarvestUnavailable`` when the provider fails to run the command or transfer the archive;
        the sandbox is then kept, so a resent verify can try again and close still stops it.
        """
        if not session.harvested:
            workdir = quote(self.config.app_workdir)
            try:
                result = await session.sandbox.exec(
                    f"[ -d {workdir} ] || exit {_NO_APP_DIR_EXIT}; cd {workdir} && tar "
                    f"{HARVEST_EXCLUDES} --exclude={PRD_FILENAME} -cf {quote(REMOTE_ARCHIVE)} .",
                    cwd="/",
                    timeout_s=self.config.harvest_timeout_s,
                )
            except Exception as error:
                raise _harvest_unavailable("archive the app", error) from error
            output = (result.stderr or result.stdout or "").strip()
            if result.error_type is not None:
                raise _HarvestUnavailable(
                    PROVIDER_REQUEST_FAILED, f"The build sandbox did not run the harvest command: {result.error_type}"
                )
            if result.return_code in (0, 1):
                if result.return_code == 1:
                    # A process the model left running (often the app server) wrote to a file while tar read it.
                    # The archive is complete, and the grader starts the app afresh from it.
                    print(f"tar reported files that changed while it read them: {output}", file=sys.stderr)
                local = session.scratch_dir / "app.tar"
                try:
                    await session.sandbox.download(REMOTE_ARCHIVE, local)
                except Exception as error:
                    raise _harvest_unavailable("transfer the app archive", error) from error
                session.archive = local
            elif result.return_code in (_NO_APP_DIR_EXIT, 2):
                print(
                    f"The model left no app tar could archive (exit {result.return_code}): {output}", file=sys.stderr
                )
            else:
                # Not tar's own outcome: the image lacks tar, or the shell itself failed.
                raise _HarvestUnavailable(
                    VERIFIER_ERROR, f"The harvest command exited {result.return_code}: {output[:2000]}"
                )
            session.harvested = True
            # The tarball is the only thing that survives; grading happens in a fresh stack.
            try:
                await self._release_sandbox(session)
            except Exception:
                # The app is already out, so a failed stop must not cost its grade. The sandbox stays on the
                # session, and close stops it, raising if that fails too so the caller can retry.
                print("Failed to stop the build sandbox after the harvest", format_exc(), file=sys.stderr)
        return session.archive

    @staticmethod
    def _ungraded(body: VibenchVerifyRequest, masked: Dict[str, Any]) -> VibenchVerifyResponse:
        """A masked result: the environment, not the model, kept the app from being graded."""
        return VibenchVerifyResponse(
            **(body.model_dump() | masked),
            reward_components={},
            build_failed=False,
            seeding_failure_rate=0.0,
            test_plans_graded=0,
            test_plans_total=len(body.test_plans),
            results=[],
            artifact_extraction_time_s=0.0,
            grading_time_s=0.0,
        )

    @classmethod
    def _lost(cls, body: VibenchVerifyRequest) -> VibenchVerifyResponse:
        return cls._ungraded(
            body, lost_session_verification("No live ViBench session: there is no built app to grade")
        )

    async def verify(self, request: Request, body: VibenchVerifyRequest) -> VibenchVerifyResponse:
        if "artifact_path" in body.model_fields_set:
            return await self._verify_artifact(body)
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id not in self._sessions:
            # A session that is gone (a restart, a checkpoint restore) took its sandbox with it; a 0 would read
            # as an app that failed to build.
            return self._lost(body)

        started = time.time()
        work_dir = Path(tempfile.mkdtemp(prefix=f"vibench-{_label(body.app, body.artifact)}-"))
        try:
            # Only the harvest holds the session's lock. Without it a second verify would tar into the same remote
            # file and download over the same copy, and a close would remove that copy while it is unpacked, which
            # grades as a build failure. Grading runs outside the lock, on this verify's own copy in work_dir.
            async with self._locks.locked(session_id):
                session = self._sessions.get(session_id)
                if session is None or not session.ready:
                    # Closed while this verify waited for the lock.
                    return self._lost(body)
                try:
                    harvested = await self._harvest_app(session)
                except _HarvestUnavailable as error:
                    print(f"Masking the verify of {body.app}/{body.artifact}: {error}", file=sys.stderr)
                    return self._ungraded(
                        body,
                        {
                            "reward": 0.0,
                            "mask_sample": True,
                            "failure_kind": error.failure_kind,
                            "failure_reason": str(error),
                        },
                    )
                archive: Optional[Path] = None
                if harvested is not None:
                    archive = work_dir / "app.tar"
                    await asyncio.to_thread(shutil.copyfile, harvested, archive)
            return await self._grade(body, archive, work_dir, started)
        finally:
            if not self.config.keep_evaluation_artifacts:
                # Blocking filesystem work off the event loop: this server handles concurrent
                # rollouts and a large app tree takes real time to remove.
                await asyncio.to_thread(shutil.rmtree, work_dir, ignore_errors=True)

    async def _verify_artifact(self, body: VibenchVerifyRequest) -> VibenchVerifyResponse:
        """Grade the tarball vibench_agent wrote into artifact_dir (the legacy relay path).

        No tarball, or one outside artifact_dir, is a build failure, as before sessions.
        """
        started = time.time()
        archive: Optional[Path] = None
        if body.artifact_path:
            # Resolve first and never touch the raw path: deleting an unvalidated,
            # agent-supplied path would remove arbitrary files even when the path was
            # rejected for reading.
            try:
                archive = self._resolve_artifact(body.artifact_path)
            except ValueError:
                print(f"Refusing the artifact of {body.app}/{body.artifact}", format_exc(), file=sys.stderr)

        work_dir = Path(tempfile.mkdtemp(prefix=f"vibench-{_label(body.app, body.artifact)}-"))
        try:
            return await self._grade(body, archive, work_dir, started)
        finally:
            if archive is not None and self.config.remove_artifact_after_grading:
                with suppress(Exception):
                    archive.unlink(missing_ok=True)
            if not self.config.keep_evaluation_artifacts:
                await asyncio.to_thread(shutil.rmtree, work_dir, ignore_errors=True)

    async def _grade(
        self, body: VibenchVerifyRequest, archive: Optional[Path], work_dir: Path, started: float
    ) -> VibenchVerifyResponse:
        app_dir = work_dir / "app"
        build_failed = False
        if archive is None:
            # The model left nothing to archive, or the legacy relay sent no usable tarball.
            build_failed = True
        else:
            try:
                await asyncio.to_thread(self._unpack_artifact, archive, app_dir)
            except Exception:
                print(f"Failed to unpack the app of {body.app}/{body.artifact}", format_exc(), file=sys.stderr)
                build_failed = True

        # An agent that produced nothing runnable is a build failure, not a 0-scoring app.
        # The contract is the two scripts ViBench's prompt requires and its grading stack
        # invokes -- not package.json, which would misjudge any app that is not Node.
        if not build_failed and not self._looks_buildable(app_dir):
            build_failed = True
        artifact_extraction_time_s = time.time() - started

        results: List[PlanResult] = []
        grading_started = time.time()
        if not build_failed:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_test_plans)

            async def run(plan: str) -> PlanResult:
                async with semaphore:
                    return await self._grade_one_test_plan(app_dir, plan, work_dir, body.test_assets_dir)

            gathered = await asyncio.gather(*(run(p) for p in body.test_plans), return_exceptions=True)
            # A misconfigured grading environment affects every plan and every rollout;
            # zeroing it would yield a whole dataset of silent zeros.
            for outcome in gathered:
                if isinstance(outcome, GraderConfigError):
                    raise outcome
            results = [
                r
                if isinstance(r, PlanResult)
                else PlanResult(
                    test_plan=Path(plan).stem,
                    score=0.0,
                    full_points=0.0,
                    normalized_score=0.0,
                    steps_total=0,
                    steps_passed=0,
                    seeding_failed=False,
                    error=f"{type(r).__name__}: {r}",
                    duration_s=0.0,
                )
                for plan, r in zip(body.test_plans, gathered)
            ]
        grading_time_s = time.time() - grading_started

        total = len(body.test_plans)
        graded = [r for r in results if not r.seeding_failed and r.error is None]
        # Every test plan counts toward the denominator: a build that cannot be seeded
        # scores zero rather than being silently dropped from the average.
        reward = (sum(r.normalized_score for r in results) / total) if total else 0.0

        return VibenchVerifyResponse(
            **body.model_dump(),
            reward=reward,
            reward_components={r.test_plan: r.normalized_score for r in results},
            build_failed=build_failed,
            seeding_failure_rate=(sum(1 for r in results if r.seeding_failed) / total) if total else 0.0,
            test_plans_graded=len(graded),
            test_plans_total=total,
            results=results,
            artifact_extraction_time_s=artifact_extraction_time_s,
            grading_time_s=grading_time_s,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Report where rollouts fail, not just what they scored.

        A mean reward alone cannot distinguish "the model wrote a weak app" from "the app
        never built" or "grading could not seed it", and those need completely different
        responses. Every zero in this environment has one of those causes.

        ``tasks`` arrives grouped by task -- tasks[i] is the list of rollouts for task i --
        per AggregateMetricsMixin, so it is flattened before counting.
        """
        rollouts = [r for task in tasks for r in task]
        if not rollouts:
            return {}
        n = len(rollouts)
        rewards = [float(r.get("reward", 0) or 0) for r in rollouts]
        plans = sum(int(r.get("test_plans_total", 0) or 0) for r in rollouts)
        graded = sum(int(r.get("test_plans_graded", 0) or 0) for r in rollouts)
        return {
            "mean_reward": sum(rewards) / n,
            "perfect_rate": sum(1 for x in rewards if x >= 1.0) / n,
            "zero_rate": sum(1 for x in rewards if x <= 0.0) / n,
            "build_failure_rate": sum(1 for r in rollouts if r.get("build_failed")) / n,
            "mean_seeding_failure_rate": sum(float(r.get("seeding_failure_rate", 0) or 0) for r in rollouts) / n,
            # The share of plans that produced a scorecard at all: the health check that
            # separates a working verifier from a broken one.
            "plans_graded_rate": (graded / plans) if plans else 0.0,
        }

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: the score, plus whether grading actually happened."""
        headline = ["mean_reward", "plans_graded_rate", "build_failure_rate"]
        selected = {k: agent_metrics[k] for k in headline if k in agent_metrics}
        # Keep the framework default (mean/*) so nothing standard disappears.
        selected.update({k: v for k, v in agent_metrics.items() if k.startswith("mean/")})
        return selected


if __name__ == "__main__":
    VibenchResourcesServer.run_webserver()
