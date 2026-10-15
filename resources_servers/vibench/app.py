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

The rollout copies an artifact out of its own box and this server grades it in a fresh one,
so it never touches the agent's sandbox. Reaching into the agent's box instead would need
serialize()/connect(), which only the OpenSandbox provider implements -- node-local providers
such as Docker cannot support it at all.

  * ``seed_session`` returns the PRD text and asset directory for the task. It creates no
    sandbox and hands out no handle. The agent never sees the test plans.
  * ``responses_api_agents/vibench_agent`` owns the build sandbox, runs a coding harness in
    it, tars the finished app, and writes it to ``artifact_dir``.
  * ``verify`` unpacks that tarball and grades it with ViBench's ``run-seed.py`` then
    ``run-evaluate-post-seeding.py``, once per test plan. Each run stands up the app +
    postgres + code-browse in a *fresh* compose project, seeds it with the seeding agent,
    then scores it with the evaluation agent. (The single-call run-seed-then-evaluate.py is
    not used; see the comment above SEED_SCRIPT.)

``artifact_dir`` is a plain filesystem path shared by the agent and this server. That is not
a new constraint: grading already shells into a local Docker daemon, so both processes are
on one host either way.

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
from contextlib import suppress
from pathlib import Path
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Optional

from fastapi import Request
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    ReverifyMode,
    SimpleResourcesServer,
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
# the results tree; we do it here because we feed test plans straight from prds/.
_SKIPPABLE_RE = re.compile(r"(<skippable>[^<]*</skippable>)")


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


class VibenchResourcesServerConfig(BaseResourcesServerConfig):
    # Absolute path to a ViBench checkout on this host. Grading shells into it.
    vibench_repo_root: str

    # Directory the agent drops built-app tarballs into, shared with this server.
    artifact_dir: str

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

    # Delete the agent's tarball once it has been unpacked.
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


class VibenchRunRequest(VibenchTaskRequest, BaseRunRequest):
    pass


class VibenchSeedSessionRequest(VibenchTaskRequest, BaseSeedSessionRequest):
    pass


class VibenchSeedSessionResponse(BaseSeedSessionResponse):
    """Everything the agent needs to set its box up. No sandbox handle: the agent owns
    the box, so there is nothing here for it to attach to."""

    prd_text: str
    asset_paths: List[str]


class VibenchVerifyRequest(VibenchTaskRequest, BaseVerifyRequest):
    # Tarball of the built app, written by the agent into the shared artifact_dir.
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


class VibenchResourcesServer(SimpleResourcesServer):
    config: VibenchResourcesServerConfig

    # Derived once per server: env_creator is a subprocess and the result is identical for
    # every plan, so recomputing it per grading call is pure overhead.
    _cached_grader_env: Optional[Dict[str, str]] = None

    # ---------------------------------------------------------------- helpers

    @property
    def _repo_root(self) -> Path:
        return Path(self.config.vibench_repo_root).expanduser().resolve()

    @property
    def _vibench_python(self) -> str:
        """ViBench's own interpreter when the checkout has one.

        Its scripts import ViBench's dependencies, which are not in the Gym component venv;
        prepare.py already resolves the same checkout this way.
        """
        candidate = self._repo_root / ".venv" / "bin" / "python"
        return str(candidate) if candidate.exists() else sys.executable

    @property
    def _artifact_root(self) -> Path:
        return Path(self.config.artifact_dir).expanduser().resolve()

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a dataset path against the ViBench checkout, refusing escapes."""
        candidate = (self._repo_root / rel_path).resolve()
        if not candidate.is_relative_to(self._repo_root):
            raise ValueError(f"Path {rel_path!r} escapes vibench_repo_root")
        return candidate

    def _resolve_artifact(self, artifact_path: str) -> Path:
        """Resolve an agent-supplied tarball path, refusing anything outside artifact_dir."""
        candidate = Path(artifact_path).expanduser().resolve()
        if not candidate.is_relative_to(self._artifact_root):
            raise ValueError(f"Artifact {artifact_path!r} escapes artifact_dir")
        return candidate

    def _unpack_artifact(self, artifact: Path, dest: Path) -> None:
        """Unpack the agent's app tarball, refusing members that escape ``dest``.

        The tarball is written by the agent from a sandbox the model controlled, so its
        members are untrusted: a path like ``../../etc`` would otherwise write outside the
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

    async def seed_session(self, request: Request, body: VibenchSeedSessionRequest) -> VibenchSeedSessionResponse:
        """Hand the agent the task brief. No sandbox is created here -- the agent owns it.

        Feature artifacts concatenate their base PRDs in order, matching how ViBench's
        build-feature path presents prior context to the coding agent.
        """
        prd_text = "\n\n".join(self._resolve(prd).read_text() for prd in body.prd_files)
        # Only static fixtures the PRD refers to. test_assets/ belongs to the grader and is
        # deliberately never offered to the builder.
        asset_paths = [str(self._resolve(d)) for d in body.asset_dirs if self._resolve(d).is_dir()]
        return VibenchSeedSessionResponse(prd_text=prd_text, asset_paths=asset_paths)

    async def verify(self, request: Request, body: VibenchVerifyRequest) -> VibenchVerifyResponse:
        work_dir = Path(tempfile.mkdtemp(prefix=f"vibench-{body.app}-{body.artifact}-"))
        app_dir = work_dir / "app"

        started = time.time()
        build_failed = False
        artifact: Optional[Path] = None
        if not body.artifact_path:
            # The agent could not produce a tarball at all (sandbox died, harness crashed).
            build_failed = True
        else:
            # Resolve first and never touch the raw path: deleting an unvalidated,
            # agent-supplied path would remove arbitrary files even when the path was
            # rejected for reading.
            try:
                artifact = self._resolve_artifact(body.artifact_path)
                await asyncio.to_thread(self._unpack_artifact, artifact, app_dir)
            except Exception:
                print(f"Failed to unpack artifact for {body.app}/{body.artifact}", format_exc(), file=sys.stderr)
                build_failed = True
            # Deleted after grading, not here -- see the end of this method.

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

        if artifact is not None and self.config.remove_artifact_after_grading:
            with suppress(Exception):
                artifact.unlink(missing_ok=True)

        if not self.config.keep_evaluation_artifacts:
            # Blocking filesystem work off the event loop: this server handles concurrent
            # rollouts and a large app tree takes real time to remove.
            await asyncio.to_thread(shutil.rmtree, work_dir, ignore_errors=True)

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
