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

ViBench is "case 2" in NVIDIA-NeMo/Gym#2082: the rollout copies an artifact out of its own
box and the verifier grades it in a fresh one. So this server never touches the agent's
sandbox -- that would be case 3, which is the one shape that needs the (unmerged) sandbox
server, and which node-local providers like Docker deliberately cannot support.

  * ``seed_session`` returns the PRD text and asset directory for the task. It creates no
    sandbox and hands out no handle. The agent never sees the test plans.
  * ``responses_api_agents/vibench_agent`` owns the build sandbox, runs a coding harness in
    it, tars the finished app, and writes it to ``artifact_dir``.
  * ``verify`` unpacks that tarball and grades it by shelling out to ViBench's existing
    ``run-seed-then-evaluate.py`` once per test plan. Each run stands up the app + postgres
    + code-browse in a *fresh* compose project, seeds it with the seeding agent, then scores
    it with the evaluation agent.

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
import sys
import tarfile
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, List, Optional

from fastapi import Request
from pydantic import BaseModel

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
    """Insert the ``<pass>``/``<comment>`` scaffolding the evaluation agent fills in."""
    return _SKIPPABLE_RE.sub(lambda m: m.group(1) + "\n<pass>Y/N</pass>\n<comment></comment>", test_plan_text)


class VibenchResourcesServerConfig(BaseResourcesServerConfig):
    # Absolute path to a ViBench checkout on this host. Grading shells into it.
    vibench_repo_root: str

    # Directory the agent drops built-app tarballs into, shared with this server.
    artifact_dir: str

    # Wall-clock ceiling for a single test plan's seed+evaluate run.
    evaluation_timeout_s: int = 5400
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

    REVERIFY_MODE = ReverifyMode.UNSUPPORTED


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
            sys.executable,
            "-c",
            probe,
            cwd=str(self._repo_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            # Failing loudly matters: degrading to an empty grader env makes every plan die
            # inside the container with "LLM API KEYS is not set", which points debugging at
            # credentials rather than at this call.
            raise RuntimeError(
                f"env_creator failed (rc={proc.returncode}) for grader_model_name="
                f"{self.config.grader_model_name!r}: {stderr.decode(errors='replace')[-500:]}"
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

    async def _run_vibench_script(self, cmd: List[str]) -> tuple[int, str]:
        """Run one ViBench grading script, returning (return_code, merged output).

        A timeout is a failed grade, not an exception: one wedged compose stack should score
        zero rather than take the whole rollout down.
        """
        env = await self._grader_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._repo_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception:
            return 1, format_exc()

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self.config.evaluation_timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 1, f"timed out after {self.config.evaluation_timeout_s}s: {' '.join(cmd)}"

        code = proc.returncode if proc.returncode is not None else 1
        return code, self._redact((stdout or b"").decode(errors="replace"), env)

    async def _grade_one_test_plan(
        self,
        app_dir: Path,
        test_plan_rel: str,
        work_dir: Path,
        test_assets_dir: Optional[str],
    ) -> PlanResult:
        """Seed, then evaluate, a single test plan and parse its score."""
        started = time.time()
        name = Path(test_plan_rel).stem
        out_dir = work_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        plan_path = out_dir / "test-plan.txt"
        plan_path.write_text(add_evaluation_tags(self._resolve(test_plan_rel).read_text()))

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

        seed_dir = out_dir / "seed"
        seed_dir.mkdir(parents=True, exist_ok=True)
        code, log = await self._run_vibench_script(
            [
                sys.executable,
                str(self._repo_root / SEED_SCRIPT),
                "--app-dir",
                str(app_dir),
                "--test-plan",
                str(plan_path),
                "--output-dir",
                str(seed_dir),
            ]
        )
        # ViBench refuses to evaluate an app it could not seed; that is a real zero, tracked
        # separately from a bad build so aggregate metrics can tell them apart.
        if code != 0 or not (seed_dir / "seeding").is_dir():
            return fail(seeding_failed=True, error=log[-2000:])

        evaluate_cmd = [
            sys.executable,
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

        code, log = await self._run_vibench_script(evaluate_cmd)

        report_path = out_dir / "evaluation-finished.json"
        if not report_path.exists():
            # Seeding already succeeded to get here, so this is an evaluation failure.
            # Reporting it as a seeding failure would point debugging at the wrong stage --
            # which is exactly what it did the first time this fired.
            return fail(seeding_failed=False, error=log[-4000:])

        data = json.loads(report_path.read_text())
        score = float(data.get("score", 0) or 0)
        full_points = float(data.get("full_points", 0) or 0)
        steps = data.get("steps", []) or []

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
        if not body.artifact_path:
            # The agent could not produce a tarball at all (sandbox died, harness crashed).
            build_failed = True
        else:
            # Resolve first and never touch the raw path: deleting an unvalidated,
            # agent-supplied path would remove arbitrary files even when the path was
            # rejected for reading.
            artifact: Optional[Path] = None
            try:
                artifact = self._resolve_artifact(body.artifact_path)
                self._unpack_artifact(artifact, app_dir)
            except Exception:
                print(f"Failed to unpack artifact for {body.app}/{body.artifact}", format_exc(), file=sys.stderr)
                build_failed = True
            finally:
                if artifact is not None and self.config.remove_artifact_after_grading:
                    with suppress(Exception):
                        artifact.unlink(missing_ok=True)

        # An agent that produced nothing runnable is a build failure, not a 0-scoring app.
        if not build_failed and not (app_dir / "package.json").exists():
            build_failed = True
        artifact_extraction_time_s = time.time() - started

        results: List[PlanResult] = []
        grading_started = time.time()
        if not build_failed:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_test_plans)

            async def run(plan: str) -> PlanResult:
                async with semaphore:
                    return await self._grade_one_test_plan(app_dir, plan, work_dir, body.test_assets_dir)

            results = list(await asyncio.gather(*(run(p) for p in body.test_plans)))
        grading_time_s = time.time() - grading_started

        if not self.config.keep_evaluation_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)

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

    def compute_metrics(self, verify_responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Report where rollouts fail, not just what they scored.

        A mean reward alone cannot distinguish "the model wrote a weak app" from "the app
        never built" or "grading could not seed it" -- and those need completely different
        responses. Every zero in this environment has one of three causes, so they are
        surfaced separately.
        """
        if not verify_responses:
            return {}
        n = len(verify_responses)
        rewards = [float(r.get("reward", 0) or 0) for r in verify_responses]
        plans = sum(int(r.get("test_plans_total", 0) or 0) for r in verify_responses)
        graded = sum(int(r.get("test_plans_graded", 0) or 0) for r in verify_responses)
        return {
            "mean_reward": sum(rewards) / n,
            "perfect_rate": sum(1 for x in rewards if x >= 1.0) / n,
            "zero_rate": sum(1 for x in rewards if x <= 0.0) / n,
            "build_failure_rate": sum(1 for r in verify_responses if r.get("build_failed")) / n,
            "mean_seeding_failure_rate": sum(float(r.get("seeding_failure_rate", 0) or 0) for r in verify_responses)
            / n,
            # The share of plans that produced a scorecard at all: the health check that
            # separates a working verifier from a broken one.
            "plans_graded_rate": (graded / plans) if plans else 0.0,
        }

    def get_key_metrics(self) -> List[str]:
        return ["mean_reward", "plans_graded_rate", "build_failure_rate"]


if __name__ == "__main__":
    VibenchResourcesServer.run_webserver()
