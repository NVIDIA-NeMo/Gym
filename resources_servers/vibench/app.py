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
from typing import Dict, List, Optional

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


SEED_THEN_EVALUATE_SCRIPT = Path("_harness/runner/scripts/run-seed-then-evaluate.py")

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

    # .env supplying AGENT_SEEDING_LLM_* / AGENT_EVALUATION_LLM_* keys for the grader agents.
    # These are the *verifier's* models and are deliberately not the policy model.
    vibench_env_file: Optional[str] = None

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
            tar.extractall(dest)

    def _grader_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env_file = self.config.vibench_env_file
        if env_file:
            for line in Path(env_file).expanduser().read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
        return env

    async def _grade_one_test_plan(
        self,
        app_dir: Path,
        test_plan_rel: str,
        prd_paths: List[Path],
        work_dir: Path,
    ) -> PlanResult:
        """Run ViBench's seed-then-evaluate for a single test plan and parse its score."""
        started = time.time()
        name = Path(test_plan_rel).stem
        out_dir = work_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        plan_path = out_dir / "test-plan.txt"
        plan_path.write_text(add_evaluation_tags(self._resolve(test_plan_rel).read_text()))

        cmd = [
            sys.executable,
            str(self._repo_root / SEED_THEN_EVALUATE_SCRIPT),
            "--app-dir",
            str(app_dir),
            "--test-plan",
            str(plan_path),
            "--output-dir",
            str(out_dir),
            "--prd-files",
            *[str(p) for p in prd_paths],
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._repo_root),
                env=self._grader_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self.config.evaluation_timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return PlanResult(
                    test_plan=name,
                    score=0.0,
                    full_points=0.0,
                    normalized_score=0.0,
                    steps_total=0,
                    steps_passed=0,
                    seeding_failed=False,
                    error=f"timed out after {self.config.evaluation_timeout_s}s",
                    duration_s=time.time() - started,
                )
        except Exception:
            return PlanResult(
                test_plan=name,
                score=0.0,
                full_points=0.0,
                normalized_score=0.0,
                steps_total=0,
                steps_passed=0,
                seeding_failed=False,
                error=format_exc(),
                duration_s=time.time() - started,
            )

        report_path = out_dir / "evaluation-finished.json"
        if not report_path.exists():
            # ViBench refuses to evaluate when seeding fails; that is a real zero, but we
            # track it separately so aggregate metrics can tell it apart from a bad build.
            tail = (stdout or b"").decode(errors="replace")[-2000:]
            return PlanResult(
                test_plan=name,
                score=0.0,
                full_points=0.0,
                normalized_score=0.0,
                steps_total=0,
                steps_passed=0,
                seeding_failed=True,
                error=tail,
                duration_s=time.time() - started,
            )

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
            try:
                self._unpack_artifact(self._resolve_artifact(body.artifact_path), app_dir)
            except Exception:
                print(f"Failed to unpack artifact for {body.app}/{body.artifact}", format_exc(), file=sys.stderr)
                build_failed = True
            finally:
                if self.config.remove_artifact_after_grading:
                    with suppress(Exception):
                        Path(body.artifact_path).unlink(missing_ok=True)

        # An agent that produced nothing runnable is a build failure, not a 0-scoring app.
        if not build_failed and not (app_dir / "package.json").exists():
            build_failed = True
        artifact_extraction_time_s = time.time() - started

        results: List[PlanResult] = []
        grading_started = time.time()
        if not build_failed:
            prd_paths = [self._resolve(p) for p in body.prd_files]
            semaphore = asyncio.Semaphore(self.config.max_concurrent_test_plans)

            async def run(plan: str) -> PlanResult:
                async with semaphore:
                    return await self._grade_one_test_plan(app_dir, plan, prd_paths, work_dir)

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


if __name__ == "__main__":
    VibenchResourcesServer.run_webserver()
