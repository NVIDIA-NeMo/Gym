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

Topology mirrors ``resources_servers/swebench``:

  * ``seed_session`` starts the build sandbox from ViBench's codegen image, drops the
    PRD at ``/app/prd.txt``, and hands the sandbox id back to the agent. Any agent that
    consumes ``sandbox_handle`` works here -- see ``responses_api_agents/opencode_sandboxed_agent``.
  * The agent builds the app inside that sandbox. It never sees the test plans.
  * ``verify`` pulls ``/app`` out of the agent's sandbox and grades it by shelling out to
    ViBench's existing ``run-seed-then-evaluate.py`` once per test plan. Each run stands up
    the app + postgres + code-browse in a *fresh* compose project, seeds it with the seeding
    agent, then scores it with the evaluation agent.

P0 scope and known gaps are tracked in README.md. The two that matter most: grading runs
on the resources server's own Docker daemon (not in a Gym sandbox), and the verifier is
itself an LLM agent, so reward is stochastic -- profile its variance before using this for
training. ``REVERIFY_MODE`` is ``UNSUPPORTED`` for the same reason.
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
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY


# ViBench's coding agent reads its brief from this path inside the build container; the
# seeding/evaluation agents read the same PRD text from /app/prd/ at grade time.
PRD_PATH_IN_SANDBOX = "prd.txt"

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

    # Image the coding agent builds in. ViBench's Dockerfile.base, pre-built and tagged.
    build_image: str
    app_workdir: str = "/app"

    # Sandbox config, shaped like resources_servers/swebench/configs/swebench.yaml.
    sandbox_provider: str
    sandbox_config: Dict[str, Any]

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
    sandbox_spec: Optional[Dict[str, Any]] = None


class VibenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    workdir: str


class VibenchVerifyRequest(VibenchTaskRequest, BaseVerifyRequest):
    pass


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

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = dict()

    # ---------------------------------------------------------------- helpers

    @property
    def _repo_root(self) -> Path:
        return Path(self.config.vibench_repo_root).expanduser().resolve()

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a dataset path against the ViBench checkout, refusing escapes."""
        candidate = (self._repo_root / rel_path).resolve()
        if not candidate.is_relative_to(self._repo_root):
            raise ValueError(f"Path {rel_path!r} escapes vibench_repo_root")
        return candidate

    async def _create_build_sandbox(self, body: VibenchSeedSessionRequest) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)

        spec = SandboxSpec(
            image=self.config.build_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=self.config.app_workdir,
            env=self.config.sandbox_config.get("env", {}),
            files=dict(),
            metadata=provider_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "vibench_app": body.app[:63],
                "vibench_artifact": body.artifact[:63],
            },
            resources=SandboxResources.from_mapping(dict(self.config.sandbox_config.get("resources", {}))),
            entrypoint=None,
            provider_options=self.config.sandbox_config.get("provider_options", {}),
        )
        sandbox = AsyncSandbox(provider)
        await sandbox.start(spec)
        return sandbox

    async def _stage_prd(self, sandbox: AsyncSandbox, body: VibenchSeedSessionRequest) -> None:
        """Write the PRD (and any static assets) into the build sandbox.

        Feature artifacts concatenate their base PRDs in order, matching how ViBench's
        build-feature path presents prior context to the coding agent.
        """
        prd_text = "\n\n".join(self._resolve(p).read_text() for p in body.prd_files)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(prd_text)
            local_prd = Path(fh.name)
        try:
            await sandbox.upload(local_prd, f"{self.config.app_workdir}/{PRD_PATH_IN_SANDBOX}")
        finally:
            local_prd.unlink(missing_ok=True)

        for asset_dir in body.asset_dirs:
            src = self._resolve(asset_dir)
            if not src.is_dir():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                bundle = Path(tmp) / "assets.tar"
                with tarfile.open(bundle, "w") as tar:
                    tar.add(src, arcname="assets")
                remote_bundle = f"{self.config.app_workdir}/assets.tar"
                await sandbox.upload(bundle, remote_bundle)
                await sandbox.exec(
                    f"cd {self.config.app_workdir} && tar -xf assets.tar && rm assets.tar",
                )

    async def _extract_app(self, sandbox: AsyncSandbox, dest: Path) -> None:
        """Copy the built app out of the agent's sandbox into ``dest``."""
        remote_tar = "/tmp/vibench-app.tar"
        # Exclude the things that would poison a fresh grading build: node_modules is huge
        # and platform-specific, and prd.txt/assets are re-supplied by the grader.
        result = await sandbox.exec(
            f"cd {self.config.app_workdir} && tar "
            f"--exclude=./node_modules --exclude=./.git --exclude=./prd.txt "
            f"-cf {remote_tar} .",
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to tar app dir: {result.stderr or result.stdout}")

        with tempfile.TemporaryDirectory() as tmp:
            local_tar = Path(tmp) / "app.tar"
            await sandbox.download(remote_tar, local_tar)
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local_tar) as tar:
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
        sandbox = await self._create_build_sandbox(body)
        await self._stage_prd(sandbox, body)
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox
        return VibenchSeedSessionResponse(
            sandbox_handle=sandbox._handle.sandbox_id,
            workdir=self.config.app_workdir,
        )

    async def verify(self, request: Request, body: VibenchVerifyRequest) -> VibenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        sandbox = self._session_id_to_sandbox.pop(session_id, None)

        work_dir = Path(tempfile.mkdtemp(prefix=f"vibench-{body.app}-{body.artifact}-"))
        app_dir = work_dir / "app"

        started = time.time()
        build_failed = False
        if sandbox is None:
            build_failed = True
        else:
            try:
                await self._extract_app(sandbox, app_dir)
            except Exception:
                print(f"Failed to extract app for {body.app}/{body.artifact}", format_exc(), file=sys.stderr)
                build_failed = True
            finally:
                try:
                    await sandbox.stop()
                except Exception:
                    print("Failed to stop build sandbox", format_exc(), file=sys.stderr)

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
