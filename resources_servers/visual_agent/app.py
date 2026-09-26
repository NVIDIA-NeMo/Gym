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
"""Visual agent tasks (MiMo-V2.6 §4.2.3): websites, interactive apps, games, 3D scenes, slides,
SVG, videos and Figma-style designs, graded by an agentic verifier.

Rollout lifecycle
1. `seed_session` starts the policy sandbox from the task's Docker image. It uploads the task
   files and vendored libraries, then returns the sandbox handle to the OpenCode agent. The
   agent gets no grader tooling: `vtools.py` and the rubric live only in the grader sandbox.
2. The agent works in `/workspace/output/`.
3. `verify` tars the output folder and stops the policy sandbox. It then grades in a fresh
   sandbox so nothing the policy left behind can influence grading:
   a. a deterministic pass (`vtools.py measure`): renders, runtime and layout checks, and for
      replication tasks pixel similarity to the reference plus reference-copy detection;
   b. an agentic judge: OpenCode running the judge model views the renders, exercises
      interactive artifacts and decides each rubric item, writing `verdict.json`;
   c. `grading.compute_reward` combines the two.
4. Optionally, for open-ended tasks, a groupwise judge compares the rendered artifacts of one
   rollout group and shifts clearly stronger/weaker candidates' rewards (`groupwise`).
"""

import asyncio
import hashlib
import json
import logging
import random
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from typing import Any, Awaitable, Callable, ClassVar, Dict, List, Literal, Optional

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import JUDGE_FAILED, SESSION_LOST, VERIFIER_ERROR
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY, get_server_url
from resources_servers.visual_agent.grading import (
    ItemResult,
    ParsedVerdict,
    RewardConfig,
    RubricItem,
    compute_reward,
    groupwise_adjust,
    parse_group_verdict,
    parse_verdict,
    runtime_gate,
)
from resources_servers.visual_agent.opencode_runner import (
    JudgeOpenCodeConfig,
    OpenCodeRunResult,
    exec_checked,
    list_files,
    run_opencode_judge,
    upload_tree,
)
from resources_servers.visual_agent.prompts import (
    GRADER_ARTIFACT_DIR,
    GRADER_DIR,
    GRADER_MEASUREMENTS,
    GRADER_REFERENCE_DIR,
    GRADER_RENDER_DIR,
    GRADER_TOOLS_PATH,
    GRADER_VERDICT,
    GROUP_DIR,
    GROUP_VERDICT,
    OUTPUT_DIR,
    group_judge_prompt,
    judge_prompt,
    request_text_from_input,
)


LOG = logging.getLogger(__name__)
SERVER_DIR = Path(__file__).parent
VTOOLS_LOCAL = SERVER_DIR / "sandbox_tools" / "vtools.py"
WORKSPACE = "/workspace"


class ArtifactSpec(BaseModel):
    kind: Literal["html", "slides", "svg", "video"]
    entry: str


class VisualTask(BaseModel):
    """Task fields every dataset row carries (next to `responses_create_params`)."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    category: Literal["website", "interactive_app", "game", "3d_scene", "slides", "svg", "video", "figma"]
    mode: Literal["open_ended", "replication"]
    artifact: ArtifactSpec
    rubric: List[RubricItem]
    viewport: Optional[Dict[str, int]] = None
    checks: Dict[str, bool] = Field(default_factory=dict)
    vendor: List[Literal["three", "chartjs"]] = Field(default_factory=list)
    assets: List[str] = Field(default_factory=list)
    reference_images: List[str] = Field(default_factory=list)
    reference_frame_times: Optional[List[float]] = None
    # Responsive replication: the page is captured full-page at each viewport and compared with the
    # reference image at the same index.
    reference_viewports: Optional[List[Dict[str, int]]] = None
    video_spec: Optional[Dict[str, Any]] = None
    slides_spec: Optional[Dict[str, Any]] = None
    interaction_hints: Optional[str] = None
    docker_image: Optional[str] = None


class GroupwiseConfig(BaseModel):
    # Compare the renders of one rollout group and shift clearly stronger/weaker candidates.
    enabled: bool = False
    # Rollouts per task; must equal the dataset's num_repeats.
    group_size: int = Field(default=1, ge=1)
    # How long a verify waits for the rest of its group before keeping its pointwise reward.
    collection_timeout_s: float = Field(default=3600, gt=0)
    bonus: float = Field(default=0.15, ge=0, le=1)
    min_gated_candidates: int = Field(default=2, ge=2)


class VisualAgentResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    sandbox_provider: str
    sandbox_config: Dict[str, Any]
    default_docker_image: str = "apify/actor-python-playwright:3.12-1.63.0"
    # Bash script run once in every policy and grader sandbox (path relative to the Gym repo root).
    # It must leave numpy, Pillow, ffmpeg, node and the fonts the prompts mention available.
    sandbox_setup_script: str = "resources_servers/visual_agent/sandbox_tools/setup_sandbox.sh"
    setup_timeout_s: float = 900

    # Relative paths resolve against the Gym repo root.
    data_dir: str = "resources_servers/visual_agent/data"
    grading_output_dir: str = "resources_servers/visual_agent/results"
    max_artifact_bytes: int = 200 * 1024 * 1024
    measure_timeout_s: float = 900
    max_concurrent_graders: int = Field(default=1024, ge=1)

    judge_model_server: ModelServerRef
    # Explicit OpenAI-compatible base URL (".../v1") for the judge; overrides judge_model_server.
    judge_base_url: Optional[str] = None
    judge: JudgeOpenCodeConfig = Field(default_factory=JudgeOpenCodeConfig)
    judge_max_attempts: int = Field(default=2, ge=1)
    # Mark samples whose judge never produced a usable verdict as not validly measured.
    mask_on_judge_failure: bool = True

    reward: RewardConfig = Field(default_factory=RewardConfig)
    groupwise: GroupwiseConfig = Field(default_factory=GroupwiseConfig)

    debug: bool = False


class VisualAgentSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str


class VisualAgentVerifyRequest(VisualTask, BaseVerifyRequest):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task_index: Optional[int] = Field(default=None, alias=TASK_INDEX_KEY_NAME)
    rollout_index: Optional[int] = Field(default=None, alias=ROLLOUT_INDEX_KEY_NAME)


class JudgeOutcome(BaseModel):
    status: Literal["ok", "failed", "skipped"]
    attempts: int = 0
    verdict: Optional[ParsedVerdict] = None
    runs: List[OpenCodeRunResult] = Field(default_factory=list)


class VisualAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    category: str
    mode: str
    pointwise_reward: float
    rubric_score: Optional[float] = None
    similarity: Optional[float] = None
    gate_passed: bool
    gate_reasons: List[str] = Field(default_factory=list)
    hack_detected: bool = False
    hack_reasons: List[str] = Field(default_factory=list)
    rubric_results: List[ItemResult] = Field(default_factory=list)
    judge_status: str
    judge_attempts: int = 0
    judge_summary: str = ""
    judge_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    judge_runs: List[OpenCodeRunResult] = Field(default_factory=list)
    groupwise_status: str = "disabled"
    groupwise_tier: Optional[int] = None
    measurements_summary: Dict[str, Any] = Field(default_factory=dict)
    grading_dir: str = ""
    verification_time_s: float = 0.0


@dataclass
class _GroupMember:
    pointwise: float
    gate_passed: bool
    rubric_score: float
    grading_dir: Path


@dataclass
class _Cohort:
    members: Dict[int, _GroupMember] = field(default_factory=dict)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    rewards: Dict[int, float] = field(default_factory=dict)
    tiers: Dict[int, int] = field(default_factory=dict)
    status: str = "collecting"
    task: Optional[asyncio.Task] = None
    created_at: float = field(default_factory=time.monotonic)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PARENT_DIR / p


def _measurement_summary(m: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("artifact_found", "load_ok", "blank", "animated", "slide_count", "file_count", "artifact_bytes")
    summary = {k: m[k] for k in keys if k in m}
    runtime = m.get("runtime") or {}
    summary["page_errors"] = (runtime.get("page_errors") or [])[:3]
    summary["external_requests"] = len(runtime.get("external_requests") or [])
    if m.get("similarity"):
        summary["similarity"] = {k: m["similarity"].get(k) for k in ("normalized", "combined_mean")}
    if m.get("video"):
        summary["video"] = {k: m["video"].get(k) for k in ("width", "height", "fps", "duration_s", "codec")}
    if m.get("reference_copy", {}).get("detected"):
        summary["reference_copy"] = m["reference_copy"]["findings"][:3]
    return summary


class VisualAgentResourcesServer(SimpleResourcesServer):
    config: VisualAgentResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: Dict[str, AsyncSandbox] = {}
        self._grader_semaphore = asyncio.Semaphore(self.config.max_concurrent_graders)
        self._cohorts: Dict[str, _Cohort] = {}
        self._cohort_lock = asyncio.Lock()
        self._setup_command = f"bash -c {quote(_resolve(self.config.sandbox_setup_script).read_text())}"

    # ------------------------------------------------------------------------------------------
    # Sandboxes
    # ------------------------------------------------------------------------------------------
    async def _create_sandbox(self, image: str, role: str, task_id: str) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config_dict))
        default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        resources = SandboxResources.from_mapping(self.config.sandbox_config.get("resources", {}))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(resources.cpu) | env
        spec = SandboxSpec(
            image=image,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=None,
            env=env,
            files={},
            metadata=default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {"nemo_gym_agent": self.config.name, "role": role, "instance_id": task_id[:63]},
            resources=resources,
            entrypoint=None,
            provider_options=dict(self.config.sandbox_config.get("provider_options") or {}),
        )
        sandbox = AsyncSandbox(provider)

        async def setup(sb: AsyncSandbox) -> None:
            await exec_checked(sb, self._setup_command, timeout_s=self.config.setup_timeout_s, what="sandbox setup")

        await sandbox.start_with_setup(spec, setup)
        return sandbox

    @staticmethod
    async def _stop(sandbox: Optional[AsyncSandbox]) -> None:
        if sandbox is None:
            return
        try:
            await sandbox.stop()
        except Exception:
            LOG.warning("Failed to stop sandbox", exc_info=True)

    def _task_files(self, task: VisualTask) -> Dict[str, Path]:
        data_dir = _resolve(self.config.data_dir)
        files: Dict[str, Path] = {}
        for name in list(task.reference_images) + list(task.assets):
            path = data_dir / "assets" / task.task_id / name
            if not path.is_file():
                raise FileNotFoundError(f"Task {task.task_id} asset {path} is missing; run the data build script")
            files[f"task/{name}"] = path
        for lib in task.vendor:
            lib_dir = data_dir / "vendor" / lib
            if not lib_dir.is_dir():
                raise FileNotFoundError(f"Vendor library {lib_dir} is missing; run fetch_vendor.py")
            files |= list_files(lib_dir, prefix=f"output/vendor/{lib}/")
        return files

    # ------------------------------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------------------------------
    async def seed_session(self, request: Request, body: VisualTask) -> VisualAgentSeedSessionResponse:
        image = body.docker_image or self.config.default_docker_image
        sandbox = await self._create_sandbox(image, "policy", body.task_id)
        try:
            files = self._task_files(body)
            if files:
                await upload_tree(sandbox, files, WORKSPACE)
            await exec_checked(sandbox, f"mkdir -p {OUTPUT_DIR}", what="workspace dirs")
        except Exception:
            await self._stop(sandbox)
            raise
        self._session_id_to_sandbox[request.session[SESSION_ID_KEY]] = sandbox
        return VisualAgentSeedSessionResponse(sandbox_handle=sandbox._handle.sandbox_id)

    # ------------------------------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------------------------------
    async def _collect_artifact(self, sandbox: AsyncSandbox, dest: Path) -> Path:
        size_out = await exec_checked(sandbox, f"du -sb {OUTPUT_DIR} | cut -f1", timeout_s=120, what="artifact size")
        size = int((size_out.strip().splitlines() or ["0"])[-1])
        if size > self.config.max_artifact_bytes:
            raise ValueError(
                f"Output folder is {size} bytes, above max_artifact_bytes={self.config.max_artifact_bytes}"
            )
        remote = f"/tmp/artifact_{dest.parent.name}.tar.gz"
        await exec_checked(
            sandbox,
            f"tar czf {remote} --exclude=node_modules --exclude=.git -C {OUTPUT_DIR} .",
            timeout_s=600,
            what="pack artifact",
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        await sandbox.download(remote, dest)
        return dest

    def _grading_dir(self, body: VisualAgentVerifyRequest, session_id: str) -> Path:
        slot = f"t{body.task_index}_r{body.rollout_index}" if body.task_index is not None else "t_r"
        digest = hashlib.sha1(session_id.encode()).hexdigest()[:10]
        return _resolve(self.config.grading_output_dir) / body.task_id / f"{slot}_{digest}"

    def _judge_base_url(self) -> str:
        if self.config.judge_base_url:
            return self.config.judge_base_url.rstrip("/")
        return get_server_url(self.config.judge_model_server.name) + "/v1"

    async def _run_judge(
        self,
        graders: List[AsyncSandbox],
        restart: Callable[[], Awaitable[AsyncSandbox]],
        body: VisualAgentVerifyRequest,
        measurements: Dict[str, Any],
        grading_dir: Path,
    ) -> JudgeOutcome:
        """Run judge attempts in `graders[-1]`; a sandbox that died during an attempt (e.g. OOM-killed)
        is replaced through `restart` before the next one."""
        prompt = judge_prompt(
            body.model_dump(),
            request_text_from_input(body.responses_create_params.input),
            body.rubric,
            measurements,
        )
        (grading_dir / "judge_prompt.md").write_text(prompt)
        outcome = JudgeOutcome(status="failed")
        for attempt in range(1, self.config.judge_max_attempts + 1):
            outcome.attempts = attempt
            if attempt > 1 and outcome.runs[-1].error_type is not None:
                await self._stop(graders[-1])
                graders.append(await restart())
            grader = graders[-1]
            try:
                await grader.exec(f"rm -f {GRADER_VERDICT}", timeout_s=60)
            except Exception:
                # The sandbox died between attempts; the judge run below records the failure.
                LOG.warning("Grader sandbox for %s is unreachable before judge attempt %d", body.task_id, attempt)
            run = await run_opencode_judge(
                grader,
                prompt=prompt,
                model_base_url=self._judge_base_url(),
                config=self.config.judge,
                workdir=GRADER_DIR,
                export_local_path=grading_dir / f"judge_export_{attempt}.json",
                title=f"judge-{body.task_id}-{attempt}",
            )
            outcome.runs.append(run)
            local_verdict = grading_dir / f"verdict_{attempt}.json"
            try:
                await grader.download(GRADER_VERDICT, local_verdict)
                verdict = parse_verdict(local_verdict.read_text(), body.rubric)
            except Exception as exc:
                verdict = ParsedVerdict(valid=False, error=f"no verdict file: {type(exc).__name__}")
            outcome.verdict = verdict
            if verdict.valid:
                outcome.status = "ok"
                break
            LOG.warning("Judge attempt %d for %s produced no usable verdict: %s", attempt, body.task_id, verdict.error)
        return outcome

    async def _start_grader(
        self, body: VisualAgentVerifyRequest, artifact_tgz: Path, grading_dir: Path, *, restore: bool
    ) -> AsyncSandbox:
        """A fresh grader sandbox with the unpacked artifact, references, input files and task.

        With `restore`, it also gets the renders and measurements saved by an earlier grader, so a
        judge attempt can continue after that sandbox died.
        """
        grader = await self._create_sandbox(
            body.docker_image or self.config.default_docker_image, "grader", body.task_id
        )
        try:
            task_dict = body.model_dump(exclude={"responses_create_params", "response"})
            files: Dict[str, Path] = {"tools/vtools.py": VTOOLS_LOCAL, "artifact.tar.gz": artifact_tgz}
            data_dir = _resolve(self.config.data_dir)
            for name in body.reference_images:
                files[f"reference/{name}"] = data_dir / "assets" / body.task_id / name
            for name in body.assets:
                files[f"task/{name}"] = data_dir / "assets" / body.task_id / name
            if restore:
                files["measurements.json"] = grading_dir / "measurements.json"
                files |= list_files(grading_dir / "renders", prefix="renders/")
            with TemporaryDirectory() as tmp:
                task_json = Path(tmp) / "task.json"
                task_json.write_text(json.dumps(task_dict, indent=1, default=str))
                files["task.json"] = task_json
                await upload_tree(grader, files, GRADER_DIR)
            await exec_checked(
                grader,
                f"mkdir -p {GRADER_ARTIFACT_DIR} && tar xzf {GRADER_DIR}/artifact.tar.gz -C {GRADER_ARTIFACT_DIR} "
                f"&& rm {GRADER_DIR}/artifact.tar.gz",
                what="unpack artifact",
            )
        except Exception:
            await self._stop(grader)
            raise
        return grader

    async def _grade(
        self, body: VisualAgentVerifyRequest, artifact_tgz: Path, grading_dir: Path
    ) -> tuple[Dict[str, Any], JudgeOutcome]:
        graders = [await self._start_grader(body, artifact_tgz, grading_dir, restore=False)]
        try:
            grader = graders[0]
            reference_arg = f"--reference-dir {GRADER_REFERENCE_DIR}" if body.reference_images else ""
            await exec_checked(
                grader,
                f"cd {GRADER_DIR} && python3 {GRADER_TOOLS_PATH} measure --task {GRADER_DIR}/task.json "
                f"--artifact-dir {GRADER_ARTIFACT_DIR} {reference_arg} --out-dir {GRADER_RENDER_DIR} "
                f"--report {GRADER_MEASUREMENTS}",
                timeout_s=self.config.measure_timeout_s,
                what="measure",
            )
            await grader.download(GRADER_MEASUREMENTS, grading_dir / "measurements.json")
            measurements = json.loads((grading_dir / "measurements.json").read_text())
            await exec_checked(grader, f"tar czf /tmp/renders.tgz -C {GRADER_RENDER_DIR} .", what="pack renders")
            await grader.download("/tmp/renders.tgz", grading_dir / "renders.tgz")
            with tarfile.open(grading_dir / "renders.tgz") as tar:
                tar.extractall(grading_dir / "renders", filter="data")

            if runtime_gate(measurements):
                return measurements, JudgeOutcome(status="skipped")

            async def restart() -> AsyncSandbox:
                return await self._start_grader(body, artifact_tgz, grading_dir, restore=True)

            return measurements, await self._run_judge(graders, restart, body, measurements, grading_dir)
        finally:
            await self._stop(graders[-1])

    async def verify(self, request: Request, body: VisualAgentVerifyRequest) -> VisualAgentVerifyResponse:
        started = time.monotonic()
        session_id = request.session[SESSION_ID_KEY]
        policy_sandbox = self._session_id_to_sandbox.pop(session_id, None)
        grading_dir = self._grading_dir(body, session_id)
        if grading_dir.exists():
            shutil.rmtree(grading_dir)
        grading_dir.mkdir(parents=True)

        common = dict(task_id=body.task_id, category=body.category, mode=body.mode, grading_dir=str(grading_dir))
        async with self._grader_semaphore:
            try:
                if policy_sandbox is None:
                    raise RuntimeError(
                        "No policy sandbox for this session (seed_session was not called or it expired)"
                    )
                artifact_tgz = await self._collect_artifact(policy_sandbox, grading_dir / "artifact.tar.gz")
            except Exception as exc:
                LOG.exception("Could not collect the artifact for %s", body.task_id)
                return await self._failure(
                    body, common, started, SESSION_LOST, f"artifact collection failed: {exc}", masked=True
                )
            finally:
                await self._stop(policy_sandbox)

            try:
                measurements, judge = await self._grade(body, artifact_tgz, grading_dir)
            except Exception as exc:
                LOG.exception("Grading failed for %s", body.task_id)
                return await self._failure(
                    body, common, started, VERIFIER_ERROR, f"grading failed: {exc}", masked=True
                )

        breakdown = compute_reward(body.model_dump(), body.rubric, measurements, judge.verdict, self.config.reward)
        verdict = judge.verdict
        mask = False
        failure_kind = None
        failure_reason = None
        if judge.status == "failed":
            failure_kind = JUDGE_FAILED
            failure_reason = (verdict.error if verdict else None) or "judge produced no usable verdict"
            mask = self.config.mask_on_judge_failure

        reward = breakdown.reward
        groupwise_status = "disabled"
        groupwise_tier = None
        if self.config.groupwise.enabled:
            if body.mode != "open_ended":
                groupwise_status = "not_applicable"
            else:
                reward, groupwise_status, groupwise_tier = await self._groupwise(
                    body,
                    breakdown.reward,
                    breakdown.gate_passed and not mask,
                    breakdown.rubric_score or 0.0,
                    grading_dir,
                )

        response = VisualAgentVerifyResponse(
            **(body.model_dump() | common),
            reward=reward,
            pointwise_reward=breakdown.reward,
            rubric_score=breakdown.rubric_score,
            similarity=breakdown.similarity,
            gate_passed=breakdown.gate_passed,
            gate_reasons=breakdown.gate_reasons,
            hack_detected=breakdown.hack_detected,
            hack_reasons=breakdown.hack_reasons,
            rubric_results=breakdown.items,
            judge_status=judge.status,
            judge_attempts=judge.attempts,
            judge_summary=verdict.summary if verdict else "",
            judge_scores={"aesthetics": verdict.aesthetics, "fidelity": verdict.fidelity} if verdict else {},
            judge_runs=judge.runs,
            groupwise_status=groupwise_status,
            groupwise_tier=groupwise_tier,
            measurements_summary=_measurement_summary(measurements),
            verification_time_s=round(time.monotonic() - started, 2),
            mask_sample=mask,
            failure_kind=failure_kind,
            failure_reason=failure_reason,
        )
        (grading_dir / "result.json").write_text(
            response.model_dump_json(indent=1, exclude={"responses_create_params", "response"})
        )
        return response

    async def _failure(
        self,
        body: VisualAgentVerifyRequest,
        common: Dict[str, Any],
        started: float,
        kind: str,
        reason: str,
        *,
        masked: bool,
    ) -> VisualAgentVerifyResponse:
        groupwise_status = "disabled"
        if self.config.groupwise.enabled:
            groupwise_status = "not_applicable"
            if body.mode == "open_ended":
                # Join as an ungated member so the rest of the group is not left waiting for it.
                member = _GroupMember(0.0, False, 0.0, Path(common["grading_dir"]))
                cohort, status = await self._join_cohort(body, member)
                groupwise_status = "excluded" if cohort is not None else status
        return VisualAgentVerifyResponse(
            **(body.model_dump() | common),
            reward=0.0,
            pointwise_reward=0.0,
            gate_passed=False,
            gate_reasons=[reason],
            judge_status="skipped",
            groupwise_status=groupwise_status,
            verification_time_s=round(time.monotonic() - started, 2),
            mask_sample=masked,
            failure_kind=kind,
            failure_reason=reason[:2000],
        )

    # ------------------------------------------------------------------------------------------
    # Groupwise grading (open-ended tasks)
    # ------------------------------------------------------------------------------------------
    async def _join_cohort(
        self, body: VisualAgentVerifyRequest, member: _GroupMember
    ) -> tuple[Optional[_Cohort], str]:
        """Register a rollout with its group; the last member to arrive starts the group judge."""
        cfg = self.config.groupwise
        if body.rollout_index is None or cfg.group_size <= 1:
            return None, "not_applicable"
        key = f"{body.task_index if body.task_index is not None else body.task_id}"
        async with self._cohort_lock:
            cohort = self._cohorts.setdefault(key, _Cohort())
            if cohort.status != "collecting" and body.rollout_index not in cohort.members:
                # A late member from an earlier, already-evaluated group (e.g. a retried rollout).
                return None, "late"
            cohort.members[body.rollout_index] = member
            if len(cohort.members) >= cfg.group_size and cohort.status == "collecting":
                cohort.status = "evaluating"
                cohort.task = asyncio.create_task(self._evaluate_cohort(key, body, cohort))
        return cohort, "joined"

    async def _groupwise(
        self, body: VisualAgentVerifyRequest, pointwise: float, gated: bool, rubric_score: float, grading_dir: Path
    ) -> tuple[float, str, Optional[int]]:
        cfg = self.config.groupwise
        cohort, status = await self._join_cohort(body, _GroupMember(pointwise, gated, rubric_score, grading_dir))
        if cohort is None:
            return pointwise, status, None
        remaining = cfg.collection_timeout_s - (time.monotonic() - cohort.created_at)
        try:
            await asyncio.wait_for(asyncio.shield(cohort.done.wait()), timeout=max(1.0, remaining))
        except asyncio.TimeoutError:
            return pointwise, "timeout", None
        if cohort.status != "applied":
            return pointwise, cohort.status, None
        index = body.rollout_index
        return cohort.rewards.get(index, pointwise), "applied", cohort.tiers.get(index)

    async def _evaluate_cohort(self, key: str, body: VisualAgentVerifyRequest, cohort: _Cohort) -> None:
        cfg = self.config.groupwise
        try:
            gated = {i: m for i, m in cohort.members.items() if m.gate_passed}
            if len(gated) < cfg.min_gated_candidates:
                cohort.status = "too_few_candidates"
                return
            order = sorted(gated)
            random.Random(key).shuffle(order)
            labels = {index: chr(ord("A") + n) for n, index in enumerate(order)}
            tiers_by_label = await self._run_group_judge(body, {labels[i]: gated[i] for i in order})
            if tiers_by_label is None:
                cohort.status = "failed"
                return
            cohort.tiers = {i: tiers_by_label[labels[i]] for i in order}
            pointwise = {i: m.pointwise for i, m in cohort.members.items()}
            cohort.rewards = groupwise_adjust(pointwise, cohort.tiers, cfg.bonus)
            cohort.status = "applied"
        except Exception:
            LOG.exception("Groupwise grading failed for cohort %s", key)
            cohort.status = "failed"
        finally:
            cohort.done.set()
            # Keep finished cohorts briefly for late waiters, then drop them.
            asyncio.get_running_loop().call_later(cfg.collection_timeout_s, self._cohorts.pop, key, None)

    async def _run_group_judge(
        self, body: VisualAgentVerifyRequest, candidates: Dict[str, _GroupMember]
    ) -> Optional[Dict[str, int]]:
        image = body.docker_image or self.config.default_docker_image
        sandbox = await self._create_sandbox(image, "group-grader", body.task_id)
        group_dir = next(iter(candidates.values())).grading_dir.parent / f"group_{int(time.time())}"
        group_dir.mkdir(parents=True, exist_ok=True)
        try:
            files: Dict[str, Path] = {"vtools.py": VTOOLS_LOCAL}
            info: List[Dict[str, Any]] = []
            for label, member in candidates.items():
                files |= list_files(member.grading_dir / "renders", prefix=f"{label}/renders/")
                files[f"{label}/artifact.tar.gz"] = member.grading_dir / "artifact.tar.gz"
                files[f"{label}/measurements.json"] = member.grading_dir / "measurements.json"
                render_names = sorted(p.name for p in (member.grading_dir / "renders").glob("*.png"))
                info.append({"label": label, "rubric_score": member.rubric_score, "render_names": render_names})
            await upload_tree(sandbox, files, GROUP_DIR)
            unpack = " && ".join(
                f"mkdir -p {GROUP_DIR}/{label}/artifact && tar xzf {GROUP_DIR}/{label}/artifact.tar.gz "
                f"-C {GROUP_DIR}/{label}/artifact"
                for label in candidates
            )
            await exec_checked(sandbox, unpack, what="unpack group artifacts")
            prompt = group_judge_prompt(
                body.model_dump(),
                request_text_from_input(body.responses_create_params.input),
                sorted(info, key=lambda c: c["label"]),
            )
            (group_dir / "group_prompt.md").write_text(prompt)
            labels = sorted(candidates)
            for attempt in range(1, self.config.judge_max_attempts + 1):
                await run_opencode_judge(
                    sandbox,
                    prompt=prompt,
                    model_base_url=self._judge_base_url(),
                    config=self.config.judge,
                    workdir=GROUP_DIR,
                    export_local_path=group_dir / f"group_export_{attempt}.json",
                    title=f"group-judge-{body.task_id}-{attempt}",
                )
                local = group_dir / f"ranking_{attempt}.json"
                try:
                    await sandbox.download(GROUP_VERDICT, local)
                except Exception:
                    continue
                tiers = parse_group_verdict(local.read_text(), labels)
                if tiers is not None:
                    (group_dir / "tiers.json").write_text(json.dumps(tiers, indent=1))
                    return tiers
            return None
        finally:
            await self._stop(sandbox)

    # ------------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------------
    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task in tasks for r in task]
        if not rows:
            return {}
        metrics: Dict[str, Any] = {}

        def mean(values: List[float]) -> Optional[float]:
            return round(sum(values) / len(values), 6) if values else None

        scored = [r for r in rows if not r.get("mask_sample")]
        for group_key in ("category", "mode"):
            for value in sorted({str(r.get(group_key)) for r in scored}):
                subset = [r for r in scored if str(r.get(group_key)) == value]
                metrics[f"pointwise_reward/{group_key}/{value}"] = mean(
                    [r.get("pointwise_reward", 0.0) for r in subset]
                )
        metrics["rate/gate_failed"] = mean([0.0 if r.get("gate_passed") else 1.0 for r in rows])
        metrics["rate/judge_failed"] = mean([1.0 if r.get("judge_status") == "failed" else 0.0 for r in rows])
        metrics["rate/hack_detected"] = mean([1.0 if r.get("hack_detected") else 0.0 for r in rows])
        similarities = [r["similarity"] for r in scored if r.get("similarity") is not None]
        metrics["mean_similarity/replication"] = mean(similarities)
        applied = [r for r in rows if r.get("groupwise_status") == "applied"]
        if applied:
            metrics["rate/groupwise_applied"] = mean(
                [1.0 if r in applied else 0.0 for r in rows if r.get("mode") == "open_ended"]
            )
            metrics["groupwise/mean_abs_shift"] = mean([abs(r["reward"] - r["pointwise_reward"]) for r in applied])
        return {k: v for k, v in metrics.items() if v is not None}

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("mean/reward", "mean/pointwise_reward", "mean/rubric_score", "mean/similarity")
        key_metrics = {k: agent_metrics[k] for k in keys if k in agent_metrics}
        key_metrics |= {k: v for k, v in agent_metrics.items() if k.startswith("pointwise_reward/")}
        return key_metrics


if __name__ == "__main__":
    VisualAgentResourcesServer.run_webserver()
