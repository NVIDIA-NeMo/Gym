# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ORAgentBench resources server.

Each task is a Harbor task directory from https://github.com/ORAgentBench/ORAgentBench (MIT):
the agent works inside a per-task container and writes decision artefacts under
``/app/submissions``; upstream's own validator (``tests/test.sh`` + ``evaluate_solution.py``)
then runs inside the same container and writes ``/logs/verifier/reward.json`` with
``feasibility`` in {0, 1} and ``quality`` on [0, 2] (1.0 = matches the reference objective).

The reward is the paper's pass predicate, ported literally from upstream's
``scripts/summarize_results.py``: a task passes iff ``feasibility > 0`` and
``quality / 2 > 0.4``. Eight tasks are Harbor multi-step tasks (``[[steps]]`` in
``task.toml``); for those the step verifiers run between agent phases inside one container
(they write the carried-forward state the next step reads), feasibility is the conjunction
over steps and quality the mean over steps with a missing step counting 0, again as upstream
aggregates them.
"""

import json
import math
import tomllib
from copy import deepcopy
from enum import Enum
from glob import glob
from pathlib import Path
from shlex import quote
from sys import stderr
from tempfile import TemporaryDirectory
from time import time
from traceback import format_exc
from typing import Any, ClassVar, Dict, List, Literal, Optional

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
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY


# Upstream ``scripts/summarize_results.py`` at c9eb952435a4352f33daa2a35efe0f8c76d31b28:
# PASS_QUALITY_NORMALIZED_CUTOFF = 0.4; is_pass = feasibility > 0 and quality_raw / 2 > cutoff.
PASS_QUALITY_NORMALIZED_CUTOFF = 0.4
QUALITY_RAW_MAX = 2.0
DIFFICULTIES = ("easy", "medium", "hard")

# Container paths, matching Harbor's EnvironmentPaths so upstream scripts find what they expect.
TESTS_DIR = "/tests"
SOLUTION_DIR = "/solution"
VERIFIER_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"


class Status(str, Enum):
    # Outcomes that are judgements on the policy: the agent can cause every one of them.
    SCORED = "scored"
    MISSING_SOLUTION = "missing_solution"
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_OUTPUT_MISSING = "verifier_output_missing"
    STEP_ABORTED = "step_aborted"
    STEP_INCOMPLETE = "step_incomplete"
    # Harness faults: provably not the policy's doing. ``failure_reason`` is set for exactly these.
    BAD_TASK_FOLDER = "bad_task_folder"
    SANDBOX_FAILED = "sandbox_failed"
    STEP_SETUP_FAILED = "step_setup_failed"
    TESTS_UPLOAD_FAILED = "tests_upload_failed"
    NO_SESSION = "no_session"


HARNESS_FAULTS = {
    Status.BAD_TASK_FOLDER: "task.toml, instruction.md or tests/ under task_folder is missing or unreadable",
    Status.SANDBOX_FAILED: "the task container could not be started (model-free validation mode)",
    Status.STEP_SETUP_FAILED: "upstream's step workdir/setup.sh exited non-zero before the agent ran",
    Status.TESTS_UPLOAD_FAILED: "the task's tests/ could not be uploaded into the container",
    Status.NO_SESSION: "verify() was called without a seeded session for this rollout",
}


# Fixes to upstream *reference solutions* applied only when the model-free ``reference`` and
# ``wrong_file`` modes upload ``solution/`` into the container. They never touch a validator, the
# task data, or anything the agent sees; each entry documents a defect in upstream's own solver
# at the pinned commit. Keyed by task name, then relative path inside ``solution/``.
REFERENCE_SOLUTION_PATCHES: Dict[str, Dict[str, List[tuple[str, str]]]] = {
    # solve_reference.py::build_improved_case_schedule references two names that do not exist
    # (``env_dir_plan``, ``greedy_plan_eval``); the surrounding lines show the intended call.
    "oragentbench/sterile_processing_robust_schedule": {
        "solve_reference.py": [
            (
                'greedy_eval, _ = evaluate_case_plan(data, env_dir_plan, scratch_dir / "greedy_seed.csv")',
                'greedy_eval, _ = evaluate_case_plan(data, env_dir, greedy_plan, scratch_dir / "greedy_seed.csv")',
            ),
            (
                "initial_plan, initial_eval = greedy_plan_eval",
                "initial_plan, initial_eval = greedy_plan, greedy_eval",
            ),
        ],
    },
}


def _clean(text: Any) -> str:
    """Return ``text`` as a wire-safe string (lone surrogates replaced)."""
    return str(text if text is not None else "").encode("utf-8", "replace").decode("utf-8")


def _clean_json(value: Any) -> Any:
    """Recursively sanitize strings inside parsed JSON so the response can be re-encoded."""
    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, dict):
        return {_clean(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    return value


def _numeric(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_pass(feasibility: float, quality_raw: float) -> bool:
    """Upstream ``summarize_results.is_pass``."""
    return feasibility > 0.0 and (quality_raw / QUALITY_RAW_MAX) > PASS_QUALITY_NORMALIZED_CUTOFF


def scalar_reward(feasibility: float, quality_raw: float) -> float:
    """Upstream ``summarize_results.scalar_reward``: the value ``test.sh`` writes to reward.txt."""
    if feasibility <= 0.0:
        return 0.0
    return (feasibility + quality_raw) / 3.0


class StepSpec(BaseModel):
    """One agent/verifier phase of a task. Single-step tasks have exactly one with ``name=None``."""

    name: Optional[str]
    instruction: str
    agent_timeout_s: float
    verifier_timeout_s: float
    min_reward: Optional[float]
    tests_dir: str
    solution_dir: Optional[str]
    workdir_dir: Optional[str]
    solution_env: Dict[str, str] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    name: str
    steps: List[StepSpec]


def load_task(task_folder: Path) -> TaskSpec:
    """Parse a Harbor task directory into the step list; raises on anything missing."""
    with open(task_folder / "task.toml", "rb") as f:
        toml = tomllib.load(f)
    solution_env = {k: str(v) for k, v in (toml.get("solution", {}).get("env") or {}).items()}
    steps: List[StepSpec] = []
    raw_steps = toml.get("steps") or []
    if raw_steps:
        for raw in raw_steps:
            step_dir = task_folder / "steps" / raw["name"]
            tests_dir = step_dir / "tests"
            if not (tests_dir / "test.sh").is_file():
                raise FileNotFoundError(f"missing {tests_dir / 'test.sh'}")
            solution_dir = step_dir / "solution"
            workdir_dir = step_dir / "workdir"
            min_reward = raw.get("min_reward")
            if min_reward is not None and not isinstance(min_reward, (int, float)):
                raise ValueError(f"unsupported min_reward {min_reward!r} for step {raw['name']}")
            steps.append(
                StepSpec(
                    name=raw["name"],
                    instruction=(step_dir / "instruction.md").read_text(),
                    agent_timeout_s=float(
                        raw.get("agent", {}).get("timeout_sec", toml.get("agent", {}).get("timeout_sec"))
                    ),
                    verifier_timeout_s=float(
                        raw.get("verifier", {}).get("timeout_sec", toml.get("verifier", {}).get("timeout_sec"))
                    ),
                    min_reward=None if min_reward is None else float(min_reward),
                    tests_dir=str(tests_dir),
                    solution_dir=str(solution_dir) if (solution_dir / "solve.sh").is_file() else None,
                    workdir_dir=str(workdir_dir) if workdir_dir.is_dir() else None,
                    solution_env=solution_env,
                )
            )
    else:
        tests_dir = task_folder / "tests"
        if not (tests_dir / "test.sh").is_file():
            raise FileNotFoundError(f"missing {tests_dir / 'test.sh'}")
        solution_dir = task_folder / "solution"
        steps.append(
            StepSpec(
                name=None,
                instruction=(task_folder / "instruction.md").read_text(),
                agent_timeout_s=float(toml["agent"]["timeout_sec"]),
                verifier_timeout_s=float(toml["verifier"]["timeout_sec"]),
                min_reward=None,
                tests_dir=str(tests_dir),
                solution_dir=str(solution_dir) if (solution_dir / "solve.sh").is_file() else None,
                workdir_dir=None,
                solution_env=solution_env,
            )
        )
    return TaskSpec(name=toml["task"]["name"], steps=steps)


class StepResult(BaseModel):
    name: Optional[str]
    status: str
    feasibility: float = 0.0
    quality_raw: float = 0.0
    quality_status: Optional[str] = None
    verifier_return_code: Optional[int] = None
    verifier_time_taken: float = 0.0
    test_output: str = ""
    reward_details: Optional[Dict[str, Any]] = None
    # Model-free validation only.
    control_output: Optional[str] = None


class ORAgentBenchResourcesServerConfig(BaseResourcesServerConfig):
    # Verification depends on the state of the task container; a stored rollout cannot be re-verified.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    sandbox_provider: str
    sandbox_config: Dict[str, Any] = Field(default_factory=dict)

    # Model-free validation. ``reference`` runs upstream's solution/solve.sh in place of the
    # agent; the others are negative controls. Every mode creates its own container in verify().
    validation_mode: Literal["none", "reference", "no_action", "wrong_file", "hung_process"] = "none"
    # Ceiling on any single solve.sh in ``reference`` mode; the per-step agent budget applies when lower.
    reference_solve_timeout_s: float = 2700.0

    debug: bool = False


class ORAgentBenchSeedSessionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_name: str
    docker_image: str
    task_folder: str


class ORAgentBenchStepInfo(BaseModel):
    name: Optional[str]
    agent_timeout_s: float


class ORAgentBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    steps: List[ORAgentBenchStepInfo]


class ORAgentBenchStepRequest(BaseModel):
    step_index: int


class ORAgentBenchPrepareStepResponse(BaseModel):
    step_index: int
    name: Optional[str]
    instruction: str
    agent_timeout_s: float
    setup_ok: bool
    setup_output: str


class ORAgentBenchVerifyStepResponse(BaseModel):
    step_index: int
    name: Optional[str]
    status: str
    feasibility: float
    quality_raw: float
    stop: bool


class ORAgentBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    task_name: str
    docker_image: str
    task_folder: str
    # Provenance label: a wrong type costs the label, not the row.
    difficulty: Any = None
    # Model-free sweeps only: overrides the server's validation_mode for this row, and is ignored
    # unless the server was started in a validation mode (a rollout row cannot switch a live server).
    validation_mode: Optional[Literal["reference", "no_action", "wrong_file", "hung_process"]] = None


class ORAgentBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    status: str
    harness_failure: float
    task_name: str
    difficulty: Optional[str]
    # Upstream quantities. ``quality_raw`` is reward.json's [0, 2] value; ``quality`` is the paper's q.
    feasibility: float
    quality_raw: float
    quality: float
    upstream_scalar_reward: float
    num_steps: int
    steps_completed: int
    step_results: List[StepResult]
    evaluation_completed: bool
    verification_time_taken: float
    test_output: str


class _Session(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sandbox: Any
    task: TaskSpec
    step_results: List[StepResult] = Field(default_factory=list)
    prepared_index: int = -1
    aborted: bool = False
    harness_status: Optional[Status] = None


class ORAgentBenchResourcesServer(SimpleResourcesServer):
    config: ORAgentBenchResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sessions: Dict[str, _Session] = {}

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/prepare_step")(self.prepare_step)
        app.post("/verify_step")(self.verify_step)
        return app

    # --- metrics -----------------------------------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Per-stratum pass, feasibility and quality; upstream reports each band separately."""
        by_band: Dict[str, List[Dict[str, Any]]] = {}
        for rollouts in tasks:
            for r in rollouts:
                band = r.get("difficulty")
                if band in DIFFICULTIES:
                    by_band.setdefault(band, []).append(r)
        out: Dict[str, Any] = {}
        for band, rows in by_band.items():
            out[f"pass_rate/{band}"] = sum(float(r.get("reward") or 0.0) for r in rows) / len(rows)
            out[f"feasibility_rate/{band}"] = sum(float(r.get("feasibility") or 0.0) for r in rows) / len(rows)
            out[f"mean_quality/{band}"] = sum(float(r.get("quality") or 0.0) for r in rows) / len(rows)
            out[f"count/{band}"] = len(rows)
        return out

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keep = {
            "mean/reward",
            "mean/feasibility",
            "mean/quality",
            "mean/harness_failure",
            "mean/input_tokens",
            "mean/output_tokens",
            "mean/total_tokens",
        }
        keep |= {
            k
            for k in agent_metrics
            if k.split("/")[0] in {"pass_rate", "feasibility_rate", "mean_quality", "count"}
            and k.split("/")[-1] in DIFFICULTIES
        }
        return {k: v for k, v in agent_metrics.items() if k in keep}

    # --- sandbox -----------------------------------------------------------------------------

    def _resolve_task_folder(self, task_folder: str) -> Path:
        path = Path(task_folder)
        return path if path.is_absolute() else PARENT_DIR / path

    async def _create_sandbox(self, task_name: str, docker_image: str) -> AsyncSandbox:
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        resources = SandboxResources.from_mapping(dict(self.config.sandbox_config.get("resources", {})))
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            # BLAS/OpenMP thread caps sized to the container's CPU limit, set before any solver loads.
            env = cpu_cap_env(resources.cpu) | env
        spec = SandboxSpec(
            image=docker_image,
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=None,
            env=env,
            files={},
            metadata=provider_metadata
            | dict(self.config.sandbox_config.get("metadata", {}))
            | {"nemo_gym_agent": self.config.name, "instance_id": task_name},
            resources=resources,
            entrypoint=None,
            provider_options=deepcopy(self.config.sandbox_config.get("provider_options") or {}),
        )
        sandbox = AsyncSandbox(provider_config)

        async def _setup(sb: AsyncSandbox) -> None:
            result = await sb.exec(f"mkdir -p {AGENT_LOGS_DIR} {VERIFIER_DIR} /app/submissions", user="root")
            if result.return_code != 0:
                raise RuntimeError(f"failed to prepare task container: {result}")

        await sandbox.start_with_setup(spec, _setup)
        return sandbox

    async def _upload_dir(
        self,
        sandbox: AsyncSandbox,
        local_dir: Path,
        target_dir: str,
        patches: Optional[Dict[str, List[tuple[str, str]]]] = None,
    ) -> None:
        for rel in sorted(glob("**", root_dir=str(local_dir), recursive=True, include_hidden=True)):
            local_path = local_dir / rel
            if not local_path.is_file() or "__pycache__" in local_path.parts:
                continue
            target = f"{target_dir}/{rel}"
            if patches and rel in patches:
                content = local_path.read_text()
                for old, new in patches[rel]:
                    if content.count(old) != 1:
                        raise RuntimeError(f"reference patch for {rel} did not match exactly once: {old!r}")
                    content = content.replace(old, new)
                with TemporaryDirectory(prefix="nemo-gym-oragentbench-patch-") as tmp:
                    patched = Path(tmp) / Path(rel).name
                    patched.write_text(content)
                    await sandbox.upload(local_path=patched, remote_path=target)
                continue
            await sandbox.upload(local_path=local_path, remote_path=target)

    async def _stop(self, sandbox: Optional[AsyncSandbox]) -> None:
        """Teardown never raises and never blocks the caller on a wedged container."""
        if sandbox is None:
            return
        try:
            await sandbox.stop()
        except BaseException:
            print(f"Exception stopping sandbox: {format_exc()}", file=stderr)

    # --- endpoints ---------------------------------------------------------------------------

    async def seed_session(
        self, request: Request, body: ORAgentBenchSeedSessionRequest
    ) -> ORAgentBenchSeedSessionResponse:
        task = load_task(self._resolve_task_folder(body.task_folder))
        sandbox = await self._create_sandbox(body.task_name, body.docker_image)
        self._sessions[request.session[SESSION_ID_KEY]] = _Session(sandbox=sandbox, task=task)
        return ORAgentBenchSeedSessionResponse(
            sandbox_handle=sandbox._handle.sandbox_id,
            steps=[ORAgentBenchStepInfo(name=s.name, agent_timeout_s=s.agent_timeout_s) for s in task.steps],
        )

    async def prepare_step(self, request: Request, body: ORAgentBenchStepRequest) -> ORAgentBenchPrepareStepResponse:
        session = self._sessions[request.session[SESSION_ID_KEY]]
        return await self._prepare_step(session, body.step_index)

    async def _prepare_step(self, session: _Session, step_index: int) -> ORAgentBenchPrepareStepResponse:
        """Harbor's per-step preparation: upload ``steps/<name>/workdir`` into the container's
        working directory and run its ``setup.sh`` there. A non-zero setup is a harness fault."""
        step = session.task.steps[step_index]
        session.prepared_index = step_index
        setup_ok, setup_output = True, ""
        if step.workdir_dir is not None:
            cwd = ((await session.sandbox.exec("pwd", user="root")).stdout or "/app").strip()
            await self._upload_dir(session.sandbox, Path(step.workdir_dir), cwd)
            if (Path(step.workdir_dir) / "setup.sh").is_file():
                result = await session.sandbox.exec(
                    f"bash {quote(cwd.rstrip('/') + '/setup.sh')}", user="root", timeout_s=step.verifier_timeout_s
                )
                setup_output = _clean(result.stderr) + _clean(result.stdout)
                setup_ok = result.return_code == 0
                if not setup_ok:
                    session.harness_status = Status.STEP_SETUP_FAILED
                    session.aborted = True
        return ORAgentBenchPrepareStepResponse(
            step_index=step_index,
            name=step.name,
            instruction=step.instruction,
            agent_timeout_s=step.agent_timeout_s,
            setup_ok=setup_ok,
            setup_output=setup_output,
        )

    async def verify_step(self, request: Request, body: ORAgentBenchStepRequest) -> ORAgentBenchVerifyStepResponse:
        session = self._sessions[request.session[SESSION_ID_KEY]]
        result = await self._verify_step(session, body.step_index)
        return ORAgentBenchVerifyStepResponse(
            step_index=body.step_index,
            name=result.name,
            status=result.status,
            feasibility=result.feasibility,
            quality_raw=result.quality_raw,
            stop=session.aborted,
        )

    async def _verify_step(self, session: _Session, step_index: int) -> StepResult:
        """Run upstream's validator for one step inside the task container and record the result."""
        if step_index != len(session.step_results):
            raise ValueError(f"step {step_index} verified out of order; {len(session.step_results)} steps recorded")
        step = session.task.steps[step_index]
        sandbox = session.sandbox
        start = time()
        result = StepResult(name=step.name, status=Status.SCORED.value)
        try:
            reset = await sandbox.exec(
                f"rm -rf {TESTS_DIR} {VERIFIER_DIR} && mkdir -p {TESTS_DIR} {VERIFIER_DIR}", user="root"
            )
            if reset.return_code != 0:
                raise RuntimeError(f"could not reset verifier directories: {reset}")
            await self._upload_dir(sandbox, Path(step.tests_dir), TESTS_DIR)
        except BaseException:
            print(f"Exception uploading tests for {session.task.name}: {format_exc()}", file=stderr)
            result.status = Status.TESTS_UPLOAD_FAILED.value
            session.harness_status = Status.TESTS_UPLOAD_FAILED
            session.aborted = True
            result.verifier_time_taken = time() - start
            session.step_results.append(result)
            return result

        exec_result = await sandbox.exec(f"bash {TESTS_DIR}/test.sh", user="root", timeout_s=step.verifier_timeout_s)
        result.verifier_return_code = exec_result.return_code
        result.test_output = _clean(exec_result.stderr) + _clean(exec_result.stdout)
        if exec_result.error_type == "timeout":
            result.status = Status.VERIFIER_TIMEOUT.value
        else:
            reward, details = await self._download_rewards(sandbox)
            if reward is None:
                result.status = Status.VERIFIER_OUTPUT_MISSING.value
            else:
                result.feasibility = 1.0 if (_numeric(reward.get("feasibility")) or 0.0) > 0.0 else 0.0
                result.quality_raw = _numeric(reward.get("quality")) or 0.0
                result.reward_details = details
                result.quality_status = _clean(details.get("quality_status")) if isinstance(details, dict) else None
                if result.quality_status == "missing_solution":
                    result.status = Status.MISSING_SOLUTION.value
        result.verifier_time_taken = time() - start
        session.step_results.append(result)

        # Harbor gates on the step reward; upstream's reward.txt scalar is (F + q) / 3.
        if step.min_reward is not None and scalar_reward(result.feasibility, result.quality_raw) < step.min_reward:
            session.aborted = True
        return result

    async def _download_rewards(self, sandbox: AsyncSandbox) -> tuple[Optional[Dict[str, Any]], Any]:
        with TemporaryDirectory(prefix="nemo-gym-oragentbench-") as tmp:
            reward: Optional[Dict[str, Any]] = None
            details: Any = None
            try:
                await sandbox.download(f"{VERIFIER_DIR}/reward.json", Path(tmp) / "reward.json")
                loaded = _clean_json(json.loads((Path(tmp) / "reward.json").read_text()))
                reward = loaded if isinstance(loaded, dict) else None
            except BaseException:
                if self.config.debug:
                    print(f"Could not read reward.json: {format_exc()}", file=stderr)
            try:
                await sandbox.download(f"{VERIFIER_DIR}/reward_details.json", Path(tmp) / "reward_details.json")
                details = _clean_json(json.loads((Path(tmp) / "reward_details.json").read_text()))
            except BaseException:
                details = None
            return reward, details

    async def verify(self, request: Request, body: ORAgentBenchVerifyRequest) -> ORAgentBenchVerifyResponse:
        start = time()
        if self.config.validation_mode != "none":
            session = await self._run_model_free(body)
        else:
            session = self._sessions.pop(request.session[SESSION_ID_KEY], None)
            if session is not None and not session.aborted and len(session.step_results) < len(session.task.steps):
                # The stock single-step agent never calls /verify_step: finalize the pending step here.
                pending = len(session.step_results)
                if pending == 0 or session.prepared_index == pending:
                    try:
                        await self._verify_step(session, pending)
                    except BaseException:
                        print(f"Exception verifying {body.task_name}: {format_exc()}", file=stderr)
        try:
            return self._aggregate(body, session, verification_time_taken=time() - start)
        finally:
            if session is not None:
                await self._stop(session.sandbox)

    async def _run_model_free(self, body: ORAgentBenchVerifyRequest) -> Optional[_Session]:
        """Drive the whole task without a model: reference solution or a negative control per step."""
        mode = body.validation_mode or self.config.validation_mode
        try:
            task = load_task(self._resolve_task_folder(body.task_folder))
        except BaseException:
            print(f"Bad task folder for {body.task_name}: {format_exc()}", file=stderr)
            return _Session(
                sandbox=None, task=TaskSpec(name=body.task_name, steps=[]), harness_status=Status.BAD_TASK_FOLDER
            )
        try:
            sandbox = await self._create_sandbox(body.task_name, body.docker_image)
        except BaseException:
            print(f"Could not start container for {body.task_name}: {format_exc()}", file=stderr)
            return _Session(sandbox=None, task=task, harness_status=Status.SANDBOX_FAILED)
        session = _Session(sandbox=sandbox, task=task)
        for index, step in enumerate(task.steps):
            prepared = await self._prepare_step(session, index)
            if not prepared.setup_ok:
                break
            control_output = await self._apply_control(sandbox, task.name, step, mode)
            result = await self._verify_step(session, index)
            result.control_output = control_output
            if session.aborted:
                break
        return session

    async def _apply_control(self, sandbox: AsyncSandbox, task_name: str, step: StepSpec, mode: str) -> str:
        if mode == "no_action":
            return ""
        if mode == "hung_process":
            # An agent that leaves a runaway process behind and overruns its budget: the exec times
            # out, the verifier must still run, and teardown must still succeed.
            await sandbox.exec("nohup sleep 100000 >/dev/null 2>&1 &", user="root", timeout_s=10)
            timed_out = await sandbox.exec("sleep 30", user="root", timeout_s=1)
            return f"exec error_type={timed_out.error_type}"
        # reference and wrong_file both run upstream's solve.sh from the step's solution/.
        if step.solution_dir is None:
            return "no solution/solve.sh for this step"
        before = await sandbox.exec("find /app/submissions -type f | sort", user="root")
        await sandbox.exec(f"rm -rf {SOLUTION_DIR} && mkdir -p {SOLUTION_DIR}", user="root")
        patches = REFERENCE_SOLUTION_PATCHES.get(task_name) if step.name is None else None
        await self._upload_dir(sandbox, Path(step.solution_dir), SOLUTION_DIR, patches=patches)
        timeout_s = min(step.agent_timeout_s, self.config.reference_solve_timeout_s)
        solve = await sandbox.exec(
            f"bash {SOLUTION_DIR}/solve.sh", user="root", env=step.solution_env or None, timeout_s=timeout_s
        )
        output = f"solve.sh return_code={solve.return_code} error_type={solve.error_type} patched={bool(patches)}\n"
        output += _clean(solve.stderr)[-4000:] + _clean(solve.stdout)[-4000:]
        if mode == "wrong_file":
            after = await sandbox.exec("find /app/submissions -type f | sort", user="root")
            new_files = sorted(set((after.stdout or "").split()) - set((before.stdout or "").split()))
            for path in new_files:
                await sandbox.exec(f"mv {quote(path)} {quote(path + '.wrong')}", user="root")
            output += f"\nrenamed: {new_files}"
        return output

    def _aggregate(
        self, body: ORAgentBenchVerifyRequest, session: Optional[_Session], verification_time_taken: float
    ) -> ORAgentBenchVerifyResponse:
        steps = session.task.steps if session is not None else []
        results = list(session.step_results) if session is not None else []
        num_steps = len(steps)
        if session is None:
            status = Status.NO_SESSION
        elif session.harness_status is not None:
            status = session.harness_status
        elif session.aborted and len(results) < num_steps:
            status = Status.STEP_ABORTED
        elif len(results) < num_steps:
            status = Status.STEP_INCOMPLETE
        else:
            non_scored = [r.status for r in results if r.status != Status.SCORED.value]
            status = Status(non_scored[0]) if non_scored else Status.SCORED
        harness_fault = status in HARNESS_FAULTS

        # Upstream ``summarize_trial``: all-step feasibility, mean step quality with missing steps as 0.
        if num_steps and not harness_fault:
            feasibility = 1.0 if len(results) == num_steps and all(r.feasibility > 0.0 for r in results) else 0.0
            quality_raw = sum(r.quality_raw for r in results) / num_steps
        else:
            feasibility, quality_raw = 0.0, 0.0
        reward = 1.0 if is_pass(feasibility, quality_raw) else 0.0

        computed = dict(
            reward=reward,
            failure_reason=HARNESS_FAULTS.get(status),
            status=status.value,
            harness_failure=1.0 if harness_fault else 0.0,
            task_name=body.task_name,
            difficulty=body.difficulty if body.difficulty in DIFFICULTIES else None,
            feasibility=feasibility,
            quality_raw=quality_raw,
            quality=quality_raw / QUALITY_RAW_MAX,
            upstream_scalar_reward=scalar_reward(feasibility, quality_raw),
            num_steps=num_steps,
            steps_completed=len(results),
            step_results=results,
            evaluation_completed=status == Status.SCORED,
            verification_time_taken=verification_time_taken,
            test_output=results[-1].test_output if results else "",
        )
        # Rows may carry provenance extras (e.g. ``num_steps`` from the preparer) that share a name
        # with a computed field; the computed value wins and the request copy is dropped.
        echoed = {k: v for k, v in body.model_dump().items() if k not in computed}
        return ORAgentBenchVerifyResponse(**echoed, **computed)


if __name__ == "__main__":
    ORAgentBenchResourcesServer.run_webserver()
