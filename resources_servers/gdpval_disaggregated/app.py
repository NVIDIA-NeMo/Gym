# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDPVal task state and grading for remotely executed agent harnesses."""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any, ClassVar, Literal

from artifact_io import ArtifactIO
from fastapi import HTTPException, Request
from judge_runtime import ensure_judge_runtime
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
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
from pydantic import ConfigDict, Field, PrivateAttr
from stage_gdpval_task import stage

logger = logging.getLogger(__name__)


class GDPValResourcesConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    dataset_path: str
    data_root: str
    artifact_root: str
    sandbox_provider: str = "sandbox"
    sandbox_image: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    judge_timeout: int = Field(default=900, ge=1)
    judge_max_attempts: int = Field(default=2, ge=1, le=3)
    judge_retry_delay_s: float = Field(default=5.0, ge=0)
    judge_concurrency: int = Field(default=32, ge=1)
    judge_media_mode: str = "images_and_text"
    judge_url: str = "https://inference-api.nvidia.com/v1"
    judge_model: str = "openai/openai/gpt-5.5"
    # Policy failure reasons (agent_context_limit, agent_iteration_limit, agent_stuck_in_loop)
    # that produce a MASKED row instead of a valid reward of 0. Default keeps the historical
    # behaviour (valid 0). 2026-10-14: with 6-37% of rollouts ending at the context wall, the
    # zero was learned as a penalty on length rather than on quality.
    mask_policy_failures: list[str] = Field(default_factory=list)


class GDPValSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    # The staged row's task fields, so a harness that renders its own prompt
    # (gdpval_stirrup_agent) does not depend on the run body carrying them.
    task: dict[str, Any] | None = None


class GDPValVerifyRequest(BaseVerifyRequest):
    task_id: str
    agent_finished: bool = True
    incomplete_reason: Literal["agent_exec_timeout", "agent_exec_incomplete", "model_response_timeout", "sandbox_ended"] | None = None
    sandbox_error: str | None = None
    agent_error: str | None = None
    policy_failure_reason: (
        Literal["agent_context_limit", "agent_iteration_limit", "agent_stuck_in_loop"]
        | None
    ) = None
    # Context-compaction harnesses: one response per context segment (NeMo RL multi-trace
    # training). Passed through unchanged on judged rows; masked rows carry none.
    responses: list[dict[str, Any]] | None = None


class GDPValVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@dataclass
class TaskSession:
    task_id: str
    sandbox: AsyncSandbox
    run: Path
    task: Path
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready: bool = False
    verify_started: bool = False
    result: GDPValVerifyResponse | None = None


class GDPValResourcesServer(SimpleResourcesServer, ArtifactIO):
    config: GDPValResourcesConfig
    _rows: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _sessions: dict[str, TaskSession] = PrivateAttr(default_factory=dict)
    _manifest: str = PrivateAttr(default="")
    _judge_sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._judge_sem = asyncio.Semaphore(self.config.judge_concurrency)
        for line in Path(self.config.dataset_path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["task_id"] in self._rows:
                raise ValueError(f"Duplicate task_id: {row['task_id']}")
            self._rows[row["task_id"]] = row
        self._manifest = hashlib.sha256(
            (Path(self.config.data_root) / "manifest.json").read_bytes()
        ).hexdigest()
        if not os.environ.get("JUDGE_API_KEY"):
            raise ValueError("JUDGE_API_KEY must be exported to the resources server")
        ensure_judge_runtime()

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/cleanup_session")(self.cleanup_session)
        return app

    def _new_sandbox(self) -> AsyncSandbox:
        cfg = self.config
        global_config = get_global_config_dict()
        raw = cfg.sandbox_config
        resources = SandboxResources.from_mapping(raw.get("resources", {}))
        # Match SWE: libraries may see the host CPU count instead of the
        # sandbox quota. Bound their pools; explicit environment settings win.
        env = {str(k): str(v) for k, v in raw.get("env", {}).items()}
        if raw.get("derive_cpu_env", True):
            env = cpu_cap_env(resources.cpu) | env
        spec = SandboxSpec(
            image=cfg.sandbox_image,
            ttl_s=raw.get("ttl_s", 7200),
            ready_timeout_s=raw.get("ready_timeout_s", 1200),
            workdir="/workspace",
            env=env,
            metadata=resolve_provider_metadata(cfg.sandbox_provider, global_config)
            | raw.get("metadata", {})
            | {"nemo_gym_agent": cfg.name, "benchmark": "gdpval"},
            resources=resources,
            provider_options=raw.get("provider_options", {}),
        )
        return AsyncSandbox(
            resolve_provider_config(cfg.sandbox_provider, global_config), spec
        )

    async def seed_session(
        self, request: Request, body: BaseRunRequest
    ) -> GDPValSeedResponse:
        task_id = (body.responses_create_params.metadata or {}).get("task_id")
        if task_id not in self._rows:
            raise HTTPException(
                422, "Unknown task_id in responses_create_params.metadata"
            )
        key = request.session[SESSION_ID_KEY]
        state = self._sessions.get(key)
        if state is None:
            root = (
                Path(self.config.artifact_root)
                / os.environ.get("SLURM_JOB_ID", "local")
                / self.config.name
            )
            root.mkdir(parents=True, exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix="gdpval-", dir=root))
            state = TaskSession(task_id, self._new_sandbox(), run, run / "task")
            self._sessions[key] = state
        if state.task_id != task_id:
            raise HTTPException(409, "Session already belongs to another task")
        async with state.lock:
            if not state.ready:
                try:
                    row = self._rows[task_id]
                    references = (
                        Path(self.config.data_root)
                        / "references"
                        / row["reference_root_key"]
                    )
                    stage(row, references, state.task, self._manifest)
                    (state.task / "trace").mkdir()
                    await state.sandbox.start()
                    result = await state.sandbox.exec(
                        "mkdir -p /workspace/reference_files /workspace/deliverables /trace /control /runner"
                    )
                    if result.return_code:
                        raise RuntimeError(
                            f"Failed to initialize GDPVal sandbox: {result.stderr}"
                        )
                    await self._upload_tree(
                        state.sandbox, state.task / "agent", "/workspace"
                    )
                    state.ready = True
                except BaseException:
                    try:
                        await state.sandbox.stop()
                    except Exception:
                        logger.exception("Sandbox cleanup failed during GDPVal setup")
                    self._sessions.pop(key, None)
                    raise
            row = self._rows[task_id]
            names = row.get("reference_files", [])
            if isinstance(names, str):
                try:
                    names = json.loads(names)
                except json.JSONDecodeError:
                    names = []
            return GDPValSeedResponse(
                sandbox_handle=state.sandbox._handle.sandbox_id,
                task={
                    "task_id": task_id,
                    "prompt": row.get("prompt", ""),
                    "sector": row.get("sector", ""),
                    "occupation": row.get("occupation", ""),
                    "reference_files": [str(n) for n in (names or [])],
                },
            )

    async def _download_outputs(self, state: TaskSession) -> None:
        # Skips (and lists) symlinks and unreadable files instead of failing the
        # whole verify: a single odd entry used to raise, fail /run, and restart
        # every sibling of the group (12 times in 3.5 h on 2026-10-05).
        pack_script = """from pathlib import Path
import json, sys, zipfile
skipped = []
with zipfile.ZipFile('/tmp/gdpval-output.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root in (Path('/workspace/deliverables'), Path('/trace')):
        if root.is_symlink() or not root.exists():
            skipped.append({'path': str(root), 'reason': 'symlink or missing root'})
            continue
        for p in root.rglob('*'):
            try:
                if p.is_symlink():
                    skipped.append({'path': str(p), 'reason': 'symlink'})
                elif p.is_file():
                    z.write(p, p.relative_to('/'))
            except Exception as error:
                skipped.append({'path': str(p), 'reason': repr(error)[:200]})
print(json.dumps({'skipped': skipped}))
"""
        packed = await state.sandbox.exec(
            "python3 -c " + quote(pack_script), timeout_s=300
        )
        if packed.return_code:
            raise RuntimeError(
                f"Failed to package GDPVal outputs: exit {packed.return_code}: "
                f"{(packed.stderr or packed.stdout or '')[-800:]}"
            )
        try:
            skipped = json.loads((packed.stdout or "").strip().splitlines()[-1]).get("skipped", [])
        except Exception:
            skipped = []
        if skipped:
            logger.warning("GDPVal output packaging skipped %d entries in %s", len(skipped), state.task)
            (state.task / "packaging-skipped.json").write_text(json.dumps(skipped) + "\n")
        archive = state.run / "sandbox-output.zip"
        await state.sandbox.download("/tmp/gdpval-output.zip", archive)
        self._extract_results(archive, state.task)

    @staticmethod
    def _save_masked_result(state: TaskSession, body: GDPValVerifyRequest, reason: str) -> GDPValVerifyResponse:
        result = GDPValVerifyResponse(
            responses_create_params=body.responses_create_params, response=body.response,
            reward=0.0, reward_valid=False, mask_sample=True,
            evaluation_completed=False, agent_finished=body.agent_finished,
            failure_kind=("agent_budget" if reason in ("agent_exec_timeout", "agent_context_limit",
                                                      "agent_iteration_limit", "agent_stuck_in_loop")
                          else "infrastructure"),
            failure_reason=reason, judge_skipped=not body.agent_finished,
            # Context-compaction harnesses export their trainable tokens as one
            # response per context segment and leave ``response`` text-only; a
            # masked row still needs those tokens (NeMo RL rejects a result with
            # no generation data), so pass them through exactly like the judged path.
            **({"responses": body.responses} if body.responses else {}),
        )
        (state.task / "rollout.json").write_text(result.model_dump_json() + "\n")
        (state.task / "reward.json").write_text(json.dumps({
            "valid": False, "reward": None, "mask_sample": True, "failure_reason": reason,
        }) + "\n")
        return result

    async def _judge_saved_artifacts(self, state: TaskSession, config_path: Path, config: dict) -> None:
        # Queueing, backoff, and all attempts share the original judge deadline.
        # Cancellation propagates to _run_judge, which kills its subprocess group.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.judge_timeout
        async with asyncio.timeout(self.config.judge_timeout):
            async with self._judge_sem:
                for attempt in range(1, self.config.judge_max_attempts + 1):
                    remaining = deadline - loop.time()
                    budget = remaining / (self.config.judge_max_attempts - attempt + 1)
                    config_path.write_text(json.dumps(config | {"judge_timeout": budget}) + "\n")
                    try:
                        async with asyncio.timeout(budget):
                            await self._run_judge(state.task, config_path, budget)
                        return
                    except (RuntimeError, TimeoutError) as error:
                        with (state.task / "judge-attempts.jsonl").open("a") as log:
                            log.write(json.dumps({"attempt": attempt, "error_type": type(error).__name__,
                                                  "budget_s": budget}) + "\n")
                        if attempt == self.config.judge_max_attempts:
                            raise
                        await asyncio.sleep(min(self.config.judge_retry_delay_s, max(0, deadline-loop.time())))
                    finally:
                        log_path = state.task / "judge.log"
                        if log_path.exists():
                            (state.task / f"judge-attempt-{attempt}.log").write_bytes(log_path.read_bytes())

    async def verify(
        self, request: Request, body: GDPValVerifyRequest
    ) -> GDPValVerifyResponse:
        state = self._sessions.get(request.session[SESSION_ID_KEY])
        if state is None or state.task_id != body.task_id:
            raise HTTPException(409, "No matching GDPVal task session")
        async with state.lock:
            if state.result is not None:
                return state.result
            if not state.ready or state.verify_started:
                raise HTTPException(
                    409, "GDPVal session is not available for verification"
                )
            state.verify_started = True
            try:
                outputs_unavailable = None
                try:
                    await self._download_outputs(state)
                except Exception as error:
                    # Nothing to download (sandbox died, packaging failed): the
                    # row is masked instead of failing /verify, which would abort
                    # the group's rollout stream and re-dispatch every sibling.
                    outputs_unavailable = f"{type(error).__name__}: {str(error)[:300]}"
                    logger.warning("GDPVal outputs unavailable (%s, agent_finished=%s) in %s: %s",
                                   body.incomplete_reason, body.agent_finished, state.task, outputs_unavailable)
                (state.task / "trajectory.json").write_text(
                    body.model_dump_json() + "\n"
                )
            finally:
                try:
                    await state.sandbox.stop()
                except Exception:
                    logger.exception("GDPVal sandbox stop failed in verify (sandbox may already have ended)")
            if not body.agent_finished:
                if body.incomplete_reason is None:
                    raise ValueError("Incomplete agent attempt requires an explicit reason")
                state.result = self._save_masked_result(state, body, body.incomplete_reason)
                return state.result
            if outputs_unavailable is not None:
                (state.task / "outputs-unavailable.txt").write_text(outputs_unavailable + "\n")
                state.result = self._save_masked_result(state, body, "outputs_unavailable")
                return state.result
            if body.policy_failure_reason and body.policy_failure_reason in self.config.mask_policy_failures:
                self._collect_artifacts(state.task)   # keep the deliverables for inspection
                state.result = self._save_masked_result(state, body, body.policy_failure_reason)
                return state.result
            manifest = self._collect_artifacts(state.task)
            config = {
                "judge_timeout": self.config.judge_timeout,
                "judge_url": self.config.judge_url,
                "judge_model": self.config.judge_model,
                "judge_media_mode": self.config.judge_media_mode,
                "gym": {},
                "policy_failure_reason": body.policy_failure_reason,
                "empty_submission_zero_reward": not any(
                    item["bytes"] for item in manifest
                ),
            }
            config_path = state.run / "judge-config.json"
            try:
                await self._judge_saved_artifacts(state, config_path, config)
            except (RuntimeError, TimeoutError):
                logger.warning("Judge exhausted its budget/attempts; masking saved rollout %s", state.task)
                state.result = self._save_masked_result(state, body, "judge_unavailable")
                return state.result
            state.result = GDPValVerifyResponse.model_validate_json(
                (state.task / "rollout.json").read_text()
            )
            state.result = state.result.model_copy(update={
                "evaluation_completed": True, "agent_finished": True, "reward_valid": True,
                **({"responses": body.responses} if body.responses else {}),
            })
            (state.task / "rollout.json").write_text(state.result.model_dump_json() + "\n")
            return state.result

    async def cleanup_session(self, request: Request) -> dict[str, bool]:
        key = request.session[SESSION_ID_KEY]
        state = self._sessions.get(key)
        if state is not None:
            async with state.lock:
                try:
                    if state.ready and not state.verify_started:
                        # Preserve controller diagnostics when /run fails before /verify.
                        await self._download_outputs(state)
                except Exception:
                    logger.exception(
                        "Failed to preserve GDPVal failure outputs in %s", state.run
                    )
                finally:
                    await state.sandbox.stop()
                    self._sessions.pop(key, None)
        return {"cleaned": True}


if __name__ == "__main__":
    GDPValResourcesServer.run_webserver()
