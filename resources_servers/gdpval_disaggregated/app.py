# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDPVal task state and grading for remotely executed agent harnesses."""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from shlex import quote
from typing import Any, ClassVar, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.client import SESSION_ID_KEY, get_server_url
from nemo_gym.config import ModelServerRef, get_global_config_dict
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxResources,
    SandboxSpec,
    cpu_cap_env,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.gdpval.setup_libreoffice import ensure_libreoffice
from resources_servers.gdpval_disaggregated.artifact_io import ArtifactIO, ArtifactProcessingError
from resources_servers.gdpval_disaggregated.stage_gdpval_task import reference_name, stage


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
    judge_model_server: ModelServerRef
    judge_media_mode: Literal["native_pdf", "images_and_text"] = "images_and_text"
    judge_pdf_render_dpi: int = Field(default=144, ge=1)
    judge_pdf_max_pages: int = Field(default=120, ge=1)
    judge_handles_audio: bool = False
    judge_handles_video: bool = False
    judge_create_params: dict[str, Any] = Field(default_factory=dict)
    libreoffice_command: list[str] | None = Field(default=None, min_length=1)
    # These completed policy failures are valid zeroes unless explicitly masked.
    mask_policy_failures: list[str] = Field(default_factory=list)


class GDPValSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str
    # Task fields for a harness that renders its own prompt.
    task: dict[str, Any] | None = None


class GDPValVerifyRequest(BaseVerifyRequest):
    task_id: str
    agent_finished: bool = True
    incomplete_reason: (
        Literal["agent_exec_timeout", "agent_exec_incomplete", "model_response_timeout", "sandbox_ended"] | None
    ) = None
    sandbox_error: str | None = None
    agent_error: str | None = None
    policy_failure_reason: Literal["agent_context_limit", "agent_iteration_limit", "agent_stuck_in_loop"] | None = None
    # Compaction segments remain available on both judged and masked rows.
    responses: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def require_incomplete_reason(self) -> "GDPValVerifyRequest":
        """Reject incomplete attempts without a reason before touching their session."""
        if not self.agent_finished and self.incomplete_reason is None:
            raise ValueError("Incomplete agent attempt requires an explicit reason")
        return self


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
    setup_failed: bool = False
    verify_started: bool = False
    result: GDPValVerifyResponse | None = None


class GDPValResourcesServer(SimpleResourcesServer, ArtifactIO):
    config: GDPValResourcesConfig
    # No Ray inside this server (Gym would otherwise initialize it in every worker).
    ray_enabled: ClassVar[bool] = False
    _rows: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _sessions: dict[str, TaskSession] = PrivateAttr(default_factory=dict)
    _manifest: str = PrivateAttr(default="")
    _judge_sem: asyncio.Semaphore = PrivateAttr()
    _office_rendering_available: bool = PrivateAttr(default=False)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("GDPVal resources sessions require num_workers=1")
        self._judge_sem = asyncio.Semaphore(self.config.judge_concurrency)
        for line in Path(self.config.dataset_path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["task_id"] in self._rows:
                raise ValueError(f"Duplicate task_id: {row['task_id']}")
            self._rows[row["task_id"]] = row
        self._manifest = hashlib.sha256((Path(self.config.data_root) / "manifest.json").read_bytes()).hexdigest()
        if self.config.libreoffice_command is not None:
            self._office_rendering_available = ensure_libreoffice(
                self.config.libreoffice_command, probe_dirs=[Path(self.config.artifact_root)]
            )
            if not self._office_rendering_available:
                raise RuntimeError(
                    "libreoffice_command failed its startup conversion check; the command must see "
                    "artifact_root and the temporary directory at the same paths"
                )
        else:
            self._office_rendering_available = ensure_libreoffice()
            if not self._office_rendering_available:
                logger.warning(
                    "LibreOffice could not be provisioned; Office submissions will be masked as "
                    "artifact_processing_failed. Text submissions remain available."
                )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
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
            | {"nemo_gym_agent": cfg.name[:63], "benchmark": "gdpval"},
            resources=resources,
            provider_options=raw.get("provider_options", {}),
        )
        return AsyncSandbox(resolve_provider_config(cfg.sandbox_provider, global_config), spec)

    async def seed_session(self, request: Request, body: BaseRunRequest) -> GDPValSeedResponse:
        task_id = (body.responses_create_params.metadata or {}).get("task_id")
        if task_id not in self._rows:
            raise HTTPException(422, "Unknown task_id in responses_create_params.metadata")
        key = request.session[SESSION_ID_KEY]
        state = self._sessions.get(key)
        if state is None:
            root = Path(self.config.artifact_root) / os.environ.get("SLURM_JOB_ID", "local") / self.config.name
            root.mkdir(parents=True, exist_ok=True)
            run = Path(tempfile.mkdtemp(prefix="gdpval-", dir=root))
            state = TaskSession(task_id, self._new_sandbox(), run, run / "task")
            self._sessions[key] = state
        if state.task_id != task_id:
            raise HTTPException(409, "Session already belongs to another task")
        async with state.lock:
            if state.setup_failed:
                raise HTTPException(409, "GDPVal setup failed; close the session before retrying")
            if not state.ready:
                try:
                    row = self._rows[task_id]
                    references = (
                        Path(self.config.data_root) / "references" / str(reference_name(row["reference_root_key"]))
                    )
                    stage(row, references, state.task, self._manifest)
                    (state.task / "trace").mkdir()

                    async def setup(sandbox: AsyncSandbox) -> None:
                        result = await sandbox.exec(
                            "mkdir -p /workspace/reference_files /workspace/deliverables /trace /control /runner"
                        )
                        if result.error_type is not None or result.return_code != 0:
                            raise RuntimeError(f"Failed to initialize GDPVal sandbox: {result.stderr}")
                        await self._upload_tree(sandbox, state.task / "agent", "/workspace")

                    await state.sandbox.start_with_setup(None, setup)
                    state.ready = True
                except BaseException:
                    state.setup_failed = True
                    try:
                        await state.sandbox.stop()
                    except Exception:
                        # Keep ownership so /cleanup_session or shutdown can retry.
                        logger.exception("Sandbox cleanup failed during GDPVal setup")
                    else:
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
                sandbox_handle=state.sandbox.sandbox_id,
                task={
                    "task_id": task_id,
                    "prompt": row.get("prompt", ""),
                    "sector": row.get("sector", ""),
                    "occupation": row.get("occupation", ""),
                    "reference_files": [str(n) for n in (names or [])],
                },
            )

    async def _download_outputs(self, state: TaskSession) -> None:
        # Preserve readable files and report omissions. A skipped deliverable makes
        # the measurement inconclusive; a missing diagnostic trace does not.
        pack_script = """from pathlib import Path
import json, sys, zipfile
skipped = []
with zipfile.ZipFile('/tmp/gdpval-output.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root in (Path('/workspace/deliverables'), Path('/trace')):
        if root.is_symlink() or not root.is_dir():
            skipped.append({'path': str(root), 'reason': 'symlink or non-directory root'})
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
        packed = await state.sandbox.exec("python3 -c " + quote(pack_script), timeout_s=300)
        if packed.error_type is not None or packed.return_code != 0:
            raise RuntimeError(
                f"Failed to package GDPVal outputs: exit {packed.return_code}: "
                f"{(packed.stderr or packed.stdout or '')[-800:]}"
            )
        try:
            report = json.loads((packed.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError) as error:
            raise ValueError("GDPVal output packaging did not return a valid JSON report") from error
        if not isinstance(report, dict) or not isinstance(report.get("skipped"), list):
            raise ValueError("GDPVal output packaging report requires a skipped list")
        skipped = report["skipped"]
        if any(not isinstance(entry, dict) or not isinstance(entry.get("path"), str) for entry in skipped):
            raise ValueError("GDPVal output packaging report requires string paths for skipped entries")
        if skipped:
            logger.warning("GDPVal output packaging skipped %d entries in %s", len(skipped), state.task)
            (state.task / "packaging-skipped.json").write_text(json.dumps(skipped) + "\n")
        archive = state.run / "sandbox-output.zip"
        await state.sandbox.download("/tmp/gdpval-output.zip", archive)
        self._extract_results(archive, state.task)
        if any(PurePosixPath(entry["path"]).is_relative_to("/workspace/deliverables") for entry in skipped):
            raise RuntimeError("GDPVal output packaging skipped deliverables; refusing partial grading")

    @staticmethod
    def _save_masked_result(state: TaskSession, body: GDPValVerifyRequest, reason: str) -> GDPValVerifyResponse:
        result = GDPValVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=0.0,
            reward_valid=False,
            mask_sample=True,
            evaluation_completed=False,
            agent_finished=body.agent_finished,
            failure_kind=f"gdpval:{reason}",
            failure_reason=reason,
            judge_skipped=not body.agent_finished,
            **body.model_dump(
                include={"agent_error", "sandbox_error", "incomplete_reason", "policy_failure_reason"},
                exclude_none=True,
            ),
            # Context-compaction harnesses export their trainable tokens as one
            # response per context segment and leave ``response`` text-only; a
            # masked row still needs those tokens (NeMo RL rejects a result with
            # no generation data), so pass them through exactly like the judged path.
            **({"responses": body.responses} if body.responses else {}),
        )
        (state.task / "rollout.json").write_text(result.model_dump_json() + "\n")
        (state.task / "reward.json").write_text(
            json.dumps(
                {
                    "valid": False,
                    "reward": None,
                    "mask_sample": True,
                    "failure_reason": reason,
                }
            )
            + "\n"
        )
        return result

    async def _judge_saved_artifacts(self, state: TaskSession, config_path: Path, config: dict[str, Any]) -> None:
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
                    except ArtifactProcessingError:
                        raise
                    except (RuntimeError, TimeoutError) as error:
                        with (state.task / "judge-attempts.jsonl").open("a") as log:
                            log.write(
                                json.dumps(
                                    {"attempt": attempt, "error_type": type(error).__name__, "budget_s": budget}
                                )
                                + "\n"
                            )
                        if attempt == self.config.judge_max_attempts:
                            raise
                        await asyncio.sleep(min(self.config.judge_retry_delay_s, max(0, deadline - loop.time())))
                    finally:
                        log_path = state.task / "judge.log"
                        if log_path.exists():
                            (state.task / f"judge-attempt-{attempt}.log").write_bytes(log_path.read_bytes())

    async def verify(self, request: Request, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        state = self._sessions.get(request.session[SESSION_ID_KEY])
        if state is None or state.task_id != body.task_id:
            raise HTTPException(409, "No matching GDPVal task session")
        async with state.lock:
            if state.result is not None:
                return state.result
            if not state.ready or state.verify_started:
                raise HTTPException(409, "GDPVal session is not available for verification")
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
                    logger.warning(
                        "GDPVal outputs unavailable (%s, agent_finished=%s) in %s: %s",
                        body.incomplete_reason,
                        body.agent_finished,
                        state.task,
                        outputs_unavailable,
                    )
                (state.task / "trajectory.json").write_text(body.model_dump_json() + "\n")
            finally:
                try:
                    await state.sandbox.stop()
                except Exception:
                    logger.exception("GDPVal sandbox stop failed in verify (sandbox may already have ended)")
            if not body.agent_finished:
                state.result = self._save_masked_result(state, body, body.incomplete_reason)
                return state.result
            if outputs_unavailable is not None:
                (state.task / "outputs-unavailable.txt").write_text(outputs_unavailable + "\n")
                state.result = self._save_masked_result(state, body, "outputs_unavailable")
                return state.result
            try:
                manifest = self._collect_artifacts(state.task)
            except (OSError, ValueError) as error:
                logger.warning("GDPVal deliverables unavailable: %s", error)
                state.result = self._save_masked_result(state, body, "outputs_unavailable")
                return state.result
            if body.policy_failure_reason and body.policy_failure_reason in self.config.mask_policy_failures:
                state.result = self._save_masked_result(state, body, body.policy_failure_reason)
                return state.result
            config = {
                "judge_timeout": self.config.judge_timeout,
                "judge_url": get_server_url(self.config.judge_model_server.name) + "/v1",
                "judge_model": self.config.judge_model_server.name,
                "judge_create_params": self.config.judge_create_params,
                "judge_pdf_render_dpi": self.config.judge_pdf_render_dpi,
                "judge_pdf_max_pages": self.config.judge_pdf_max_pages,
                "judge_handles_audio": self.config.judge_handles_audio,
                "judge_handles_video": self.config.judge_handles_video,
                "libreoffice_command": self.config.libreoffice_command,
                "office_rendering_available": self._office_rendering_available,
                "judge_media_mode": self.config.judge_media_mode,
                "policy_failure_reason": body.policy_failure_reason,
                "empty_submission_zero_reward": not any(item["bytes"] for item in manifest),
            }
            config_path = state.run / "judge-config.json"
            config_path.touch(mode=0o600, exist_ok=True)
            try:
                await self._judge_saved_artifacts(state, config_path, config)
            except ArtifactProcessingError:
                state.result = self._save_masked_result(state, body, "artifact_processing_failed")
                return state.result
            except (RuntimeError, TimeoutError):
                logger.warning("Judge exhausted its budget/attempts; masking saved rollout %s", state.task)
                state.result = self._save_masked_result(state, body, "judge_unavailable")
                return state.result
            state.result = GDPValVerifyResponse.model_validate_json((state.task / "rollout.json").read_text())
            state.result = state.result.model_copy(
                update={
                    "evaluation_completed": True,
                    "agent_finished": True,
                    "reward_valid": True,
                    **body.model_dump(
                        include={"agent_error", "sandbox_error", "incomplete_reason", "policy_failure_reason"},
                        exclude_none=True,
                    ),
                    **({"responses": body.responses} if body.responses else {}),
                }
            )
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
                    logger.exception("Failed to preserve GDPVal failure outputs in %s", state.run)
                finally:
                    await state.sandbox.stop()
                    self._sessions.pop(key, None)
        return {"cleaned": True}

    async def shutdown(self) -> None:
        """Retry release of every resources-owned sandbox on server shutdown."""
        for key, state in list(self._sessions.items()):
            async with state.lock:
                try:
                    await state.sandbox.stop()
                except Exception:
                    logger.exception("GDPVal sandbox cleanup failed for session %s", key)
                else:
                    self._sessions.pop(key, None)


if __name__ == "__main__":
    GDPValResourcesServer.run_webserver()


elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GDPValResourcesServer.run_webserver()
