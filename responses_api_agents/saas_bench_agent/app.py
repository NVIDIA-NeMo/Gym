# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session adapter for the native SaaS external-harness worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    maybe_rollout_id_from_run_body,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.saas_bench_agent.response import actor_response
from responses_api_agents.saas_bench_agent.runtime import (
    DEFAULT_BUNDLE_ROOT,
    HARNESS_VERSIONS,
    check_bundle_code,
    configure_cache,
)
from responses_api_agents.saas_bench_agent.schemas import SaasBenchRunRequest


class SaasBenchConfig(BaseResponsesAPIAgentConfig):
    num_workers: int = 1
    session_close_timeout_seconds: float = 180
    source_root: str = str(DEFAULT_BUNDLE_ROOT)
    harness: Literal["opencode", "browser_use"] = "opencode"
    cache_root: str
    tasks_dir: str = "tasks"
    task_selector: str = "all"
    model_server: ModelServerRef
    model_name: str
    output_root: str
    worker_python: str | None = None
    command: str = "opencode"
    opencode_version: Literal["1.17.11"] = "1.17.11"
    concurrency: int = Field(default=1, ge=1)
    slot_capacity: int = Field(default=4, ge=1, le=40)
    slot_lock_root: str | None = None
    hostname: str = "localhost"
    max_steps: int = Field(default=200, ge=1)
    context_tokens: int = 262144
    output_tokens: int = 32768
    viewport_width: int = 1280
    viewport_height: int = 960
    mcp_timeout_ms: int = Field(default=180000, ge=1000)
    agent_timeout: int = Field(default=3600, ge=1)
    episode_timeout: int = Field(default=7200, ge=1)
    inconclusive_verification_retries: int = Field(default=1, ge=0, le=3)


class SaasBenchSessionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode: dict | None = None


@dataclass
class SaasBenchSession(AgentSessionState):
    body: SaasBenchRunRequest
    task: asyncio.Task | None = None
    activation: dict | None = None


class SaasBenchAgent(SimpleResponsesAPIAgent):
    """Execute episodes in separate processes so runtime globals cannot cross slots."""

    agent_context_model: ClassVar[type[BaseModel]] = SaasBenchSessionContext
    ray_enabled = False
    config: SaasBenchConfig
    _prepared: bool = PrivateAttr(default=False)
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)
    _fingerprints: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        if self.config.num_workers != 1 or self.config.concurrency > self.config.slot_capacity:
            raise ValueError("SaaS episodes require num_workers=1 and concurrency <= slot_capacity")
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    def _prepare_runtime(self) -> None:
        if self._prepared:
            return
        if not (os.environ.get("LLM_BASE_URL") or os.environ.get("MINDRA_BASE_URL")):
            raise ValueError("Set LLM_BASE_URL or MINDRA_BASE_URL to the judge endpoint before starting SaaS Bench")
        configure_cache(self.config.cache_root)
        if self.config.harness == "opencode" and self.config.command == "opencode":
            from responses_api_agents.opencode_agent.setup_opencode import ensure_opencode

            ensure_opencode(self.config.opencode_version)
        configured_source = Path(self.config.source_root)
        if not configured_source.is_absolute():
            configured_source = Path(__file__).resolve().parents[2] / configured_source
        source = configured_source.resolve(strict=True)
        self.config.source_root = str(source)
        if not (source / "saas_bench/opencode/worker.py").is_file():
            raise ValueError(f"Missing SaaS OpenCode implementation at {source}")
        check_bundle_code(source)
        # The prepared bundle joins shipped runtime code with verified task data.
        sys.path.insert(0, str(source))
        self._prepared = True

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> SaasBenchSession:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("SaaS native app adapter does not support sandbox access")
        if body.tool_accesses or body.continuation is not None:
            raise AgentSeedRejectedError("SaaS sessions accept neither external tool grants nor continuation")
        context = self.parse_agent_context(body)
        if not isinstance(context, SaasBenchSessionContext) or context.episode is None:
            raise AgentSeedRejectedError("SaaS requires its environment's episode context")
        episode = SaasBenchRunRequest.model_validate(context.episode)
        if maybe_rollout_id_from_run_body(episode) != body.episode_id.capture_key:
            raise AgentSeedRejectedError("SaaS session episode identity does not match its context")
        if episode.verifier_metadata["task_id"] != body.task_id.task_id:
            raise AgentSeedRejectedError("SaaS session task identity does not match its context")
        self._prepare_runtime()
        return SaasBenchSession(body, episode)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(422, "SaaS activation requires an environment-seeded agent session")
        state = self._require_agent_session(session_id)
        if not isinstance(state, SaasBenchSession):
            raise HTTPException(409, "Unexpected SaaS session state")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "SaaS activation does not match the session rollout")
        activation = body.model_dump(mode="json")
        if activation != state.body.responses_create_params.model_dump(mode="json"):
            raise HTTPException(409, "SaaS activation differs from its prepared episode input")
        if state.task is None:
            state.activation = activation
            state.task = asyncio.create_task(self._execute(state.body))
        elif state.activation != activation:
            raise HTTPException(409, "SaaS session already has a different activation")
        native = await asyncio.shield(state.task)
        response = actor_response(state.body, native, harness=self.config.harness, model_name=self.config.model_name)
        return response.model_copy(update={"saas_bench_native_result": native})

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, SaasBenchSession):
            raise RuntimeError("Unexpected SaaS session state")
        if state.task is not None and not state.task.done():
            state.task.cancel()
            try:
                await state.task
            except asyncio.CancelledError:
                pass
        if state.task is not None:
            output = (
                Path(self.config.output_root).resolve()
                / hashlib.sha256(state.request.episode_id.capture_key.encode()).hexdigest()
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(Path(self.config.source_root).resolve())
                + os.pathsep
                + str(Path(__file__).resolve().parents[2]),
            }
            await self._stop_recorded_apps(output, env)
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id, cleanup_confirmed=True)

    async def _execute(self, body: SaasBenchRunRequest) -> dict:
        # Capture IDs distinguish repeats and make HTTP retries idempotent.
        rollout_id = maybe_rollout_id_from_run_body(body)
        if not rollout_id:
            raise HTTPException(400, "SaaS Bench requires _ng_rollout_id; enable Gym observability")
        identity = hashlib.sha256(rollout_id.encode()).hexdigest()

        requested_version = body.verifier_metadata.get("harness_version")
        if body.verifier_metadata.get("harness", self.config.harness) != self.config.harness:
            raise HTTPException(409, "Dataset harness differs; prepare the selected harness again")
        if requested_version and requested_version != HARNESS_VERSIONS[self.config.harness]:
            raise HTTPException(
                409,
                "Prepared dataset targets a different harness version; prepare it again",
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "request": body.model_dump(),
                    "config": self.config.model_dump(),
                    "harness_version": HARNESS_VERSIONS[self.config.harness],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        output = Path(self.config.output_root).resolve() / identity
        if identity in self._fingerprints and self._fingerprints[identity] != fingerprint:
            raise HTTPException(409, "Rollout ID reused with different request")
        if identity not in self._inflight:
            self._fingerprints[identity] = fingerprint
            self._inflight[identity] = asyncio.create_task(self._episode(body, output, fingerprint, rollout_id))
        # Duplicate requests share completion; cancelling the owning request cleans up the episode.
        task = self._inflight[identity]
        try:
            return await task
        finally:
            if task.done():
                self._inflight.pop(identity, None)
                self._fingerprints.pop(identity, None)

    async def _episode(self, body, output: Path, fingerprint: str, rollout_id: str) -> dict:
        async with self._semaphore:
            output.mkdir(parents=True, exist_ok=True)
            # A persisted binding prevents retries from silently changing the task.
            binding = output / "binding.json"
            try:
                with binding.open("x") as stream:
                    json.dump({"fingerprint": fingerprint, "rollout_id": rollout_id}, stream)
            except FileExistsError:
                if json.loads(binding.read_text())["fingerprint"] != fingerprint:
                    raise HTTPException(409, "Rollout ID reused with different request")
                result_path = output / "result.json"
                if not result_path.exists():
                    raise HTTPException(
                        409,
                        "Incomplete episode requires inspection; use a new rollout ID to retry",
                    )
                return json.loads(result_path.read_text())
            config = self.config.model_dump()
            config.update(
                output=str(output),
                task_id=body.verifier_metadata["task_id"],
                source_sha256=body.verifier_metadata["source_sha256"],
                model=self.config.model_name,
                endpoint=self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            )
            config["harness"] = self.config.harness
            config["harness_version"] = HARNESS_VERSIONS[self.config.harness]
            config["slot_lock_root"] = self.config.slot_lock_root or str(
                Path(self.config.cache_root) / "slot-locks" / __import__("socket").gethostname()
            )
            config_path = output / "worker.json"
            config_path.write_text(json.dumps(config, indent=2))
            env = dict(os.environ)
            env["PYTHONPATH"] = (
                str(Path(self.config.source_root).resolve()) + os.pathsep + str(Path(__file__).resolve().parents[2])
            )
            process = None
            with (output / "worker.log").open("w") as log:
                try:
                    process = await asyncio.create_subprocess_exec(
                        self.config.worker_python or sys.executable,
                        "-m",
                        "saas_bench.opencode.worker"
                        if self.config.harness == "opencode"
                        else "responses_api_agents.saas_bench_agent.browser_worker",
                        str(config_path),
                        env=env,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    await asyncio.wait_for(process.wait(), timeout=self.config.episode_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    if process and process.returncode is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=45)
                        except asyncio.TimeoutError:
                            os.killpg(process.pid, signal.SIGKILL)
                            await process.wait()
                    await self._stop_recorded_apps(output, env)
                    raise
            result_path = output / "result.json"
            if not result_path.is_file():
                raise HTTPException(500, f"Worker exited without result; inspect {output}")
            return json.loads(result_path.read_text())

    async def _stop_recorded_apps(self, output: Path, env: dict[str, str]) -> None:
        """Reap detached Enroot processes even when a worker died during app readiness."""
        state = output / "enroot-state"
        if not any(state.glob("*.enroot.pid")):
            return
        script = (
            "import os\nfrom pathlib import Path\n"
            "from saas_bench.enroot_slot import stop_enroot_app\n"
            "for path in Path(os.environ['SAAS_ENROOT_STATE_DIR']).glob('*.enroot.pid'):\n"
            "    stop_enroot_app(path.name.removesuffix('.enroot.pid'))\n"
        )
        with (output / "cleanup.log").open("a") as log:
            cleanup = await asyncio.create_subprocess_exec(
                self.config.worker_python or sys.executable,
                "-c",
                script,
                env={**env, "SAAS_ENROOT_STATE_DIR": str(state)},
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(cleanup.wait(), timeout=120)
                if cleanup.returncode:
                    raise RuntimeError(f"App cleanup failed; inspect {output / 'cleanup.log'}")
            except BaseException:
                if cleanup.returncode is None:
                    os.killpg(cleanup.pid, signal.SIGKILL)
                    await cleanup.wait()
                raise


if __name__ == "__main__":
    SaasBenchAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SaasBenchAgent.run_webserver()
