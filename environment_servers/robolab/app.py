# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the assigned RoboLab condition and return its audited native score."""

import asyncio
import os
import secrets
import signal
import tempfile
from contextlib import suppress
from pathlib import Path
from statistics import mean
from typing import Annotated, Self

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, JsonValue, PrivateAttr, model_validator

from benchmarks.robolab.episode_binding import RoboLabEpisodeBinding
from benchmarks.robolab.prepare import Manifest, load_case_manifest
from environment_servers.robolab.opencode_relay import mount_opencode_relay
from environment_servers.robolab.terminal import SHA256, Score, read_condition
from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import apply_rollout_prefix
from nemo_gym.config import (
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    AGENT_TIMEOUT,
    ENVIRONMENT_SERVER_FAILED,
    MODEL_CONTEXT_LIMIT,
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_runtime import (
    AgentContext,
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    BaseVerifyResponse,
    CleanupContext,
    observe_verifier,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


POLICY_FAILURE_KINDS = {
    "policy_failure.decision_budget": AGENT_LIMIT,
    "policy_failure.deadline": AGENT_TIMEOUT,
    "policy_failure.context": MODEL_CONTEXT_LIMIT,
    "policy_failure.no_progress": AGENT_INVALID_OUTPUT,
}


class RoboLabTaskInput(BaseModel):
    case_id: str
    verifier_metadata: dict[str, JsonValue]
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class RoboLabVerifyResponse(BaseVerifyResponse):
    case_id: str
    seed: int
    manifest_sha256: str
    official_manifest_sha256: str
    native_source_sha256: str
    terminal_sha256: str
    native_score: Score | None = None
    native_success: bool | None = None
    native_outcome: str
    cleanup_confirmed: bool


class RoboLabEnvironmentConfig(BaseEnvironmentServerConfig):
    agent_server: AgentServerRef
    model_server: ModelServerRef | None = None
    case_manifest: Path
    backend: Path | None = None
    output_root: Path | None = None
    native_source_sha256: SHA256 | None = None
    use_policy_model_relay: bool = False
    episode_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    cleanup_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 180.0

    @model_validator(mode="after")
    def validate_backend(self) -> Self:
        if self.backend is not None and (
            not self.backend.is_absolute() or self.output_root is None or self.native_source_sha256 is None
        ):
            raise ValueError("RoboLab backend requires an absolute executable path, output root and native source SHA")
        if self.use_policy_model_relay and self.model_server is None:
            raise ValueError("RoboLab policy_model relay requires model_server to be configured")
        return self


class ScoredResult(BaseModel):
    native_score: Score
    native_success: Annotated[bool, Field(strict=True)]


class RoboLabEnvironmentServer(
    BaseEnvironmentServer[BaseEpisodeRequest[RoboLabTaskInput], BaseEpisodeResponse[dict[str, JsonValue]]]
):
    config: RoboLabEnvironmentConfig
    request_model = BaseEpisodeRequest[RoboLabTaskInput]
    response_model = BaseEpisodeResponse[dict[str, JsonValue]]
    _relays: dict[str, str] = PrivateAttr(default_factory=dict)
    _relay_mounted: bool = PrivateAttr(default=False)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        server_name = self.config.model_server.name if self.config.model_server is not None else "policy_model"
        mount_opencode_relay(app, self._relays, server_name=server_name)
        self._relay_mounted = True
        return app

    def multiple_workers_unsupported_reason(self) -> str:
        return "one RoboLab environment server instance owns one local GPU"

    def _resolve_model_base_url(self, rollout_id: str | None) -> str:
        assert self.config.model_server is not None
        server_config = get_first_server_config_dict(
            self.server_client.global_config_dict, self.config.model_server.name
        )
        base_url = self.server_client._build_server_base_url(server_config)
        prefixed = apply_rollout_prefix(base_url, rollout_id)
        return f"{prefixed}/v1"

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        if self.config.backend is None:
            raise HTTPException(422, "RoboLab metrics require the pinned Bridge runtime adapter")
        manifest, digest = load_case_manifest(self.config.case_manifest)
        scores = []
        try:
            for row in body.verify_responses:
                if row.get("mask_sample") or row.get("native_score") is None:
                    continue
                if (
                    row.get("case_id") != manifest.case.id
                    or type(row.get("seed")) is not int
                    or row["seed"] != 0
                    or row.get("manifest_sha256") != digest
                    or row.get("official_manifest_sha256") != manifest.official_manifest_sha256
                    or row.get("native_source_sha256") != self.config.native_source_sha256
                    or row.get("cleanup_confirmed") is not True
                    or row.get("native_outcome") != "scored"
                ):
                    raise ValueError("RoboLab metrics row differs from the assigned native condition")
                scores.append(ScoredResult.model_validate(row))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        metrics: dict[str, float | int] = {
            "num/scored_cases": len(scores),
            "num/unscored_cases": len(body.verify_responses) - len(scores),
        }
        if scores:
            metrics["mean/native_score"] = mean(row.native_score for row in scores)
            metrics["mean/native_success"] = mean(row.native_success for row in scores)
        return AggregateMetrics(agent_metrics=metrics, key_metrics=metrics)

    def bind_case(self, request: BaseEpisodeRequest[RoboLabTaskInput]) -> tuple[Manifest, RoboLabEpisodeBinding]:
        task = request.task
        if task.task_id.taskset != "robolab":
            raise ValueError("RoboLab requires the robolab taskset")
        manifest, digest = load_case_manifest(self.config.case_manifest)
        metadata = task.task_input.verifier_metadata
        binding = RoboLabEpisodeBinding.model_validate(
            {
                "case_id": task.task_input.case_id,
                "seed": metadata.get("seed"),
                "manifest_sha256": metadata.get("manifest_sha256"),
            }
        )
        if (binding.case_id, binding.seed, binding.manifest_sha256) != (manifest.case.id, 0, digest):
            raise ValueError("RoboLab case, seed or manifest SHA differs from the staged manifest")
        return manifest, binding

    def agent_seed_request(self, request: BaseEpisodeRequest[RoboLabTaskInput]) -> AgentSeedSessionRequest:
        """Bind the prepared case and make the Gym agent-session seed."""
        _, binding = self.bind_case(request)
        return AgentSeedSessionRequest(
            agent_session_id=request.episode_id.capture_key,
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            agent_context=AgentContext(values=binding.model_dump(mode="json")),
        )

    async def run(
        self, request: BaseEpisodeRequest[RoboLabTaskInput], cleanup: CleanupContext
    ) -> BaseEpisodeResponse[dict[str, JsonValue]]:
        try:
            manifest, binding = self.bind_case(request)
            backend = self.config.backend
            root = self.config.output_root
            native_source_sha256 = self.config.native_source_sha256
            if (
                backend is None
                or root is None
                or native_source_sha256 is None
                or not backend.is_file()
                or not os.access(backend, os.X_OK)
            ):
                raise ValueError("RoboLab rollout requires the pinned Bridge runtime adapter")
            timeout = self.config.episode_timeout_seconds
            if timeout is None:
                bridge = manifest.bridge_config
                native = bridge["native"]
                startup_seconds = native.get("startup_seconds") if isinstance(native, dict) else None
                episode_seconds = bridge.get("episode_seconds")
                if (
                    type(startup_seconds) is not int
                    or startup_seconds <= 0
                    or type(episode_seconds) is not int
                    or episode_seconds <= 0
                ):
                    raise ValueError("RoboLab requires finite startup and episode limits in the case manifest")
                timeout = startup_seconds + episode_seconds
            root.mkdir(parents=True, exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix="robolab-", dir=root)).resolve()
            staged_manifest = attempt / "case.json"
            staged_manifest.write_bytes(self.config.case_manifest.read_bytes())
            if load_case_manifest(staged_manifest)[1] != binding.manifest_sha256:
                raise ValueError("RoboLab manifest changed before launch")
        except (OSError, ValueError) as error:
            return self.failure_response(
                request, EpisodeFailure(failure_reason=str(error), terminal=True, stage="admission")
            )
        condition = attempt / "condition"
        process = None
        relay_id: str | None = None
        child_env: dict[str, str] | None = None
        if self.config.use_policy_model_relay and self._relay_mounted and self.config.model_server is not None:
            model_url = self._resolve_model_base_url(request.episode_id.capture_key)
            relay_id = secrets.token_urlsafe(18)
            self._relays[relay_id] = model_url
            relay_host = "127.0.0.1" if self.config.host in ("0.0.0.0", "::", "") else self.config.host
            relay_url = f"http://{relay_host}:{self.config.port}/_opencode_relay/{relay_id}/v1"
            child_env = {**os.environ, "ROBOLAB_POLICY_MODEL_RELAY_URL": relay_url}

        async def stop_backend() -> None:
            if process is None:
                return
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            cleanup_timeout = self.config.cleanup_timeout_seconds
            kill_and_reap_reserve = min(5, cleanup_timeout / 10)
            try:
                async with asyncio.timeout(cleanup_timeout - kill_and_reap_reserve):
                    await process.wait()
            except TimeoutError:
                pass
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

        stopped = cleanup.register_cleanup("RoboLab Bridge runtime", stop_backend)
        try:
            with (attempt / "bridge.log").open("xb") as log:
                launch = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        str(backend),
                        "--case-manifest",
                        str(staged_manifest),
                        "--condition-output",
                        str(condition),
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        env=child_env,
                        start_new_session=True,
                    )
                )
                try:
                    process = await asyncio.shield(launch)
                except asyncio.CancelledError:
                    process = await asyncio.shield(launch)
                    raise
                async with asyncio.timeout(timeout):
                    returncode = await process.wait()
            if returncode:
                raise RuntimeError(f"RoboLab Bridge runtime exited {returncode}; see {attempt / 'bridge.log'}")
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("RoboLab Bridge process group remained live after runtime exit")
            await stopped.close()
        except (OSError, RuntimeError, TimeoutError) as error:
            return self.failure_response(
                request,
                EpisodeFailure(failure_reason=str(error) or "RoboLab runtime timed out", terminal=True, stage="agent"),
            )
        finally:
            if relay_id is not None:
                self._relays.pop(relay_id, None)
        try:
            result = read_condition(condition, manifest, binding.manifest_sha256, native_source_sha256)
        except (OSError, ValueError) as error:
            return self.failure_response(
                request, EpisodeFailure(failure_reason=str(error), terminal=True, stage="verification")
            )
        model_info = manifest.bridge_config.get("model")
        model_id = (
            model_info.get("id")
            if isinstance(model_info, dict) and isinstance(model_info.get("id"), str) and model_info.get("id")
            else "robolab"
        )
        response = NeMoGymResponse(
            id=attempt.name,
            created_at=0,
            model=model_id,
            object="response",
            output=[],
            parallel_tool_calls=False,
            tools=[],
            tool_choice="none",
            metadata={"episode_engine": "robolab_bridge"},
        )
        common = {
            "responses_create_params": request.task.task_input.responses_create_params,
            "response": response,
            "case_id": result["case_id"],
            "seed": result["seed"],
            "manifest_sha256": result["manifest_sha256"],
            "official_manifest_sha256": result["official_manifest_sha256"],
            "native_source_sha256": result["native_source_sha256"],
            "terminal_sha256": result["terminal_sha256"],
            "native_outcome": result["native_outcome"],
            "cleanup_confirmed": result["cleanup_confirmed"],
        }
        if result["native_score"] is None:
            outcome = str(result["native_outcome"])
            verify_response = RoboLabVerifyResponse(
                **common,
                reward=0.0,
                native_score=None,
                native_success=None,
                mask_sample=True,
                failure_kind=POLICY_FAILURE_KINDS.get(outcome, ENVIRONMENT_SERVER_FAILED),
                failure_reason=outcome,
            )
            return self.response_model(
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                result=verify_response.model_dump(mode="json"),
            )

        async def verified_grade() -> RoboLabVerifyResponse:
            return RoboLabVerifyResponse(
                **common,
                reward=float(result["reward"]),
                native_score=result["native_score"],
                native_success=result["native_success"],
            )

        graded = await observe_verifier(verified_grade, enabled=True, verifier_name=self.config.name)()
        return self.response_model(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=graded.model_dump(mode="json"),
        )


if __name__ == "__main__":
    RoboLabEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = RoboLabEnvironmentServer.run_webserver()
