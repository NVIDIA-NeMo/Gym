# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select prompt-group harnesses for digest-bound AVB and ClawEval tasks."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import Field, PrivateAttr, ValidationError, model_validator

from environment_servers.harbor.app import (
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborEpisodeRequest,
    HarborEpisodeResponse,
    HarborEpisodeResult,
)
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.config import AgentServerRef
from nemo_gym.core import (
    EpisodeId,
    HarnessRoutingMetadata,
    MaterializedTask,
    ModelCallSampling,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    maybe_rollout_id_from_run_body,
    model_sampling_from_run_body,
    skips_token_capture,
)
from nemo_gym.data.tasks.harbor import HarborTask, HarborTaskInput, materialize_task
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import CleanupContext
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOGGER = logging.getLogger(__name__)


class UnifiedVideoHarborConfig(HarborEnvironmentServerConfig):
    harness_agents: dict[str, AgentServerRef] = Field(default_factory=dict)
    require_harness_routing: bool = False


class UnifiedVideoTaskInput(HarborTaskInput):
    # Native EnvironmentParticipant checkpoints the task, so retain the selected route
    # there too. This field never enters the digest-covered task_data or model prompt.
    harness: str | None = None
    harness_routing: HarnessRoutingMetadata | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="before")
    @classmethod
    def accept_flat_task_input(cls, value: object) -> object:
        # Extend the native envelope explicitly: its flat-field validator would
        # otherwise fold this checkpoint routing key into digest-covered task_data.
        if not isinstance(value, Mapping) or not ({"harness", "harness_routing"} & value.keys()):
            return super().accept_flat_task_input(value)
        values = dict(value)
        route = {name: values.pop(name) for name in ("harness", "harness_routing") if name in values}
        return super().accept_flat_task_input(values) | route


class UnifiedVideoEpisodeRequest(HarborEpisodeRequest):
    task: MaterializedTask[UnifiedVideoTaskInput]
    harness: str | None = None
    harness_routing: HarnessRoutingMetadata | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def preserve_checkpoint_route(self) -> UnifiedVideoEpisodeRequest:
        saved_routing = self.task.task_input.harness_routing
        if self.harness_routing is not None and saved_routing is not None and self.harness_routing != saved_routing:
            raise ValueError("Request and checkpoint harness routing disagree")
        routing = self.harness_routing if self.harness_routing is not None else saved_routing
        if routing is not None:
            if routing.task_type != "agentic":
                raise ValueError("Harbor harness routing requires an agentic task")
            self.harness_routing = routing.select()
            self.task.task_input.harness_routing = self.harness_routing
            if self.harness is not None and self.harness != self.harness_routing.selected_harness:
                raise ValueError("Request harness differs from its prompt-group selection")
            self.harness = self.harness_routing.selected_harness
        saved = self.task.task_input.harness
        if self.harness is not None and saved is not None and self.harness != saved:
            raise ValueError("Request and checkpoint harness disagree")
        self.harness = self.harness if self.harness is not None else saved
        self.task.task_input.harness = self.harness
        return self


class UnifiedVideoEpisodeResult(HarborEpisodeResult):
    task_source: str
    harness: str | None = None
    harness_agent_ref: AgentServerRef | None = None
    harness_routing: HarnessRoutingMetadata | None = Field(default=None, exclude_if=lambda value: value is None)


class UnifiedVideoEpisodeResponse(HarborEpisodeResponse):
    result: UnifiedVideoEpisodeResult | None = None
    harness_routing: HarnessRoutingMetadata | None = Field(default=None, exclude_if=lambda value: value is None)


class _VideoTaskHarbor(HarborEnvironmentServer):
    """Native lifecycle/scoring, with only the video datasets' ungraded failure rule."""

    _expected_harness: str | None = PrivateAttr(default=None)

    async def _activate_agent_session(
        self,
        request: HarborEpisodeRequest,
        handle: Any,
        responses_create_params: Any,
        progress: Any,
        *,
        activation: int | None = None,
    ) -> NeMoGymResponse:
        response = await super()._activate_agent_session(
            request, handle, responses_create_params, progress, activation=activation
        )
        if self._expected_harness is not None:
            executed = (response.metadata or {}).get("executed_harness")
            if executed != self._expected_harness:
                raise self._failure(
                    stage="agent",
                    terminal=True,
                    failure_reason="Responding agent bridge differs from the selected harness",
                    partial_response=response,
                )
        return response

    def _score(
        self,
        task: HarborTask,
        settings: Any,
        result: SandboxExecResult,
        logs_dir: Path,
        started: float,
        *,
        timed_out: bool,
        rewards: dict[str, float] | None,
        problem: str | None,
        separate: bool,
    ) -> dict[str, Any]:
        outcome = super()._score(
            task,
            settings,
            result,
            logs_dir,
            started,
            timed_out=timed_out,
            rewards=rewards,
            problem=problem,
            separate=separate,
        )
        source = task.task_data.get("video_source")
        if source == "claweval_agent" and rewards is None:
            raise self._failure(
                stage="verification",
                terminal=False,
                failure_reason="Claw verifier produced no native reward; grading is unavailable",
            )
        if source == "claweval_agent" and rewards is not None and not 0 <= outcome["reward"] <= 1:
            raise self._failure(
                stage="verification",
                terminal=False,
                failure_reason="Native video verifier reward is outside its [0, 1] contract",
            )
        # Measured finite zero and fractional grades are exactly native outcomes.
        # Missing scores become typed failures above, never fabricated grade zero.
        return outcome


class UnifiedVideoHarborServer(_VideoTaskHarbor):
    config: UnifiedVideoHarborConfig
    request_model = UnifiedVideoEpisodeRequest
    response_model = UnifiedVideoEpisodeResponse
    _routes: dict[tuple[str | None, str], _VideoTaskHarbor] = PrivateAttr(default_factory=dict)

    def agent_server_for_episode(self, request: HarborEpisodeRequest) -> AgentServerRef:
        harness = request.harness if isinstance(request, UnifiedVideoEpisodeRequest) else None
        if isinstance(request, UnifiedVideoEpisodeRequest) and self.config.require_harness_routing:
            if request.harness_routing is None or request.harness_routing.group_id is None:
                raise HTTPException(422, "A checkpointed prompt-group harness route is required")
            if request.harness_routing.env != self.config.name:
                raise HTTPException(422, "Harness routing must name the actual environment")
        if harness is None:
            return self.config.agent_server
        if harness not in self.config.harness_agents:
            raise HTTPException(422, "Select a configured video harness")
        return self.config.harness_agents[harness]

    def _new_route(self, agent: AgentServerRef) -> _VideoTaskHarbor:
        values = self.config.model_dump(exclude={"harness_agents", "require_harness_routing"})
        values["agent_server"] = agent.model_dump()
        route = _VideoTaskHarbor(
            config=HarborEnvironmentServerConfig.model_validate(values), server_client=self.server_client
        )
        # One front-end owns one native admission limit and one duplicate-owner set.
        # The native lifecycle remains responsible for execution and cleanup.
        route._active = self._active
        return route

    def _route(self, request: HarborEpisodeRequest) -> _VideoTaskHarbor:
        agent = self.agent_server_for_episode(request)
        harness = request.harness if isinstance(request, UnifiedVideoEpisodeRequest) else None
        key = (harness, agent.name)
        if key not in self._routes:
            self._routes[key] = self._new_route(agent)
            self._routes[key]._expected_harness = harness
        return self._routes[key]

    async def run_request(self, request: UnifiedVideoEpisodeRequest) -> UnifiedVideoEpisodeResponse:
        response = await super().run_request(request)
        routing = request.harness_routing
        if routing is not None:
            agent_response = (
                response.result.response
                if response.result is not None
                else (response.failure.partial_response if response.failure is not None else None)
            )
            executed = (agent_response.metadata or {}).get("executed_harness") if agent_response is not None else None
            if response.failure is not None and executed != routing.selected_harness:
                executed = None
            stamped = routing.with_execution(executed)
            response.harness_routing = stamped
            if response.result is not None:
                response.result.harness_routing = stamped
        return response

    async def episode_timeout_seconds(self, request: HarborEpisodeRequest) -> float | None:
        return await self._route(request).episode_timeout_seconds(request)

    async def run(self, request: UnifiedVideoEpisodeRequest, cleanup: CleanupContext) -> UnifiedVideoEpisodeResponse:
        route = self._route(request)
        response = await route.run(request, cleanup)
        # Preserve every native result/trajectory/observation field. Attribution is
        # explicit routing metadata, never inferred from an arbitrary taskset name.
        result = None
        if response.result is not None:
            selected = self.agent_server_for_episode(request)
            harness = request.harness
            if harness is None:
                matches = [key for key, ref in self.config.harness_agents.items() if ref == selected]
                harness = matches[0] if len(matches) == 1 else None
            result = UnifiedVideoEpisodeResult.model_validate(
                response.result.model_dump(mode="python")
                | {
                    "task_source": self.config.agent_server.name,
                    "harness": harness,
                    "harness_agent_ref": selected.model_dump(mode="python") if harness is not None else None,
                }
            )
        return UnifiedVideoEpisodeResponse.model_validate(response.model_dump(mode="python") | {"result": result})

    async def shutdown(self) -> None:
        for route in self._routes.values():
            await route.shutdown()
        await super().shutdown()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        native = next(route for route in app.routes if getattr(route, "path", None) == "/run")
        app.routes.remove(native)
        app.add_api_route("/v1/episodes", native.endpoint, methods=["POST"], response_model=self.response_model)
        app.post("/run")(self.run_legacy)
        return app

    def _episode_request_from_row(self, row: dict[str, Any]) -> UnifiedVideoEpisodeRequest:
        source = row.get("task_source", self.config.agent_server.name)
        if source != "claweval_agent":
            raise ValueError("ClawEval rows are required; use stock Harbor task rows for Agentic-VBench")
        if source != self.config.agent_server.name:
            raise ValueError("task_source does not name this environment's agent")
        agent = row.get("agent_ref")
        if agent is not None and (not isinstance(agent, dict) or agent.get("name") != source):
            raise ValueError("agent_ref does not match task_source")
        attempt = row.get("_ng_attempt_index", 0)
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError("Invalid episode attempt")
        binding = row.get("_unified_video_task")
        if not isinstance(binding, dict):
            raise ValueError("Missing pinned _unified_video_task; prepare the training dataset")
        harness = row.get("harness")
        if self.config.harness_agents:
            choices = list(self.config.harness_agents)
            if harness is None:
                index = row.get("_ng_rollout_index")
                if isinstance(index, bool) or (index is not None and (not isinstance(index, int) or index < 0)):
                    raise ValueError("Rollout index must be a nonnegative integer")
                if index is None:
                    identity = maybe_rollout_id_from_run_body(row) or uuid4().hex
                    logical_id = EpisodeId.from_capture_key(identity).rollout_id
                    index = int(hashlib.sha256(logical_id.encode()).hexdigest(), 16)
                harness = choices[index % len(choices)]
            if harness not in self.config.harness_agents:
                raise ValueError("Unknown video harness")
        elif harness is not None:
            raise ValueError("Harness selection requires configured harness_agents")
        request = UnifiedVideoEpisodeRequest.model_validate(
            {
                "episode_id": EpisodeId.from_capture_key(maybe_rollout_id_from_run_body(row) or uuid4().hex),
                "task": binding,
                "model_sampling": model_sampling_from_run_body(row),
                "skip_token_capture": skips_token_capture(row),
                "harness": harness,
            }
        )
        taskset = request.task.task_id.taskset
        if taskset != source:
            raise ValueError("Native taskset does not match task_source")
        metadata = request.task.task_input.task_data.get("verifier_metadata")
        if not isinstance(metadata, dict) or metadata != row.get("verifier_metadata"):
            raise ValueError("Verifier metadata does not match the pinned task")
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
        inputs = params.model_dump(mode="json", exclude_none=True)["input"]
        if not isinstance(inputs, list) or len(inputs) != 1 or inputs[0].get("role") != "user":
            raise ValueError("Expected the original single user prompt")
        content = inputs[0].get("content")
        if isinstance(content, list):
            if len(content) != 1 or content[0].get("type") != "input_text":
                raise ValueError("Task media must be staged in the sandbox")
            content = content[0].get("text")
        if not isinstance(content, str) or hashlib.sha256(content.encode()).hexdigest() != metadata.get(
            "prompt_sha256"
        ):
            raise ValueError("Prompt hash does not match the pinned task")

        allowed = {"input", *ModelCallSampling.model_fields}
        unsupported = params.model_fields_set - allowed
        if unsupported:
            raise ValueError("Flat video rows may set only the prompt and native model sampling fields")
        if request.model_sampling is None:
            sampling = params.model_dump(
                include=set(ModelCallSampling.model_fields), exclude_none=True, exclude_unset=True
            )
            request.model_sampling = ModelCallSampling.model_validate(sampling) if sampling else None

        def validate_instruction(
            native: NeMoGymResponseCreateParamsNonStreaming,
        ) -> NeMoGymResponseCreateParamsNonStreaming:
            native_input = native.model_dump(mode="json", exclude_none=True).get("input")
            if not isinstance(native_input, list) or len(native_input) != 1 or native_input[0].get("role") != "user":
                raise ValueError("Pinned task must carry one user instruction")
            native_content = native_input[0].get("content")
            if isinstance(native_content, list):
                if len(native_content) != 1 or native_content[0].get("type") != "input_text":
                    raise ValueError("Pinned task must carry a text instruction")
                native_content = native_content[0].get("text")
            if native_content != content:
                raise ValueError("Pinned instruction does not match the original prompt")
            # Native model_sampling carries the explicit whitelist to trained
            # calls. The digest-bound task params, tools and deadline stay intact.
            return native

        task_input = request.task.task_input
        if task_input.responses_create_params is not None:
            task_input.responses_create_params = validate_instruction(task_input.responses_create_params)
        for turn in task_input.turns:
            turn.responses_create_params = validate_instruction(turn.responses_create_params)
        return request

    async def run_legacy(self, row: dict[str, Any]) -> dict[str, Any]:
        """Accept episode requests and strict flat rows on the shared run route."""
        if "episode_id" in row and "task" in row:
            try:
                request = UnifiedVideoEpisodeRequest.model_validate(row)
            except ValidationError as error:
                raise RequestValidationError(error.errors(include_url=False)) from None
            response = await self.run_request(request)
            return response.model_dump(mode="json")
        try:
            request = self._episode_request_from_row(row)
            pinned = await self._resolve_task(request.task.task_id, request.task.task_input.task_data)
            self._check_task_data(request.task.task_id.taskset, pinned, request.task.task_input.task_data)
            canonical = materialize_task(pinned, request.task.task_id.taskset).model_dump(
                mode="json", exclude_unset=True
            )
            if row["_unified_video_task"] != canonical:
                raise ValueError("Native task binding differs from the digest-checked prepared task")
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        response = await self.run_request(request)
        result = SingleAgentTurnLegacyEnvironmentServer._legacy_result(self, response)
        if response.result is not None:
            result["responses_create_params"] = row["responses_create_params"]
            result["verifier_metadata"] = row["verifier_metadata"]
            result["task_source"] = self.config.agent_server.name
        if request.harness is not None:
            result["harness"] = request.harness
            result["harness_agent_ref"] = self.agent_server_for_episode(request).model_dump(mode="json")
            if self.config.artifacts_dir is not None:
                directory = self.config.artifacts_dir / "harness_rollouts"
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    destination = directory / (request.episode_id.capture_key + ".json")
                    temporary = destination.with_suffix("." + uuid4().hex + ".tmp")
                    temporary.write_text(
                        json.dumps(
                            {
                                "capture_key": request.episode_id.capture_key,
                                "harness": request.harness,
                                "task_source": self.config.agent_server.name,
                                "logical_rollout_id": request.episode_id.rollout_id,
                                "task_id": request.task.task_id.task_id,
                                "rollout_index": row.get("_ng_rollout_index"),
                                "reward": result.get("reward"),
                                "mask_sample": result.get("mask_sample", True),
                                "failure_kind": result.get("failure_kind")
                                or result.get("_ng_failure_kind")
                                or result.get("_ng_failure_stage"),
                                "failure_reason": result.get("failure_reason") or result.get("_ng_failure_message"),
                                "verifier_logs_dir": result.get("verifier_logs_dir"),
                                "completed_epoch": time.time(),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    temporary.replace(destination)
                except OSError:
                    LOGGER.exception(
                        "Could not write video harness diagnostics for %s", request.episode_id.capture_key
                    )
        return result


if __name__ == "__main__":
    UnifiedVideoHarborServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = UnifiedVideoHarborServer.run_webserver()
