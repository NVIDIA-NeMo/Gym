# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""partner_if's Environment Server: one dataset and one score over tasks that run on different harnesses.

It runs the text tasks itself, as ``single_agent_turn_legacy`` does, and sends every other task to the Environment
Server of its task family, which owns that episode's sessions, sandboxes and grading.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import Field, PrivateAttr, model_validator

from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.config import EnvironmentServerRef
from nemo_gym.core import FailureReportRow, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import (
    NG_AGENT_OBSERVATIONS_KEY,
    ToolCallObservation,
    TrajectoryRecord,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


# Environment Servers whose /run takes a flat row, the only kind this server sends.
_FLAT_ROW_ENVIRONMENT_SERVERS = frozenset({"single_agent_turn_legacy", "legacy_agent"})

# Headers that describe a connection or a body, not the payload, as in the legacy_agent environment relay. Each hop
# frames its own, and `cookie` is carried separately so aiohttp does not send it twice.
_SKIPPED_HEADERS = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "cookie",
        "host",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
    }
)


def _relayed_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(name, value) for name, value in headers if name.lower() not in _SKIPPED_HEADERS]


def _with_reply_tool_observations(body: bytes) -> bytes:
    """Add the tool executions in the agent reply's trajectory to the episode's agent observations.

    In an agent session simple_agent returns only its invocations at close, but its reply keeps the whole
    trajectory, and rollout collection builds `ng_trajectory.tool_calls` only from the observations. Each
    execution is added once, without its output, which collection joins from the invocation's conversation.
    Any other body is returned unchanged.
    """
    try:
        result = json.loads(body)
        records = result[NG_AGENT_OBSERVATIONS_KEY]["records"]
        trajectory = TrajectoryRecord.model_validate(result["response"]["_ng_trajectory"])
    except (ValueError, KeyError, TypeError):
        return body
    if not isinstance(records, list):
        return body
    observed = {
        (record.get("invocation_id"), record.get("tool_call_id"))
        for record in records
        if isinstance(record, dict) and record.get("kind") == "tool_call"
    }
    added = [
        ToolCallObservation.model_validate(tool.model_dump(exclude={"output"})).model_dump(mode="json")
        for tool in trajectory.tool_calls
        if (tool.invocation_id, tool.tool_call_id) not in observed
    ]
    if not added:
        return body
    records.extend(added)
    return json.dumps(result, ensure_ascii=False).encode()


def _recorded_context_run_binding(payload: dict) -> dict:
    """Retain collector identity across stock BaseRunRequest's excluded transport fields."""
    return {
        key: payload[key]
        for key in (
            "_ng_rollout_id",
            "_ng_attempt_index",
            "task_index",
            "rollout_index",
            "_ng_task_index",
            "_ng_rollout_index",
        )
        if key in payload
    }


def _recorded_context_result(body: bytes, payload: dict, *, agent_name: str) -> bytes:
    """Lift the Resources-owned terminal report; stock Gymnasium omits terminal tool outputs.

    The route is selected from host configuration. Identity and checksum checks catch stale or malformed
    reports; the checksum is not an authentication mechanism. Policy output is never consulted for a grade.
    """
    from resources_servers.partner_if.app import VerifyResponse
    from resources_servers.partner_if.recorded_context.assets import actor_identity
    from resources_servers.partner_if.recorded_context_app import (
        recorded_context_result_sha256,
        recorded_context_task_data,
    )

    try:
        envelope = json.loads(body)
        result = envelope["info"]["partner_if_recorded_context_result"]
        if not isinstance(result, dict):
            raise ValueError("Missing Resources result")
        if result["recorded_context_result_sha256"] != recorded_context_result_sha256(result):
            raise ValueError("Resources result checksum differs")
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload["responses_create_params"])
        expected_params = params.model_dump(mode="json", exclude_unset=True, exclude_none=True)
        actual_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(result["responses_create_params"])
        if actor_identity(
            actual_params.model_dump(mode="json", exclude_unset=True, exclude_none=True)
        ) != actor_identity(expected_params):
            raise ValueError("Resources result actor differs")
        task_data = recorded_context_task_data(payload)
        task_hash = hashlib.sha256(
            json.dumps(task_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        if result["recorded_context"]["task_data_sha256"] != task_hash:
            raise ValueError("Resources result task differs")
        if any(result.get(key) != value for key, value in task_data.items()):
            raise ValueError("Resources result task data differs")
        if not {"reward", "mask_sample", "evaluation_completed", "if_result", "response"} <= result.keys():
            raise ValueError("Resources result omits grade fields")
        grade = VerifyResponse.model_validate(result)
        if not math.isfinite(grade.reward) or grade.reward not in (0.0, 1.0):
            raise ValueError("Invalid Resources reward")
        if type(result["mask_sample"]) is not bool or type(result["evaluation_completed"]) is not bool:
            raise ValueError("Invalid Resources completion flags")
        if result["evaluation_completed"] == result["mask_sample"]:
            raise ValueError("Resources completion contradicts masking")
        if envelope.get("reward") != result["reward"] or envelope.get("mask_sample", False) != result["mask_sample"]:
            raise ValueError("Gymnasium result disagrees with Resources")
        if not (envelope.get("terminated") or envelope.get("truncated")):
            raise ValueError("Resources result is not terminal")
    except (ValueError, KeyError, TypeError, AttributeError):
        raise HTTPException(502, "Invalid recorded-context Resources result") from None
    # Keep transport failure evidence and cleanup warnings from the stock agent.
    for name, value in envelope.items():
        if name.startswith("_ng_failure") and name not in FailureReportRow.KEYS:
            result[name] = value
    FailureReportRow.merged_into(result, FailureReportRow.report(envelope))
    if warning := envelope.get("info", {}).get("cleanup_warning"):
        result["recorded_context"]["cleanup_warning"] = warning
    result["agent_ref"] = {"name": agent_name}
    return json.dumps(result, ensure_ascii=False).encode()


def _task_id(payload: Any) -> str:
    try:
        task_id = payload["verifier_metadata"]["partner_if"]["task_id"]
    except (KeyError, TypeError):
        raise HTTPException(422, "Expected verifier_metadata.partner_if.task_id") from None
    if not isinstance(task_id, str):
        raise HTTPException(422, "Expected verifier_metadata.partner_if.task_id") from None
    return task_id


class PartnerIFEnvironmentServerConfig(SingleAgentTurnLegacyEnvironmentServerConfig):
    """The text tasks' pairing, plus the Environment Server of every task.

    `task_routes` names the Environment Server of each task ID. A task routed to this server runs here, with
    `agent_server` and `resources_server`; any other runs on the server it names. A task with no route is refused,
    so a new release cannot run a task on the wrong harness. Request payloads cannot change a route.
    `resources_server` also aggregates the metrics of every task, so the benchmark has one score.
    """

    task_routes: dict[str, EnvironmentServerRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_routes(self) -> "PartnerIFEnvironmentServerConfig":
        if any(not task.strip() for task in self.task_routes):
            raise ValueError("Task IDs must not be empty")
        return self


class PartnerIFEnvironmentServer(SingleAgentTurnLegacyEnvironmentServer):
    """Run the text tasks; send each other task's flat row to its task family's Environment Server."""

    config: PartnerIFEnvironmentServerConfig
    # Routed Environment Server name -> the agent it binds, which the routed row must name.
    _execution_agents: dict[str, str] = PrivateAttr(default_factory=dict)
    _recorded_context_routes: set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="after")
    def validate_destinations(self) -> "PartnerIFEnvironmentServer":
        for route in self.config.task_routes.values():
            if route.name == self.config.name:
                continue
            destination = self.server_client.global_config_dict.get(route.name, {})
            environments = destination.get("environment_servers", {}) if isinstance(destination, Mapping) else {}
            if len(environments) != 1:
                raise ValueError(f"Missing or ambiguous environment destination: {route.name}")
            implementation, environment = next(iter(environments.items()))
            if implementation not in _FLAT_ROW_ENVIRONMENT_SERVERS:
                # One hop, to a server that takes the flat row as it is: never another partner_if server.
                raise ValueError(
                    f"Route {route.name} is a {implementation} Environment Server; a route must name one of "
                    f"{sorted(_FLAT_ROW_ENVIRONMENT_SERVERS)}"
                )
            environment = environment if isinstance(environment, Mapping) else {}
            agent_name = (environment.get("agent_server") or {}).get("name")
            if not isinstance(agent_name, str) or not agent_name:
                raise ValueError(f"Environment must bind one execution agent: {route.name}")
            self._execution_agents[route.name] = agent_name
            agent = self.server_client.global_config_dict.get(agent_name, {})
            implementations = agent.get("responses_api_agents", {}) if isinstance(agent, Mapping) else {}
            gymnasium = implementations.get("gymnasium_agent")
            if implementation == "legacy_agent" and isinstance(gymnasium, Mapping):
                resources_name = (gymnasium.get("resources_server") or {}).get("name")
                resources = self.server_client.global_config_dict.get(resources_name, {}) if resources_name else {}
                block = (
                    (resources.get("resources_servers") or {}).get("partner_if")
                    if isinstance(resources, Mapping)
                    else None
                )
                if isinstance(block, Mapping) and block.get("entrypoint") == "recorded_context_app.py":
                    self._recorded_context_routes.add(route.name)
        return self

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        # As in single_agent_turn_legacy: the episodes run here checkpoint, and their cleanups finish at shutdown.
        self.setup_environment_checkpoint(app)
        self.setup_shutdown_sweep(app)
        app.post("/run")(self.run_task)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def run_task(self, request: Request) -> Response:
        """Run a text task here or send its row to the routed server, keeping row fields and that server's failures.

        Only flat rows are accepted: the task ID routes them, and the routed servers take flat rows.
        """
        raw = await request.body()
        try:
            payload = json.loads(raw)
        except ValueError:
            raise HTTPException(422, "Expected a JSON row") from None
        task_id = _task_id(payload)
        route = self.config.task_routes.get(task_id)
        if route is None:
            raise HTTPException(422, "Unknown benchmark task ID")
        if route.name == self.config.name:
            return JSONResponse(await self.run_legacy(payload))
        return await self._relay(route.name, payload, request)

    async def _relay(self, route_name: str, payload: dict, request: Request) -> Response:
        agent_name = self._execution_agents[route_name]
        # The routed server validates these two routing fields against its own pairing, so only they are adapted;
        # actor data, row extras and identity pass unchanged. Collection restores its own values on the result.
        if "agent_ref" in payload:
            agent_ref = payload["agent_ref"]
            if not isinstance(agent_ref, dict) or agent_ref.get("name") != self.config.agent_server.name:
                raise HTTPException(422, "Request must be addressed to this environment's agent")
            payload["agent_ref"] = {"name": agent_name}
        if "task_source" in payload:
            bound = (self.config.resources_server.name, self.config.agent_server.name, self.config.name)
            if payload["task_source"] not in bound:
                raise HTTPException(422, "Unexpected collection task_source")
            payload["task_source"] = agent_name
        if route_name in self._recorded_context_routes:
            # The stock agent excludes some collector fields when serializing /reset and /step.
            # Keep a host-owned reporting binding in ordinary extras, outside the actor parameters.
            payload["recorded_context_run_binding"] = _recorded_context_run_binding(payload)
        upstream = await self.server_client.post(
            server_name=route_name,
            url_path="/run",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers=dict(_relayed_headers(request.headers.items())),
            cookies=request.cookies,
        )
        body = await upstream.read()
        if upstream.status == 200:
            if route_name in self._recorded_context_routes:
                body = _recorded_context_result(body, payload, agent_name=agent_name)
            else:
                body = _with_reply_tool_observations(body)
        response = Response(content=body, status_code=upstream.status)
        response.raw_headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in _relayed_headers(upstream.headers.items())
        ]
        response.raw_headers.append((b"content-length", str(len(body)).encode("latin-1")))
        return response


if __name__ == "__main__":
    PartnerIFEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PartnerIFEnvironmentServer.run_webserver()  # noqa: F401
