# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route complete mixed-harness benchmark episodes to their existing environments."""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping

from fastapi import Body, FastAPI, HTTPException, Request, Response
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, EnvironmentServerRef, ResourcesServerRef
from nemo_gym.core import FailureReportRow, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.trajectory import (
    NG_AGENT_OBSERVATIONS_KEY,
    ToolCallObservation,
    TrajectoryRecord,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


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


class PartnerIFDispatcherConfig(BaseResponsesAPIAgentConfig):
    """Host-selected task destinations; request payloads cannot override these."""

    task_routes: dict[str, EnvironmentServerRef] = Field(min_length=1)
    # Gym aggregates this dispatcher's rows here, so the benchmark scorer computes the metrics.
    resources_server: ResourcesServerRef

    @model_validator(mode="after")
    def validate_routes(self) -> "PartnerIFDispatcherConfig":
        if any(not task.strip() for task in self.task_routes):
            raise ValueError("Task IDs must not be empty")
        if any(route.name == self.name for route in self.task_routes.values()):
            raise ValueError("Dispatcher cannot route to itself")
        return self


class PartnerIFDispatcher(SimpleResponsesAPIAgent):
    """Dispatch only; existing environments own sessions, execution and grading."""

    config: PartnerIFDispatcherConfig
    _execution_agents: dict[str, str] = PrivateAttr(default_factory=dict)
    _recorded_context_routes: set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="after")
    def validate_destinations(self) -> "PartnerIFDispatcher":
        for route in self.config.task_routes.values():
            destination = self.server_client.global_config_dict.get(route.name, {})
            environments = destination.get("environment_servers", {}) if isinstance(destination, Mapping) else {}
            if len(environments) != 1:
                raise ValueError(f"Missing or ambiguous environment destination: {route.name}")
            implementation, environment = next(iter(environments.items()))
            agent_name = environment.get("agent_server", {}).get("name")
            if not isinstance(agent_name, str) or not agent_name:
                raise ValueError(f"Environment must bind one execution agent: {route.name}")
            if agent_name == self.config.name:
                raise ValueError(f"Dispatcher cannot route back to itself through {route.name}")
            self._execution_agents[route.name] = agent_name
            agent = self.server_client.global_config_dict.get(agent_name, {})
            implementations = agent.get("responses_api_agents", {}) if isinstance(agent, Mapping) else {}
            gymnasium = implementations.get("gymnasium_agent", {})
            if implementation == "legacy_agent" and isinstance(gymnasium, Mapping):
                resources_name = gymnasium.get("resources_server", {}).get("name")
                resources = self.server_client.global_config_dict.get(resources_name, {}) if resources_name else {}
                block = (
                    resources.get("resources_servers", {}).get("partner_if", {})
                    if isinstance(resources, Mapping)
                    else {}
                )
                if block.get("entrypoint") == "recorded_context_app.py":
                    self._recorded_context_routes.add(route.name)
        return self

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/run")(self.run)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("Use /run for complete benchmark episodes")

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server that owns the benchmark metrics."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))

    async def run(self, request: Request) -> Response:
        """Preserve row fields and downstream HTTP failures."""
        raw = await request.body()
        try:
            payload = json.loads(raw)
            task_id = payload["verifier_metadata"]["partner_if"]["task_id"]
        except (ValueError, KeyError, TypeError):
            raise HTTPException(422, "Expected verifier_metadata.partner_if.task_id") from None
        if not isinstance(task_id, str) or task_id not in self.config.task_routes:
            raise HTTPException(422, "Unknown benchmark task ID")
        route = self.config.task_routes[task_id]
        # The native session environment validates these two host-routing fields.
        # Only adapt the address; preserve all actor data, native extras and IDs.
        # Gym's collector restores its own agent_ref/task_source on returned rows.
        if "agent_ref" in payload:
            agent_ref = payload["agent_ref"]
            if not isinstance(agent_ref, dict) or agent_ref.get("name") != self.config.name:
                raise HTTPException(422, "Request must be addressed to this dispatcher")
            payload["agent_ref"] = {"name": self._execution_agents[route.name]}
        if "task_source" in payload:
            if payload["task_source"] != self.config.name:
                raise HTTPException(422, "Unexpected collection task_source")
            payload["task_source"] = self._execution_agents[route.name]
        if route.name in self._recorded_context_routes:
            # The stock agent excludes some collector fields when serializing /reset and /step.
            # Keep a host-owned reporting binding in ordinary extras, outside the actor parameters.
            payload["recorded_context_run_binding"] = _recorded_context_run_binding(payload)
        raw = json.dumps(payload, ensure_ascii=False).encode()
        upstream = await self.server_client.post(
            server_name=route.name,
            url_path="/run",
            data=raw,
            headers=dict(_relayed_headers(request.headers.items())),
            cookies=request.cookies,
        )
        body = await upstream.read()
        if upstream.status == 200:
            if route.name in self._recorded_context_routes:
                body = _recorded_context_result(body, payload, agent_name=self._execution_agents[route.name])
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
    PartnerIFDispatcher.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PartnerIFDispatcher.run_webserver()
