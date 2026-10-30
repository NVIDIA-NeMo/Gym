# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route complete mixed-harness benchmark episodes to their existing environments."""

import json
from collections.abc import Iterable, Mapping

from fastapi import Body, FastAPI, HTTPException, Request, Response
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config import EnvironmentServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import NG_AGENT_OBSERVATIONS_KEY, ToolCallObservation, TrajectoryRecord
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


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

    @model_validator(mode="after")
    def validate_destinations(self) -> "PartnerIFDispatcher":
        for route in self.config.task_routes.values():
            destination = self.server_client.global_config_dict.get(route.name, {})
            environments = destination.get("environment_servers", {}) if isinstance(destination, Mapping) else {}
            if len(environments) != 1:
                raise ValueError(f"Missing or ambiguous environment destination: {route.name}")
            environment = next(iter(environments.values()))
            agent_name = environment.get("agent_server", {}).get("name")
            if not isinstance(agent_name, str) or not agent_name:
                raise ValueError(f"Environment must bind one execution agent: {route.name}")
            if agent_name == self.config.name:
                raise ValueError(f"Dispatcher cannot route back to itself through {route.name}")
            self._execution_agents[route.name] = agent_name
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
        """Preserve native request/receipt fields and downstream HTTP failures."""
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
