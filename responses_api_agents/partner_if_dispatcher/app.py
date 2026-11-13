# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route complete mixed-harness benchmark episodes to their existing environments."""

import json
from collections.abc import Iterable, Mapping

from fastapi import Body, FastAPI, HTTPException, Request, Response
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import (
    ATTEMPT_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    AggregateMetrics,
    AggregateMetricsRequest,
    EnvironmentServerRef,
    ResourcesServerRef,
)
from nemo_gym.core import (
    ENVIRONMENT_SERVER_FAILED,
    EpisodeId,
    FailureReportRow,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    maybe_rollout_id_from_run_body,
    model_sampling_from_run_body,
    skips_token_capture,
)
from nemo_gym.data import materialize_task
from nemo_gym.server_runtime import SingleAgentTurnRequest, SingleAgentTurnResponse
from nemo_gym.trajectory import (
    NG_AGENT_OBSERVATIONS_KEY,
    NG_TRAJECTORY_KEY,
    NG_VERIFIER_INVOCATIONS_KEY,
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


def _flat_episode_result(response: SingleAgentTurnResponse, *, agent_name: str) -> dict:
    """Keep the collector's flat result and failure contract at the typed episode boundary."""
    if response.failure is not None:
        failed = response.failure
        result = {
            "_ng_failure_class": failed.failure_kind or ENVIRONMENT_SERVER_FAILED,
            "_ng_failure_kind": failed.failure_kind,
            "_ng_failure_terminal": failed.terminal,
            "_ng_failure_message": failed.failure_reason,
            "_ng_failure_stage": failed.stage,
        }
        for name, evidence in (
            ("_ng_failure_partial_response", failed.partial_response),
            (NG_AGENT_OBSERVATIONS_KEY, failed.agent_observations),
            (NG_TRAJECTORY_KEY, failed.trajectory),
        ):
            if evidence is not None:
                result[name] = evidence.model_dump(mode="json")
        if failed.verifier_invocations is not None:
            result[NG_VERIFIER_INVOCATIONS_KEY] = [
                item.model_dump(mode="json") for item in failed.verifier_invocations
            ]
    else:
        assert response.result is not None  # The episode response requires exactly one result or failure.
        result = response.result.model_dump(mode="json")
        if result.get(NG_AGENT_OBSERVATIONS_KEY) is None:
            result.pop(NG_AGENT_OBSERVATIONS_KEY, None)
    result["agent_ref"] = {"name": agent_name}
    if response.failure_report is not None:
        FailureReportRow.merged_into(result, response.failure_report)
    return result


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
    _episode_resources: dict[str, str] = PrivateAttr(default_factory=dict)

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
            if implementation == "single_agent_turn":
                resources_name = environment.get("resources_server", {}).get("name")
                if not isinstance(resources_name, str) or not resources_name:
                    raise ValueError(f"Single-agent environment must bind one resources server: {route.name}")
                self._episode_resources[route.name] = resources_name
        return self

    def _episode_request_from_row(self, row: dict, *, route_name: str) -> SingleAgentTurnRequest:
        """Materialize the host-routed row without changing its actor input or capture identity."""
        attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ValueError("Episode attempt must be a non-negative integer")
        capture_key = maybe_rollout_id_from_run_body(row)
        if capture_key is None:
            raise ValueError("An episode requires a rollout ID or task and rollout indices")
        return SingleAgentTurnRequest(
            episode_id=EpisodeId.from_capture_key(capture_key),
            task=materialize_task(row, taskset=row.get(TASK_SOURCE_KEY_NAME, self._episode_resources[route_name])),
            model_sampling=model_sampling_from_run_body(row),
            skip_token_capture=skips_token_capture(row),
        )

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
        episode = None
        if route.name in self._episode_resources:
            try:
                episode = self._episode_request_from_row(payload, route_name=route.name)
            except (ValueError, TypeError) as error:
                raise HTTPException(422, str(error)) from None
            # Preserve omitted actor defaults through nested model serialization.
            raw = episode.model_dump_json(exclude_unset=True).encode()
        else:
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
            if episode is not None:
                try:
                    reply = SingleAgentTurnResponse.model_validate_json(body)
                    if reply.episode_id != episode.episode_id or reply.task_id != episode.task.task_id:
                        raise ValueError("Episode reply identity does not match its request")
                except ValueError:
                    raise HTTPException(502, "Invalid single-agent episode response") from None
                body = json.dumps(
                    _flat_episode_result(reply, agent_name=self._execution_agents[route.name]), ensure_ascii=False
                ).encode()
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
