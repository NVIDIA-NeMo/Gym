# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixture-backed tool server: Gym owns execution, a backend owns its tools, partner_if owns scoring.

Each profile instance serves one backend (a fixed tool family over a packaged fixture),
named in config by the import path of its `ToolBackend` class. Sessions, per-task tool
isolation, call receipts and evidence assembly are shared; backends supply only tool
behavior, whether a closing answer is required, and receipts.
"""

import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, ClassVar, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import Field, JsonValue, PrivateAttr, ValidationError

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
)
from nemo_gym.mcp_auto_exposure import MCPTool
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.partner_if.app import (
    PartnerIFResourcesServer,
    PartnerIFResourcesServerConfig,
    VerifyRequest,
    VerifyResponse,
)
from resources_servers.partner_if.evidence import Artifact, EvidenceBundle
from resources_servers.partner_if.grading_contract import GradingContract, StrictModel
from resources_servers.partner_if.grading_service import GradeRequest, contract_hash
from resources_servers.partner_if.native_evidence import visible_events
from resources_servers.partner_if.package import RepoPath
from resources_servers.partner_if.receipt import scoring_fields


class ToolBackend(Protocol):
    requires_final_answer: bool

    def tools(self, row: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]: ...
    def new_state(self, row: dict[str, JsonValue]) -> Any: ...
    def call(self, state: Any, name: str, arguments: dict[str, JsonValue]) -> JsonValue: ...
    def artifacts(self, state: Any) -> dict[str, str]: ...
    def receipts(self, state: Any) -> dict[str, JsonValue]: ...


def load_backend(import_path: str) -> type[ToolBackend]:
    """Import a backend class named `module:Class`, e.g. from a benchmark's plugin root."""
    module, _, name = import_path.partition(":")
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load tool backend {import_path!r}: {error}") from error


class RecordedToolCall(StrictModel):
    name: str
    arguments: dict[str, JsonValue]
    output: JsonValue


@dataclass
class MockToolEpisode:
    seed: ResourcesSeedSessionRequest
    task: str
    tools: frozenset[str]
    state: Any
    calls: list[RecordedToolCall] = field(default_factory=list)


class MockToolsServerConfig(PartnerIFResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    backend: str = Field(pattern=r"^[\w.]+:\w+$")  # the ToolBackend class, as module:Class
    fixture_path: RepoPath
    tasks_path: RepoPath


class MockToolsVerifyResponse(VerifyResponse):
    tool_calls: list[RecordedToolCall] = []
    environment_receipts: dict[str, JsonValue] = {}


def _same_output(recorded: JsonValue, text: str) -> bool:
    """Plain-text tools return their text verbatim; JSON tools are compared as parsed values."""
    if isinstance(recorded, str):
        return recorded == text
    try:
        return json.loads(text) == recorded
    except ValueError:
        return False


def to_evidence(
    initial_params: dict[str, JsonValue],
    response: NeMoGymResponse,
    calls: list[RecordedToolCall],
    *,
    requires_final_answer: bool,
    artifacts: dict[str, str] | None = None,
) -> EvidenceBundle:
    """Join captured calls/results with authoritative receipts, retaining only visible events.

    Function call IDs come from the trusted harness, not the HTTP tool arguments. Results
    match by call ID and receipts by name/arguments/output, which permits parallel result
    ordering without guessing. Calls the harness rejects before any tool runs (unadvertised
    names, non-object arguments) are actor errors without receipts, not capture gaps.

    Without a required final answer, a response cut off by the output-token cap ends the
    episode like the step cap: its calls never ran, so the completed tool exchanges before
    it are graded. Receipts still have to match, so a cutoff cannot hide an executed call.
    """
    bundle = EvidenceBundle(
        context=initial_params["input"],
        instructions=initial_params.get("instructions"),
        tools=initial_params.get("tools") or [],
        artifacts={name: Artifact(status="present", text=text) for name, text in (artifacts or {}).items()},
    )
    advertised = {tool["name"] for tool in bundle.tools}
    try:
        items = [item.model_dump(mode="json") for item in response.output]
        if not requires_final_answer and _output_cap_cutoff(response):
            last_output = max(
                (i for i, item in enumerate(items) if item["type"] == "function_call_output"), default=-1
            )
            items = items[: last_output + 1]
        elif response.error or response.incomplete_details or response.status != "completed":
            raise ValueError("Actor trajectory is incomplete")
        bundle.trajectory = visible_events(items)
        pending, seen, remaining, last_result = {}, set(), list(calls), -1
        for event in bundle.trajectory:
            if event.kind == "tool_call":
                if not event.call_id or event.call_id in seen:
                    raise ValueError("Missing or duplicate tool call ID")
                pending[event.call_id] = event
                seen.add(event.call_id)
            elif event.kind == "tool_result":
                call = pending.pop(event.call_id, None)
                if call is None:
                    raise ValueError("Tool result has no preceding unmatched call")
                last_result = event.source_index
                match = next(
                    (
                        i
                        for i, receipt in enumerate(remaining)
                        if receipt.name == call.tool_name
                        and receipt.arguments == call.arguments
                        and _same_output(receipt.output, event.text)
                    ),
                    None,
                )
                if match is not None:
                    remaining.pop(match)
                    continue
                try:
                    local = json.loads(event.text)
                except ValueError:
                    local = None
                rejected_locally = call.tool_name not in advertised or not isinstance(call.arguments, dict)
                if (
                    not rejected_locally
                    or not isinstance(local, dict)
                    or not (local.get("error") or local.get("detail"))
                ):
                    raise ValueError("Tool output lacks a matching authoritative receipt")
        if pending or remaining:
            raise ValueError("Tool calls, outputs and receipts do not form a complete trajectory")
        bundle.answer = "\n".join(
            event.text for event in bundle.trajectory if event.kind == "assistant" and event.source_index > last_result
        )
        if requires_final_answer and not bundle.answer.strip():
            raise ValueError("Actor produced no final answer after its tool calls")
    except (ValueError, TypeError) as error:
        bundle.trajectory_error = str(error)
    return bundle


def _output_cap_cutoff(response: NeMoGymResponse) -> bool:
    details = response.incomplete_details
    return response.error is None and details is not None and details.reason == "max_output_tokens"


def _actor_input(params: dict[str, JsonValue]) -> dict[str, JsonValue]:
    normalized = NeMoGymResponseCreateParamsNonStreaming.model_validate(params).model_dump(exclude_none=True)
    return {key: normalized.get(key) for key in ("input", "instructions", "tools")}


class MockToolsResourcesServer(PartnerIFResourcesServer):
    ray_enabled = False
    config: MockToolsServerConfig
    _backend: ToolBackend = PrivateAttr()
    _tasks: dict[str, dict[str, JsonValue]] = PrivateAttr(default_factory=dict)
    _tools: dict[str, dict[str, dict[str, JsonValue]]] = PrivateAttr(default_factory=dict)
    _episodes: dict[str, MockToolEpisode] = PrivateAttr(default_factory=dict)
    _closed: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._backend = load_backend(self.config.backend)(self.config.fixture_path)
        for line in self.config.tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            contract = GradingContract.model_validate(row["verifier_metadata"]["partner_if"])
            if contract.evidence_mode != "live_agent" or contract.task_id in self._tasks:
                raise ValueError("Mock-tool tasks require unique IDs and live-agent contracts")
            # A backend may serve a whole tool family; a task's actor can call only the tools it advertises.
            advertised = {tool["name"] for tool in row["responses_create_params"].get("tools") or []}
            served = self._backend.tools(row)
            self._tools[contract.task_id] = {name: tool for name, tool in served.items() if name in advertised}
            self._tasks[contract.task_id] = row
        if not self._tasks:
            raise ValueError(f"No {self.config.backend} tasks were configured")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.router.routes = [route for route in app.router.routes if getattr(route, "path", None) != "/grade"]
        app.post("/{tool_name}")(self.call_tool)
        return app

    def mcp_tools(self, harvested: list[MCPTool], catchall: Any) -> list[MCPTool]:
        """Expose only backend tools; never lifecycle or grading routes."""
        tools = {name: tool for served in self._tools.values() for name, tool in served.items()}
        return [
            catchall.tool(name, tool.get("parameters"), tool.get("description"))
            for name, tool in sorted(tools.items())
        ]

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> list[str] | None:
        try:
            task_id = seed_body["verifier_metadata"]["partner_if"]["task_id"]
        except (KeyError, TypeError):
            return []
        return sorted(self._tools.get(task_id, {}))

    def _contract(self, contract: GradingContract) -> dict[str, JsonValue]:
        row = self._tasks.get(contract.task_id)
        if row is None or contract_hash(contract) != contract_hash(
            GradingContract.model_validate(row["verifier_metadata"]["partner_if"])
        ):
            raise HTTPException(409, "Task contract does not match the configured snapshot")
        return row

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        identifier = body.resources_session_id
        if identifier in self._closed:
            raise HTTPException(409, "Session is closed")
        try:
            contract = GradingContract.model_validate(body.task_data["verifier_metadata"]["partner_if"])
        except (KeyError, TypeError, ValidationError) as error:
            raise HTTPException(422, "Missing or invalid task contract") from error
        row = self._contract(contract)
        existing = self._episodes.get(identifier)
        if existing and existing.seed != body:
            raise HTTPException(409, "Session ID belongs to another episode/task")
        if existing is None:
            self._episodes[identifier] = MockToolEpisode(
                seed=body,
                task=contract.task_id,
                tools=frozenset(self._tools[contract.task_id]),
                state=self._backend.new_state(row),
            )
        request.session[SESSION_ID_KEY] = identifier
        return ResourcesSeedSessionResponse(resources_session_id=identifier)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        episode = self._episodes.get(body.resources_session_id)
        if episode and episode.seed.episode_id != body.episode_id:
            raise HTTPException(409, "Session belongs to another episode")
        self._episodes.pop(body.resources_session_id, None)
        self._closed.add(body.resources_session_id)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def call_tool(self, request: Request, tool_name: str, body: dict[str, JsonValue]):
        """Run one tool this rollout's task may use; anything else (e.g. the host-only /grade) is absent."""
        episode = self._episodes.get(request.session.get(SESSION_ID_KEY))
        if episode is not None and tool_name not in episode.tools:
            raise HTTPException(404, "Not a tool of this task")
        if episode is None:
            raise HTTPException(409, "Session was not seeded or is closed")
        output = self._backend.call(episode.state, tool_name, body)
        episode.calls.append(RecordedToolCall(name=tool_name, arguments=body, output=output))
        return PlainTextResponse(output) if isinstance(output, str) else JSONResponse(output)

    async def verify(self, request: Request, body: VerifyRequest) -> MockToolsVerifyResponse:
        contract = body.verifier_metadata.partner_if
        row = self._contract(contract)
        params = body.responses_create_params.model_dump(exclude_none=True)
        if _actor_input(params) != _actor_input(row["responses_create_params"]):
            raise HTTPException(409, "Actor input differs from the configured task")
        episode = self._episodes.get(request.session.get(SESSION_ID_KEY))
        calls = episode.calls if episode else []
        bundle = to_evidence(
            params,
            body.response,
            calls,
            requires_final_answer=self._backend.requires_final_answer,
            artifacts=self._backend.artifacts(episode.state) if episode else None,
        )
        if episode is None or episode.task != contract.task_id:
            bundle.trajectory_error = "Missing or mismatched resource session"
        receipt = await self.grade_evidence(GradeRequest(contract=contract, evidence=bundle))
        return MockToolsVerifyResponse(
            **body.model_dump(),
            **scoring_fields(contract, receipt),
            execution_identity=self.config.execution_identity,
            tool_calls=calls,
            environment_receipts=self._backend.receipts(episode.state) if episode else {},
        )


if __name__ == "__main__":
    MockToolsResourcesServer.run_webserver()
