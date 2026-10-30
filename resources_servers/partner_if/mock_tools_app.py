# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixture-backed tool server: Gym owns execution, a backend owns its tools, partner_if owns scoring.

A backend is a fixed tool family over a packaged fixture (`tool_backend.ToolBackend`). In-process, a server
instance serves one backend, named in config by the import path of its class; `example_tools.NotesBackend` is a
small example. Sandboxed, each task row brings its own backend as a `tool_backend` bundle, which runs in a fresh
sandbox for each rollout (tool_sandbox.py), so one server can serve tasks with different backends. Sessions,
per-task tool isolation and call receipts are shared; mock_tools_evidence.py turns an episode into evidence.
Backends supply only tool behavior, whether a closing answer is required, and receipts.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Protocol
from unittest.mock import MagicMock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import Field, JsonValue, PrivateAttr, ValidationError, model_validator

from nemo_gym.base_resources_server import (
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.mcp_auto_exposure import MCPTool
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.partner_if.app import (
    NoAnswer,
    PartnerIFGradingConfig,
    PartnerIFGradingMixin,
    VerifyRequest,
    VerifyResponse,
    output_cap_cutoff,
)
from resources_servers.partner_if.checks import resolve
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import NO_ANSWER, TRUNCATED, GradingContract
from resources_servers.partner_if.grading_service import contract_hash
from resources_servers.partner_if.metrics import PartnerIFMetricsMixin
from resources_servers.partner_if.mock_tools_evidence import RecordedToolCall, to_evidence
from resources_servers.partner_if.package import RepoPath
from resources_servers.partner_if.receipt import masked_fields
from resources_servers.partner_if.task_data import ToolBackendBundle
from resources_servers.partner_if.tool_backend import ToolBackend, ToolCallError
from resources_servers.partner_if.tool_sandbox import ToolSandboxConfig, ToolSandboxes, UnservedToolsError


logger = logging.getLogger(__name__)

# A tool call the backend could not run: the environment failed, not the model.
TOOL_BACKEND_FAILED = "partner_if:tool_backend_failed"


def load_backend(import_path: str) -> type[ToolBackend]:
    """Import a backend class named `module:Class`, e.g. from a benchmark's plugin root."""
    module, _, name = import_path.partition(":")
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load tool backend {import_path!r}: {error}") from error


class EpisodeBackend(Protocol):
    """One episode's tools as the server calls them: an in-process backend with the episode's state, or the
    task's own backend in the episode's sandbox (tool_sandbox.SandboxEpisode)."""

    name: str

    @property
    def requires_final_answer(self) -> bool: ...

    @property
    def sandbox_id(self) -> str | None: ...

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> JsonValue: ...

    async def artifacts(self) -> dict[str, str]: ...

    async def receipts(self) -> dict[str, JsonValue]: ...

    async def stop(self) -> None: ...


@dataclass
class InProcessEpisode:
    """An in-process backend's synchronous protocol behind the server's async calls."""

    name: str
    backend: ToolBackend
    state: Any

    @property
    def requires_final_answer(self) -> bool:
        return self.backend.requires_final_answer

    @property
    def sandbox_id(self) -> None:
        return None

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        return self.backend.call(self.state, name, arguments)

    async def artifacts(self) -> dict[str, str]:
        return self.backend.artifacts(self.state)

    async def receipts(self) -> dict[str, JsonValue]:
        return self.backend.receipts(self.state)

    async def stop(self) -> None:
        pass


@dataclass
class MockToolEpisode:
    seed: ResourcesSeedSessionRequest
    task: str
    tools: frozenset[str]
    backend: EpisodeBackend
    calls: list[RecordedToolCall] = field(default_factory=list)
    # Why the backend failed to run a call, if it did; verify then masks the rollout.
    backend_failure: str | None = None


class MockToolsServerConfig(PartnerIFGradingConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    # In-process: the ToolBackend class, as module:Class, serves every task over fixture_path.
    backend: str | None = Field(default=None, pattern=r"^[\w.]+:\w+$")
    fixture_path: RepoPath | None = None
    # Sandboxed: each task's own `tool_backend` runs in a fresh sandbox per rollout, from this sandbox provider.
    sandbox_provider: str | None = None
    sandbox_config: ToolSandboxConfig = Field(default_factory=ToolSandboxConfig)
    tasks_path: RepoPath

    @model_validator(mode="after")
    def one_backend_mode(self):
        if (self.backend is None) == (self.sandbox_provider is None):
            raise ValueError("Set either backend, for an in-process backend, or sandbox_provider, for sandboxed ones")
        if (self.backend is None) != (self.fixture_path is None):
            raise ValueError("fixture_path goes with an in-process backend; a sandboxed task's bundle has its own")
        return self


class MockToolsVerifyResponse(VerifyResponse):
    tool_calls: list[RecordedToolCall] = []
    environment_receipts: dict[str, JsonValue] = {}
    # The sandbox that ran a sandboxed task's backend, to trace a rollout in the provider's records.
    tool_sandbox_id: str | None = None


def _actor_input(params: dict[str, JsonValue]) -> dict[str, JsonValue]:
    normalized = NeMoGymResponseCreateParamsNonStreaming.model_validate(params).model_dump(exclude_none=True)
    return {key: normalized.get(key) for key in ("input", "instructions", "tools")}


class MockToolsResourcesServer(PartnerIFGradingMixin, PartnerIFMetricsMixin, SimpleResourcesServer):
    # Episodes live in this process and are not exported, so a checkpoint restarts their rollouts.
    checkpoint_mode = "restart_only"
    checkpoint_verify = "wait"
    # Episodes live in worker memory, so the server runs one worker (see model_post_init) and no Ray.
    ray_enabled = False

    config: MockToolsServerConfig
    _backend: ToolBackend | None = PrivateAttr(default=None)
    _sandboxes: ToolSandboxes | None = PrivateAttr(default=None)
    _bundles: dict[str, ToolBackendBundle] = PrivateAttr(default_factory=dict)
    _tasks: dict[str, dict[str, JsonValue]] = PrivateAttr(default_factory=dict)
    _tools: dict[str, dict[str, dict[str, JsonValue]]] = PrivateAttr(default_factory=dict)
    _episodes: dict[str, MockToolEpisode] = PrivateAttr(default_factory=dict)
    # Seed, calls, verify, close and retire hold the session's lock. A sandbox start awaits, and a close or a
    # repeated seed that ran meanwhile would leak its sandbox or start a second one.
    _session_locks: dict[str | None, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _closed: ClosedResourcesSessionRecords[EpisodeId | None] = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Mock-tool episodes live in the server's memory, so it requires num_workers=1")
        self._closed = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)
        if self.config.backend is not None:
            self._backend = load_backend(self.config.backend)(self.config.fixture_path)
        else:
            self._sandboxes = ToolSandboxes(
                self.config.sandbox_provider, self.config.sandbox_config, server_name=self.config.name
            )
        for line in self.config.tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            contract = GradingContract.model_validate(row["verifier_metadata"]["partner_if"])
            if contract.evidence_mode != "live_agent" or contract.task_id in self._tasks:
                raise ValueError("Mock-tool tasks require unique IDs and live-agent contracts")
            for item in contract.checks:
                resolve(item.verifier)  # An unknown verifier or invalid params fails here, not on every rollout.
            # A backend may serve a whole tool family; a task's actor can call only the tools it advertises.
            advertised = {tool["name"]: tool for tool in row["responses_create_params"].get("tools") or []}
            if self._backend is not None:
                served = self._backend.tools(row)
                if unserved := advertised.keys() - set(served):
                    raise ValueError(
                        f"Task {contract.task_id} advertises tools the backend does not serve: {', '.join(sorted(unserved))}"
                    )
                self._tools[contract.task_id] = {name: tool for name, tool in served.items() if name in advertised}
            else:
                # Only a sandbox can run the task's own backend, so each seed checks the tools it serves.
                if row.get("tool_backend") is None:
                    raise ValueError(f"Task {contract.task_id} has no tool_backend for the sandboxed server to run")
                self._bundles[contract.task_id] = ToolBackendBundle.model_validate(row["tool_backend"])
                self._tools[contract.task_id] = advertised
            self._tasks[contract.task_id] = row
        if not self._tasks:
            raise ValueError(f"No mock-tool tasks were configured in {self.config.tasks_path}")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{tool_name}")(self.call_tool)
        lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def stop_episodes_at_shutdown(application: FastAPI):
            async with lifespan(application) as state:
                try:
                    yield state
                finally:
                    await self._stop_all()

        app.router.lifespan_context = stop_episodes_at_shutdown
        return app

    def mcp_tools(self, harvested: list[MCPTool], catchall: Any) -> list[MCPTool]:
        """Expose only backend tools; never lifecycle or grading routes."""
        tools = {}
        for task_id, served in self._tools.items():
            for name, tool in served.items():
                exposed = tool.get("parameters"), tool.get("description")
                if tools.setdefault(name, exposed) != exposed:
                    raise ValueError(f"Task {task_id} gives tool {name} another schema, but MCP exposes one per name")
        return [
            catchall.tool(name, parameters, description) for name, (parameters, description) in sorted(tools.items())
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

    def _session_lock(self, session_id: str | None) -> asyncio.Lock:
        return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        identifier = body.resources_session_id
        try:
            contract = GradingContract.model_validate(body.task_data["verifier_metadata"]["partner_if"])
        except (KeyError, TypeError, ValidationError) as error:
            raise HTTPException(422, "Missing or invalid task contract") from error
        row = self._contract(contract)
        if (
            self._sandboxes is not None
            and body.task_data.get("tool_backend", row["tool_backend"]) != row["tool_backend"]
        ):
            raise HTTPException(409, "Task tool backend does not match the configured snapshot")
        async with self._session_lock(identifier):
            if identifier in self._closed:
                raise HTTPException(409, "Session is closed")
            episode = self._episodes.get(identifier)
            if episode is None:
                # A start that fails or is cancelled stops its own sandbox, so a failed seed leaves nothing behind.
                backend = await self._start_backend(contract.task_id, row)
                self._episodes[identifier] = MockToolEpisode(
                    seed=body, task=contract.task_id, tools=frozenset(self._tools[contract.task_id]), backend=backend
                )
            elif episode.seed != body:
                raise HTTPException(409, "Session ID belongs to another episode/task")
        request.session[SESSION_ID_KEY] = identifier
        return ResourcesSeedSessionResponse(resources_session_id=identifier)

    async def _start_backend(self, task_id: str, row: dict[str, JsonValue]) -> EpisodeBackend:
        if self._backend is not None:
            return InProcessEpisode(self.config.backend, self._backend, self._backend.new_state(row))
        try:
            return await self._sandboxes.start(task_id, self._bundles[task_id], row, set(self._tools[task_id]))
        except UnservedToolsError as error:
            raise HTTPException(422, str(error)) from error

    @staticmethod
    async def _stop_quietly(episode: MockToolEpisode) -> None:
        try:
            await episode.backend.stop()
        except Exception:
            logger.exception("Could not stop the tool backend of task %s; its sandbox's ttl_s will", episode.task)

    async def _stop_all(self) -> None:
        episodes, self._episodes = list(self._episodes.values()), {}
        await asyncio.gather(*(self._stop_quietly(episode) for episode in episodes))

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        identifier = body.resources_session_id
        async with self._session_lock(identifier):
            closed_episode = self._closed.get(identifier)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise HTTPException(409, "Session belongs to another episode")
            episode = self._episodes.get(identifier)
            if episode and episode.seed.episode_id != body.episode_id:
                raise HTTPException(409, "Session belongs to another episode")
            if episode is not None:
                # A backend that fails to stop raises and keeps its episode, so a retried close stops it again.
                await episode.backend.stop()
                del self._episodes[identifier]
            self._closed.remember(identifier, body.episode_id)
        # Later requests for a closed session are refused, so it needs no lock; any waiting on this one see the close.
        self._session_locks.pop(identifier, None)
        return ResourcesCloseSessionResponse(resources_session_id=identifier)

    async def retire_session_state(self, session_id: str) -> None:
        async with self._session_lock(session_id):
            episode = self._episodes.pop(session_id, None)
            self._closed.remember(session_id, episode.seed.episode_id if episode is not None else None)
            if episode is not None:
                await self._stop_quietly(episode)
        self._session_locks.pop(session_id, None)

    async def call_tool(self, request: Request, tool_name: str, body: dict[str, JsonValue]):
        """Run one tool this rollout's task may use; any other name is absent."""
        identifier = request.session.get(SESSION_ID_KEY)
        async with self._session_lock(identifier):
            episode = self._episodes.get(identifier)
            if episode is not None and tool_name not in episode.tools:
                raise HTTPException(404, "Not a tool of this task")
            if episode is None:
                raise HTTPException(409, "Session was not seeded or is closed")
            try:
                output = await episode.backend.call(tool_name, body)
            except ToolCallError as error:
                output = {"error": str(error)}
            except Exception:
                # Unrecorded, so the call has no receipt, and verify masks the rollout as the backend's failure.
                # The details stay in this log: the model sees the response body, and Gym's exception middleware
                # would put the exception in it.
                logger.exception("Tool backend %s failed in %s", episode.backend.name, tool_name)
                episode.backend_failure = episode.backend_failure or f"The tool backend failed in {tool_name}"
                return PlainTextResponse("Internal Server Error", status_code=500)
            episode.calls.append(RecordedToolCall(name=tool_name, arguments=body, output=output))
        return PlainTextResponse(output) if isinstance(output, str) else JSONResponse(output)

    async def verify(self, request: Request, body: VerifyRequest) -> MockToolsVerifyResponse:
        contract = body.verifier_metadata.partner_if
        row = self._contract(contract)
        params = body.responses_create_params.model_dump(exclude_none=True)
        if _actor_input(params) != _actor_input(row["responses_create_params"]):
            raise HTTPException(409, "Actor input differs from the configured task")
        identifier = request.session.get(SESSION_ID_KEY)
        async with self._session_lock(identifier):
            episode = self._episodes.get(identifier)
            if episode is not None and episode.task != contract.task_id:
                episode = None
            calls = list(episode.calls) if episode else []
            failure = episode.backend_failure if episode else None
            # A failed backend may be gone; its rollout is masked without its artifacts.
            artifacts, receipts = None, {}
            if episode and not failure:
                try:
                    artifacts = await episode.backend.artifacts()
                    receipts = await episode.backend.receipts()
                except Exception:
                    # A backend or sandbox lost after the last call is the environment's failure too.
                    logger.exception("Tool backend %s failed before verify", episode.backend.name)
                    artifacts, receipts = None, {}
                    failure = "The tool backend failed before verify"
        recorded = {
            "tool_calls": calls,
            "environment_receipts": receipts,
            "tool_sandbox_id": episode.backend.sandbox_id if episode else None,
        }
        if failure is not None:
            return MockToolsVerifyResponse(
                **body.model_dump(), **masked_fields(contract, TOOL_BACKEND_FAILED, failure), **recorded
            )

        def evidence() -> EvidenceBundle:
            bundle = to_evidence(params, body.response, calls, artifacts=artifacts)
            if episode is None:
                bundle.trajectory_error = "Missing or mismatched resource session"
            if bundle.trajectory_error is None and episode.backend.requires_final_answer and not bundle.answer.strip():
                if output_cap_cutoff(body.response):
                    reason = "The episode was cut off by the output-token cap before its final answer"
                    raise NoAnswer(TRUNCATED, reason)
                raise NoAnswer(NO_ANSWER, "The episode ended without a final answer")
            return bundle

        return MockToolsVerifyResponse(
            **body.model_dump(), **await self.grade_response(contract, evidence), **recorded
        )


def _verifier_fixture_server() -> MockToolsResourcesServer:
    """The fixture's server: the example NotesBackend in-process over the example tasks, graded by code checks only,
    so no judge is called. Its cases raise no ToolCallError: the fixture loader imports this module apart from the
    copy of tool_backend.py that the backend, imported by name, uses."""
    data = Path(__file__).parent / "data"
    config = MockToolsServerConfig(
        name="partner_if_mock_tools",
        host="127.0.0.1",
        port=0,
        entrypoint="mock_tools_app.py",
        backend="resources_servers.partner_if.example_tools:NotesBackend",
        fixture_path=data / "mock_tools_notes.json",
        tasks_path=data / "mock_tools_example.jsonl",
        judge_model_server={"type": "responses_api_models", "name": "partner_if_judge"},
    )
    return MockToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def replay_and_verify(server: MockToolsResourcesServer, body: VerifyRequest) -> MockToolsVerifyResponse:
    """Verify a recorded episode as a live one is verified: seed its task, run its tool calls through the backend,
    which records their receipts, and grade it. A recorded output the backend does not reproduce is masked."""
    request = SimpleNamespace(session={})  # The handlers read only the request's session.
    await server.seed_session(
        request,
        ResourcesSeedSessionRequest(
            resources_session_id="verifier-fixture",
            episode_id={"rollout_id": "verifier-fixture", "attempt": 0},
            task_id={"taskset": "verifier-fixture", "task_id": body.verifier_metadata.partner_if.task_id},
            task_data={"verifier_metadata": body.verifier_metadata.model_dump(mode="json")},
        ),
    )
    for item in body.response.output:
        if item.type == "function_call":
            await server.call_tool(request, item.name, json.loads(item.arguments))
    return await server.verify(request, body)


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_verifier_fixture_server,
    request_model=VerifyRequest,
    cases_path=Path(__file__).parent / "tests/mock_tools_verifier_cases.jsonl",
    invoke=replay_and_verify,
)


if __name__ == "__main__":
    MockToolsResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MockToolsResourcesServer.run_webserver()
