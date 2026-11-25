# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attempt sandbox: what the Resources Server declares at seed, and how the base Environment Server plans,
starts, lends and stops each attempt's sandbox group."""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from fastapi import Body
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId, MaterializedTask, TaskId
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.providers.base import (
    SandboxCapabilities,
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSidecarSpec,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.server_runtime.attempt_sandbox import (
    PREPARE_SANDBOX_PATH,
    ResourcesPrepareSandboxRequest,
    SandboxGroupSidecar,
    SandboxGroupSpec,
)
from nemo_gym.server_runtime.base_environment_server import (
    ATTEMPT_SANDBOX_BLOCKS,
    AttemptSandboxEnvironmentServerConfig,
    BaseEnvironmentServer,
    CleanupContext,
)
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime.owned_sandboxes import OwnedSandboxes


SEED = {
    "resources_session_id": "resources-session",
    "episode_id": {"rollout_id": "rollout", "attempt": 0},
    "task_id": {"taskset": "tasks", "task_id": "task"},
    "task_data": {"question": "2+2"},
}
ACCESS = {
    "connection": {"kind": "direct", "provider_config_ref": "attempt_sandboxes", "descriptor": {"sandbox_id": "sb"}},
    "workdir": "/workspace",
}
PREPARE = {
    "resources_session_id": "resources-session",
    "episode_id": SEED["episode_id"],
    "task_id": SEED["task_id"],
    "task_data": SEED["task_data"],
    "sandbox_access": ACCESS,
}


# --- What the Resources Server declares at seed ---------------------------------------------------------------------


class _Verifier(SimpleResourcesServer):
    async def verify(self, body: Any) -> Any:
        raise NotImplementedError


class _CodeVerifier(_Verifier):
    def sandbox_group(self, body: ResourcesSeedSessionRequest) -> SandboxGroupSpec | None:
        return SandboxGroupSpec(image="python:3.13-slim", files={"tests/test_task.py": body.task_data["question"]})


class _AnswerReader(_Verifier):
    verifier_executes_nothing = True


class _OverriddenSeed(_AnswerReader):
    """A server that builds its own seed reply and forgets the sandbox declaration."""

    async def seed_session(self, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)


class _Preparer(_Verifier):
    prepared: ClassVar[list[ResourcesPrepareSandboxRequest]] = []

    async def prepare_sandbox(self, body: ResourcesPrepareSandboxRequest) -> None:
        self.prepared.append(body)


def _resources(cls: type[SimpleResourcesServer]) -> TestClient:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="resources")
    return TestClient(cls(config=config, server_client=MagicMock(spec=ServerClient)).setup_webserver())


def test_a_resources_server_that_declares_nothing_keeps_the_seed_reply_and_serves_no_prepare() -> None:
    client = _resources(_Verifier)

    reply = client.post("/seed_session", json=SEED)

    # The reply keeps its wire shape: no sandbox keys, so an older Environment Server still validates it.
    assert reply.json() == {
        "resources_session_id": "resources-session",
        "resources_tools": None,
        "sandbox_access": None,
    }
    assert client.post(PREPARE_SANDBOX_PATH, json=PREPARE).status_code == 404


def test_a_server_declares_the_group_its_hook_describes_and_no_verifier_opt_out() -> None:
    reply = _resources(_CodeVerifier).post("/seed_session", json=SEED).json()

    group = SandboxGroupSpec.model_validate(reply["sandbox_group"])
    assert (group.image, group.files, group.prepare) == ("python:3.13-slim", {"tests/test_task.py": "2+2"}, False)
    # Not declared: an Environment Server keeps this verifier in a sandbox.
    assert "verifier_executes_nothing" not in reply


def test_a_verifier_that_executes_nothing_is_declared_on_the_seed_reply_without_a_group() -> None:
    reply = _resources(_AnswerReader).post("/seed_session", json=SEED).json()

    assert reply["verifier_executes_nothing"] is True
    assert "sandbox_group" not in reply
    assert ResourcesSeedSessionResponse.model_validate(reply).verifier_executes_nothing is True


def test_a_seed_override_that_drops_the_declaration_declares_nothing() -> None:
    reply = _resources(_OverriddenSeed).post("/seed_session", json=SEED).json()

    # Fails closed: the verifier of a server whose seed reply drops the declaration cannot opt out.
    assert ResourcesSeedSessionResponse.model_validate(reply).verifier_executes_nothing is False


def test_a_server_that_prepares_the_group_declares_it_and_serves_the_prepare_route() -> None:
    _Preparer.prepared.clear()
    client = _resources(_Preparer)

    seed = client.post("/seed_session", json=SEED).json()
    prepared = client.post(PREPARE_SANDBOX_PATH, json=PREPARE)

    # Only prepare is declared; the group itself comes from the Environment Server's sandbox_config.
    assert seed["sandbox_group"] == SandboxGroupSpec(prepare=True).model_dump(mode="json")
    assert prepared.status_code == 200
    assert prepared.json() == {"resources_session_id": "resources-session"}
    # It knows the task it prepares without session state of its own.
    assert [(body.task_data, body.sandbox_access.workdir) for body in _Preparer.prepared] == [
        ({"question": "2+2"}, "/workspace")
    ]


def test_a_seed_reply_lends_a_sandbox_or_declares_one_but_not_both() -> None:
    with pytest.raises(ValidationError, match="not both"):
        ResourcesSeedSessionResponse(
            resources_session_id="s",
            sandbox_access=SandboxAccess.model_validate(ACCESS),
            sandbox_group=SandboxGroupSpec(),
        )


def test_a_group_refuses_two_sidecars_of_one_name_and_starts_each_from_its_own_spec() -> None:
    sidecar = SandboxGroupSidecar(name="weather", image="weather:1", env={"PORT": "8080"}, resources={"cpu": 1})

    with pytest.raises(ValidationError, match="unique"):
        SandboxGroupSpec(sidecars=[sidecar, sidecar])
    with pytest.raises(ValidationError):
        SandboxGroupSidecar(name="weather", image="weather:1", resources={"gpus": 1})
    assert sidecar.spec() == SandboxSidecarSpec(
        name="weather", image="weather:1", env={"PORT": "8080"}, resources={"cpu": 1}
    )


# --- The base Environment Server's lifecycle -------------------------------------------------------------------------


class _Provider:
    """A provider that lends sandboxes to other processes and applies the model-only rule."""

    name = "fake"
    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "model_only"}))

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.fail_create: Exception | None = None

    def network_options(self, rule: str, hosts: Sequence[str], provider_options: Mapping[str, Any]) -> dict[str, Any]:
        return {**provider_options, "network_policy": {"allow": list(hosts)}}

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if self.fail_create is not None:
            raise self.fail_create
        sandbox_id = f"sb-{len(self.boxes) + 1}"
        self.boxes[sandbox_id] = {"spec": spec, "state": "running"}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Any, target_path: str) -> None:
        pass

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Any) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING if self.boxes[handle.sandbox_id]["state"] == "running" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id, "scope": scope}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)

    def running(self) -> list[str]:
        return [sandbox_id for sandbox_id, box in self.boxes.items() if box["state"] == "running"]


class _SidecarProvider(_Provider):
    """A provider that also starts the sidecars a spec declares."""

    async def sidecar_exec(
        self, handle: SandboxHandle, sidecar: str, argv: list[str], **kwargs: Any
    ) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def download_sidecar_file(
        self, handle: SandboxHandle, sidecar: str, source_path: str, target_path: Path
    ) -> None:
        pass


class _TaskInput(BaseModel):
    value: str = "task"


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[str]):
    pass


class _Server(BaseEnvironmentServer[_Request, _Response]):
    """An Environment Server type other than single_agent_turn: its agent turn is a wait while it holds the group."""

    request_model = _Request
    response_model = _Response
    lends_sandboxes = True
    owned_sandbox_blocks = ATTEMPT_SANDBOX_BLOCKS
    turn: ClassVar[asyncio.Event | None] = None

    async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
        key = f"attempt/{request.episode_id.capture_key}"
        cleanup.register_cleanup("attempt sandbox", lambda: self.stop_attempt_sandbox(key))
        attempt, access = await self.seed_attempt_sandbox(request, key=key)
        if self.turn is not None:
            await self.turn.wait()
        lent = await self.verifier_sandbox_access(request, attempt)
        await self.release_attempt_sandbox(key)
        return _Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=f"{access.connection.descriptor['sandbox_id']} {lent.connection.descriptor['sandbox_id']}",
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


SANDBOX_CONFIG = {
    "image": "python:3.13-slim",
    "workdir": "/workspace",
    "env": {"FROM_CONFIG": "1", "SHARED": "config"},
    "ttl_s": 3600,
    "network": "inherit",
}


def _server(
    provider: _Provider | None = None, *, configured: bool = True, server_type: type[_Server] = _Server, **config: Any
) -> _Server:
    settings = AttemptSandboxEnvironmentServerConfig.model_validate(
        {"name": "environment", "host": "", "port": 0, "entrypoint": "", "cleanup_timeout_seconds": 10}
        | ({"sandbox_provider": "attempt_sandboxes", "sandbox_config": SANDBOX_CONFIG} if configured else {})
        | config
    )
    server = server_type(config=settings, server_client=MagicMock(spec=ServerClient))
    if provider is not None:
        server._owned_sandboxes["sandbox_provider"] = OwnedSandboxes(
            provider, state_dir=None, server_name="environment", provider_config_ref="attempt_sandboxes"
        )
    return server


def _request() -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task=MaterializedTask(task_id=TaskId(taskset="tasks", task_id="task"), task_input=_TaskInput()),
    )


def _refusal(error: pytest.ExceptionInfo[NgFailureEvent]) -> tuple[str, bool]:
    return error.value.stage, error.value.episode_terminal


@pytest.mark.parametrize(
    "opt_out, lends",
    [
        ({}, (True, True)),
        ({"task": True}, (False, True)),
        ({"verifier": True}, (True, False)),
    ],
)
def test_the_agent_and_the_verifier_each_borrow_the_group_unless_they_opt_out(
    opt_out: dict[str, bool], lends: tuple[bool, bool]
) -> None:
    attempt = _server(sandbox_opt_out=opt_out).plan_attempt_sandbox(key="attempt/a", verifier_executes_nothing=True)

    assert attempt is not None
    assert (attempt.key, (attempt.lend_to_agent, attempt.lend_to_verifier)) == ("attempt/a", lends)
    assert (attempt.image, attempt.workdir) == ("python:3.13-slim", "/workspace")


def test_nothing_starts_when_both_opt_out_or_the_server_starts_no_sandboxes() -> None:
    both = _server(sandbox_opt_out={"task": True, "verifier": True})
    assert both.plan_attempt_sandbox(key="a", verifier_executes_nothing=True) is None
    assert _server(configured=False).plan_attempt_sandbox(key="a") is None


@pytest.mark.parametrize(
    "plan, reason",
    [
        (
            {"opt_out": {"task": True}, "task_tools": ["get_weather"]},
            "only for a pure QA task, whose model calls no tools, but this task has ['get_weather']",
        ),
        (
            {
                "opt_out": {"task": True},
                "group": SandboxGroupSpec(sidecars=[SandboxGroupSidecar(name="db", image="db")]),
            },
            "this task has ['sidecar db']",
        ),
        (
            {"opt_out": {"verifier": True}},
            "'resources' does not declare that its verifier does (verifier_executes_nothing)",
        ),
        (
            {"configured": False, "group": SandboxGroupSpec(image="verifier:1")},
            "declares a sandbox group at seed, but environment sets no sandbox_provider",
        ),
        (
            {"lent": SandboxAccess.model_validate(ACCESS)},
            "a Resources Server only borrows the sandbox the Environment Server lends",
        ),
        (
            {"sandbox_config": SANDBOX_CONFIG | {"image": None}},
            "the attempt's sandbox group has no image",
        ),
        (
            {"group": SandboxGroupSpec(sidecars=[SandboxGroupSidecar(name="db", image="db")])},
            "declares sidecars, which provider _Provider of 'attempt_sandboxes' cannot start",
        ),
    ],
)
def test_a_refused_opt_out_or_an_unstartable_group_fails_the_attempt_at_seed_for_good(
    plan: dict[str, Any], reason: str
) -> None:
    provider = _Provider()
    config = {"sandbox_config": plan.pop("sandbox_config")} if "sandbox_config" in plan else {}
    server = _server(
        provider, configured=plan.pop("configured", True), sandbox_opt_out=plan.pop("opt_out", {}), **config
    )

    with pytest.raises(NgFailureEvent) as error:
        server.plan_attempt_sandbox(key="a", resources_server="'resources'", **plan)

    assert reason in error.value.failure_reason
    assert _refusal(error) == ("seed", True)
    assert provider.boxes == {}


def test_a_sandbox_the_resources_seed_lends_is_lent_on_by_a_server_that_starts_none() -> None:
    assert _server(configured=False).plan_attempt_sandbox(key="a", lent=SandboxAccess.model_validate(ACCESS)) is None


async def test_the_group_starts_from_the_config_with_the_declared_group_over_it_and_stops_on_every_repeat() -> None:
    provider = _SidecarProvider()
    server = _server(provider)
    group = SandboxGroupSpec(
        image="task:1",
        workdir="/task",
        env={"SHARED": "group"},
        files={"task.txt": "2+2"},
        sidecars=[SandboxGroupSidecar(name="weather", image="weather:1")],
    )
    prepared: list[SandboxAccess] = []

    async def prepare(access: SandboxAccess) -> None:
        prepared.append(access)

    attempt, access = await server.seed_attempt_sandbox(_request(), key="attempt/a", group=group, prepare=prepare)

    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.workdir, spec.ttl_s) == ("task:1", "/task", 3600)
    assert spec.env == {"FROM_CONFIG": "1", "SHARED": "group"}
    assert spec.files == {"task.txt": "2+2"}
    assert spec.sidecars == (SandboxSidecarSpec(name="weather", image="weather:1"),)
    assert (spec.metadata["task_id"], spec.metadata["rollout_id"]) == ("task", "rollout-a1")
    # Lent with operate scope, through the provider block a borrower resolves, with the declared sidecars, so the
    # borrower reaches each one with sidecar(name); prepare was not declared.
    descriptor = {
        "sandbox_id": "sb-1",
        "scope": "operate",
        "workdir": "/task",
        "sidecars": [{"name": "weather", "image": "weather:1"}],
    }
    assert access.connection.provider_config_ref == "attempt_sandboxes"
    assert access.connection.descriptor == descriptor
    assert access.workdir == "/task"
    assert prepared == []
    # The verifier is lent the same group after the agent.
    assert (await server.verifier_sandbox_access(_request(), attempt)).connection.descriptor == descriptor

    await server.release_attempt_sandbox(attempt.key)
    await server.stop_attempt_sandbox(attempt.key)
    assert provider.running() == []


async def test_a_declared_prepare_runs_once_the_group_has_started() -> None:
    provider = _Provider()
    server = _server(provider)
    seen: list[list[str]] = []

    async def prepare(access: SandboxAccess) -> None:
        seen.append(provider.running())

    await server.seed_attempt_sandbox(_request(), key="a", group=SandboxGroupSpec(prepare=True), prepare=prepare)

    assert seen == [["sb-1"]]


async def test_a_group_that_cannot_start_fails_the_attempt_for_a_retry_and_leaves_nothing_running() -> None:
    provider = _Provider()
    provider.fail_create = SandboxCreateError("provider unreachable")
    server = _server(provider, sandbox_opt_out={"task": True})
    attempt, _ = await server.seed_attempt_sandbox(_request(), key="a")

    with pytest.raises(NgFailureEvent) as error:
        await server.verifier_sandbox_access(_request(), attempt, partial_response="kept")

    # An infrastructure failure: retried, never scored, and the evidence of the turn is kept.
    assert _refusal(error) == ("verification", False)
    assert "SandboxCreateError: provider unreachable" in error.value.failure_reason
    assert error.value.episode_evidence == {"partial_response": "kept"}
    assert provider.running() == []
    assert server.sandboxes.keys == []


async def test_a_failed_prepare_is_retried_only_when_the_resources_servers_error_is() -> None:
    provider = _Provider()
    server = _server(provider)

    async def prepare(access: SandboxAccess) -> None:
        raise TimeoutError("resources server did not answer")

    with pytest.raises(NgFailureEvent) as error:
        await server.seed_attempt_sandbox(_request(), key="a", group=SandboxGroupSpec(prepare=True), prepare=prepare)

    assert _refusal(error) == ("seed", False)
    # The started group is the episode's cleanup's to stop.
    assert provider.running() == ["sb-1"]
    await server.stop_attempt_sandbox("a")
    assert provider.running() == []


async def test_seeding_marks_the_episode_a_restart_and_starts_the_group_only_for_the_agent() -> None:
    provider = _Provider()
    server = _server(provider, sandbox_opt_out={"task": True})
    restarts: list[EpisodeId] = []

    async def restart(request: _Request) -> None:
        restarts.append(request.episode_id)

    object.__setattr__(server, "checkpoint_restart", restart)

    attempt, access = await server.seed_attempt_sandbox(_request(), key="a", group=SandboxGroupSpec(env={"T": "1"}))

    # The plan holds what Resources declared, so no checkpoint may export the episode.
    assert restarts == [_request().episode_id]
    assert (attempt.lend_to_agent, attempt.lend_to_verifier, access) == (False, True, None)
    assert provider.boxes == {}
    lent = await server.verifier_sandbox_access(_request(), attempt)
    assert lent.connection.descriptor["sandbox_id"] == "sb-1"
    assert provider.boxes["sb-1"]["spec"].env["T"] == "1"
    await server.release_attempt_sandbox(attempt.key)
    assert provider.running() == []


async def test_a_group_only_the_agent_borrowed_is_released_before_verify() -> None:
    provider = _Provider()
    server = _server(provider, sandbox_opt_out={"verifier": True})

    attempt, access = await server.seed_attempt_sandbox(_request(), key="a", verifier_executes_nothing=True)

    assert access.connection.descriptor["sandbox_id"] == "sb-1"
    assert await server.verifier_sandbox_access(_request(), attempt) is None
    assert provider.running() == []


async def test_a_failed_release_is_left_to_the_episodes_cleanup() -> None:
    provider = _Provider()
    server = _server(provider)
    attempt, _ = await server.seed_attempt_sandbox(_request(), key="a")
    close = provider.close
    calls: list[str] = []

    async def failing_close(handle: SandboxHandle) -> None:
        calls.append(handle.sandbox_id)
        if len(calls) == 1:
            raise ConnectionResetError("provider went away")
        await close(handle)

    provider.close = failing_close

    await server.release_attempt_sandbox(attempt.key)
    assert provider.running() == ["sb-1"]
    await server.stop_attempt_sandbox(attempt.key)
    assert provider.running() == []


async def test_stopping_an_attempt_that_started_nothing_builds_no_provider() -> None:
    server = _server()

    await server.stop_attempt_sandbox("attempt/a")

    assert server._owned_sandboxes == {}


def test_startup_refuses_a_server_that_starts_attempt_sandboxes_without_declaring_it_owns_and_lends_them() -> None:
    undeclared = type("Undeclared", (_Server,), {"lends_sandboxes": False})
    server = undeclared(config=_server().config, server_client=MagicMock(spec=ServerClient))

    with pytest.raises(ValueError, match="sets lends_sandboxes = True and lists ATTEMPT_SANDBOX_BLOCKS"):
        server.check_sandbox_provider()


async def test_another_environment_server_type_runs_an_attempt_in_a_group_it_starts_and_stops() -> None:
    provider = _Provider()
    server = _server(provider)

    response = await server.run_request(_request())

    assert response.failure is None, response.failure
    # The agent and the verifier borrowed one group, which stopped once the verifier was done.
    assert response.result == "sb-1 sb-1"
    assert provider.running() == []


async def test_another_environment_server_types_cancelled_attempt_leaves_no_group_running() -> None:
    provider = _Provider()
    server = _server(provider, server_type=type("Waiting", (_Server,), {"turn": asyncio.Event()}))

    episode = asyncio.create_task(server.run_request(_request()))
    while provider.running() != ["sb-1"]:
        await asyncio.sleep(0.01)
    episode.cancel()
    with pytest.raises(asyncio.CancelledError):
        await episode

    assert provider.running() == []
    assert server.sandboxes.keys == []
