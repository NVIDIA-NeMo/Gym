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
    AttemptSandbox,
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
    verifier_mode = "separate"

    def sandbox_group(self, body: ResourcesSeedSessionRequest) -> SandboxGroupSpec | None:
        return SandboxGroupSpec(image="python:3.13-slim", files={"tests/test_task.py": body.task_data["question"]})


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


def test_a_server_declares_the_group_its_hook_describes_and_where_its_verifier_runs() -> None:
    reply = _resources(_CodeVerifier).post("/seed_session", json=SEED).json()

    group = SandboxGroupSpec.model_validate(reply["sandbox_group"])
    assert (group.image, group.files, group.prepare) == ("python:3.13-slim", {"tests/test_task.py": "2+2"}, False)
    assert reply["verifier_mode"] == "separate"
    assert ResourcesSeedSessionResponse.model_validate(reply).verifier_mode == "separate"


def test_an_undeclared_verifier_mode_is_none_and_stays_off_the_wire() -> None:
    reply = ResourcesSeedSessionResponse(resources_session_id="s")

    assert reply.verifier_mode == "none"
    assert "verifier_mode" not in reply.model_dump(mode="json")
    with pytest.raises(ValidationError):
        ResourcesSeedSessionResponse(resources_session_id="s", verifier_mode="mixed")


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


class _GraderThatAlsoRunsInProcess(_Verifier):
    """A verifier that grades in a sandbox when it is lent one, and in its own process otherwise."""

    supported_verifier_modes = frozenset({"separate", "none"})


def test_a_server_says_every_mode_its_verifier_runs_in_and_only_then_the_seed_reply_carries_them() -> None:
    reply = _resources(_GraderThatAlsoRunsInProcess).post("/seed_session", json=SEED).json()

    # In VerifierMode's order; verifier_mode itself stays the default, so it is left off the wire.
    assert reply["supported_verifier_modes"] == ["none", "separate"]
    assert "verifier_mode" not in reply
    # A server that supports only its verifier_mode keeps the reply's shape.
    assert "supported_verifier_modes" not in _resources(_CodeVerifier).post("/seed_session", json=SEED).json()


def test_a_seed_reply_or_a_class_that_does_not_support_its_own_verifier_mode_is_refused() -> None:
    with pytest.raises(ValidationError, match="not among supported_verifier_modes"):
        ResourcesSeedSessionResponse(
            resources_session_id="s", verifier_mode="shared", supported_verifier_modes=["none", "separate"]
        )
    with pytest.raises(TypeError, match="'separate' is not among its supported_verifier_modes \\['none'\\]"):
        type(
            "_Contradiction",
            (_Verifier,),
            {"verifier_mode": "separate", "supported_verifier_modes": frozenset({"none"})},
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
    """A provider that lends sandboxes to other processes and applies the restricted network rules."""

    name = "fake"
    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit", "no_network", "model_only", "hosts_only"}))

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.fail_create: Exception | None = None
        self.commands: list[tuple[str, str | None]] = []
        self.exec_result = SandboxExecResult(stdout="", stderr=None, return_code=0)

    def network_options(self, rule: str, hosts: Sequence[str], provider_options: Mapping[str, Any]) -> dict[str, Any]:
        return {**provider_options, "network_policy": {"rule": rule, "allow": list(hosts)}}

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if self.fail_create is not None:
            raise self.fail_create
        sandbox_id = f"sb-{len(self.boxes) + 1}"
        self.boxes[sandbox_id] = {"spec": spec, "state": "running"}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        self.commands.append((command, kwargs.get("cwd")))
        return self.exec_result

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


class _OpenProvider(_Provider):
    """A provider that applies no restricted network rule."""

    capabilities = SandboxCapabilities(network_rules=frozenset({"inherit"}))


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
        attempt, access = await self.seed_attempt_sandbox(request, key=key, verifier_mode="shared")
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


class _ToolGrantingServer(_Server):
    """A server whose protocol grants the agent the tools a sidecar serves."""

    grants_tool_sidecars = True


SANDBOX_CONFIG = {
    "image": "python:3.13-slim",
    "workdir": "/workspace",
    "env": {"FROM_CONFIG": "1", "SHARED": "config"},
    "ttl_s": 3600,
    "network": "inherit",
}


def _server(
    provider: _Provider | None = None,
    *,
    configured: bool = True,
    server_type: type[_Server] = _Server,
    **config: Any,
) -> _Server:
    settings = AttemptSandboxEnvironmentServerConfig.model_validate(
        {"name": "environment", "host": "", "port": 0, "entrypoint": "", "cleanup_timeout_seconds": 10}
        | ({"sandbox_provider": "attempt_sandboxes", "sandbox_config": SANDBOX_CONFIG} if configured else {})
        | config
    )
    server = server_type(config=settings, server_client=MagicMock(spec=ServerClient))
    if provider is not None:
        server._owned_sandboxes["sandbox_provider"] = OwnedSandboxes(
            provider,
            state_dir=None,
            server_name="environment",
            provider_config_ref="attempt_sandboxes",
            global_config={},
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
    "tools, runs_without_sandbox, lends_to_agent",
    [
        # A task with tools runs in the group, whatever the agent.
        (["get_weather"], True, True),
        (["get_weather"], False, True),
        # A task without tools: only an agent that cannot run its loop without a sandbox borrows a group.
        ([], False, True),
        ([], True, False),
    ],
)
def test_the_agent_borrows_a_group_when_the_task_has_tools_or_it_cannot_run_without_one(
    tools: list[str], runs_without_sandbox: bool, lends_to_agent: bool
) -> None:
    attempt = _server(_Provider()).plan_attempt_sandbox(
        key="attempt/a",
        task_tools=tools,
        agent_runs_loop_without_sandbox=runs_without_sandbox,
        verifier_mode="separate",
    )

    assert (attempt.key, attempt.lend_to_agent, attempt.verifier_mode) == ("attempt/a", lends_to_agent, "separate")
    # The agent's group has an image only when it starts; a separate verifier's always does.
    agent_group = ("python:3.13-slim", "/workspace") if lends_to_agent else (None, None)
    assert (attempt.image, attempt.workdir) == agent_group
    assert (attempt.verifier_image, attempt.verifier_workdir) == ("python:3.13-slim", "/workspace")


def test_a_sidecar_makes_the_agent_borrow_a_group_whatever_the_agent() -> None:
    server = _server(_SidecarProvider(), server_type=_ToolGrantingServer)
    group = SandboxGroupSpec(sidecars=[SandboxGroupSidecar(name="db", image="db")])

    assert server.plan_attempt_sandbox(key="a", group=group, agent_runs_loop_without_sandbox=True).lend_to_agent


def test_nothing_starts_when_neither_the_agent_nor_the_verifier_borrows_a_group() -> None:
    server = _server()

    assert server.plan_attempt_sandbox(key="a", agent_runs_loop_without_sandbox=True) is None
    # A declared group that nobody borrows starts nothing either.
    assert (
        server.plan_attempt_sandbox(key="a", group=SandboxGroupSpec(image="x:1"), agent_runs_loop_without_sandbox=True)
        is None
    )


def test_without_a_sandbox_provider_the_agent_runs_where_it_ran_before() -> None:
    server = _server(configured=False)

    assert server.plan_attempt_sandbox(key="a", task_tools=["get_weather"]) is None
    assert server.plan_attempt_sandbox(key="a", lent=SandboxAccess.model_validate(ACCESS)) is None


@pytest.mark.parametrize(
    "plan, reason",
    [
        (
            {"configured": False, "group": SandboxGroupSpec(image="verifier:1")},
            "'resources' declares a sandbox group at seed, but environment sets no sandbox_provider",
        ),
        (
            {"configured": False, "verifier_mode": "separate"},
            "'resources' declares verifier mode 'separate' at seed, but environment sets no sandbox_provider",
        ),
        (
            {"verifier_mode": "shared", "agent_runs_loop_without_sandbox": True},
            "the verifier mode is 'shared', but the agent borrows no group to share",
        ),
        (
            {"provider": _OpenProvider(), "verifier_mode": "separate"},
            "a separate verifier's group starts with network 'no_network', but sandbox provider",
        ),
        (
            {"lent": SandboxAccess.model_validate(ACCESS)},
            "a Resources Server only borrows the sandbox the Environment Server lends",
        ),
        (
            {"sandbox_config": SANDBOX_CONFIG | {"image": None}},
            "the agent's sandbox group has no image",
        ),
        (
            {"group": SandboxGroupSpec(sidecars=[SandboxGroupSidecar(name="db", image="db")])},
            "declares sidecars, which provider _Provider of 'attempt_sandboxes' cannot start",
        ),
        (
            {
                "provider": _SidecarProvider(),
                "group": SandboxGroupSpec(sidecars=[SandboxGroupSidecar(name="tools", image="t", tools_port=8080)]),
            },
            "sidecars ['tools'] serve tools on a tools_port, but _Server does not grant a sidecar's tools",
        ),
        (
            {
                "provider": type("_OpenSidecarProvider", (_SidecarProvider, _OpenProvider), {})(),
                "group": SandboxGroupSpec(
                    sidecars=[SandboxGroupSidecar(name="api", image="api", egress_hosts=["a.b"])]
                ),
            },
            "sidecar 'api' reaches only its own hosts, but sandbox provider",
        ),
    ],
)
def test_a_group_no_attempt_can_have_fails_the_attempt_at_seed_for_good(plan: dict[str, Any], reason: str) -> None:
    provider = plan.pop("provider", _Provider())
    config = {"sandbox_config": plan.pop("sandbox_config")} if "sandbox_config" in plan else {}
    server = _server(provider, configured=plan.pop("configured", True), **config)

    with pytest.raises(NgFailureEvent) as error:
        server.plan_attempt_sandbox(key="a", resources_server="'resources'", **plan)

    assert reason in error.value.failure_reason
    assert _refusal(error) == ("seed", True)
    assert provider.boxes == {}


async def test_the_group_starts_from_the_config_with_the_declared_group_over_it_and_stops_on_every_repeat() -> None:
    provider = _SidecarProvider()
    server = _server(provider, server_type=_ToolGrantingServer)
    group = SandboxGroupSpec(
        image="task:1",
        workdir="/task",
        env={"SHARED": "group"},
        files={"task.txt": "2+2"},
        sidecars=[
            SandboxGroupSidecar(
                name="weather", image="weather:1", egress_hosts=["api.weather.example"], tools_port=8080
            ),
            SandboxGroupSidecar(name="db", image="db:1"),
        ],
    )
    prepared: list[SandboxAccess] = []

    async def prepare(access: SandboxAccess) -> None:
        prepared.append(access)

    attempt, access = await server.seed_attempt_sandbox(
        _request(), key="attempt/a", group=group, verifier_mode="shared", prepare=prepare
    )

    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.workdir, spec.ttl_s) == ("task:1", "/task", 3600)
    assert spec.env == {"FROM_CONFIG": "1", "SHARED": "group"}
    assert spec.files == {"task.txt": "2+2"}
    # Each sidecar starts with its own rule, never the main container's: the weather API only, or nothing.
    assert spec.sidecars == (
        SandboxSidecarSpec(
            name="weather",
            image="weather:1",
            provider_options={"network_policy": {"rule": "hosts_only", "allow": ["api.weather.example"]}},
        ),
        SandboxSidecarSpec(
            name="db", image="db:1", provider_options={"network_policy": {"rule": "no_network", "allow": []}}
        ),
    )
    assert (spec.metadata["task_id"], spec.metadata["rollout_id"]) == ("task", "rollout-a1")
    # Lent with operate scope, through the provider block a borrower resolves, with the declared sidecars, so the
    # borrower reaches each one with sidecar(name); prepare was not declared.
    descriptor = {
        "sandbox_id": "sb-1",
        "scope": "operate",
        "workdir": "/task",
        "sidecars": [{"name": "weather", "image": "weather:1"}, {"name": "db", "image": "db:1"}],
    }
    assert access.connection.provider_config_ref == "attempt_sandboxes"
    assert access.connection.descriptor == descriptor
    assert access.workdir == "/task"
    assert prepared == []
    # A shared verifier is lent the same group after the agent.
    assert (await server.verifier_sandbox_access(_request(), attempt)).connection.descriptor == descriptor

    await server.release_attempt_sandbox(attempt.key)
    await server.stop_attempt_sandbox(attempt.key)
    assert provider.running() == []


async def test_a_separate_verifier_without_an_agent_group_gets_a_late_group_with_no_egress() -> None:
    provider = _Provider()
    server = _server(
        provider, sandbox_config=SANDBOX_CONFIG | {"network": "model_only", "model_server": "policy_model"}
    )
    attempt, access = await server.seed_attempt_sandbox(
        _request(), key="a", verifier_mode="separate", agent_runs_loop_without_sandbox=True
    )

    # Nothing starts during the agent's turn.
    assert (attempt.lend_to_agent, access, provider.boxes) == (False, None, {})
    lent = await server.verifier_sandbox_access(_request(), attempt)

    assert lent.connection.descriptor["sandbox_id"] == "sb-1"
    assert server.sandboxes.keys == ["a/verifier"]
    assert provider.boxes["sb-1"]["spec"].provider_options["network_policy"] == {"rule": "no_network", "allow": []}
    await server.release_attempt_sandbox(attempt.key)
    assert provider.running() == []


async def test_a_separate_verifier_after_an_agent_group_gets_a_fresh_group_once_the_agents_has_stopped() -> None:
    provider = _SidecarProvider()
    server = _server(provider, server_type=_ToolGrantingServer)
    group = SandboxGroupSpec(prepare=True, sidecars=[SandboxGroupSidecar(name="tools", image="t", tools_port=8080)])
    prepared: list[list[str]] = []

    async def prepare(access: SandboxAccess) -> None:
        prepared.append(provider.running())

    attempt, access = await server.seed_attempt_sandbox(
        _request(), key="a", group=group, verifier_mode="separate", prepare=prepare
    )
    lent = await server.verifier_sandbox_access(_request(), attempt, prepare=prepare)

    assert (access.connection.descriptor["sandbox_id"], lent.connection.descriptor["sandbox_id"]) == ("sb-1", "sb-2")
    # The agent's group stopped before the verifier's started, so the verifier gets the response, not its files.
    assert provider.running() == ["sb-2"]
    verifier_spec = provider.boxes["sb-2"]["spec"]
    assert verifier_spec.sidecars == ()
    assert verifier_spec.provider_options["network_policy"] == {"rule": "no_network", "allow": []}
    # The task was prepared once, in the agent's group.
    assert prepared == [["sb-1"]]
    await server.stop_attempt_sandbox(attempt.key)
    assert provider.running() == []


class _StuckProvider(_Provider):
    """A provider whose close of ``stuck`` keeps failing."""

    stuck = "sb-1"

    async def close(self, handle: SandboxHandle) -> None:
        if handle.sandbox_id == self.stuck:
            raise RuntimeError("provider close failed")
        await super().close(handle)


async def test_a_separate_verifiers_group_stops_even_while_the_agents_group_fails_to_stop() -> None:
    provider = _StuckProvider()
    server = _server(provider)

    attempt, _ = await server.seed_attempt_sandbox(
        _request(), key="a", verifier_mode="separate", task_tools=["run_python"]
    )
    # The agent's group fails to stop when it is released for the verifier; the verifier's group starts anyway.
    lent = await server.verifier_sandbox_access(_request(), attempt)
    assert lent.connection.descriptor["sandbox_id"] == "sb-2"
    assert provider.running() == ["sb-1", "sb-2"]

    # Released after /verify and stopped in cleanup: the agent's failed stop never keeps the verifier's running.
    await server.release_attempt_sandbox(attempt.key)
    assert provider.running() == ["sb-1"]
    with pytest.raises(RuntimeError, match="provider close failed"):
        await server.stop_attempt_sandbox(attempt.key)
    assert provider.running() == ["sb-1"]
    provider.stuck = ""
    await server.stop_attempt_sandbox(attempt.key)
    assert provider.running() == []


async def test_a_verifier_that_runs_in_the_resources_server_is_lent_nothing_and_the_agents_group_stops_first() -> None:
    provider = _Provider()
    server = _server(provider)

    attempt, access = await server.seed_attempt_sandbox(_request(), key="a", task_tools=["run_python"])

    assert (attempt.verifier_mode, access.connection.descriptor["sandbox_id"]) == ("none", "sb-1")
    assert await server.verifier_sandbox_access(_request(), attempt) is None
    assert provider.running() == []


async def test_the_working_directory_is_created_before_anyone_borrows_the_group() -> None:
    provider = _Provider()
    server = _server(provider)

    await server.seed_attempt_sandbox(_request(), key="a", group=SandboxGroupSpec(workdir="/task dir"))

    # From the root, since a provider that does not create it cannot run a command in it either.
    assert provider.commands == [("mkdir -p '/task dir'", "/")]


async def test_a_working_directory_that_cannot_be_created_fails_the_attempt_for_a_retry() -> None:
    provider = _Provider()
    provider.exec_result = SandboxExecResult(stdout="", stderr="read-only file system", return_code=1)
    server = _server(provider)

    with pytest.raises(NgFailureEvent) as error:
        await server.seed_attempt_sandbox(_request(), key="a")

    assert _refusal(error) == ("seed", False)
    assert "'/workspace' could not be created (return code 1, no error type): read-only file system" in (
        error.value.failure_reason
    )
    # The started group is the episode's cleanup's to stop.
    await server.stop_attempt_sandbox("a")
    assert provider.running() == []


async def test_a_declared_prepare_runs_once_the_group_has_started() -> None:
    provider = _Provider()
    server = _server(provider)
    seen: list[list[str]] = []

    async def prepare(access: SandboxAccess) -> None:
        seen.append(provider.running())

    await server.seed_attempt_sandbox(_request(), key="a", group=SandboxGroupSpec(prepare=True), prepare=prepare)

    assert seen == [["sb-1"]]


async def test_a_separate_group_that_cannot_start_fails_the_attempt_for_a_retry_and_leaves_nothing_running() -> None:
    provider = _Provider()
    provider.fail_create = SandboxCreateError("provider unreachable")
    server = _server(provider)
    attempt, _ = await server.seed_attempt_sandbox(
        _request(), key="a", verifier_mode="separate", agent_runs_loop_without_sandbox=True
    )

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


async def test_seeding_a_plan_marks_the_episode_a_restart() -> None:
    provider = _Provider()
    server = _server(provider)
    restarts: list[EpisodeId] = []

    async def restart(request: _Request) -> None:
        restarts.append(request.episode_id)

    object.__setattr__(server, "checkpoint_restart", restart)

    await server.seed_attempt_sandbox(
        _request(),
        key="a",
        group=SandboxGroupSpec(env={"T": "1"}),
        verifier_mode="separate",
        agent_runs_loop_without_sandbox=True,
    )

    # The plan holds what Resources declared, so no checkpoint may export the episode.
    assert restarts == [_request().episode_id]


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


# --- The verifier block --------------------------------------------------------------------------------------------

PROVIDER = {"sandbox_provider": "attempt_sandboxes"}
# A grader's whole group, as code_gen configures it: no task sandbox, since its tasks have no tools.
GRADER_ALONE = {"mode": "separate", "sandbox": {"image": "grader:1", "workdir": "/grade", "ttl_s": 600}}


def _settings(**config: Any) -> AttemptSandboxEnvironmentServerConfig:
    return AttemptSandboxEnvironmentServerConfig.model_validate(
        {"name": "environment", "host": "", "port": 0, "entrypoint": "", "cleanup_timeout_seconds": 10} | config
    )


@pytest.mark.parametrize(
    "config, refusal",
    [
        # verifier.sandbox describes a separate verifier's group, so it says so.
        ({"verifier": {"sandbox": {"image": "grader:1"}}} | PROVIDER, "set verifier.mode: separate"),
        (
            {"verifier": {"mode": "none", "sandbox": {"image": "grader:1"}}, "sandbox_config": SANDBOX_CONFIG}
            | PROVIDER,
            "set verifier.mode: separate",
        ),
        # No task sandbox means no agent group to share.
        ({"verifier": {"mode": "shared"}}, "verifier.mode shared runs /verify in the agent's group"),
        # Without a task sandbox, verifier.sandbox is the group's whole settings.
        ({"verifier": {"mode": "separate"}} | PROVIDER, "needs a complete verifier.sandbox; set ['image', 'ttl_s']"),
        ({"verifier": {"mode": "separate", "sandbox": {"image": "grader:1"}}} | PROVIDER, "set ['ttl_s']"),
        ({"verifier": {"mode": "separate", "sandbox": {"ttl_s": 600}}} | PROVIDER, "set ['image']"),
        # Every group starts on sandbox_provider, and a provider alone starts nothing.
        (PROVIDER, "set sandbox_provider together with sandbox_config or verifier.sandbox"),
        ({"sandbox_config": SANDBOX_CONFIG}, "set sandbox_provider together with sandbox_config or verifier.sandbox"),
        ({"verifier": GRADER_ALONE}, "set sandbox_provider together with sandbox_config or verifier.sandbox"),
        # The verifier's group must outlive the episode too.
        (
            {"verifier": GRADER_ALONE, "default_episode_timeout_seconds": 600} | PROVIDER,
            "the separate verifier's ttl_s (600) must exceed default_episode_timeout_seconds",
        ),
        # A host goes with the rule that allows it.
        (
            {
                "verifier": {"mode": "separate", "sandbox": {"allowed_hosts": ["pypi.org"]}},
                "sandbox_config": SANDBOX_CONFIG,
            }
            | PROVIDER,
            "allowed_hosts add to a restricted network rule",
        ),
    ],
)
def test_startup_refuses_verifier_settings_no_attempt_can_start(config: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValidationError) as error:
        _settings(**config)

    assert refusal in str(error.value)


def test_a_separate_verifiers_group_takes_what_verifier_sandbox_leaves_unset_from_the_task_sandbox_but_no_egress() -> (
    None
):
    task_sandbox = SANDBOX_CONFIG | {"network": "model_only", "model_server": "policy_model", "resources": {"cpu": 2}}
    replaced = {"image": "grader:1", "resources": {"cpu": 1, "memory_mib": 2048}}

    verifier = _settings(
        **PROVIDER, sandbox_config=task_sandbox, verifier={"mode": "separate", "sandbox": replaced}
    ).verifier_sandbox_config

    assert (verifier.image, verifier.resources) == ("grader:1", {"cpu": 1, "memory_mib": 2048})
    assert (verifier.workdir, verifier.ttl_s, verifier.env) == ("/workspace", 3600, SANDBOX_CONFIG["env"])
    # The task sandbox's rule is the agent's: the verifier's group reaches nothing unless it sets a rule of its own.
    assert (verifier.network, verifier.model_server) == ("no_network", None)
    own = {"network": "hosts_only", "allowed_hosts": ["pypi.org"]}
    verifier = _settings(
        **PROVIDER, sandbox_config=task_sandbox, verifier={"mode": "separate", "sandbox": own}
    ).verifier_sandbox_config
    assert (verifier.network, verifier.allowed_hosts, verifier.image) == (
        "hosts_only",
        ["pypi.org"],
        "python:3.13-slim",
    )
    # Without a task sandbox, verifier.sandbox is the whole group; with neither, there is none.
    verifier = _settings(**PROVIDER, verifier=GRADER_ALONE).verifier_sandbox_config
    assert (verifier.image, verifier.workdir, verifier.ttl_s, verifier.network) == (
        "grader:1",
        "/grade",
        600,
        "no_network",
    )
    assert _settings().verifier_sandbox_config is None


@pytest.mark.parametrize(
    "configured, declared, expected",
    [
        ("grader:1", "task:1", "grader:1"),
        (None, "task:1", "task:1"),
        (None, None, "python:3.13-slim"),
    ],
    ids=["verifier.sandbox.image", "declared at seed", "sandbox_config.image"],
)
def test_a_separate_verifiers_image_is_its_own_then_the_declared_one_then_the_task_sandboxes(
    configured: str | None, declared: str | None, expected: str
) -> None:
    verifier = {"mode": "separate"} | ({"sandbox": {"image": configured}} if configured else {})
    server = _server(_Provider(), verifier=verifier)

    attempt = server.plan_attempt_sandbox(
        key="a", group=SandboxGroupSpec(image=declared, workdir="/task"), verifier_mode="separate"
    )

    assert (attempt.verifier_image, attempt.verifier_workdir) == (expected, "/task")
    # The agent's group never takes the verifier's image.
    assert attempt.image == (declared or "python:3.13-slim")
    with pytest.raises(ValidationError, match="a separate verifier's group needs an image"):
        AttemptSandbox.model_validate(attempt.model_dump() | {"verifier_image": None})
    with pytest.raises(ValidationError, match="the agent's group needs an image"):
        AttemptSandbox(key="a", lend_to_agent=True)


@pytest.mark.parametrize(
    "provider, verifier, refused",
    [
        (_OpenProvider(), {"mode": "separate"}, True),
        (_Provider(), {"mode": "separate"}, False),
        # An open network in verifier.sandbox does not get a separate verifier onto a provider that cannot cut it.
        (_OpenProvider(), {"mode": "separate", "sandbox": {"network": "inherit"}}, True),
        (_Provider(), {"mode": "separate", "sandbox": {"network": "inherit"}}, False),
    ],
)
def test_startup_refuses_a_configured_separate_verifier_on_a_provider_that_cannot_apply_its_rule(
    provider: _Provider, verifier: dict[str, Any], refused: bool
) -> None:
    server = _server(provider, verifier=verifier)
    # A provider block whose class passes the base checks, so the started provider's own capabilities decide.
    server.server_client.global_config_dict = {"attempt_sandboxes": {"docker": {}}}

    if refused:
        with pytest.raises(ValueError, match="sets verifier.mode separate, whose group starts with network '.*', but"):
            server.check_sandbox_provider()
    else:
        server.check_sandbox_provider()
    # A server that starts no group builds no provider at startup.
    idle = _server(configured=False)
    idle.server_client.global_config_dict = {}
    idle.check_sandbox_provider()
    assert idle._owned_sandboxes == {}


async def test_without_a_task_sandbox_a_separate_verifier_starts_from_verifier_sandbox_alone() -> None:
    provider = _Provider()
    server = _server(provider, configured=False, **PROVIDER, verifier=GRADER_ALONE)
    group = SandboxGroupSpec(image="task:1", files={"tests/test_task.py": "assert True"})

    attempt, access = await server.seed_attempt_sandbox(
        _request(), key="a", group=group, verifier_mode="separate", agent_runs_loop_without_sandbox=True
    )
    assert (access, provider.boxes) == (None, {})
    lent = await server.verifier_sandbox_access(_request(), attempt)

    spec = provider.boxes["sb-1"]["spec"]
    assert (spec.image, spec.workdir, spec.ttl_s) == ("grader:1", "/grade", 600)
    assert spec.files == {"tests/test_task.py": "assert True"}
    assert spec.provider_options["network_policy"] == {"rule": "no_network", "allow": []}
    assert (lent.workdir, provider.commands) == ("/grade", [("mkdir -p /grade", "/")])
    await server.release_attempt_sandbox(attempt.key)
    assert provider.running() == []


@pytest.mark.parametrize(
    "config, plan, mode",
    [
        # The configured mode wins over the declared one, among the modes the Resources Server supports.
        ({"verifier": {"mode": "separate"}}, {"supported_verifier_modes": ["none", "separate"]}, "separate"),
        (
            {"verifier": {"mode": "none"}},
            {"verifier_mode": "separate", "supported_verifier_modes": ["none", "separate"]},
            None,
        ),
        # Unset, the declared mode stands.
        ({}, {"verifier_mode": "separate"}, "separate"),
        # Without a provider, a configured none needs no group.
        (
            {"configured": False, "verifier": {"mode": "none"}},
            {"verifier_mode": "separate", "supported_verifier_modes": ["none", "separate"]},
            None,
        ),
    ],
)
def test_the_configured_verifier_mode_decides_where_verify_runs(
    config: dict[str, Any], plan: dict[str, Any], mode: str | None
) -> None:
    server = _server(_Provider(), configured=config.pop("configured", True), **config)

    attempt = server.plan_attempt_sandbox(key="a", agent_runs_loop_without_sandbox=True, **plan)

    assert (attempt.verifier_mode if attempt else None) == mode


@pytest.mark.parametrize(
    "config, plan, reason",
    [
        (
            {"verifier": {"mode": "none"}},
            {"verifier_mode": "separate"},
            "environment sets verifier.mode 'none', but 'resources' says its /verify runs only in ['separate']",
        ),
        (
            {"verifier": {"mode": "separate"}},
            {"supported_verifier_modes": ["none", "shared"]},
            "environment sets verifier.mode 'separate', but 'resources' says its /verify runs only in "
            "['none', 'shared']",
        ),
        (
            {"configured": False, **PROVIDER, "verifier": GRADER_ALONE},
            {"verifier_mode": "separate", "task_tools": ["run_python"]},
            "the agent needs a sandbox group (the task has tools ['run_python']), but environment sets no "
            "sandbox_config",
        ),
        (
            {"configured": False, **PROVIDER, "verifier": GRADER_ALONE},
            {"verifier_mode": "separate", "agent_runs_loop_without_sandbox": False},
            "(the agent cannot run its loop without a sandbox)",
        ),
        (
            {"configured": False, **PROVIDER, "verifier": GRADER_ALONE},
            {
                "verifier_mode": "separate",
                "group": SandboxGroupSpec(sidecars=[SandboxGroupSidecar(name="db", image="db")]),
            },
            "(the task has sidecars)",
        ),
        (
            {"sandbox_config": SANDBOX_CONFIG | {"workdir": None}},
            {"verifier_mode": "separate", "agent_runs_loop_without_sandbox": True},
            "a separate verifier's sandbox group has no workdir: set it in verifier.sandbox or sandbox_config",
        ),
    ],
)
def test_a_verifier_mode_or_an_agent_group_the_configuration_cannot_give_fails_the_attempt_at_seed_for_good(
    config: dict[str, Any], plan: dict[str, Any], reason: str
) -> None:
    provider = _Provider()
    server = _server(provider, configured=config.pop("configured", True), **config)

    with pytest.raises(NgFailureEvent) as error:
        server.plan_attempt_sandbox(
            key="a", resources_server="'resources'", **({"agent_runs_loop_without_sandbox": True} | plan)
        )

    assert reason in error.value.failure_reason
    assert _refusal(error) == ("seed", True)
    assert provider.boxes == {}


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
    # The agent and the shared verifier borrowed one group, which stopped once the verifier was done.
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
