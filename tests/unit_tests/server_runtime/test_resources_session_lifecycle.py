# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import Field, JsonValue, PrivateAttr

from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime.mcp_auto_exposure import maybe_auto_expose
from nemo_gym.server_runtime.testing.session_conformance import check_resources_session_contract


class TaskSeed(BaseSeedSessionRequest):
    value: int


class AsyncResourceServer(SimpleResourcesServer):
    seed_request_model = TaskSeed
    state: dict[str, int] = Field(default_factory=dict)
    released: list[str] = Field(default_factory=list)
    _initializing: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _initialization_gate: asyncio.Event | None = PrivateAttr(default=None)
    _fail_release: bool = PrivateAttr(default=False)

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self.state

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        assert isinstance(body, TaskSeed)
        self._initializing.set()
        if self._initialization_gate is not None:
            await self._initialization_gate.wait()
        self.state[session_id] = body.value

    async def release_session_state(self, session_id: str) -> None:
        await asyncio.sleep(0)
        if self._fail_release:
            raise RuntimeError("resource still running")
        self.released.append(session_id)
        self.state.pop(session_id, None)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0)


def _server(*, expose_tools_over_mcp: bool = False) -> AsyncResourceServer:
    return AsyncResourceServer(
        config=BaseResourcesServerConfig(
            host="", port=0, entrypoint="", name="async-resource", expose_tools_over_mcp=expose_tools_over_mcp
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _request() -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: "caller-cookie"}})


SEED = ResourcesSeedSessionRequest(
    resources_session_id="session",
    episode_id=EpisodeId(rollout_id="rollout"),
    task_id=TaskId(taskset="tasks", task_id="one"),
    task_data={"value": 42},
)
CLOSE = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=SEED.episode_id)


async def _release(server: AsyncResourceServer, operation: str) -> None:
    if operation == "close":
        await server.close_resources_session(CLOSE)
    else:
        await server.retire_session_state("session")


class FlakyInitializationServer(AsyncResourceServer):
    """Initialization fails after creating part of its resources, as a sandbox provider outage would."""

    _fail_initialization: bool = PrivateAttr(default=True)
    partial: dict[str, int] = Field(default_factory=dict)
    initializations: int = 0

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        self.initializations += 1
        if self._fail_initialization:
            # Kept apart from live state (has_session_state) and reachable only by release_session_state.
            self.partial[session_id] = -1
            raise RuntimeError("provider unavailable")
        await super().initialize_session_state(session_id, body)

    async def release_session_state(self, session_id: str) -> None:
        self.partial.pop(session_id, None)
        await super().release_session_state(session_id)


def _flaky_server() -> FlakyInitializationServer:
    return FlakyInitializationServer(config=_server().config, server_client=MagicMock(spec=ServerClient))


async def test_failed_typed_initialization_binds_nothing_and_a_retry_initializes_again() -> None:
    server = _flaky_server()
    request = _request()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await server.seed_session(SEED, request=request)
    assert request.session[SESSION_ID_KEY] == "caller-cookie"
    assert server._session_episodes == {}

    server._fail_initialization = False
    await server.seed_session(SEED, request=request)
    assert server.initializations == 2
    assert server.state == {"session": 42}
    assert request.session[SESSION_ID_KEY] == "session"


async def test_close_releases_what_a_failed_initialization_left_behind() -> None:
    server = _flaky_server()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await server.seed_session(SEED, request=_request())
    assert server.partial == {}  # The base cleans a failed initialization immediately.

    await server.close_resources_session(CLOSE)

    assert server.partial == {}
    assert server.session_is_closed("session")


async def test_typed_seed_after_verification_is_refused_and_does_not_reinitialize() -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    await _verify_and_release(server)

    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(SEED, request=_request())
    assert server.state == {}


async def _verify_and_release(server: AsyncResourceServer) -> None:
    async with server.verifying("session") as verification:
        assert verification.is_live
        verification.retain("receipt")


async def test_close_after_verification_checks_the_verified_episode() -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    await _verify_and_release(server)

    with pytest.raises(ResourcesSessionConflictError):
        await server.close_resources_session(CLOSE.model_copy(update={"episode_id": EpisodeId(rollout_id="other")}))
    assert not server.session_is_closed("session")

    await server.close_resources_session(CLOSE)
    assert server.session_is_closed("session")


async def test_cookie_seed_cannot_replace_a_verified_typed_session() -> None:
    server = _server()
    request = _request()
    await server.seed_session(SEED, request=request)
    await _verify_and_release(server)

    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(TaskSeed(value=2), request=request)
    assert server.state == {}
    assert server.session_identity("session") == (SEED.episode_id, SEED.task_id)
    async with server.verifying("session") as verification:
        assert verification.retained == "receipt"


async def test_cookie_seed_after_verification_keeps_the_session_id_and_starts_fresh_state() -> None:
    server = _server()
    request = _request()
    await server.seed_session(TaskSeed(value=1), request=request)
    async with server.verifying("caller-cookie") as verification:
        verification.retain("receipt")

    await server.seed_session(TaskSeed(value=2), request=request)

    assert request.session[SESSION_ID_KEY] == "caller-cookie"
    assert server.state == {"caller-cookie": 2}
    # The new run's verify must score its own state, not replay the finished run's receipt.
    assert not server.session_is_verified("caller-cookie")


async def test_a_failed_cookie_reseed_keeps_the_finished_runs_receipt() -> None:
    server = _flaky_server()
    async with server.verifying("caller-cookie", allow_missing=True) as verification:
        verification.retain("receipt")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await server.seed_session(TaskSeed(value=2), request=_request())

    async with server.verifying("caller-cookie") as verification:
        assert not verification.is_live
        assert verification.retained == "receipt"


@pytest.mark.parametrize("operation", ["close", "retire"])
async def test_release_waits_for_in_flight_initialization(operation: str) -> None:
    server = _server()
    request = _request()
    server._initialization_gate = asyncio.Event()
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(server.seed_session(SEED, request=request))
        await asyncio.wait_for(server._initializing.wait(), timeout=1)
        closing = tasks.create_task(_release(server, operation))
        await asyncio.sleep(0)
        assert not closing.done()
        assert request.session[SESSION_ID_KEY] == "caller-cookie"
        server._initialization_gate.set()
    assert request.session[SESSION_ID_KEY] == "session"
    assert server.state == {}
    assert server._session_episodes == {}
    # Only a close fences the session; retirement leaves refusing late requests to the checkpoint participant.
    assert server.session_is_closed("session") == (operation == "close")


@pytest.mark.parametrize("operation", ["close", "retire"])
async def test_failed_release_retains_state_and_identity_for_retry(operation: str) -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    server._fail_release = True
    with pytest.raises(RuntimeError, match="resource still running"):
        await _release(server, operation)
    assert server.state == {"session": 42}
    assert server._session_episodes == {"session": (SEED.episode_id, SEED.task_id)}
    assert not server.session_is_closed("session")
    server._fail_release = False
    await _release(server, operation)
    assert server.state == {}
    assert server._session_episodes == {}
    assert server.session_is_closed("session") == (operation == "close")


async def test_invalid_task_never_initializes_or_binds_the_cookie() -> None:
    server = _server()
    request = _request()
    with pytest.raises(ResourcesSeedRejectedError) as error:
        await server.seed_session(SEED.model_copy(update={"task_data": {"value": "not an integer"}}), request=request)
    assert error.value.status_code == 422
    assert request.session[SESSION_ID_KEY] == "caller-cookie"
    assert not server._initializing.is_set()
    assert server.state == {}
    assert server._session_episodes == {}


def test_inherited_seed_route_validates_task_data_and_preserves_retries() -> None:
    server = _server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json={"value": "invalid"}).status_code == 422
        assert server.state == {}
        assert client.post("/seed_session", json={"value": 7}).status_code == 200
        assert list(server.state.values()) == [7]

        seed = SEED.model_dump(mode="json")
        assert client.post("/seed_session", json=seed).status_code == 200
        server.state["session"] = 99
        assert client.post("/seed_session", json=seed).status_code == 200
        assert server.state["session"] == 99
        assert client.post("/seed_session", json={"value": 8}).status_code == 409
        assert client.post("/close_session", json=CLOSE.model_dump(mode="json")).status_code == 200
        assert "session" not in server.state
        assert client.post("/seed_session", json=seed).status_code == 409


@pytest.mark.parametrize("expose_tools_over_mcp", [False, True])
@pytest.mark.parametrize("missing", ["resources_session_id", "episode_id", "task_id", "task_data"])
def test_inherited_seed_route_rejects_partial_typed_requests(missing: str, expose_tools_over_mcp: bool) -> None:
    # MCP auto-exposure wraps /seed_session; the cookie arm's validator must survive the wrapper.
    server = _server(expose_tools_over_mcp=expose_tools_over_mcp)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    seed = SEED.model_dump(mode="json") | {"value": 7}
    del seed[missing]
    with TestClient(app) as client:
        assert client.post("/seed_session", json=seed).status_code == 422
        assert server.state == {}
        assert client.post("/seed_session", json=SEED.model_dump(mode="json")).status_code == 200
    assert server.state == {"session": 42}


class StatelessResourceServer(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0)


async def test_stateless_seed_keeps_the_body_only_call_and_permissive_http_schema() -> None:
    server = StatelessResourceServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    assert await server.seed_session(BaseSeedSessionRequest()) == BaseSeedSessionResponse()
    assert await server.seed_session(SEED) == ResourcesSeedSessionResponse(resources_session_id="session")
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=SEED.model_dump(mode="json")).status_code == 200
        # These were accepted as extra task fields before the shared lifecycle was introduced.
        assert client.post("/seed_session", json={"task_id": "task", "other": 42}).status_code == 200
        assert client.post("/seed_session", json={"resources_session_id": "row-7", "value": 1}).status_code == 200
        typed_looking = {"episode_id": {"rollout_id": "r"}, "task_id": {"taskset": "t", "task_id": "1"}}
        assert client.post("/seed_session", json=typed_looking).status_code == 200
        assert client.post("/close_session", json=CLOSE.model_dump(mode="json")).status_code == 200
        assert client.post("/seed_session", json=SEED.model_dump(mode="json")).status_code == 200
    assert server._session_episodes == {}


def test_incomplete_state_hooks_fail_at_class_definition() -> None:
    with pytest.raises(TypeError, match="has_session_state.*release_session_state"):

        class Incomplete(StatelessResourceServer):
            async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
                raise AssertionError("must not reach initialization")


async def test_exported_server_can_inherit_retirement_from_the_state_hooks() -> None:
    class ExportedResourceServer(AsyncResourceServer):
        checkpoint_mode: ClassVar[str] = "exported"

        async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
            raise AssertionError("export is outside this test")

        async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
            raise AssertionError("restore is outside this test")

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    server = ExportedResourceServer(config=_server().config, server_client=server_client)
    server.setup_webserver()
    await server.seed_session(SEED, request=_request())
    await server.retire_session_state(SEED.resources_session_id)
    assert server.state == {}
    # The participant also retires a session to undo a restore, and the same session may be installed again.
    assert not server.session_is_closed(SEED.resources_session_id)
    await server.seed_session(SEED, request=_request())
    assert server.state == {SEED.resources_session_id: 42}


async def test_cookie_reseed_releases_previous_resources_before_initializing() -> None:
    server = _server()
    request = _request()
    await server.seed_session(TaskSeed(value=1), request=request)
    assert server.released == []
    server._fail_release = True
    with pytest.raises(RuntimeError, match="resource still running"):
        await server.seed_session(TaskSeed(value=2), request=request)
    assert server.state == {"caller-cookie": 1}
    server._fail_release = False
    await server.seed_session(TaskSeed(value=2), request=request)
    assert server.released == ["caller-cookie"]
    assert server.state == {"caller-cookie": 2}


class PublishedPartialServer(AsyncResourceServer):
    fail_initialization: bool = True

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        await super().initialize_session_state(session_id, body)
        if self.fail_initialization:
            raise RuntimeError("upload unavailable")


async def test_failed_typed_seed_cleans_published_state_and_retry_does_not_conflict() -> None:
    server = PublishedPartialServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    with pytest.raises(RuntimeError, match="upload unavailable"):
        await server.seed_session(SEED, request=_request())
    assert server.state == {}
    assert server.session_identity("session") is None
    server.fail_initialization = False
    await server.seed_session(SEED, request=_request())
    assert server.state == {"session": 42}


async def test_failed_initialization_cleanup_keeps_owner_and_retries_before_allocating() -> None:
    server = PublishedPartialServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    server._fail_release = True
    with pytest.raises(RuntimeError, match="upload unavailable"):
        await server.seed_session(SEED, request=_request())
    assert server.session_identity("session") == (SEED.episode_id, SEED.task_id)
    with pytest.raises(ResourcesSessionConflictError):
        await server.close_resources_session(CLOSE.model_copy(update={"episode_id": EpisodeId(rollout_id="other")}))
    with pytest.raises(ResourcesSessionConflictError):
        await server.seed_session(
            SEED.model_copy(update={"episode_id": EpisodeId(rollout_id="other")}), request=_request()
        )
    server.fail_initialization = False
    server._fail_release = False
    await server.seed_session(SEED, request=_request())
    assert server.released == ["session"]
    assert server.state == {"session": 42}
    assert server._initializing_sessions == {}


async def test_cancelled_initialization_releases_partial_state() -> None:
    class CancelledInitializationServer(AsyncResourceServer):
        async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
            self.state[session_id] = 1
            self._initializing.set()
            await asyncio.Event().wait()

    server = CancelledInitializationServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    task = asyncio.create_task(server.seed_session(SEED, request=_request()))
    await asyncio.wait_for(server._initializing.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server.state == {}
    assert server.session_identity("session") is None


async def test_failed_retire_keeps_ownership_for_the_close_that_follows() -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    server._fail_release = True
    with pytest.raises(RuntimeError):
        await server.retire_session_state("session")
    server._fail_release = False
    with pytest.raises(ResourcesSessionConflictError):
        await server.close_resources_session(CLOSE.model_copy(update={"episode_id": EpisodeId(rollout_id="other")}))
    assert server.state == {"session": 42}
    await server.close_resources_session(CLOSE)
    assert server.state == {}
    assert server.session_is_closed("session")


async def test_restore_identity_replays_the_continuation_seed_without_resetting_state() -> None:
    server = _server()
    server.state["session"] = 99
    continuation = SEED.episode_id.model_copy(update={"attempt": SEED.episode_id.attempt + 1})
    server.restore_session_identity("session", episode_id=continuation, task_id=SEED.task_id)
    request = _request()
    await server.seed_session(SEED.model_copy(update={"episode_id": continuation}), request=request)
    assert server.state == {"session": 99}
    assert request.session[SESSION_ID_KEY] == "session"
    with pytest.raises(ResourcesSessionConflictError):
        await server.close_resources_session(CLOSE)
    await server.close_resources_session(CLOSE.model_copy(update={"episode_id": continuation}))
    assert server.state == {}


async def test_restore_batch_conflict_installs_no_partial_bindings() -> None:
    server = _server()
    await server.close_resources_session(CLOSE)
    with pytest.raises(ResourcesSessionConflictError):
        server.restore_session_identities({"other": (SEED.episode_id, SEED.task_id), "session": None})
    assert server.session_identity("other") is None


async def test_subclass_without_super_post_init_still_initializes_lifecycle() -> None:
    class OwnPostInit(AsyncResourceServer):
        def model_post_init(self, context: object, /) -> None:
            pass

    server = OwnPostInit(config=_server().config, server_client=MagicMock(spec=ServerClient))
    await server.seed_session(SEED, request=_request())
    await server.close_resources_session(CLOSE)
    assert server.state == {}
    assert server.session_is_closed("session")


@pytest.mark.parametrize("name", ["initialize_session_state", "release_session_state"])
def test_synchronous_state_hook_is_rejected_before_instantiation(name: str) -> None:
    with pytest.raises(TypeError, match=f"{name} must be async"):
        type("SynchronousHook", (AsyncResourceServer,), {name: lambda *args: None})


def test_async_has_state_hook_is_rejected_before_instantiation() -> None:
    with pytest.raises(TypeError, match="has_session_state must be synchronous"):

        class AsyncHasState(AsyncResourceServer):
            async def has_session_state(self, session_id: str) -> bool:
                return True


class RequestFirstSeedServer(StatelessResourceServer):
    seeded: dict[str, dict] = Field(default_factory=dict)

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        self.seeded[request.session[SESSION_ID_KEY]] = body.model_dump()
        return BaseSeedSessionResponse()


def test_request_first_seed_override_is_served_unchanged() -> None:
    server = RequestFirstSeedServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json={"verifier_metadata": {"city": "Paris"}})
    assert response.status_code == 200
    assert len(server.seeded) == 1


def test_conformance_rejects_seed_override_that_bypasses_base_ownership() -> None:
    class UnboundSeedServer(AsyncResourceServer):
        async def seed_session(
            self, request: Request, body: ResourcesSeedSessionRequest
        ) -> ResourcesSeedSessionResponse:
            self.state[body.resources_session_id] = 42
            request.session[SESSION_ID_KEY] = body.resources_session_id
            return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)

    server = UnboundSeedServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    with pytest.raises(AssertionError, match="close of active session.*another episode"):
        check_resources_session_contract(
            server.setup_webserver(), SEED, keeps_state=True, holds_state=server.has_session_state
        )


def test_conformance_accepts_seed_override_delegating_to_base() -> None:
    class DelegatingSeedServer(AsyncResourceServer):
        async def seed_session(
            self, request: Request, body: ResourcesSeedSessionRequest
        ) -> ResourcesSeedSessionResponse:
            return await super().seed_session(body, request=request)

    server = DelegatingSeedServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    check_resources_session_contract(
        server.setup_webserver(), SEED, keeps_state=True, holds_state=server.has_session_state
    )


async def test_verifying_serializes_close_and_retains_evidence_for_retries() -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    async with server.verifying("session") as verification:
        assert verification.is_live and not verification.is_lost
        verification.retain({"answer": server.state["session"]})
        close = asyncio.create_task(server.close_resources_session(CLOSE))
        await asyncio.sleep(0)
        assert not close.done()
        assert server.state == {"session": 42}
    await asyncio.wait_for(close, timeout=1)
    assert server.state == {}
    async with server.verifying("session") as verification:
        assert verification.is_lost
        assert verification.retained is None


async def test_verifying_retains_snapshot_on_score_error_and_release_failure(monkeypatch) -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server._fail_release = True
    with pytest.raises(RuntimeError, match="resource still running"):
        async with server.verifying("session") as verification:
            verification.retain(42)
            raise ValueError("judge unavailable")
    clock[0] += 1000
    assert server.session_is_verified("session")
    with pytest.raises(ResourcesSessionConflictError):
        await server.close_resources_session(CLOSE.model_copy(update={"episode_id": EpisodeId(rollout_id="other")}))
    server._fail_release = False
    async with server.verifying("session") as verification:
        assert not verification.is_live and verification.retained == 42
    assert server.state == {}
    assert server._session_episodes == {}
    assert server.session_identity("session") == (SEED.episode_id, SEED.task_id)
    clock[0] += 200
    async with server.verifying("session") as verification:
        assert verification.retained == 42
    clock[0] += 100
    assert not server.session_is_verified("session")
    async with server.verifying("session") as verification:
        assert verification.is_lost


async def test_scoring_exception_still_releases_and_retains_payload() -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    with pytest.raises(ValueError, match="judge unavailable"):
        async with server.verifying("session") as verification:
            verification.retain("captured before judge")
            raise ValueError("judge unavailable")
    assert server.state == {}
    async with server.verifying("session") as verification:
        assert not verification.is_live
        assert verification.retained == "captured before judge"


async def test_request_scored_unknown_verify_is_opt_in_but_closed_is_always_lost() -> None:
    server = _server()
    async with server.verifying("session") as verification:
        assert verification.is_lost
    assert server.released == []
    async with server.verifying("session", allow_missing=True) as verification:
        assert not verification.is_lost
        assert not verification.is_live
    assert server.released == ["session"]
    await server.close_resources_session(CLOSE)
    async with server.verifying("session", allow_missing=True) as verification:
        assert verification.is_lost


async def test_verification_retry_updates_payload_without_extending_expiry(monkeypatch) -> None:
    server = _server()
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    await server.seed_session(SEED, request=_request())
    async with server.verifying("session") as verification:
        verification.retain("first")
    clock[0] = 200.0
    async with server.verifying("session") as verification:
        assert verification.retained == "first"
        verification.retain("refreshed")
    async with server.verifying("session") as verification:
        assert verification.retained == "refreshed"
    clock[0] = 400.0
    assert not server.session_is_verified("session")


async def test_retained_evidence_wins_over_a_mutable_archive_left_for_shutdown() -> None:
    class ArchivingServer(AsyncResourceServer):
        async def release_session_state(self, session_id: str) -> None:
            self.released.append(session_id)

    server = ArchivingServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    await server.seed_session(SEED, request=_request())
    async with server.verifying("session") as verification:
        verification.retain(server.state["session"])
    server.state["session"] = 999
    async with server.verifying("session") as verification:
        assert not verification.is_live
        assert verification.retained == 42
    await server.close_resources_session(CLOSE)
    async with server.verifying("session") as verification:
        assert verification.is_lost
        assert verification.retained is None


@pytest.mark.parametrize("typed", [False, True])
async def test_partially_initialized_state_cannot_be_verified_even_when_missing_is_allowed(typed: bool) -> None:
    server = PublishedPartialServer(config=_server().config, server_client=MagicMock(spec=ServerClient))
    server._fail_release = True
    seed = SEED if typed else TaskSeed(value=42)
    session_id = "session" if typed else "caller-cookie"
    with pytest.raises(RuntimeError, match="upload unavailable"):
        await server.seed_session(seed, request=_request())
    assert server.has_session_state(session_id)
    async with server.verifying(session_id, allow_missing=True) as verification:
        assert verification.is_lost
    assert server.has_session_state(session_id)


async def test_failed_retirement_after_verify_keeps_retained_owner() -> None:
    server = _server()
    await server.seed_session(SEED, request=_request())
    await _verify_and_release(server)
    server._fail_release = True
    with pytest.raises(RuntimeError):
        await server.retire_session_state("session")
    with pytest.raises(ResourcesSessionConflictError):
        await server.close_resources_session(CLOSE.model_copy(update={"episode_id": EpisodeId(rollout_id="other")}))
    server._fail_release = False
    await server.retire_session_state("session")
    assert not server.session_is_verified("session")
    assert not server.session_is_closed("session")
    server.state["session"] = 17
    server.restore_session_identity("session", episode_id=SEED.episode_id, task_id=SEED.task_id)
    await server.seed_session(SEED, request=_request())
    assert server.state["session"] == 17


async def test_a_retry_finishing_after_expiry_does_not_resurrect_an_evicted_receipt(monkeypatch) -> None:
    server = _server()
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    await server.seed_session(SEED, request=_request())
    async with server.verifying("session") as verification:
        verification.retain(42)
    async with server.verifying("session") as verification:
        assert verification.retained == 42
        clock[0] += server.config.session_close_retry_window_seconds
        # Expiring records is global, so a request for another session can evict this one while scoring awaits.
        assert not server.session_is_verified("unrelated")
        await asyncio.sleep(0)
    assert not server.session_is_verified("session")
    assert server.session_identity("session") is None
    async with server.verifying("session") as verification:
        assert verification.is_lost
