# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sidecars from ``provider_options["sidecars"]`` for rollout sandboxes, through real server apps and AsyncSandbox."""

import asyncio
import contextvars
import logging
import subprocess
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

import nemo_gym.sandbox.rollout_sidecars as rollout_sidecars
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.mcp_auto_exposure import harvest_tools
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.sandbox import AsyncSandbox, SandboxHandle, SandboxSpec
from nemo_gym.sandbox.api import _SPEC_TRANSFORMS, register_spec_transform
from nemo_gym.sandbox.providers import SandboxExecResult, SandboxSidecarSpec
from nemo_gym.sandbox.rollout_sidecars import (
    SeedSessionMiddleware,
    apply_rollout_sidecars,
    is_seeding_session,
    register_rollout_sidecars,
    seeding_session,
)
from nemo_gym.server_utils import ServerClient


RECORDER = {"name": "recorder", "image": "registry/recorder:1", "resources": {"cpu": 0.1, "memory_mib": 1024}}


class SidecarProvider:
    """A fake provider with sidecar support that records every spec it is asked to create."""

    name = "sidecar_fake"

    def __init__(self) -> None:
        self.created: list[SandboxSpec] = []

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created.append(spec)
        return SandboxHandle(sandbox_id=f"sb-{len(self.created)}", provider_name=self.name, raw=None)

    async def sidecar_exec(self, handle, sidecar, argv, *, timeout_s, on_stdout=None) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def download_sidecar_file(self, handle, sidecar, source_path, target_path: Path) -> None:
        return None

    async def serialize_handle(self, handle, *, scope=None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)

    async def close(self, handle: SandboxHandle) -> None:
        return None

    async def aclose(self) -> None:
        return None


class PlainProvider(SidecarProvider):
    """A provider without sidecar support."""

    sidecar_exec = None  # type: ignore[assignment]
    download_sidecar_file = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    """Start every test from a process with nothing registered, and restore the process-wide state afterwards."""
    saved_transforms, saved_mode = list(_SPEC_TRANSFORMS), rollout_sidecars._mode
    _SPEC_TRANSFORMS.clear()
    rollout_sidecars._mode = None
    yield
    _SPEC_TRANSFORMS[:] = saved_transforms
    rollout_sidecars._mode = saved_mode


def _spec(**kwargs: Any) -> SandboxSpec:
    return SandboxSpec(image="task:latest", provider_options={"keep": True, "sidecars": [RECORDER]}, **kwargs)


def _role_spec(role: str) -> SandboxSpec:
    return SandboxSpec(image="task:latest", metadata={"role": role}, provider_options={"sidecars": [RECORDER]})


def _sidecars_by_role(provider: SidecarProvider) -> dict[str, tuple[str, ...]]:
    return {spec.metadata["role"]: tuple(sidecar.name for sidecar in spec.sidecars) for spec in provider.created}


VERIFY_BODY = {
    "responses_create_params": {"input": []},
    "response": {
        "id": "",
        "created_at": 0,
        "model": "",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}


def _resources_server(server_cls: type[SimpleResourcesServer]) -> SimpleResourcesServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
    return server_cls(config=config, server_client=MagicMock(spec=ServerClient))


def _agent(agent_cls: type[SimpleResponsesAPIAgent]) -> SimpleResponsesAPIAgent:
    config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")
    return agent_cls(config=config, server_client=MagicMock(spec=ServerClient))


class _NoopResourcesServer(SimpleResourcesServer):
    async def verify(self, body):
        pass


class _NoopAgent(SimpleResponsesAPIAgent):
    async def responses(self, body=None):
        pass

    async def run(self, body=None):
        pass


# The generic spec-transform hook


async def test_spec_transforms_run_in_order_once_each_before_the_provider_creates() -> None:
    def tag(suffix: str) -> Any:
        def transform(spec: SandboxSpec) -> SandboxSpec:
            return SandboxSpec(image=f"{spec.image}+{suffix}")

        return transform

    first, second = tag("a"), tag("b")
    for transform in (first, second, first):
        register_spec_transform(transform)
    provider = SidecarProvider()
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="task"))

    assert [spec.image for spec in provider.created] == ["task+a+b"]
    assert sandbox._spec is provider.created[0]  # serialize() and sidecar() see the transformed spec


# Registration


def test_importing_the_server_bases_registers_nothing() -> None:
    """Environment servers and scripts import these modules but must pass provider_options through unchanged."""
    code = (
        "import nemo_gym.base_responses_api_agent, nemo_gym.base_resources_server, nemo_gym.mcp_auto_exposure\n"
        "import nemo_gym.sandbox.rollout_sidecars as rollout_sidecars\n"
        "from nemo_gym.sandbox.api import _SPEC_TRANSFORMS\n"
        "print(len(_SPEC_TRANSFORMS), rollout_sidecars._mode)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "0 None"


@pytest.mark.parametrize(
    ("build", "mode"),
    [(lambda: _resources_server(_NoopResourcesServer), "scoped"), (lambda: _agent(_NoopAgent), "unscoped")],
    ids=["resources_server", "agent"],
)
def test_building_a_server_app_registers_the_transform_once(build: Any, mode: str) -> None:
    for _ in range(2):
        build().setup_webserver()

    assert _SPEC_TRANSFORMS == [apply_rollout_sidecars]
    assert rollout_sidecars._mode == mode


@pytest.mark.parametrize("agent_first", [True, False])
async def test_scoped_wins_when_both_server_kinds_register_in_one_process(
    agent_first: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """A verification sandbox never gets the sidecars, whichever app is built first; the conflict is logged."""
    builders = [lambda: _agent(_NoopAgent), lambda: _resources_server(_NoopResourcesServer)]
    with caplog.at_level(logging.WARNING, logger=rollout_sidecars.__name__):
        for build in builders if agent_first else reversed(builders):
            build().setup_webserver()

    assert rollout_sidecars._mode == "scoped"
    assert _SPEC_TRANSFORMS == [apply_rollout_sidecars]
    assert "Both a resources server and an agent server registered rollout sidecars" in caplog.text
    provider = SidecarProvider()
    await AsyncSandbox(provider).start(_spec())
    assert provider.created[0].sidecars == ()


# The transform in AsyncSandbox.start


async def test_unregistered_process_passes_the_option_to_the_provider() -> None:
    """With nothing registered the option reaches the provider unchanged, so one that validates its options rejects it."""
    provider = SidecarProvider()
    with seeding_session():
        await AsyncSandbox(provider).start(_spec())

    assert provider.created[0].provider_options == {"keep": True, "sidecars": [RECORDER]}
    assert provider.created[0].sidecars == ()


async def test_seed_session_sandbox_gets_the_option_sidecars_and_serialize_lists_them() -> None:
    register_rollout_sidecars(scoped=True)
    provider = SidecarProvider()
    options = {"keep": True, "sidecars": [dict(RECORDER)]}
    with seeding_session():
        sandbox = await AsyncSandbox(provider).start(SandboxSpec(image="task:latest", provider_options=options))

    assert provider.created[0].sidecars == (SandboxSidecarSpec(**RECORDER),)
    assert provider.created[0].provider_options == {"keep": True}  # the option never reaches the provider
    assert options == {"keep": True, "sidecars": [RECORDER]}  # the configuration is not modified
    assert sandbox.sidecar("recorder").name == "recorder"
    assert (await sandbox.serialize())["sidecars"] == [{"name": "recorder", "image": RECORDER["image"]}]


async def test_scoped_sandbox_outside_seed_session_gets_no_sidecars() -> None:
    register_rollout_sidecars(scoped=True)
    provider = PlainProvider()  # e.g. a verification sandbox, even on a provider without sidecars
    await AsyncSandbox(provider).start(_spec())

    assert provider.created[0].sidecars == ()
    assert provider.created[0].provider_options == {"keep": True}


async def test_explicit_sidecars_are_kept_next_to_the_option() -> None:
    register_rollout_sidecars(scoped=True)
    provider = SidecarProvider()
    with seeding_session():
        await AsyncSandbox(provider).start(_spec(sidecars=(SandboxSidecarSpec(name="logs", image="logs:1"),)))

    assert [s.name for s in provider.created[0].sidecars] == ["logs", "recorder"]


@pytest.mark.parametrize("scoped", [True, False])
@pytest.mark.parametrize("seeding", [True, False])
async def test_null_provider_options_reach_the_provider_unchanged(scoped: bool, seeding: bool) -> None:
    """Servers pass ``sandbox_config.get("provider_options", {})``, which is None for a null in YAML."""
    register_rollout_sidecars(scoped=scoped)
    provider = SidecarProvider()
    with seeding_session() if seeding else nullcontext():
        await AsyncSandbox(provider).start(SandboxSpec(image="task:latest", provider_options=None))

    assert provider.created[0].provider_options is None
    assert provider.created[0].sidecars == ()


@pytest.mark.parametrize("scoped", [True, False])
@pytest.mark.parametrize("value", [None, [], ()])
async def test_null_or_empty_option_adds_no_sidecars(scoped: bool, value: Any) -> None:
    register_rollout_sidecars(scoped=scoped)
    provider = PlainProvider()  # no sidecar support, so an added sidecar would fail the start
    with seeding_session():
        await AsyncSandbox(provider).start(
            SandboxSpec(image="task:latest", provider_options={"keep": True, "sidecars": value})
        )

    assert provider.created[0].sidecars == ()
    assert provider.created[0].provider_options == {"keep": True}


@pytest.mark.parametrize("scoped", [True, False])
@pytest.mark.parametrize("seeding", [True, False])
@pytest.mark.parametrize("value", [RECORDER, "recorder", ["recorder"], 1])
async def test_malformed_option_fails_with_an_actionable_error(value: Any, seeding: bool, scoped: bool) -> None:
    """A mapping instead of a list fails at the first sandbox, also out of scope where the option would be dropped."""
    register_rollout_sidecars(scoped=scoped)
    provider = SidecarProvider()
    with seeding_session() if seeding else nullcontext():
        with pytest.raises(ValueError, match=r"provider_options\['sidecars'\] must be a list of sidecar specs"):
            await AsyncSandbox(provider).start(SandboxSpec(image="task:latest", provider_options={"sidecars": value}))

    assert provider.created == []


# Resources server: scoped to /seed_session


async def _seen_while_serving(path: str) -> tuple[str, bool]:
    seen = []

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        seen.append((scope["path"], is_seeding_session()))

    await RolloutContextMiddleware(SeedSessionMiddleware(app))({"type": "http", "path": path}, None, None)
    return seen[0]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/seed_session", ("/seed_session", True)),
        ("/ng-rollout/0-0/seed_session", ("/seed_session", True)),
        ("/verify", ("/verify", False)),
        ("/ng-rollout/0-0/verify", ("/verify", False)),
    ],
)
async def test_middleware_scopes_only_seed_session(path: str, expected: tuple[str, bool]) -> None:
    assert await _seen_while_serving(path) == expected
    assert not is_seeding_session()


def test_resources_server_seed_sandbox_gets_the_sidecar_and_verify_sandbox_does_not() -> None:
    """Through the full middleware stack and a real AsyncSandbox.start, with and without the rollout prefix."""
    provider = SidecarProvider()

    class Server(SimpleResourcesServer):
        async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
            await AsyncSandbox(provider, owns_provider=False).start(_role_spec(f"seed-{len(provider.created)}"))
            return BaseSeedSessionResponse()

        async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
            await AsyncSandbox(provider, owns_provider=False).start(_role_spec(f"verify-{len(provider.created)}"))
            return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    client = TestClient(_resources_server(Server).setup_webserver())
    assert client.post("/seed_session", json={}).status_code == 200
    assert client.post("/verify", json=VERIFY_BODY).status_code == 200
    assert client.post("/ng-rollout/r-1/seed_session", json={}).status_code == 200
    assert client.post("/ng-rollout/r-1/verify", json=VERIFY_BODY).status_code == 200

    assert _sidecars_by_role(provider) == {
        "seed-0": ("recorder",),
        "verify-1": (),
        "seed-2": ("recorder",),
        "verify-3": (),
    }
    assert all("sidecars" not in spec.provider_options for spec in provider.created)


async def test_overlapping_seed_session_and_verify_do_not_share_the_scope() -> None:
    """/verify creates its sandbox while a /seed_session on the same app is suspended mid-request."""
    provider = SidecarProvider()
    seed_entered, verify_done = asyncio.Event(), asyncio.Event()

    class Server(SimpleResourcesServer):
        async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
            seed_entered.set()
            await verify_done.wait()
            await AsyncSandbox(provider, owns_provider=False).start(_role_spec("seed"))
            return BaseSeedSessionResponse()

        async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
            await seed_entered.wait()
            await AsyncSandbox(provider, owns_provider=False).start(_role_spec("verify"))
            verify_done.set()
            return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    transport = httpx.ASGITransport(app=_resources_server(Server).setup_webserver())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        seed, verify = await asyncio.gather(
            client.post("/ng-rollout/r-1/seed_session", json={}),
            client.post("/ng-rollout/r-1/verify", json=VERIFY_BODY),
        )

    assert (seed.status_code, verify.status_code) == (200, 200)
    assert [spec.metadata["role"] for spec in provider.created] == ["verify", "seed"]
    assert _sidecars_by_role(provider) == {"verify": (), "seed": ("recorder",)}


async def test_task_spawned_during_seed_session_keeps_the_scope_after_the_response() -> None:
    """Intended inheritance: a task spawned in /seed_session copies the request's context, so a sandbox it starts
    after the response still gets the sidecars. Spawn it with a fresh ``contextvars.Context()`` to opt out."""
    provider = SidecarProvider()
    release = asyncio.Event()
    background: list[asyncio.Task] = []

    class Server(SimpleResourcesServer):
        async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
            async def start_later(role: str) -> None:
                await release.wait()
                await AsyncSandbox(provider, owns_provider=False).start(_role_spec(role))

            background.append(asyncio.create_task(start_later("inherits")))
            background.append(asyncio.create_task(start_later("fresh-context"), context=contextvars.Context()))
            return BaseSeedSessionResponse()

        async def verify(self, body):
            pass

    transport = httpx.ASGITransport(app=_resources_server(Server).setup_webserver())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/seed_session", json={})).status_code == 200
    release.set()  # the /seed_session request is over
    await asyncio.gather(*background)

    assert _sidecars_by_role(provider) == {"inherits": ("recorder",), "fresh-context": ()}


def test_resources_server_with_the_middleware_can_expose_tools_over_mcp() -> None:
    """MCP tool exposure refuses middleware it does not know; the /seed_session middleware is allowlisted."""
    server = _resources_server(_NoopResourcesServer)
    app = server.setup_webserver()

    assert SeedSessionMiddleware in [m.cls for m in app.user_middleware]
    harvest_tools(app, server)


# Agent server: unscoped


def test_agent_process_gives_every_sandbox_the_sidecar() -> None:
    """An agent only starts sandboxes its harness runs in, so each gets the sidecars, with no /seed_session."""
    provider = SidecarProvider()

    class Agent(SimpleResponsesAPIAgent):
        async def responses(self, body=None):
            pass

        async def run(self, body: BaseRunRequest) -> BaseVerifyResponse:
            for role in ("harness", "second"):
                await AsyncSandbox(provider, owns_provider=False).start(_role_spec(role))
            return BaseVerifyResponse(**VERIFY_BODY, reward=1.0)

    client = TestClient(_agent(Agent).setup_webserver())
    assert client.post("/run", json={"responses_create_params": {"input": []}}).status_code == 200

    assert _sidecars_by_role(provider) == {"harness": ("recorder",), "second": ("recorder",)}
    assert all("sidecars" not in spec.provider_options for spec in provider.created)
