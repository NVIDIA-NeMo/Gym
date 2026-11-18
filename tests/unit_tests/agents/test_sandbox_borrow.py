# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent borrows a lent sandbox: it works in it as the access says, cannot stop it, and releases its client."""

from typing import Any
from unittest.mock import MagicMock

import anyio
import pytest

from nemo_gym.agents.utils import AgentSandboxConfig, BorrowedSandbox, BorrowedSandboxError, borrow
from nemo_gym.agents.utils import sandbox_borrow as borrow_module
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess, SandboxSpec
from nemo_gym.server_runtime import OwnedSandboxes
from tests.unit_tests.server_runtime.test_checkpoint_sandbox import FakeSnapshotProvider


class _RecordingProvider(FakeSnapshotProvider):
    """Records the user, directory and environment each command runs with, and whether its client was closed."""

    def __init__(self) -> None:
        super().__init__()
        self.commands: list[dict[str, Any]] = []
        self.client_closed = False

    async def exec(self, handle: Any, command: str, **kwargs: Any) -> Any:
        self.commands.append({"command": command, **kwargs})
        return await super().exec(handle, command, **kwargs)

    async def aclose(self) -> None:
        self.client_closed = True


@pytest.fixture
def provider(monkeypatch) -> _RecordingProvider:
    """One backend for owner and borrower, as two processes would share one sandbox service."""
    shared = _RecordingProvider()
    borrower_clients: list[_RecordingProvider] = []

    def create(config: dict[str, Any]) -> _RecordingProvider:
        # The borrower gets its own client of the same backend.
        client = _RecordingProvider()
        client.boxes, client.commands = shared.boxes, shared.commands
        borrower_clients.append(client)
        return client

    monkeypatch.setattr(borrow_module, "create_provider", create)
    monkeypatch.setattr(borrow_module, "resolve_provider_config", lambda reference, configs: {"fake": {}})
    shared.borrower_clients = borrower_clients
    return shared


async def _lend(provider: _RecordingProvider, **access: Any) -> SandboxAccess:
    owned = OwnedSandboxes(provider, state_dir=None, server_name="owner", provider_config_ref="fake_sandbox")
    await owned.create("session", SandboxSpec(image="image", ttl_s=60))
    lent = await owned.access("session", workdir="/repo")
    return lent.model_copy(update=access)


async def test_a_borrower_runs_commands_as_the_access_says(provider) -> None:
    access = await _lend(provider, user="agent", env={"TOKEN": "lent", "MODE": "lent"})

    sandbox = await borrow(access, named_configs={})
    await sandbox.exec("make test", env={"MODE": "command"})

    (command,) = provider.commands
    assert command["cwd"] == "/repo"
    assert command["user"] == "agent"
    assert command["env"] == {"TOKEN": "lent", "MODE": "command"}


async def test_a_borrower_cannot_stop_the_sandbox() -> None:
    sandbox = BorrowedSandbox(FakeSnapshotProvider(), SandboxSpec(), owns_provider=False)

    with pytest.raises(BorrowedSandboxError, match="only the server that lent it may stop it"):
        await sandbox.stop()


async def test_leaving_the_block_releases_the_client_and_keeps_the_sandbox_running(provider) -> None:
    access = await _lend(provider)

    async with await borrow(access, named_configs={}) as sandbox:
        await sandbox.exec("true")

    (client,) = provider.borrower_clients
    assert client.client_closed
    assert [box["state"] for box in provider.boxes.values()] == ["running"]


async def test_a_borrower_passes_on_only_operate_scope(provider) -> None:
    sandbox = await borrow(await _lend(provider), named_configs={})

    assert (await sandbox.serialize())["scope"] == "operate"
    with pytest.raises(ValueError, match="only operate scope"):
        await sandbox.serialize(scope="full")


async def test_a_failed_connect_releases_the_client(provider) -> None:
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="fake_sandbox", descriptor={"sandbox_id": "gone"}),
        workdir="/repo",
    )

    with pytest.raises(RuntimeError, match="gone"):
        await borrow(access, named_configs={})

    (client,) = provider.borrower_clients
    assert client.client_closed


class TestAgentSandboxConfig:
    def test_the_keys_agent_recipes_set_today_load(self) -> None:
        config = AgentSandboxConfig.model_validate(
            {
                "image": "harness:1",
                "workdir": "/app",
                "files": {"/app/AGENTS.md": "be brief"},
                "entrypoint": ["sleep", "infinity"],
                "derive_cpu_env": False,
                "ttl_s": 3600,
                "ready_timeout_s": 300,
                "resources": {"cpu": 2},
                "env": {"A": "1"},
                "metadata": {"team": "gym"},
                "provider_options": {"volumes": []},
                "network": "inherit",
            }
        )

        assert (config.image, config.workdir, config.entrypoint) == ("harness:1", "/app", ["sleep", "infinity"])

    def test_an_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            AgentSandboxConfig.model_validate({"imagee": "harness:1"})

    def test_the_spec_carries_the_agents_own_image_files_and_entrypoint(self) -> None:
        config = AgentSandboxConfig(
            image="harness:1",
            workdir="/app",
            files={"/app/a": "a"},
            entrypoint=["sleep", "infinity"],
            resources={"cpu": 2},
            env={"OMP_NUM_THREADS": "8"},
            derive_cpu_env=True,
        )

        spec = config.spec({"fake": {}}, files={"/app/b": "b"})

        assert (spec.image, spec.workdir, spec.entrypoint) == ("harness:1", "/app", ["sleep", "infinity"])
        assert spec.files == {"/app/a": "a", "/app/b": "b"}
        assert spec.env["OMP_NUM_THREADS"] == "8", "the recipe's own value wins over the CPU caps"
        assert spec.env["GOMAXPROCS"] == "2"

    def test_the_cpu_caps_are_off_unless_asked_for(self) -> None:
        # Most agents set no CPU caps today, so moving onto this config must not add them.
        spec = AgentSandboxConfig(resources={"cpu": 2}).spec({"fake": {}})

        assert spec.env == {}

    def test_the_user_and_ports_reach_the_spec(self) -> None:
        spec = AgentSandboxConfig(user="agent", ports=[8080]).spec({"fake": {}})

        assert (spec.user, list(spec.ports)) == ("agent", [8080])


class _SlowClient(_RecordingProvider):
    """A provider client whose connect and close yield to the loop, as real HTTP clients do."""

    async def connect(self, descriptor: Any) -> Any:
        await anyio.sleep(0)
        return await super().connect(descriptor)

    async def aclose(self) -> None:
        await anyio.sleep(0)
        self.client_closed = True


@pytest.fixture
def slow_clients(monkeypatch, provider) -> list[_SlowClient]:
    clients: list[_SlowClient] = []

    def create(config: dict[str, Any]) -> _SlowClient:
        client = _SlowClient()
        client.boxes, client.commands = provider.boxes, provider.commands
        clients.append(client)
        return client

    monkeypatch.setattr(borrow_module, "create_provider", create)
    return clients


class TestReleaseUnderCancellation:
    """A client that disconnects cancels its request's cancel scope; the borrower's client must still be released."""

    async def test_leaving_the_block_in_a_cancelled_request_still_releases_the_client(
        self, provider, slow_clients
    ) -> None:
        access = await _lend(provider)
        sandbox = await borrow(access, named_configs={})

        with anyio.CancelScope() as request_scope:
            request_scope.cancel()
            async with sandbox:
                pass

        (client,) = slow_clients
        assert client.client_closed

    async def test_a_cancelled_connect_still_releases_the_client(self, provider, slow_clients) -> None:
        access = await _lend(provider)

        with anyio.CancelScope() as request_scope:
            request_scope.cancel()
            await borrow(access, named_configs={})

        (client,) = slow_clients
        assert client.client_closed


@pytest.mark.parametrize(
    ("operation", "verb"),
    [
        (lambda sandbox: sandbox.start(), "start"),
        (lambda sandbox: sandbox.start_with_setup(None, lambda _: None), "start"),
        (lambda sandbox: sandbox.pause(), "pause"),
        (lambda sandbox: sandbox.resume(), "resume"),
        (lambda sandbox: sandbox.snapshot(name="mine"), "snapshot"),
    ],
)
async def test_the_owners_lifecycle_operations_refuse(provider, operation, verb) -> None:
    sandbox = await borrow(await _lend(provider), named_configs={})

    with pytest.raises(BorrowedSandboxError, match=f"only the server that lent it may {verb} it"):
        await operation(sandbox)

    assert [box["state"] for box in provider.boxes.values()] == ["running"]
    assert not provider.snapshots


class _PtyProvider(_RecordingProvider):
    """Records the spec of every terminal session it opens."""

    def __init__(self) -> None:
        super().__init__()
        self.pty_specs: list[Any] = []

    async def create_pty(self, handle: Any, spec: Any) -> Any:
        self.pty_specs.append(spec)
        return MagicMock(closed=False)


async def test_a_terminal_session_gets_the_access_env_under_its_own(monkeypatch) -> None:
    pty_provider = _PtyProvider()
    owner = _PtyProvider()
    monkeypatch.setattr(borrow_module, "create_provider", lambda config: pty_provider)
    monkeypatch.setattr(borrow_module, "resolve_provider_config", lambda reference, configs: {"fake": {}})
    pty_provider.boxes = owner.boxes
    access = await _lend(owner, user="agent", env={"TOKEN": "lent", "MODE": "lent"})
    sandbox = await borrow(access, named_configs={})

    await sandbox.pty.create(env={"MODE": "session"})

    (spec,) = pty_provider.pty_specs
    assert spec.env == {"TOKEN": "lent", "MODE": "session"}
    assert (spec.user, spec.cwd) == ("agent", "/repo")


async def test_a_borrower_passes_its_access_on_to_borrow_again(provider) -> None:
    sandbox = await borrow(await _lend(provider, user="agent", env={"TOKEN": "lent"}), named_configs={})

    passed_on = await sandbox.access()
    again = await borrow(passed_on, named_configs={})
    await again.exec("true")

    assert passed_on.connection.descriptor["scope"] == "operate"
    assert passed_on.connection.provider_config_ref == "fake_sandbox"
    assert (passed_on.workdir, passed_on.user, passed_on.env) == ("/repo", "agent", {"TOKEN": "lent"})
    assert provider.commands[-1]["env"] == {"TOKEN": "lent"}


async def test_a_one_shot_terminal_command_gets_the_access_env_under_its_own(provider, monkeypatch) -> None:
    from nemo_gym.sandbox import SandboxPty

    sandbox = await borrow(await _lend(provider, env={"TOKEN": "lent", "MODE": "lent"}), named_configs={})
    opened: list[dict[str, Any]] = []

    async def create(self, command=None, **kwargs):
        opened.append(kwargs)
        raise RuntimeError("stop here: the session's env is what this test checks")

    monkeypatch.setattr(SandboxPty, "create", create)
    with pytest.raises(RuntimeError, match="stop here"):
        await sandbox.pty.exec("make test", env={"MODE": "command"})

    assert opened[0]["env"] == {"TOKEN": "lent", "MODE": "command"}
