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

"""Sidecars on OpenSandbox: each one is a sandbox of its own, in the main sandbox's placement group."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


pytestmark = pytest.mark.sandbox

pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")
pytest.importorskip("opensandbox", reason="opensandbox optional sandbox dependency is not installed")

from nemo_gym.core.failure_kinds import PROVIDER_SETUP_FAILED
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import PLACEMENT_GROUP_KEY, SandboxSpec, SupportsSandboxSidecars
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


class FakeSandbox:
    """The SDK sandbox: records creates and kills, and runs commands by echoing them.

    A background command stays running for two status polls and writes one line before each poll: ``log_prefix``
    and the poll's number. Its log is read like execd's, from a byte offset, and the reply names the offset after it.
    """

    created: list[dict[str, Any]] = []
    killed: list[str] = []
    closed: list[str] = []
    fail_image: str | None = None
    fail_kill: str | None = None
    fail_connect: str | None = None
    log_prefix = b"line"

    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id
        self.polls = 0
        self.commands = SimpleNamespace(
            run=self._run, get_command_status=self._status, get_background_command_logs=self._logs
        )
        self.files = SimpleNamespace(read_bytes_stream=self._read)

    @classmethod
    async def create(cls, **kwargs: Any) -> "FakeSandbox":
        if kwargs["image"] == cls.fail_image:
            raise ValueError("no such image")
        cls.created.append(kwargs)
        return cls(f"sandbox-{len(cls.created)}")

    @classmethod
    async def connect(cls, sandbox_id: str, **_kwargs: Any) -> "FakeSandbox":
        if sandbox_id == cls.fail_connect:
            raise ValueError("no such sandbox")
        return cls(sandbox_id)

    async def check_ready(self, *_args: Any) -> None:
        return None

    async def get_info(self) -> Any:
        return SimpleNamespace(status=SimpleNamespace(state="RUNNING"))

    async def _run(self, command: str, *, opts: Any, handlers: Any = None) -> Any:
        if getattr(opts, "background", False):
            return SimpleNamespace(id="execution-1", logs=SimpleNamespace(stdout=[], stderr=[]), error=None)
        if handlers is not None:
            # Older execd strips each line's ending from the event; newer execd keeps it.
            await handlers.on_stdout(SimpleNamespace(text="PORT 8080"))
            await handlers.on_stdout(SimpleNamespace(text="ready\n"))
        line = SimpleNamespace(text=f"{self.id}: {command}")
        return SimpleNamespace(logs=SimpleNamespace(stdout=[line], stderr=[]), exit_code=0, error=None)

    async def _status(self, _execution_id: str) -> Any:
        self.polls += 1
        return SimpleNamespace(running=self.polls < 3, exit_code=None if self.polls < 3 else 0, error=None)

    async def _logs(self, _execution_id: str, cursor: int | None = None) -> Any:
        output = b"".join(type(self).log_prefix + f" {poll}\n".encode() for poll in range(1, self.polls + 1))
        return SimpleNamespace(content=output[cursor or 0 :].decode(errors="replace"), cursor=len(output))

    async def _read(self, path: str) -> Any:
        async def chunks() -> Any:
            yield f"{self.id}:{path}".encode()

        return chunks()

    async def kill(self) -> None:
        # A closed SDK handle has no client left to send the delete with.
        if self.id == type(self).fail_kill or self.id in type(self).closed:
            raise RuntimeError("delete refused")
        type(self).killed.append(self.id)

    async def close(self) -> None:
        type(self).closed.append(self.id)


class FakeConnectionConfig:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def with_transport_if_missing(self) -> "FakeConnectionConfig":
        return self

    async def close_transport_if_owned(self) -> None:
        return None


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> opensandbox_provider.OpenSandboxProvider:
    FakeSandbox.created, FakeSandbox.killed, FakeSandbox.closed = [], [], []
    FakeSandbox.fail_image = FakeSandbox.fail_kill = FakeSandbox.fail_connect = None
    FakeSandbox.log_prefix = b"line"
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSandbox, FakeConnectionConfig, SimpleNamespace, object, object),
    )
    monkeypatch.setattr(opensandbox_provider, "_to_image_spec", lambda image, _auth: image)
    return opensandbox_provider.OpenSandboxProvider(
        probe={"command": None}, operations={"retries": 0, "close_timeout_s": 1}, create={"retries": 0}
    )


_TOOLS = {"name": "tools", "image": "tools:2"}
_SPEC = SandboxSpec(
    image="task:1",
    ttl_s=600,
    metadata={"task": "t-1"},
    provider_options={"extensions": {"pool": "tasks"}},
    sidecars=[{"name": "tools", "image": "tools:2", "env": {"MODE": "capture"}, "resources": {"cpu": 0.5}}],
)


async def test_a_sidecar_is_its_own_sandbox_in_the_main_sandboxs_placement_group(provider) -> None:
    assert isinstance(provider, SupportsSandboxSidecars)

    handle = await provider.create(_SPEC)

    # The sidecar starts first, so the main sandbox's egress policy can name its address.
    sidecar, main = FakeSandbox.created
    assert handle.sandbox_id == "sandbox-2"
    assert (main["image"], sidecar["image"]) == ("task:1", "tools:2")
    assert sidecar["env"] == {"MODE": "capture"} and sidecar["resource"] == {"cpu": "0.5"}
    assert sidecar["timeout"] == main["timeout"]
    # The task's own provider options (mounts, a snapshot) describe the main sandbox only.
    assert main["extensions"]["pool"] == "tasks" and "pool" not in sidecar["extensions"]
    assert "network_policy" not in main and "network_policy" not in sidecar
    group = main["metadata"][PLACEMENT_GROUP_KEY]
    assert group.startswith("sidecars-") and len(group) <= 63
    assert sidecar["metadata"][PLACEMENT_GROUP_KEY] == group
    assert sidecar["metadata"]["task"] == "t-1"


async def test_a_placement_group_the_spec_already_names_is_kept(provider) -> None:
    await provider.create(SandboxSpec(image="task:1", metadata={PLACEMENT_GROUP_KEY: "project-7"}, sidecars=[_TOOLS]))

    assert [created["metadata"][PLACEMENT_GROUP_KEY] for created in FakeSandbox.created] == ["project-7"] * 2


async def test_a_sandbox_without_sidecars_gets_no_placement_group(provider) -> None:
    await provider.create(SandboxSpec(image="task:1"))

    assert len(FakeSandbox.created) == 1
    assert PLACEMENT_GROUP_KEY not in FakeSandbox.created[0]["metadata"]


async def test_sidecar_commands_and_downloads_go_to_the_sidecars_sandbox(provider, tmp_path: Path) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    streamed: list[str] = []

    result = await sandbox.sidecar("tools").exec("serve", "--dir", "/my data", timeout_s=5, on_stdout=streamed.append)
    await sandbox.sidecar("tools").download("/out/log.txt", tmp_path / "log.txt")

    # The argv is quoted into one shell command, and stdout arrives line by line, each line ended once.
    assert result.stdout == "sandbox-1: serve --dir '/my data'"
    assert streamed == ["PORT 8080\n", "ready\n"]
    assert (tmp_path / "log.txt").read_text() == "sandbox-1:/out/log.txt"


def _polled_provider() -> opensandbox_provider.OpenSandboxProvider:
    return opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        create={"retries": 0},
        operations={"background_exec": True, "background_poll_initial_s": 0.01, "background_poll_interval_s": 0.01},
    )


async def test_a_polled_sidecar_command_hands_over_its_output_as_it_arrives(provider) -> None:
    sandbox = await AsyncSandbox(_polled_provider(), _SPEC).start()
    streamed: list[str] = []

    result = await sandbox.sidecar("tools").exec("serve", timeout_s=5, on_stdout=streamed.append)

    # Two polls while it runs, then the rest once it has ended; no request stays open for the command.
    assert streamed == ["line 1\n", "line 2\n", "line 3\n"]
    assert result.stdout == "line 1\nline 2\nline 3\n" and result.return_code == 0


async def test_a_polled_command_whose_output_is_not_utf8_is_read_from_execds_cursor(provider) -> None:
    # One invalid byte per line decodes to a three-byte replacement character, so the text forwarded so far is
    # longer than the bytes execd sent: the next read starts where execd says, not after the text's length.
    FakeSandbox.log_prefix = b"\xff"
    sandbox = await AsyncSandbox(_polled_provider(), _SPEC).start()
    streamed: list[str] = []

    result = await sandbox.sidecar("tools").exec("serve", timeout_s=5, on_stdout=streamed.append)

    assert streamed == ["\ufffd 1\n", "\ufffd 2\n", "\ufffd 3\n"]
    assert "".join(streamed) == result.stdout


async def test_closing_the_sandbox_deletes_its_sidecars(provider) -> None:
    handle = await provider.create(_SPEC)

    await provider.close(handle)

    assert sorted(FakeSandbox.killed) == ["sandbox-1", "sandbox-2"]
    assert provider._sidecars == {}


async def test_a_sidecar_that_cannot_be_deleted_is_retried_by_the_next_close(provider) -> None:
    handle = await provider.create(_SPEC)
    FakeSandbox.fail_kill = "sandbox-1"

    with pytest.raises(NgFailureEvent, match="Sandbox kill failed"):
        await provider.close(handle)
    # The main sandbox is deleted all the same.
    assert FakeSandbox.killed == ["sandbox-2"]
    FakeSandbox.fail_kill = None
    # The repeated close deletes the sidecar only: the main sandbox is gone, and its SDK handle is closed.
    await provider.close(handle)

    assert FakeSandbox.killed == ["sandbox-2", "sandbox-1"]
    assert provider._sidecars == {} and provider._deleted_with_sidecars_left == set()


async def test_a_main_sandbox_that_cannot_start_deletes_its_sidecars(provider) -> None:
    FakeSandbox.fail_image = "task:1"

    with pytest.raises(ValueError, match="no such image"):
        await provider.create(_SPEC)

    assert FakeSandbox.killed == ["sandbox-1"]
    assert provider._sidecars == {}


async def test_a_sidecar_that_cannot_start_deletes_the_sidecars_before_it(provider) -> None:
    FakeSandbox.fail_image = "judge:3"
    spec = SandboxSpec(image="task:1", sidecars=[_TOOLS, {"name": "judge", "image": "judge:3"}])

    with pytest.raises(ValueError, match="no such image"):
        await provider.create(spec)

    assert [created["image"] for created in FakeSandbox.created] == ["tools:2"]
    assert FakeSandbox.killed == ["sandbox-1"]
    assert provider._sidecars == {}


async def test_a_sidecar_shares_the_main_sandboxs_egress_policy_and_the_main_sandbox_may_reach_it(
    provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    hosts: dict[str, dict[str, str]] = {}

    async def network_address(handle: Any) -> str:
        return "10.0.0.5"

    async def set_hosts(handle: Any, addresses: dict[str, str]) -> None:
        hosts[handle.sandbox_id] = addresses

    monkeypatch.setattr(opensandbox_provider, "_to_network_policy", lambda policy: policy)
    monkeypatch.setattr(provider, "network_address", network_address)
    monkeypatch.setattr(provider, "set_hosts", set_hosts)
    provider._networking.enabled = True
    policy = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "models.example.com"}]}

    await provider.create(SandboxSpec(image="task:1", provider_options={"network_policy": policy}, sidecars=[_TOOLS]))

    sidecar, main = FakeSandbox.created
    assert sidecar["network_policy"] == policy
    # The sidecar's rule leads, so no rule of the task's own denies it.
    assert main["network_policy"] == {
        "defaultAction": "deny",
        "egress": [*_reach("10.0.0.5"), {"action": "allow", "target": "models.example.com"}],
    }
    assert hosts == {"sandbox-2": {"tools": "10.0.0.5"}}

    # An empty policy denies by default, so it gets the same treatment.
    await provider.create(SandboxSpec(image="task:1", provider_options={"network_policy": {}}, sidecars=[_TOOLS]))

    sidecar, main = FakeSandbox.created[2:]
    assert sidecar["network_policy"] == {}
    assert main["network_policy"] == {"egress": _reach("10.0.0.5")}


async def test_a_reconnected_sandbox_reaches_its_sidecars(provider) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await sandbox.serialize()
    assert descriptor["sidecar_ids"] == {"tools": "sandbox-1"}

    borrower = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    borrowed = await AsyncSandbox.connect(descriptor, provider=borrower)
    result = await borrowed.sidecar("tools").exec("status", timeout_s=5)

    assert result.stdout == "sandbox-1: status"


async def test_a_sandbox_whose_sidecar_is_gone_is_not_reattached(provider) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await sandbox.serialize()
    FakeSandbox.fail_connect = "sandbox-1"
    borrower = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    with pytest.raises(NgFailureEvent, match="Sandbox connect failed"):
        await AsyncSandbox.connect(descriptor, provider=borrower)

    # The main sandbox's local handle is released, and the borrower keeps no record of it.
    assert FakeSandbox.closed == ["sandbox-2"]
    assert borrower._sidecars == {}
    assert FakeSandbox.killed == []


async def test_a_borrower_releases_the_sidecar_handles_it_connected(provider) -> None:
    owner = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await owner.serialize()
    borrower = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    borrowed = await AsyncSandbox.connect(descriptor, provider=borrower, owns_provider=False)
    assert list(borrower._sidecars) == ["sandbox-2"]
    await borrowed.disconnect()

    # Released, not deleted: the owner still reaches the sidecar.
    assert borrower._sidecars == {} and borrower._connected_sidecars == {}
    assert FakeSandbox.closed == ["sandbox-1"] and FakeSandbox.killed == []
    assert (await owner.sidecar("tools").exec("status", timeout_s=5)).stdout == "sandbox-1: status"


async def test_a_reconnect_on_the_owners_provider_keeps_the_owners_sidecar_handles(provider) -> None:
    owner = await AsyncSandbox(provider, _SPEC).start()
    handles = dict(provider._sidecars["sandbox-2"])

    reattached = await AsyncSandbox.connect(await owner.serialize(), provider=provider, owns_provider=False)
    await reattached.disconnect()

    assert provider._sidecars == {"sandbox-2": handles}
    assert FakeSandbox.closed == []


def _peers(provider: opensandbox_provider.OpenSandboxProvider, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Turn on networking; each sandbox's address ends in its number, and the hosts entries are recorded."""
    hosts: dict[str, dict[str, str]] = {}

    async def network_address(handle: Any) -> str:
        return f"10.0.0.{handle.sandbox_id.removeprefix('sandbox-')}"

    async def set_hosts(handle: Any, addresses: dict[str, str]) -> None:
        hosts[handle.sandbox_id] = addresses

    monkeypatch.setattr(opensandbox_provider, "_to_network_policy", lambda policy: policy)
    monkeypatch.setattr(provider, "network_address", network_address)
    monkeypatch.setattr(provider, "set_hosts", set_hosts)
    provider._networking.enabled = True
    return hosts


def _reach(address: str) -> list[dict[str, Any]]:
    """The rule that lets the main sandbox reach a sidecar."""
    return [{"action": "allow", "target": address}]


def _only(host: str) -> dict[str, Any]:
    return {"defaultAction": "deny", "egress": [{"action": "allow", "target": host}]}


async def test_a_sidecar_with_its_own_egress_policy_reaches_only_its_own_hosts(
    provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    hosts = _peers(provider, monkeypatch)
    tools = {**_TOOLS, "provider_options": {"network_policy": _only("api.example.com"), "extensions": {"pool": "t"}}}
    spec = SandboxSpec(
        image="task:1",
        provider_options={"network_policy": _only("models.example.com")},
        sidecars=[tools, {"name": "judge", "image": "judge:3"}],
    )

    await provider.create(spec)

    tools_created, judge_created, main = FakeSandbox.created
    # Its own policy and options replace what it would get from the main sandbox; a sidecar without one inherits.
    assert tools_created["network_policy"] == _only("api.example.com")
    assert tools_created["extensions"]["pool"] == "t"
    assert judge_created["network_policy"] == _only("models.example.com")
    # The main sandbox reaches the model and both sidecars, nothing else.
    assert main["network_policy"]["egress"] == [
        *_reach("10.0.0.1"),
        *_reach("10.0.0.2"),
        {"action": "allow", "target": "models.example.com"},
    ]
    assert hosts == {"sandbox-3": {"tools": "10.0.0.1", "judge": "10.0.0.2"}}


async def test_a_misspelled_sidecar_option_starts_no_sandbox(provider) -> None:
    judge = {"name": "judge", "image": "judge:3", "provider_options": {"network_polcy": _only("api.example.com")}}

    with pytest.raises(ValueError, match="Unknown OpenSandbox provider option"):
        await provider.create(SandboxSpec(image="task:1", sidecars=[_TOOLS, judge]))

    assert FakeSandbox.created == []


async def test_a_sidecar_the_main_sandbox_cannot_reach_fails_the_create_on_the_infrastructure(
    provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    _peers(provider, monkeypatch)

    async def mapped_address(_handle: Any) -> str:
        raise ValueError("OpenSandbox networking requires a direct, unmapped IP endpoint without routing headers")

    monkeypatch.setattr(provider, "network_address", mapped_address)

    with pytest.raises(NgFailureEvent) as raised:
        await provider.create(_SPEC)

    # An infrastructure failure, so the attempt is retried or masked, never scored; nothing is left running.
    assert raised.value.failure_kind == PROVIDER_SETUP_FAILED
    assert "Sidecar 'tools'" in raised.value.failure_reason
    assert [created["image"] for created in FakeSandbox.created] == ["tools:2"]
    assert FakeSandbox.killed == ["sandbox-1"]


async def test_a_sidecar_the_provider_does_not_hold_is_refused_by_name(provider) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await sandbox.serialize()
    # A descriptor that declares the sidecar but carries no id to connect it with.
    del descriptor["sidecar_ids"]
    borrowed = await AsyncSandbox.connect(
        descriptor, provider=opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    )

    with pytest.raises(RuntimeError, match="holds no sidecar 'tools' of sandbox 'sandbox-2'"):
        await borrowed.sidecar("tools").exec("status", timeout_s=5)


async def test_a_sidecar_name_the_hosts_file_cannot_take_starts_no_sandbox(provider, monkeypatch) -> None:
    _peers(provider, monkeypatch)

    with pytest.raises(ValueError, match="Invalid sandbox hostname"):
        await provider.create(SandboxSpec(image="task:1", sidecars=[_TOOLS, {"name": "web search", "image": "s:1"}]))

    assert FakeSandbox.created == []


async def test_borrowers_on_one_provider_share_the_sidecar_handles_until_the_last_disconnects(provider) -> None:
    owner = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await owner.serialize()
    shared = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    first = await AsyncSandbox.connect(descriptor, provider=shared, owns_provider=False)
    second = await AsyncSandbox.connect(descriptor, provider=shared, owns_provider=False)
    await first.disconnect()

    # The second borrower still reaches the sidecar; the handles go only with the last disconnect.
    assert (await second.sidecar("tools").exec("status", timeout_s=5)).stdout == "sandbox-1: status"
    assert FakeSandbox.closed == []
    await second.disconnect()
    assert shared._sidecars == {} and shared._connected_sidecars == {}
    assert FakeSandbox.closed == ["sandbox-1"] and FakeSandbox.killed == []


async def test_an_owner_that_connected_the_sandbox_still_deletes_the_sidecars_after_a_borrower_leaves(
    provider,
) -> None:
    created = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await created.serialize()
    # A restarted owner holds the sandbox only through its descriptor, and lends it on the same provider.
    restarted = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    owner = await AsyncSandbox.connect(descriptor, provider=restarted, owns_provider=False)
    borrowed = await AsyncSandbox.connect(descriptor, provider=restarted, owns_provider=False)
    await borrowed.disconnect()

    await owner.stop()

    assert sorted(FakeSandbox.killed) == ["sandbox-1", "sandbox-2"]
