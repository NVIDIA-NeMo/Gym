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

from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import PLACEMENT_GROUP_KEY, SandboxSpec, SupportsSandboxSidecars
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


class FakeSandbox:
    """The SDK sandbox: records creates and kills, and runs commands by echoing them."""

    created: list[dict[str, Any]] = []
    killed: list[str] = []
    closed: list[str] = []
    fail_image: str | None = None
    fail_kill: str | None = None
    fail_connect: str | None = None

    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id
        self.commands = SimpleNamespace(run=self._run)
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
        if handlers is not None:
            await handlers.on_stdout(SimpleNamespace(text="PORT 8080"))
        line = SimpleNamespace(text=f"{self.id}: {command}")
        return SimpleNamespace(logs=SimpleNamespace(stdout=[line], stderr=[]), exit_code=0, error=None)

    async def _read(self, path: str) -> Any:
        async def chunks() -> Any:
            yield f"{self.id}:{path}".encode()

        return chunks()

    async def kill(self) -> None:
        if self.id == type(self).fail_kill:
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

    # The argv is quoted into one shell command, and stdout arrives line by line.
    assert result.stdout == "sandbox-1: serve --dir '/my data'"
    assert streamed == ["PORT 8080\n"]
    assert (tmp_path / "log.txt").read_text() == "sandbox-1:/out/log.txt"


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
    await provider.close(handle)

    assert FakeSandbox.killed == ["sandbox-2", "sandbox-2", "sandbox-1"]
    assert provider._sidecars == {}


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
        "egress": [{"action": "allow", "target": "10.0.0.5"}, {"action": "allow", "target": "models.example.com"}],
    }
    assert hosts == {"sandbox-2": {"tools": "10.0.0.5"}}

    # An empty policy denies by default, so it gets the same treatment.
    await provider.create(SandboxSpec(image="task:1", provider_options={"network_policy": {}}, sidecars=[_TOOLS]))

    sidecar, main = FakeSandbox.created[2:]
    assert sidecar["network_policy"] == {}
    assert main["network_policy"] == {"egress": [{"action": "allow", "target": "10.0.0.5"}]}


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
