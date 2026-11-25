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

import asyncio
import contextlib
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
from nemo_gym.sandbox.providers.base import (
    PLACEMENT_GROUP_KEY,
    SandboxExecArgvUnsupportedError,
    SandboxSpec,
    SupportsSandboxSidecars,
)
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


# The SDK's own conversion, which most tests here replace to compare plain mappings.
_SDK_NETWORK_POLICY = opensandbox_provider._to_network_policy


class FakeSandbox:
    """The SDK sandbox: records creates and kills, and runs commands by echoing them.

    A background command stays running for two status polls and writes one line before each poll: ``log_prefix``
    and the poll's number; an interrupt ends it at the next poll. Its log is read like execd's, from a byte offset
    and at most ``read_cap`` bytes at a time, and the reply names the offset after it.
    """

    created: list[dict[str, Any]] = []
    killed: list[str] = []
    closed: list[str] = []
    renewed: list[tuple[str, float]] = []
    ran: list[tuple[str, Any]] = []
    interrupted: list[str] = []
    read_cap: int | None = None
    fail_image: str | None = None
    exits_image: str | None = None
    exited = 0
    fail_kill: str | None = None
    fail_connect: str | None = None
    gone: str | None = None
    log_prefix = b"line"

    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id
        self.polls = 0
        self.commands = SimpleNamespace(
            run=self._run,
            get_command_status=self._status,
            get_background_command_logs=self._logs,
            interrupt=self._interrupt,
        )
        self.files = SimpleNamespace(read_bytes_stream=self._read)

    @classmethod
    async def create(cls, **kwargs: Any) -> "FakeSandbox":
        if kwargs["image"] == cls.fail_image:
            raise ValueError("no such image")
        if kwargs["image"] == cls.exits_image:
            cls.exited += 1
            raise RuntimeError(
                "Create sandbox failed: Sandbox s-1 failed: 1/1 observed pods failed; primary reason=ContainerFailed; "
                "sample pod=s-1-0; sample detail=Pod s-1-0 container sandbox exited with code 255 (Error)"
            )
        cls.created.append(kwargs)
        return cls(f"sandbox-{len(cls.created)}")

    @classmethod
    async def connect(cls, sandbox_id: str, **_kwargs: Any) -> "FakeSandbox":
        # A real connect waits on the network, so concurrent connects interleave here.
        await asyncio.sleep(0)
        if sandbox_id == cls.fail_connect:
            raise ValueError("connection reset")
        if sandbox_id == cls.gone:
            raise ValueError(f"Sandbox {sandbox_id!r} not found")
        return cls(sandbox_id)

    @classmethod
    async def resume(cls, sandbox_id: str, **_kwargs: Any) -> "FakeSandbox":
        return cls(sandbox_id)

    async def check_ready(self, *_args: Any) -> None:
        return None

    async def get_info(self) -> Any:
        return SimpleNamespace(status=SimpleNamespace(state="RUNNING"))

    async def renew(self, timeout: Any) -> None:
        type(self).renewed.append((self.id, timeout.total_seconds()))

    async def _interrupt(self, _execution_id: str) -> None:
        type(self).interrupted.append(self.id)

    async def _run(self, command: str, *, opts: Any, handlers: Any = None) -> Any:
        type(self).ran.append((self.id, command))
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
        running = self.polls < 3 and self.id not in type(self).interrupted
        return SimpleNamespace(running=running, exit_code=None if running else 0, error=None)

    async def _logs(self, _execution_id: str, cursor: int | None = None) -> Any:
        output = b"".join(type(self).log_prefix + f" {poll}\n".encode() for poll in range(1, self.polls + 1))
        start = cursor or 0
        chunk = output[start : len(output) if type(self).read_cap is None else start + type(self).read_cap]
        return SimpleNamespace(content=chunk.decode(errors="replace"), cursor=start + len(chunk))

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
    FakeSandbox.renewed, FakeSandbox.ran, FakeSandbox.interrupted = [], [], []
    FakeSandbox.fail_image = FakeSandbox.fail_kill = FakeSandbox.fail_connect = FakeSandbox.gone = None
    FakeSandbox.exits_image, FakeSandbox.exited = None, 0
    FakeSandbox.log_prefix, FakeSandbox.read_cap = b"line", None
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
# A sidecar that serves the main sandbox: its service and the port the main sandbox may reach.
_SERVICE = {**_TOOLS, "entrypoint": ["tool-server", "--port", "8080"], "ports": [8080]}
_SIDECAR_LABEL = "nemo-gym.nvidia.com/sidecar"
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
    # A label names the sidecar, so a listing by run label can tell it from a task sandbox.
    assert sidecar["metadata"][_SIDECAR_LABEL] == "tools" and _SIDECAR_LABEL not in main["metadata"]
    # Without a service of its own it idles, like the main sandbox.
    assert sidecar["entrypoint"] == main["entrypoint"] == list(opensandbox_provider.DEFAULT_KEEPALIVE_ENTRYPOINT)
    # The sidecar started first, so its TTL starts again once the main sandbox exists: it does not end before it.
    assert FakeSandbox.renewed == [("sandbox-1", 600)]


async def test_a_leased_sidecars_lifetime_is_counted_from_the_main_sandboxs_create(provider) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        operations={"retries": 0, "close_timeout_s": 1},
        create={"retries": 0, "renew_ttl_s": 60, "renew_interval_s": 30},
    )

    handle = await provider.create(_SPEC)

    try:
        # The sidecar's lease starts again once the main sandbox exists, and its renewals with it: the renewal made
        # at the sidecar's own create would end the sidecar's lifetime before the main sandbox's.
        assert FakeSandbox.renewed == [("sandbox-1", 60)]
        assert set(provider._renewals) == {"sandbox-1", "sandbox-2"}
    finally:
        await provider.close(handle)
    assert provider._renewals == {}


async def test_a_sidecar_gets_none_of_the_deployments_setup_or_shared_storage_labels(provider, monkeypatch) -> None:
    _peers(provider, monkeypatch)
    provider._setup.env, provider._setup.commands = {"PROXY": "http://proxy"}, ["install-tools"]
    provider._shared_storage.metadata = {"storage": "shared"}
    spec = SandboxSpec(
        image="task:1", workdir="/work", metadata={"storage": "shared", "task": "t-1"}, sidecars=[_SERVICE]
    )

    await provider.create(spec)

    sidecar, main = FakeSandbox.created
    # The setup is written for task images and needs a shell; a service image may have none.
    assert sidecar["env"] is None and main["env"] == {"PROXY": "http://proxy"}
    assert [sandbox_id for sandbox_id, _command in FakeSandbox.ran] == ["sandbox-2", "sandbox-2"]
    assert sidecar["entrypoint"] == ["tool-server", "--port", "8080"]
    # What grants the main sandbox shared storage is not the sidecar's.
    assert "storage" not in sidecar["metadata"] and main["metadata"]["storage"] == "shared"


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

    # The argv reaches execd as a list, with no shell to quote it for, and stdout arrives line by line, each line
    # ended once.
    assert result.stdout == "sandbox-1: ['serve', '--dir', '/my data']"
    assert streamed == ["PORT 8080\n", "ready\n"]
    assert (tmp_path / "log.txt").read_text() == "sandbox-1:/out/log.txt"


async def test_a_sidecar_command_needs_an_sdk_that_sends_argv(provider, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    monkeypatch.setattr(opensandbox_provider, "_sdk_sends_argv", lambda: False)

    # No shell to fall back on: the sidecar image may have none.
    with pytest.raises(SandboxExecArgvUnsupportedError):
        await sandbox.sidecar("tools").exec("serve", timeout_s=5)


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


async def test_polled_output_past_execds_read_bound_still_reaches_the_callback(provider) -> None:
    # execd returns a bounded amount per read. The last line is longer than the bound, so one read at the
    # command's end does not bring all of it.
    FakeSandbox.read_cap = 4
    sandbox = await AsyncSandbox(_polled_provider(), _SPEC).start()
    streamed: list[str] = []

    await sandbox.sidecar("tools").exec("serve", timeout_s=5, on_stdout=streamed.append)

    assert "".join(streamed) == "line 1\nline 2\nline 3\n"


async def test_an_error_of_the_callers_callback_is_raised_unchanged(provider) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    failure = KeyError("no handler for this line")
    seen: list[str] = []

    def on_stdout(text: str) -> None:
        seen.append(text)
        raise failure

    # The caller's own error, not a provider failure that would be retried as an infrastructure one.
    with pytest.raises(KeyError) as raised:
        await sandbox.sidecar("tools").exec("serve", timeout_s=5, on_stdout=on_stdout)

    assert raised.value is failure
    # The callback gets nothing after it failed.
    assert seen == ["PORT 8080\n"]


async def test_a_polled_command_is_interrupted_once_the_callers_callback_fails(provider) -> None:
    sandbox = await AsyncSandbox(_polled_provider(), _SPEC).start()
    failure = KeyError("no handler for this line")

    def on_stdout(_text: str) -> None:
        raise failure

    with pytest.raises(KeyError) as raised:
        await sandbox.sidecar("tools").exec("serve", timeout_s=5, on_stdout=on_stdout)

    assert raised.value is failure
    # Nothing takes its output any more, so the command is not left to run to its timeout.
    assert FakeSandbox.interrupted == ["sandbox-1"]


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


async def test_a_sidecar_whose_container_exits_at_start_is_a_setup_failure_and_not_tried_again(provider) -> None:
    FakeSandbox.exits_image = "distroless:1"
    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None}, create={"retries": 2, "retry_delay_s": 0, "retry_max_delay_s": 0}
    )

    with pytest.raises(NgFailureEvent) as raised:
        await provider.create(SandboxSpec(image="task:1", sidecars=[{"name": "tools", "image": "distroless:1"}]))

    # Its image or entrypoint is at fault, for example an image without a shell: no retry would change that.
    assert raised.value.failure_kind == PROVIDER_SETUP_FAILED
    assert "exited at start with code 255: image 'distroless:1' cannot run" in raised.value.failure_reason
    assert FakeSandbox.exited == 1 and FakeSandbox.created == []


async def test_a_sidecar_that_cannot_start_deletes_the_sidecars_before_it(provider) -> None:
    FakeSandbox.fail_image = "judge:3"
    spec = SandboxSpec(image="task:1", sidecars=[_TOOLS, {"name": "judge", "image": "judge:3"}])

    with pytest.raises(ValueError, match="no such image"):
        await provider.create(spec)

    assert [created["image"] for created in FakeSandbox.created] == ["tools:2"]
    assert FakeSandbox.killed == ["sandbox-1"]
    assert provider._sidecars == {}


async def test_the_main_sandbox_reaches_only_the_ports_a_sidecar_names(provider, monkeypatch) -> None:
    hosts = _peers(provider, monkeypatch)
    policy = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "models.example.com"}]}
    spec = SandboxSpec(
        image="task:1",
        provider_options={"network_policy": policy},
        sidecars=[_SERVICE, {"name": "grader", "image": "grader:3"}],
    )

    await provider.create(spec)

    tools, grader, main = FakeSandbox.created
    # The main sandbox's grants are not a sidecar's: one that names no policy reaches nothing.
    assert tools["network_policy"] == grader["network_policy"] == {"defaultAction": "deny", "egress": []}
    # One leading rule, for the ports the tools sidecar names: no rule of the task's own denies it, its exec daemon
    # stays closed, and the grader gets no rule at all.
    assert main["network_policy"] == {
        "defaultAction": "deny",
        "egress": [*_reach("10.0.0.1", 8080), {"action": "allow", "target": "models.example.com"}],
    }
    assert hosts == {"sandbox-3": {"tools": "10.0.0.1", "grader": "10.0.0.2"}}


async def test_a_sandbox_without_a_policy_gives_its_sidecars_none(provider, monkeypatch) -> None:
    _peers(provider, monkeypatch)

    await provider.create(SandboxSpec(image="task:1", sidecars=[_SERVICE]))

    assert all("network_policy" not in created for created in FakeSandbox.created)


@pytest.mark.parametrize(
    ("networking", "sidecar", "policy", "message"),
    [
        # Nothing in the main sandbox could reach the service.
        ({"enabled": False}, _SERVICE, None, "needs networking.enabled=true"),
        # The exec daemon and the egress API are never the main sandbox's to reach.
        ({"enabled": True}, {**_SERVICE, "ports": [8080, 44772]}, None, r"control ports \[44772\]"),
        ({"enabled": True}, {**_SERVICE, "ports": [18080]}, None, r"control ports \[18080\]"),
        # An egress layer that drops a rule's ports would open every port of the sidecar.
        ({"enabled": True}, _SERVICE, {"defaultAction": "deny"}, "networking.rule_ports_enforced=true"),
    ],
)
async def test_a_sidecar_the_sandbox_cannot_safely_reach_starts_no_sandbox(
    provider, networking: dict, sidecar: dict, policy: dict | None, message: str
) -> None:
    provider._networking.enabled = networking["enabled"]
    options = {} if policy is None else {"network_policy": policy}

    with pytest.raises(ValueError, match=message):
        await provider.create(SandboxSpec(image="task:1", provider_options=options, sidecars=[sidecar]))

    assert FakeSandbox.created == []


async def test_the_sdks_policy_keeps_a_sidecar_rules_ports(provider, monkeypatch) -> None:
    _peers(provider, monkeypatch)
    monkeypatch.setattr(opensandbox_provider, "_to_network_policy", _SDK_NETWORK_POLICY)
    spec = SandboxSpec(
        image="task:1", provider_options={"network_policy": {"defaultAction": "deny"}}, sidecars=[_SERVICE]
    )

    await provider.create(spec)

    # What the SDK sends for the main sandbox: the sidecar's declared port and no other, so no control port.
    sent = FakeSandbox.created[-1]["network_policy"].model_dump(by_alias=True, exclude_none=True)
    assert sent == {"defaultAction": "deny", "egress": _reach("10.0.0.1", 8080)}


async def test_an_sdk_that_drops_a_rules_ports_starts_no_sandbox(provider, monkeypatch) -> None:
    from pydantic import BaseModel

    class RuleWithoutPorts(BaseModel):
        action: str
        target: str

    _peers(provider, monkeypatch)
    monkeypatch.setattr("opensandbox.models.sandboxes.NetworkRule", RuleWithoutPorts)
    spec = SandboxSpec(
        image="task:1", provider_options={"network_policy": {"defaultAction": "deny"}}, sidecars=[_SERVICE]
    )

    # Such an SDK would send the rule without its ports, which opens every port of the sidecar.
    with pytest.raises(ValueError, match="installed opensandbox SDK drops a network rule's ports"):
        await provider.create(spec)
    assert FakeSandbox.created == []

    # A sandbox without a policy gets no rule, so the same SDK still creates it.
    await provider.create(SandboxSpec(image="task:1", sidecars=[_SERVICE]))
    assert [created["image"] for created in FakeSandbox.created] == ["tools:2", "task:1"]


async def test_a_create_waits_for_a_sidecars_service_to_answer_on_its_ports(provider, monkeypatch) -> None:
    _peers(provider, monkeypatch)
    asked: list[tuple[str, int]] = []

    async def answers(handle: Any, port: int) -> bool:
        asked.append((handle.sandbox_id, port))
        return len(asked) > 2

    monkeypatch.setattr(provider, "_sidecar_port_answers", answers)
    monkeypatch.setattr(opensandbox_provider, "SIDECAR_PORT_POLL_INTERVAL_S", 0)

    await provider.create(SandboxSpec(image="task:1", sidecars=[_SERVICE]))

    # Asked until the service answered, before the main sandbox was created.
    assert asked == [("sandbox-1", 8080)] * 3
    assert [created["image"] for created in FakeSandbox.created] == ["tools:2", "task:1"]


async def test_a_create_does_not_wait_for_a_service_the_owner_starts_later(provider, monkeypatch) -> None:
    _peers(provider, monkeypatch, answers=False)
    monkeypatch.setattr(opensandbox_provider, "SIDECAR_PORT_POLL_INTERVAL_S", 0)
    policy = {"defaultAction": "deny", "egress": []}
    # No entrypoint: the sidecar idles, and its owner starts the service in it once the sandbox is up.
    later = {**_TOOLS, "ports": [8080]}

    sandbox = await AsyncSandbox(
        provider,
        SandboxSpec(
            image="task:1", ready_timeout_s=0.01, provider_options={"network_policy": policy}, sidecars=[later]
        ),
    ).start()

    # Nothing answers on the port, and the create returned all the same, with the port already granted.
    assert [created["image"] for created in FakeSandbox.created] == ["tools:2", "task:1"]
    assert FakeSandbox.created[-1]["network_policy"]["egress"] == _reach("10.0.0.1", 8080)
    assert sandbox.sidecar("tools").url() == "http://tools:8080"
    await sandbox.stop()


async def test_a_sidecars_url_names_the_hosts_entry_the_main_sandbox_resolves(provider, monkeypatch) -> None:
    hosts = _peers(provider, monkeypatch)
    sandbox = await AsyncSandbox(provider, SandboxSpec(image="task:1", sidecars=[_SERVICE])).start()

    assert sandbox.sidecar("tools").url() == "http://tools:8080"
    assert hosts == {"sandbox-2": {"tools": "10.0.0.1"}}
    await sandbox.stop()


async def test_a_sidecar_whose_service_never_answers_fails_the_create_as_a_setup_failure(
    provider, monkeypatch
) -> None:
    _peers(provider, monkeypatch, answers=False)
    monkeypatch.setattr(opensandbox_provider, "SIDECAR_PORT_POLL_INTERVAL_S", 0)

    with pytest.raises(NgFailureEvent) as raised:
        await provider.create(SandboxSpec(image="task:1", ready_timeout_s=0.01, sidecars=[_SERVICE]))

    # An infrastructure failure, not a tool error the model is scored on; nothing is left running.
    assert raised.value.failure_kind == PROVIDER_SETUP_FAILED
    assert "Sidecar 'tools' did not answer on port(s) [8080]" in raised.value.failure_reason
    assert [created["image"] for created in FakeSandbox.created] == ["tools:2"]
    assert FakeSandbox.killed == ["sandbox-1"]


async def _endless() -> Any:
    while True:
        yield b"heartbeat\n"


async def _once(body: bytes) -> Any:
    yield body


async def test_a_sidecar_request_that_never_finishes_ends_at_the_readiness_deadline(provider, monkeypatch) -> None:
    _peers(provider, monkeypatch)

    async def never_answers(_handle: Any, _port: int) -> bool:
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(provider, "_sidecar_port_answers", never_answers)
    spec = SandboxSpec(image="task:1", ready_timeout_s=0.05, sidecars=[_SERVICE, {**_SERVICE, "name": "proxy"}])

    with pytest.raises(NgFailureEvent) as raised:
        # The budget is the spec's; the outer bound only keeps a regression from hanging the suite.
        await asyncio.wait_for(provider.create(spec), timeout=5)

    assert raised.value.failure_kind == PROVIDER_SETUP_FAILED
    assert "did not answer on port(s) [8080] within 0.05s" in raised.value.failure_reason
    # Every member already started is deleted, and the main sandbox never starts.
    assert [created["image"] for created in FakeSandbox.created] == ["tools:2", "tools:2"]
    assert sorted(FakeSandbox.killed) == ["sandbox-1", "sandbox-2"]


@pytest.mark.parametrize(
    ("status", "answer", "listening"),
    [
        # Any answer of the service counts, whatever its status, and its body is not read.
        (404, lambda: _once(b"<html>not found</html>"), True),
        (200, _endless, True),
        # The management API's own answer when nothing listens on the port it proxies to.
        (502, lambda: _once(b'{"code":"BACKEND_CONNECTION_FAILED","message":"..."}'), False),
        # A server error of the service's own; only the start of its body is read.
        (502, lambda: _once(b"upstream down"), True),
        (500, _endless, True),
        (None, ConnectionError("refused"), False),
    ],
)
async def test_a_sidecar_port_answers_once_its_service_does(
    provider, monkeypatch, status: int | None, answer: Any, listening: bool
) -> None:
    from opensandbox.adapters.factory import AdapterFactory

    asked: list[Any] = []

    @contextlib.asynccontextmanager
    async def stream(method: str, url: str) -> Any:
        asked.append((method, url))
        if isinstance(answer, Exception):
            raise answer
        yield SimpleNamespace(status_code=status, aiter_bytes=answer)

    async def get_client() -> Any:
        return SimpleNamespace(get_async_httpx_client=lambda: SimpleNamespace(stream=stream))

    monkeypatch.setattr(
        AdapterFactory,
        "create_health_service",
        lambda _self, endpoint: asked.append(endpoint) or SimpleNamespace(_get_client=get_client),
    )
    handle = await provider.create(SandboxSpec(image="tools:2"))
    handle.raw.get_endpoint = lambda port: _endpoint(handle.sandbox_id, port)

    assert await provider._sidecar_port_answers(handle, 8080) is listening
    # Asked at the sidecar's own endpoint for that port.
    assert asked == ["sandbox-1:8080", ("GET", "/ping")]


async def _endpoint(sandbox_id: str, port: int) -> str:
    return f"{sandbox_id}:{port}"


async def test_a_reconnected_sandbox_reaches_its_sidecars(provider) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await sandbox.serialize()
    assert descriptor["sidecar_ids"] == {"tools": "sandbox-1"}

    borrower = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    borrowed = await AsyncSandbox.connect(descriptor, provider=borrower)
    result = await borrowed.sidecar("tools").exec("status", timeout_s=5)

    assert result.stdout == "sandbox-1: ['status']"


async def test_a_sandbox_whose_sidecar_cannot_be_reached_is_not_reattached(provider) -> None:
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


async def test_a_sandbox_whose_sidecar_is_gone_is_reattached_and_can_be_stopped(provider) -> None:
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await sandbox.serialize()
    FakeSandbox.gone = "sandbox-1"
    restarted = opensandbox_provider.OpenSandboxProvider(probe={"command": None}, operations={"close_timeout_s": 1})

    reattached = await AsyncSandbox.connect(descriptor, provider=restarted)

    with pytest.raises(NgFailureEvent, match="Sidecar 'tools' of sandbox 'sandbox-2' no longer exists"):
        await reattached.sidecar("tools").exec("status", timeout_s=5)
    await restarted.close(reattached._handle)
    assert FakeSandbox.killed == ["sandbox-2"]
    assert restarted._gone_sidecars == {}


async def test_a_resumed_sandbox_gets_its_sidecars_hosts_entries_again(provider, monkeypatch) -> None:
    hosts = _peers(provider, monkeypatch)
    sandbox = await AsyncSandbox(provider, _SPEC).start()
    assert hosts.pop("sandbox-2") == {"tools": "10.0.0.1"}

    await provider.resume(sandbox._handle)

    assert hosts == {"sandbox-2": {"tools": "10.0.0.1"}}


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
    assert (await owner.sidecar("tools").exec("status", timeout_s=5)).stdout == "sandbox-1: ['status']"


async def test_a_reconnect_on_the_owners_provider_keeps_the_owners_sidecar_handles(provider) -> None:
    owner = await AsyncSandbox(provider, _SPEC).start()
    handles = dict(provider._sidecars["sandbox-2"])

    reattached = await AsyncSandbox.connect(await owner.serialize(), provider=provider, owns_provider=False)
    await reattached.disconnect()

    assert provider._sidecars == {"sandbox-2": handles}
    assert FakeSandbox.closed == []


def _peers(
    provider: opensandbox_provider.OpenSandboxProvider, monkeypatch: pytest.MonkeyPatch, *, answers: bool = True
) -> dict:
    """Turn on networking, on a deployment whose egress layer applies a rule's ports.

    Each sandbox's address ends in its number, the hosts entries are recorded, and a sidecar's service ``answers``
    on its ports or never does.
    """
    hosts: dict[str, dict[str, str]] = {}

    async def port_answers(_handle: Any, _port: int) -> bool:
        return answers

    async def network_address(handle: Any) -> str:
        return f"10.0.0.{handle.sandbox_id.removeprefix('sandbox-')}"

    async def set_hosts(handle: Any, addresses: dict[str, str]) -> None:
        hosts[handle.sandbox_id] = addresses

    monkeypatch.setattr(opensandbox_provider, "_to_network_policy", lambda policy: policy)
    monkeypatch.setattr(provider, "network_address", network_address)
    monkeypatch.setattr(provider, "set_hosts", set_hosts)
    monkeypatch.setattr(provider, "_sidecar_port_answers", port_answers)
    provider._networking.enabled = provider._networking.rule_ports_enforced = True
    return hosts


def _reach(address: str, *ports: int) -> list[dict[str, Any]]:
    """The rule that lets the main sandbox reach a sidecar on the ports it names."""
    return [{"action": "allow", "target": address, "ports": list(ports)}]


def _only(host: str) -> dict[str, Any]:
    return {"defaultAction": "deny", "egress": [{"action": "allow", "target": host}]}


async def test_a_sidecar_with_its_own_egress_policy_reaches_only_its_own_hosts(
    provider, monkeypatch: pytest.MonkeyPatch
) -> None:
    hosts = _peers(provider, monkeypatch)
    tools = {**_SERVICE, "provider_options": {"network_policy": _only("api.example.com"), "extensions": {"pool": "t"}}}
    spec = SandboxSpec(
        image="task:1",
        provider_options={"network_policy": _only("models.example.com")},
        sidecars=[tools, {"name": "judge", "image": "judge:3", "provider_options": {"network_policy": None}}],
    )

    await provider.create(spec)

    tools_created, judge_created, main = FakeSandbox.created
    # Its own policy and options are what it gets; a sidecar without one (or with `None`) reaches nothing, and
    # never the main sandbox's hosts.
    assert tools_created["network_policy"] == _only("api.example.com")
    assert tools_created["extensions"]["pool"] == "t"
    assert judge_created["network_policy"] == {"defaultAction": "deny", "egress": []}
    # The main sandbox reaches the model and the tools sidecar's port, nothing else.
    assert main["network_policy"]["egress"] == [
        *_reach("10.0.0.1", 8080),
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


@pytest.mark.parametrize(
    ("name", "message"), [("web search", "Invalid sandbox hostname"), (f"{'a' * 40}.{'b' * 40}", "longer than")]
)
async def test_a_sidecar_name_that_is_no_hostname_or_label_starts_no_sandbox(
    provider, name: str, message: str
) -> None:
    # The name is the sidecar's hostname in the main sandbox and its label value, with networking or without.
    with pytest.raises(ValueError, match=message):
        await provider.create(SandboxSpec(image="task:1", sidecars=[_TOOLS, {"name": name, "image": "s:1"}]))

    assert FakeSandbox.created == []


async def test_borrowers_on_one_provider_share_the_sidecar_handles_until_the_last_disconnects(provider) -> None:
    owner = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await owner.serialize()
    shared = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    first = await AsyncSandbox.connect(descriptor, provider=shared, owns_provider=False)
    second = await AsyncSandbox.connect(descriptor, provider=shared, owns_provider=False)
    await first.disconnect()

    # The second borrower still reaches the sidecar; the handles go only with the last disconnect.
    assert (await second.sidecar("tools").exec("status", timeout_s=5)).stdout == "sandbox-1: ['status']"
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


async def test_concurrent_connects_on_one_provider_share_one_set_of_sidecar_handles(provider) -> None:
    owner = await AsyncSandbox(provider, _SPEC).start()
    descriptor = await owner.serialize()
    shared = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    first, second = await asyncio.gather(
        AsyncSandbox.connect(descriptor, provider=shared, owns_provider=False),
        AsyncSandbox.connect(descriptor, provider=shared, owns_provider=False),
    )
    # Both connected the sidecar; the one that finished second released its own handle and shares the other.
    assert FakeSandbox.closed == ["sandbox-1"]
    await first.disconnect()

    assert (await second.sidecar("tools").exec("status", timeout_s=5)).stdout == "sandbox-1: ['status']"
    await second.disconnect()
    assert shared._sidecars == {} and shared._connected_sidecars == {}
    assert FakeSandbox.closed == ["sandbox-1", "sandbox-1"]
