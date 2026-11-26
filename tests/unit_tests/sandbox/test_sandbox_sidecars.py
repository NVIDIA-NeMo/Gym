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

"""Tests for optional sidecar containers in the sandbox API, using in-process fake providers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSidecarSpec,
    SandboxSpec,
    SandboxStatus,
    SupportsSandboxSidecars,
)


class PlainProvider:
    """A connectable provider without sidecar support; records the specs it is asked to create."""

    name = "plain"

    def __init__(self) -> None:
        self.created: list[SandboxSpec] = []

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.created.append(spec)
        sandbox_id = f"box-{uuid4().hex[:8]}"
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=sandbox_id)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        return SandboxExecResult(stdout=f"ran: {command}", stderr=None, return_code=0)

    async def upload_file(self, handle, source_path, target_path) -> None:
        return None

    async def download_file(self, handle, source_path, target_path) -> None:
        return None

    async def status(self, handle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def serialize_handle(self, handle, *, scope=None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=sandbox_id)


class SidecarProvider(PlainProvider):
    """A provider that implements `SupportsSandboxSidecars` and records every sidecar call."""

    name = "sidecar_fake"

    def __init__(self) -> None:
        super().__init__()
        self.sidecar_calls: list[tuple[Any, ...]] = []

    async def sidecar_exec(
        self,
        handle: SandboxHandle,
        sidecar: str,
        argv: list[str],
        *,
        timeout_s: int | float,
        on_stdout: Callable[[str], None] | None = None,
    ) -> SandboxExecResult:
        self.sidecar_calls.append(("exec", handle.sandbox_id, sidecar, argv, timeout_s))
        if on_stdout is not None:
            on_stdout(f"{sidecar} started\n")
        return SandboxExecResult(stdout=f"{sidecar}: {' '.join(argv)}", stderr=None, return_code=0)

    async def download_sidecar_file(
        self, handle: SandboxHandle, sidecar: str, source_path: str, target_path: Path
    ) -> None:
        self.sidecar_calls.append(("download", handle.sandbox_id, sidecar, source_path, target_path))
        target_path.write_text(f"{sidecar}:{source_path}", encoding="utf-8")

    def sidecar_host(self, handle: SandboxHandle, sidecar: str) -> str:
        # This provider's sidecars share the sandbox's loopback.
        return "127.0.0.1"


_RECORDER = {
    "name": "recorder",
    "image": "recorder:1",
    "env": {"MODE": "capture"},
    "resources": {"cpu": 0.5},
    "provider_options": {"image_auth": {"username": "robot", "password": "secret"}},
}


_RECORDER_SPEC = SandboxSidecarSpec(name="recorder", image="recorder:1")


def test_spec_coerces_sidecar_mappings_and_keeps_specs() -> None:
    explicit = SandboxSidecarSpec(name="metrics", image="metrics:2")
    spec = SandboxSpec(image="task:1", sidecars=[_RECORDER, explicit])

    assert isinstance(spec.sidecars, tuple)
    assert spec.sidecars[0] == SandboxSidecarSpec(
        name="recorder",
        image="recorder:1",
        env={"MODE": "capture"},
        resources=SandboxResources(cpu=0.5),
        provider_options={"image_auth": {"username": "robot", "password": "secret"}},
    )
    assert explicit.provider_options == {}
    assert spec.sidecars[1] is explicit


def test_spec_rejects_invalid_sidecars() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        SandboxSpec(sidecars=[_RECORDER, {"name": "recorder", "image": "other:1"}])
    with pytest.raises(ValueError, match="needs a name and an image"):
        SandboxSpec(sidecars=[{"name": "recorder", "image": ""}])
    with pytest.raises(ValueError, match="Unknown sandbox resource keys"):
        SandboxSpec(sidecars=[{"name": "recorder", "image": "recorder:1", "resources": {"cores": 1}}])
    with pytest.raises(TypeError, match="list or tuple"):
        SandboxSpec(sidecars=_RECORDER)


def test_a_sidecar_names_its_service_and_the_ports_the_sandbox_may_reach() -> None:
    service = SandboxSidecarSpec(name="tools", image="tools:2", entrypoint=["tool-server"], ports=[8080, "9000"])

    assert service.entrypoint == ["tool-server"] and service.ports == (8080, 9000)
    # Without them the sidecar idles, and the sandbox may not reach it.
    assert (_RECORDER_SPEC.entrypoint, _RECORDER_SPEC.ports) == (None, ())


@pytest.mark.parametrize(
    ("ports", "error", "message"),
    [
        ([0], ValueError, "between 1 and 65535"),
        ([8080, 8080], ValueError, "Duplicate sandbox sidecar TCP port: 8080"),
        ([True], ValueError, "Invalid sandbox sidecar TCP port"),
        (8080, TypeError, "Sandbox sidecar ports must be a list or tuple"),
    ],
)
def test_a_sidecar_refuses_ports_that_are_no_tcp_ports(ports: Any, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        SandboxSidecarSpec(name="tools", image="tools:2", ports=ports)


def test_spec_without_sidecars_is_unchanged() -> None:
    assert SandboxSpec(image="task:1").sidecars == ()


def test_sidecar_capability_is_structural() -> None:
    assert isinstance(SidecarProvider(), SupportsSandboxSidecars)
    assert not isinstance(PlainProvider(), SupportsSandboxSidecars)


async def test_start_refuses_sidecars_on_provider_without_support() -> None:
    provider = PlainProvider()
    sandbox = AsyncSandbox(provider, SandboxSpec(image="task:1", sidecars=[_RECORDER]))

    with pytest.raises(NotImplementedError, match="'plain' does not support sidecars"):
        await sandbox.start()

    # Refused before the provider creates anything, so nothing is left running.
    assert provider.created == []


async def test_start_without_sidecars_works_on_provider_without_support() -> None:
    provider = PlainProvider()
    sandbox = await AsyncSandbox(provider, SandboxSpec(image="task:1")).start()

    assert provider.created == [SandboxSpec(image="task:1")]
    descriptor = await sandbox.serialize()
    assert "sidecars" not in descriptor
    with pytest.raises(NotImplementedError, match="'plain' does not support sidecars"):
        sandbox.sidecar("recorder")
    await sandbox.stop()


async def test_sidecar_refuses_undeclared_names() -> None:
    sandbox = await AsyncSandbox(SidecarProvider(), SandboxSpec(image="task:1", sidecars=[_RECORDER])).start()

    with pytest.raises(ValueError, match=r"Sidecar 'other' was not declared .*declared: \['recorder'\]"):
        sandbox.sidecar("other")
    await sandbox.stop()


async def test_sidecar_requires_a_started_sandbox() -> None:
    sandbox = AsyncSandbox(SidecarProvider(), SandboxSpec(image="task:1", sidecars=[_RECORDER]))

    with pytest.raises(RuntimeError, match="has not been started"):
        sandbox.sidecar("recorder")


async def test_sidecar_exec_and_download_delegate_to_provider(tmp_path: Path) -> None:
    provider = SidecarProvider()
    sandbox = await AsyncSandbox(provider, SandboxSpec(image="task:1", sidecars=[_RECORDER])).start()
    sandbox_id = sandbox._require_handle().sandbox_id
    assert provider.created[0].sidecars[0].name == "recorder"

    sidecar = sandbox.sidecar("recorder")
    output: list[str] = []
    result = await sidecar.exec("serve", "--port", "9000", timeout_s=30, on_stdout=output.append)
    await sidecar.download("/var/log/record.jsonl", str(tmp_path / "record.jsonl"))

    assert result.stdout == "recorder: serve --port 9000"
    assert output == ["recorder started\n"]
    assert provider.sidecar_calls == [
        ("exec", sandbox_id, "recorder", ["serve", "--port", "9000"], 30),
        ("download", sandbox_id, "recorder", "/var/log/record.jsonl", tmp_path / "record.jsonl"),
    ]
    assert (tmp_path / "record.jsonl").read_text(encoding="utf-8") == "recorder:/var/log/record.jsonl"
    await sandbox.stop()


async def test_a_sidecar_url_is_where_the_sandbox_reaches_a_declared_port() -> None:
    tools = {"name": "tools", "image": "tools:2", "ports": [8080]}
    proxy = {"name": "proxy", "image": "proxy:3", "ports": [8080, 9090]}
    sandbox = await AsyncSandbox(SidecarProvider(), SandboxSpec(image="task:1", sidecars=[tools, proxy])).start()

    # The provider names the host; the only declared port is the default.
    assert sandbox.sidecar("tools").url() == sandbox.sidecar("tools").url(8080) == "http://127.0.0.1:8080"
    assert sandbox.sidecar("proxy").url(9090) == "http://127.0.0.1:9090"
    await sandbox.stop()


@pytest.mark.parametrize(
    ("ports", "port", "message"),
    [
        ([], None, r"declared ports \[\], got None"),
        ([8080, 9090], None, r"declared ports \[8080, 9090\], got None"),
        ([8080], 9090, r"declared ports \[8080\], got 9090"),
    ],
)
async def test_a_sidecar_has_no_url_for_a_port_it_does_not_declare(
    ports: list[int], port: int | None, message: str
) -> None:
    sidecar = {"name": "tools", "image": "tools:2", "ports": ports}
    sandbox = await AsyncSandbox(SidecarProvider(), SandboxSpec(image="task:1", sidecars=[sidecar])).start()

    with pytest.raises(ValueError, match=message):
        sandbox.sidecar("tools").url(port)
    await sandbox.stop()


async def test_serialize_and_connect_carry_declared_sidecars(tmp_path: Path) -> None:
    owner_provider = SidecarProvider()
    owner = await AsyncSandbox(
        owner_provider, SandboxSpec(image="task:1", sidecars=[{**_RECORDER, "ports": [9000]}])
    ).start()

    # The descriptor crosses a process boundary as JSON.
    descriptor = json.loads(json.dumps(await owner.serialize()))
    # Only name, image and ports travel; the sidecar's environment, resources and provider options (which may
    # hold registry credentials) stay with the owner.
    assert descriptor["sidecars"] == [{"name": "recorder", "image": _RECORDER["image"], "ports": [9000]}]

    borrower_provider = SidecarProvider()
    borrower = await AsyncSandbox.connect(descriptor, provider=borrower_provider, owns_provider=False)
    assert [(s.name, s.image) for s in borrower._spec.sidecars] == [("recorder", _RECORDER["image"])]
    assert borrower.sidecar("recorder").url() == "http://127.0.0.1:9000"

    result = await borrower.sidecar("recorder").exec("status", timeout_s=5)
    assert result.return_code == 0
    assert borrower_provider.sidecar_calls == [("exec", owner._require_handle().sandbox_id, "recorder", ["status"], 5)]
    with pytest.raises(ValueError, match="was not declared"):
        borrower.sidecar("other")

    await borrower.disconnect()
    await owner.stop()


async def test_connect_without_sidecars_declares_none() -> None:
    borrower = await AsyncSandbox.connect({"sandbox_id": "box-1"}, provider=SidecarProvider())

    assert borrower._spec.sidecars == ()
    with pytest.raises(ValueError, match=r"declared: \[\]"):
        borrower.sidecar("recorder")
    await borrower.disconnect()
