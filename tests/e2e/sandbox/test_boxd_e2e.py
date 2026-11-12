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

"""Live checks of the boxd sandbox provider against real boxd machines.

Skipped unless ``NEMO_GYM_BOXD_E2E=1``. Needs ``BOXD_API_KEY`` and the boxd extra::

    uv sync --extra dev --extra boxd
    NEMO_GYM_BOXD_E2E=1 BOXD_API_KEY=... BOXD_REGION=nyc pytest tests/e2e/sandbox/test_boxd_e2e.py -v

Each test creates its own machines and deletes them with ``stop()``. ``NEMO_GYM_BOXD_E2E_NAME_PREFIX``
sets the machine name prefix (default ``nemo-gym-e2e-``) and ``NEMO_GYM_BOXD_E2E_LABELS`` adds labels.
"""

import asyncio
import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nemo_gym.core.failure_kinds import PROVIDER_CREDENTIALS_REJECTED
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.boxd import provider as boxd_provider
from nemo_gym.sandbox.providers.boxd.provider import BoxdMachine


pytestmark = pytest.mark.skipif(
    os.getenv("NEMO_GYM_BOXD_E2E") != "1" or not os.getenv("BOXD_API_KEY"),
    reason="creates real boxd machines; set NEMO_GYM_BOXD_E2E=1 and BOXD_API_KEY to run",
)

IMAGE = "python:3.13-slim"


def _extra_labels() -> dict[str, str]:
    """``NEMO_GYM_BOXD_E2E_LABELS="k=v,k2=v2"`` adds labels to every machine the tests create."""
    pairs = [item.split("=", 1) for item in os.getenv("NEMO_GYM_BOXD_E2E_LABELS", "").split(",") if "=" in item]
    return {key.strip(): value.strip() for key, value in pairs}


def _provider_config() -> dict:
    connection = {
        key: value for key, value in (("org", os.getenv("BOXD_ORG")), ("region", os.getenv("BOXD_REGION"))) if value
    }
    return {
        "boxd": {
            "connection": connection,
            "create": {
                "name_prefix": os.getenv("NEMO_GYM_BOXD_E2E_NAME_PREFIX", "nemo-gym-e2e-"),
                "labels": {"suite": "nemo-gym-boxd-e2e", **_extra_labels()},
            },
            "attribution": {"enabled": False},
        }
    }


def _timed(label: str, started: float) -> None:
    print(f"[boxd-e2e] {label}: {time.monotonic() - started:.2f}s", flush=True)


async def _machine(sandbox: AsyncSandbox):
    """The SDK view of the sandbox's machine, read through the provider's own client."""
    provider = sandbox._provider
    return await provider._sdk_client().machines.get(sandbox._require_handle().sandbox_id)


def test_machine_mode_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        spec = SandboxSpec(
            workdir="/workspace", env={"SPEC_VAR": "from spec"}, resources={"cpu": 2, "memory_mib": 8192}
        )
        sandbox = AsyncSandbox(_provider_config(), spec)
        started = time.monotonic()
        await sandbox.start()
        _timed("create to ready (machine mode)", started)
        machine_id = sandbox._require_handle().sandbox_id
        print(f"[boxd-e2e] machine {machine_id}", flush=True)
        try:
            machine = await _machine(sandbox)
            assert machine.networking.isolated is True
            assert machine.labels["suite"] == "nemo-gym-boxd-e2e"
            # No ttl_s in the spec: the machine carries create.max_lifetime_s, one hour.
            lifetime_left = (machine.delete_at - datetime.now(timezone.utc)).total_seconds()
            assert 3300 < lifetime_left <= 3600
            print(f"[boxd-e2e] default lifetime left: {lifetime_left:.0f}s", flush=True)
            assert await sandbox.status() is SandboxStatus.RUNNING

            started = time.monotonic()
            result = await sandbox.exec("echo out; echo err >&2; exit 7", timeout_s=30)
            _timed("first exec", started)
            assert (result.stdout, result.stderr, result.return_code) == ("out\n", "err\n", 7)

            result = await sandbox.exec('echo "$SPEC_VAR|$CALL_VAR|$(pwd)"', env={"CALL_VAR": "a b"}, timeout_s=30)
            assert result.stdout == "from spec|a b|/workspace\n"

            result = await sandbox.exec("id -un", user="root", timeout_s=30)
            assert result.stdout.strip() == "root"

            started = time.monotonic()
            result = await sandbox.exec("sleep 30; echo late", timeout_s=3)
            _timed("timed-out exec", started)
            assert result.error_type == "timeout" and result.return_code == 124
            check = await sandbox.exec("pgrep -f '[s]leep 30' || echo none", timeout_s=30)
            assert check.stdout.strip() == "none"

            # A keepalive every 2 s on a silent command: the markers are removed, the command's own bytes stay.
            monkeypatch.setattr(boxd_provider, "KEEPALIVE_AFTER_S", 1.0)
            monkeypatch.setattr(boxd_provider, "KEEPALIVE_INTERVAL_S", 2)
            started = time.monotonic()
            result = await sandbox.exec("sleep 7; printf 'a\\0b' >&2; echo quiet; exit 5", timeout_s=60)
            _timed("silent command with keepalive", started)
            assert (result.stdout, result.stderr, result.return_code) == ("quiet\n", "a\0b", 5)
            monkeypatch.undo()

            data = os.urandom(5 * 1024 * 1024)
            source = tmp_path / "blob.bin"
            source.write_bytes(data)
            started = time.monotonic()
            await sandbox.upload(source, "/workspace/blob.bin")
            back = tmp_path / "back.bin"
            await sandbox.download("/workspace/blob.bin", back)
            _timed("5 MiB upload and download", started)
            assert hashlib.sha256(back.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest()
            remote = await sandbox.exec("sha256sum /workspace/blob.bin", timeout_s=30)
            assert remote.stdout.split()[0] == hashlib.sha256(data).hexdigest()

            # A path the machine's user cannot write.
            (tmp_path / "conf").write_text("root-owned\n")
            await sandbox.upload(tmp_path / "conf", "/opt/gym/conf.txt")
            await sandbox.download("/opt/gym/conf.txt", tmp_path / "conf.back")
            assert (tmp_path / "conf.back").read_text() == "root-owned\n"

            await sandbox.exec("nohup sleep 3600 >/dev/null 2>&1 &", timeout_s=30)
            await sandbox.exec("echo kept > /workspace/state.txt", timeout_s=30)

            started = time.monotonic()
            await sandbox.pause()
            _timed("pause (hibernate)", started)
            assert await sandbox.status() is SandboxStatus.PAUSED
            started = time.monotonic()
            await sandbox.resume()
            _timed("resume (wake) to ready", started)
            assert await sandbox.status() is SandboxStatus.RUNNING
            after = await sandbox.exec("pgrep -f '[s]leep 3600' >/dev/null && cat /workspace/state.txt", timeout_s=30)
            assert after.stdout == "kept\n"
        finally:
            await sandbox.stop()
        # stop() deleted the machine: boxd no longer knows the id.
        provider = AsyncSandbox(_provider_config())._provider
        gone = SandboxHandle(
            sandbox_id=machine_id, provider_name="boxd", raw=BoxdMachine(machine_id=machine_id, name="")
        )
        assert await provider.status(gone) is SandboxStatus.STOPPED
        await provider.aclose()

    asyncio.run(scenario())


def test_container_mode_runs_the_image(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec = SandboxSpec(
            image=IMAGE,
            workdir="/sandbox",
            env={"GREETING": "hello from boxd"},
            files={"/sandbox/hello.py": "import os, sys\nprint(os.environ['GREETING'], sys.version_info[:2])\n"},
        )
        sandbox = AsyncSandbox(_provider_config(), spec)
        started = time.monotonic()
        await sandbox.start()
        _timed(f"create to ready (container mode, {IMAGE})", started)
        print(f"[boxd-e2e] machine {sandbox._require_handle().sandbox_id}", flush=True)
        try:
            result = await sandbox.exec("python hello.py", timeout_s=60)
            assert result.return_code == 0, result.stderr
            assert result.stdout == "hello from boxd (3, 13)\n"

            result = await sandbox.exec("cat /etc/os-release | head -1; id -u; pwd", timeout_s=30)
            assert "Debian" in result.stdout
            assert result.stdout.splitlines()[1:] == ["0", "/sandbox"]

            result = await sandbox.exec("echo $X", env={"X": "per call"}, cwd="/tmp", timeout_s=30)
            assert result.stdout == "per call\n"

            data = os.urandom(1024 * 1024)
            (tmp_path / "in.bin").write_bytes(data)
            await sandbox.upload(tmp_path / "in.bin", "/sandbox/data/in.bin")
            listing = await sandbox.exec("stat -c '%U %s' /sandbox/data/in.bin", timeout_s=30)
            assert listing.stdout.strip() == f"root {len(data)}"
            await sandbox.download("/sandbox/data/in.bin", tmp_path / "out.bin")
            assert (tmp_path / "out.bin").read_bytes() == data

            result = await sandbox.exec("sleep 20", timeout_s=2)
            assert result.error_type == "timeout"
            assert await sandbox.status() is SandboxStatus.RUNNING

            # Another process rebuilds the handle from the serialized descriptor.
            descriptor = await sandbox._provider.serialize_handle(sandbox._require_handle())
            other = AsyncSandbox(_provider_config())
            handle = await other._provider.connect(descriptor)
            again = await other._provider.exec(handle, "python -c 'print(6*7)'", timeout_s=30)
            assert again.stdout == "42\n"
            await other._provider.aclose()
        finally:
            await sandbox.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(("image", "cancel_after_s"), [(None, 0.0), (None, 0.1), (None, 0.4), (IMAGE, 1.5)])
def test_cancelled_start_leaves_no_machine(image: str | None, cancel_after_s: float) -> None:
    """Cancel start() before and during the create call, and during the image pull: no machine is left."""
    run_label = uuid.uuid4().hex[:12]

    async def scenario() -> None:
        config = _provider_config()
        config["boxd"]["create"]["labels"]["cancel-run"] = run_label
        sandbox = AsyncSandbox(config, SandboxSpec(image=image))
        client = sandbox._provider._sdk_client()
        start = asyncio.ensure_future(sandbox.start())
        await asyncio.sleep(cancel_after_s)
        start.cancel()
        started = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await start
        _timed(f"cancel after {cancel_after_s}s, cleanup", started)
        # Past the window in which the provider looks for a machine that appears late.
        await asyncio.sleep(boxd_provider.ORPHAN_LOOKUP_S + 5)
        left = await client.machines.list(org=os.getenv("BOXD_ORG"), labels={"cancel-run": run_label})
        print(f"[boxd-e2e] machines left after cancel at {cancel_after_s}s: {[m.id for m in left]}", flush=True)
        assert list(left) == []
        await sandbox._provider.aclose()

    asyncio.run(scenario())


def test_wrong_api_key_is_a_credentials_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOXD_API_KEY", "not-a-real-key")
    monkeypatch.delenv("BOXD_TOKEN", raising=False)

    async def scenario() -> None:
        sandbox = AsyncSandbox(_provider_config(), SandboxSpec())
        with pytest.raises(NgFailureEvent) as raised:
            await sandbox.start()
        assert raised.value.failure_kind == PROVIDER_CREDENTIALS_REJECTED
        print(f"[boxd-e2e] wrong key: {raised.value.failure_reason} / {raised.value.__cause__!r}", flush=True)

    asyncio.run(scenario())


def test_missing_api_key_is_a_credentials_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOXD_API_KEY")
    monkeypatch.delenv("BOXD_TOKEN", raising=False)
    # Leave nothing in the environment that names a boxd credential.
    for name in ("BOXD_VM_ID", "BOXD_VM_NAME", "BOXD_PROXY_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)

    async def scenario() -> None:
        sandbox = AsyncSandbox(_provider_config(), SandboxSpec())
        with pytest.raises(NgFailureEvent) as raised:
            await sandbox.start()
        assert raised.value.failure_kind == PROVIDER_CREDENTIALS_REJECTED
        print(f"[boxd-e2e] missing key: {raised.value.__cause__!r}", flush=True)

    asyncio.run(scenario())
