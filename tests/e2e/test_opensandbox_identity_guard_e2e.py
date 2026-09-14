# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end check that the OpenSandbox provider refuses misrouted sandboxes (RL-1469).

The OpenSandbox server resolves a sandbox to a pod IP from an annotation that
is never cleared. When a sandbox dies (OOM) and its IP is reused, every request
for the dead sandbox is served by whichever live sandbox inherited the IP. The
harness then grades, uploads tests into, and reads results from a stranger's
sandbox.

These tests reproduce the client-visible shape of that failure without waiting
for an OOM: sandbox A's id is placed over sandbox B's connection. A provider
that enforces sandbox identity must refuse to exec, upload, or download through
the cross-wired handle, while the correctly wired handle keeps working.

Needs a reachable OpenSandbox server. Credentials come from ``OPENSANDBOX_DOMAIN``
and ``OPENSANDBOX_API_KEY``, or from a ``KEY=VALUE`` file (default ``~/.cell3``,
override with ``NEMO_GYM_OPENSANDBOX_CREDS_FILE``). Skipped when neither exists.
"""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

from nemo_gym.sandbox.providers.base import (
    SandboxEndedError,
    SandboxHandle,
    SandboxMisrouteError,
    SandboxResources,
    SandboxSpec,
)
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider


IMAGE = "docker.io/library/python:3.12-slim"


def _load_credentials() -> tuple[str, str] | None:
    domain = os.environ.get("OPENSANDBOX_DOMAIN")
    api_key = os.environ.get("OPENSANDBOX_API_KEY")
    if domain and api_key:
        return domain, api_key
    creds_path = Path(os.environ.get("NEMO_GYM_OPENSANDBOX_CREDS_FILE", "~/.cell3")).expanduser()
    if not creds_path.is_file():
        return None
    values: dict[str, str] = {}
    for line in creds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    domain = values.get("OPENSANDBOX_DOMAIN")
    api_key = values.get("OPENSANDBOX_API_KEY")
    if domain and api_key:
        return domain, api_key
    return None


pytestmark = pytest.mark.skipif(_load_credentials() is None, reason="OpenSandbox credentials are not available")


def _make_provider() -> OpenSandboxProvider:
    domain, api_key = _load_credentials()
    return OpenSandboxProvider(
        connection={
            "domain": domain,
            "api_key": api_key,
            "use_server_proxy": True,
            "transport_backend": "aiohttp",
            "tls_verify": False,
            "request_timeout_s": 120,
        },
        create={"timeout_s": 900, "connect_attempt_timeout_s": 900, "retries": 1},
    )


def _spec() -> SandboxSpec:
    return SandboxSpec(
        image=IMAGE,
        ttl_s=1200,
        ready_timeout_s=600,
        resources=SandboxResources.from_mapping({"cpu": 0.25, "memory_mib": 512, "disk_gib": 5}),
        metadata={"harness": "nemo-gym-e2e-identity-guard"},
    )


async def _close_quietly(provider: OpenSandboxProvider, handle: SandboxHandle | None) -> None:
    if handle is None:
        return
    try:
        await provider.close(handle)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


async def test_cross_wired_handle_is_refused_while_correct_handle_keeps_working(tmp_path: Path) -> None:
    provider = _make_provider()
    sandbox_a: SandboxHandle | None = None
    sandbox_b: SandboxHandle | None = None
    try:
        sandbox_a = await provider.create(_spec())
        sandbox_b = await provider.create(_spec())
        assert sandbox_a.sandbox_id != sandbox_b.sandbox_id

        # Sandbox A's identity over sandbox B's connection: what the client sees
        # when the server routes A's requests to the pod that took over A's IP.
        cross_wired = SandboxHandle(
            sandbox_id=sandbox_a.sandbox_id,
            provider_name=sandbox_a.provider_name,
            raw=sandbox_b.raw,
        )
        marker_dir = f"/tmp/ng-identity-guard-{uuid4().hex}"
        grader = tmp_path / "test.sh"
        grader.write_text("echo grader\n", encoding="utf-8")

        with pytest.raises(SandboxMisrouteError):
            await provider.exec(
                cross_wired, f"mkdir -p {marker_dir} && echo executed > {marker_dir}/exec", timeout_s=60
            )

        with pytest.raises(SandboxMisrouteError):
            await provider.upload_file(cross_wired, grader, f"{marker_dir}/test.sh")

        with pytest.raises(SandboxMisrouteError):
            await provider.download_file(cross_wired, "/etc/hostname", tmp_path / "stolen-hostname")

        leftovers = await provider.exec(sandbox_b, f"ls -A {marker_dir} 2>/dev/null | wc -l", timeout_s=60)
        assert leftovers.return_code == 0, leftovers
        assert (leftovers.stdout or "").strip() == "0", f"cross-wired traffic reached sandbox B: {leftovers!r}"

        # The correctly wired handle must be unaffected by the guard.
        mkdir = await provider.exec(sandbox_a, f"mkdir -p {marker_dir}", timeout_s=60)
        assert mkdir.return_code == 0, mkdir
        await provider.upload_file(sandbox_a, grader, f"{marker_dir}/test.sh")
        result = await provider.exec(sandbox_a, f"cat {marker_dir}/test.sh && cat /etc/hostname", timeout_s=60)
        assert result.return_code == 0, result
        assert (result.stdout or "").splitlines() == ["echo grader", f"{sandbox_a.sandbox_id}-0"]
        await provider.download_file(sandbox_a, f"{marker_dir}/test.sh", tmp_path / "roundtrip.sh")
        assert (tmp_path / "roundtrip.sh").read_text(encoding="utf-8") == "echo grader\n"
    finally:
        await _close_quietly(provider, sandbox_a)
        await _close_quietly(provider, sandbox_b)
        await provider.aclose()


async def test_operations_on_a_terminated_sandbox_raise_sandbox_ended_error() -> None:
    provider = _make_provider()
    created: SandboxHandle | None = None
    survivor: SandboxHandle | None = None
    try:
        created = await provider.create(_spec())
        # A second handle to the same sandbox keeps its own client open, so the
        # failure below comes from the server, not from locally closed resources.
        survivor = await provider.connect({"sandbox_id": created.sandbox_id})
        await provider.close(created)
        created = None

        deadline = asyncio.get_running_loop().time() + 90.0
        raised: BaseException | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                await provider.exec(survivor, "true", timeout_s=30)
            except BaseException as e:  # noqa: BLE001 - we assert on the type below
                raised = e
                break
            await asyncio.sleep(2.0)
        assert raised is not None, "exec on a terminated sandbox kept succeeding"
        assert isinstance(raised, SandboxEndedError), (
            f"expected SandboxEndedError, got {type(raised).__name__}: {raised}"
        )
    finally:
        await _close_quietly(provider, created)
        await provider.aclose()
