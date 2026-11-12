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

"""End-to-end tests for the public sandbox facade backed by the real enroot provider.

These exercise ``Sandbox``/``AsyncSandbox`` wired to ``{"enroot": {}}`` against a
real ``enroot`` binary and a tiny image (busybox, ~4 MB) so CI stays fast. They
skip gracefully when enroot is not installed or the runner cannot actually launch
enroot containers (e.g. no network to import the image, or unprivileged
namespaces), per the repo's external-tool test policy.
"""

import asyncio
import shutil

import pytest

from nemo_gym.sandbox.api import AsyncSandbox, Sandbox
from nemo_gym.sandbox.providers import SandboxSpec


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(shutil.which("enroot") is None, reason="enroot is not installed"),
]


# Smallest image that satisfies the provider's busybox-friendly init command and
# readiness probe (see DEFAULT_INIT_COMMAND / READY_PROBE_COMMAND in provider.py).
SMALL_IMAGE = "busybox:latest"


def _provider(enroot_home: str) -> dict:
    """Provider config that pins enroot's data/cache under a throwaway dir."""
    return {"enroot": {"create": {"base_dir": enroot_home}}}


@pytest.fixture(scope="module")
def enroot_home(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A private enroot home shared by the module's tests.

    Doubles as an environment guard: it boots one real busybox sandbox and runs a
    trivial command. If enroot cannot import the image or launch a container on
    this runner, the whole module is skipped instead of failing.
    """
    home = str(tmp_path_factory.mktemp("enroot-home"))
    try:
        with Sandbox(_provider(home), SandboxSpec(image=SMALL_IMAGE)) as sandbox:
            sandbox.start()
            probe = sandbox.exec("printf ok")
            if probe.return_code != 0 or (probe.stdout or "").strip() != "ok":
                pytest.skip(f"enroot present but not functional on this runner: {probe.stderr!r}")
    except Exception as exc:  # noqa: BLE001 - any boot failure means "can't run here"
        pytest.skip(f"enroot cannot launch containers on this runner: {exc}")
    return home


def test_sync_sandbox_end_to_end(enroot_home: str, tmp_path) -> None:
    """The first snippet: env, seeded files, exec, upload, download, auto-cleanup."""
    local_script = tmp_path / "local_script.sh"
    local_script.write_text("#!/bin/sh\necho 'I am a local script'\n", encoding="utf-8")
    downloaded = tmp_path / "result.txt"

    spec = SandboxSpec(
        image=SMALL_IMAGE,
        workdir="/sandbox",
        env={"GREETING": "hello"},
        files={"/sandbox/input.txt": "some seed content"},
    )

    with Sandbox(_provider(enroot_home), spec) as sandbox:
        sandbox.start()

        # Env var reaches the container (via `enroot exec -e`) and the seeded file
        # was staged into the sandbox before start.
        result = sandbox.exec("echo $GREETING && cat /sandbox/input.txt")
        assert result.return_code == 0
        assert result.stdout is not None
        assert result.stdout.splitlines() == ["hello", "some seed content"]

        # Upload a host file, have the container produce result.txt, then download it.
        sandbox.upload(str(local_script), "/sandbox/script.sh")
        produced = sandbox.exec("cat /sandbox/script.sh > /sandbox/result.txt")
        assert produced.return_code == 0

        sandbox.download("/sandbox/result.txt", str(downloaded))

    assert downloaded.read_text(encoding="utf-8") == local_script.read_text(encoding="utf-8")


def test_sync_sandbox_env_override_and_cwd(enroot_home: str) -> None:
    """Per-call env overrides create-time env, and workdir is applied to exec."""
    spec = SandboxSpec(image=SMALL_IMAGE, workdir="/sandbox", env={"GREETING": "hello"})
    with Sandbox(_provider(enroot_home), spec) as sandbox:
        sandbox.start()

        overridden = sandbox.exec("echo $GREETING", env={"GREETING": "override"})
        assert overridden.return_code == 0
        assert (overridden.stdout or "").strip() == "override"

        where = sandbox.exec("pwd")
        assert where.return_code == 0
        assert (where.stdout or "").strip() == "/sandbox"


def test_async_sandbox_end_to_end(enroot_home: str) -> None:
    """The second snippet: async facade, start, exec, context-managed cleanup."""

    async def run() -> None:
        spec = SandboxSpec(image=SMALL_IMAGE, workdir="/sandbox")
        async with AsyncSandbox(_provider(enroot_home), spec) as sandbox:
            await sandbox.start()
            result = await sandbox.exec("uname -a")
            assert result.return_code == 0
            assert "Linux" in (result.stdout or "")

    asyncio.run(run())
