# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import re
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from nemo_gym.agents import AgentSeedRejectedError
from responses_api_agents.nemo_fabric_agent import sandbox_runtime
from responses_api_agents.nemo_fabric_agent.sandbox_runtime import (
    HEALTH,
    prepare_fabric_runtime,
    runtime_archive,
    runtime_requirements,
)


INSTALLER = Path(sandbox_runtime.__file__).with_name("install_fabric_runtime.sh")
CONFIGS = Path(sandbox_runtime.__file__).parent / "configs"
DOWNLOAD = "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download"


def _result(return_code=0, stdout="", stderr="", error_type=None):
    return SimpleNamespace(return_code=return_code, error_type=error_type, stdout=stdout, stderr=stderr)


class _Sandbox:
    """Answers the installer's probe, check and install commands, and records every command and upload."""

    def __init__(self, *, target="x86_64/gnu", healthy=False, install=None, probe_error=None):
        self.target, self.healthy, self.probe_error = target, healthy, probe_error
        self.install = install or _result()
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}

    async def upload(self, source, destination):
        self.uploads[destination] = Path(source).read_bytes()

    async def exec(self, command, *, cwd, timeout_s):
        self.commands.append(command)
        if " probe" in command:
            if self.probe_error:
                return _result(124, stderr="provider timed out", error_type=self.probe_error)
            return _result(0 if "/" in self.target else 1, self.target + "\n", "" if "/" in self.target else "no")
        if " check " in command:
            return _result(0 if self.healthy else 1)
        if " install " in command:
            return self.install
        return _result()

    def run(self, verb):
        return [command for command in self.commands if f" {verb} " in command or command.endswith(f" {verb}")]


@pytest.fixture
def python_archive(monkeypatch, tmp_path):
    payload = b"python runtime archive"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(sandbox_runtime, "PYTHON_SHA256", {"x86_64/gnu": digest})
    archive = tmp_path / "python.tar.gz"
    archive.write_bytes(payload)
    download = AsyncMock(return_value=archive)
    monkeypatch.setattr(DOWNLOAD, download)
    return SimpleNamespace(digest=digest, download=download, path=archive)


async def test_unsupported_platform_is_a_final_refusal_before_any_download(python_archive):
    sandbox = _Sandbox(target="aarch64/musl")

    with pytest.raises(AgentSeedRejectedError, match="Unsupported Fabric sandbox platform"):
        await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.claude", relay=False, timeout=60)

    python_archive.download.assert_not_awaited()
    assert not sandbox.run("install")
    # Setup artifacts of a failed attempt are kept for diagnosis, never deleted.
    assert not [command for command in sandbox.commands if command.startswith("rm -rf")]


async def test_a_probe_the_provider_could_not_run_stays_a_retryable_failure(python_archive):
    sandbox = _Sandbox(probe_error="timeout")

    with pytest.raises(RuntimeError, match="could not run: timeout") as error:
        await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.claude", relay=False, timeout=60)

    assert not isinstance(error.value, AgentSeedRejectedError)
    python_archive.download.assert_not_awaited()


async def test_healthy_cached_runtime_is_reused_without_download_or_install(python_archive):
    sandbox = _Sandbox(healthy=True)

    python = await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.claude", relay=False, timeout=60)

    fingerprint = runtime_archive(runtime_requirements("nvidia.fabric.claude", relay=False))[1]
    assert python == f"/opt/nemo-gym-fabric/{fingerprint}-x86_64-gnu/python/bin/python3"
    python_archive.download.assert_not_awaited()
    assert not sandbox.run("install")
    assert sandbox.commands[-1].startswith("rm -rf --")  # a finished setup removes its scratch files


async def test_install_uploads_both_archives_and_passes_their_digests_to_the_installer(python_archive):
    sandbox = _Sandbox()

    python = await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.claude", relay=True, timeout=60)

    requirements = runtime_requirements("nvidia.fabric.claude", relay=True)
    source, fingerprint = runtime_archive(requirements)
    (install,) = sandbox.run("install")
    assert f"/opt/nemo-gym-fabric/{fingerprint}-x86_64-gnu " in install
    assert hashlib.sha256(source).hexdigest() in install and python_archive.digest in install
    (python_upload,) = [path for path in sandbox.uploads if path.endswith("/python.tar.gz")]
    (source_upload,) = [path for path in sandbox.uploads if path.endswith("/source.tar.gz")]
    assert sandbox.uploads[python_upload] == python_archive.path.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(sandbox.uploads[source_upload])) as archive:
        assert archive.getnames() == ["runtime-requirements.txt"]
        assert archive.extractfile("runtime-requirements.txt").read().decode().split() == requirements
    assert python.endswith(f"{fingerprint}-x86_64-gnu/python/bin/python3")
    assert sandbox.commands[-1].startswith("rm -rf --")


async def test_failed_install_raises_with_the_installer_output_keeps_diagnostics_and_a_retry_installs_again(
    python_archive,
):
    sandbox = _Sandbox(install=_result(1, stderr="ERROR: no matching distribution for nemo-fabric"))

    with pytest.raises(RuntimeError, match="no matching distribution"):
        await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.claude", relay=False, timeout=60)
    assert not [command for command in sandbox.commands if command.startswith("rm -rf")]

    sandbox.install = _result()
    await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.claude", relay=False, timeout=60)

    first, second = sandbox.run("install")
    assert first.split()[3] == second.split()[3]  # the same runtime path: the installer is idempotent per runtime
    scratch = {path.rsplit("/", 1)[0] for path in sandbox.uploads if path.endswith(".tar.gz")}
    assert len(scratch) == 2  # but each attempt stages in its own scratch directory


async def test_python_archive_that_fails_its_digest_is_never_uploaded_or_installed(python_archive):
    python_archive.path.write_bytes(b"tampered")
    sandbox = _Sandbox()

    with pytest.raises(RuntimeError, match="checksum"):
        await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.claude", relay=False, timeout=60)

    assert not sandbox.run("install")
    assert not [path for path in sandbox.uploads if path.endswith(".tar.gz")]


def test_importing_the_agent_needs_no_other_agent_package():
    # harness_agent bundles only nemo_gym and this package for its in-sandbox runner.
    program = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.startswith('responses_api_agents.') and not name.startswith("
        "'responses_api_agents.nemo_fabric_agent'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import responses_api_agents.nemo_fabric_agent.app\n"
    )
    root = Path(sandbox_runtime.__file__).parents[2]
    done = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"},
    )
    assert done.returncode == 0, done.stderr[-2000:]


# --- the runtime's identity --------------------------------------------------------------------------------


def test_runtime_fingerprint_covers_the_installer_interpreter_digests_and_health_check(monkeypatch):
    requirements = runtime_requirements("nvidia.fabric.claude", relay=False)
    base = runtime_archive(requirements)[1]

    with monkeypatch.context() as patch:
        patch.setattr(sandbox_runtime, "PYTHON_SHA256", {"x86_64/gnu": "0" * 64})
        assert runtime_archive(requirements)[1] != base
    with monkeypatch.context() as patch:
        patch.setattr(sandbox_runtime, "HEALTH", HEALTH + "; assert True")
        assert runtime_archive(requirements)[1] != base
    with monkeypatch.context() as patch:
        real = Path.read_bytes
        patch.setattr(
            Path, "read_bytes", lambda self: real(self) + b"\n# edit" if self.name == INSTALLER.name else real(self)
        )
        assert runtime_archive(requirements)[1] != base
    assert runtime_archive(requirements)[1] == base


@pytest.mark.parametrize(
    "preset", sorted(path.stem for path in CONFIGS.glob("nemo_fabric_*.yaml") if path.stem != "nemo_fabric_agent")
)
def test_every_shipped_preset_has_a_sandbox_runtime_profile(preset):
    document = yaml.safe_load((CONFIGS / f"{preset}.yaml").read_text())[preset]
    adapter_id = document["responses_api_agents"]["nemo_fabric_agent"]["adapter_id"]

    requirements = runtime_requirements(adapter_id, relay=False)

    assert requirements[0].startswith("nemo-fabric[") and requirements[0].endswith(
        f"]=={sandbox_runtime.FABRIC_VERSION}"
    )


def test_sandbox_runtime_pins_match_the_host_requirements():
    # Keep the sandbox runtime pins aligned with the managed host project.
    project = tomllib.loads(Path(sandbox_runtime.__file__).with_name("pyproject.toml").read_text())
    host = "\n".join(project["project"]["dependencies"])
    installed = runtime_requirements("nvidia.fabric.hermes", relay=False) + runtime_requirements(
        "nvidia.fabric.openhands", relay=False
    )

    for pin in ["openhands-sdk==1.50.0", "openhands-tools==1.50.0", "litellm>=1.83.0,<1.103"]:
        assert pin in host and pin in installed, pin
    hermes = tomllib.loads(sandbox_runtime.HERMES_PROJECT.read_text())["project"]["dependencies"]
    assert runtime_requirements("nvidia.fabric.hermes", relay=False) == hermes
    assert not any("hermes-agent @" in dependency for dependency in project["project"]["dependencies"])
    assert re.search(rf"nemo-fabric\[[^\]]+\]=={re.escape(sandbox_runtime.FABRIC_VERSION)}", host)
    assert f"'{sandbox_runtime.FABRIC_VERSION}'" in HEALTH


async def test_hermes_installs_python314_with_its_own_dependency_overrides(python_archive, monkeypatch):
    monkeypatch.setattr(sandbox_runtime, "HERMES_PYTHON_SHA256", {"x86_64/gnu": python_archive.digest})
    sandbox = _Sandbox()

    await prepare_fabric_runtime(sandbox, adapter_id="nvidia.fabric.hermes", relay=True, timeout=60)

    url = python_archive.download.call_args.args[0]
    assert "cpython-3.14.9%2B20261021-x86_64" in url
    assert "(3,14,9)" in sandbox.run("install")[0]
    source = next(data for path, data in sandbox.uploads.items() if path.endswith("source.tar.gz"))
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        requirements = archive.extractfile("runtime-requirements.txt").read().decode()
        overrides = archive.extractfile("runtime-overrides.txt").read().decode()
    assert "nemo-relay==0.11.1" in requirements
    assert "hermes-agent[anthropic] @" in requirements
    assert "b200f5bc60e31588ba2cb60e288029c4a54ba5fa.tar.gz" in requirements
    assert "litellm>=1.83.0,<1.103" not in requirements  # belongs to the Python 3.13 adapter environment
    assert "cryptography==50.0.2" in overrides and "Pillow==13.0.0" in overrides


def test_hermes_overrides_are_part_of_the_cached_runtime_identity(monkeypatch):
    requirements = runtime_requirements("nvidia.fabric.hermes", relay=True)
    before = runtime_archive(requirements, hermes=True)[1]
    project = sandbox_runtime._hermes_project()
    project["tool"]["uv"]["override-dependencies"].append("example==1")
    monkeypatch.setattr(sandbox_runtime, "_hermes_project", lambda: project)
    assert runtime_archive(requirements, hermes=True)[1] != before
