# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Setup without network: a prepared bootstrap directory supplies uv, Python and every wheel mini-SWE installs."""

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.miniswe_sandboxed_agent import bootstrap
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig, MiniSWEHarness


class _RecordingSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: list[str] = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        stdout = "x86_64\ngnu\n" if "uname -m" in command else ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def upload(self, local, remote):
        self.uploads.append(remote)


def _offline_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "bootstrap"
    directory.mkdir()
    (directory / "uv-x86_64-unknown-linux-musl.tar.gz").write_bytes(b"uv")
    python = directory / "cpython-3.13.12%2B20260310-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    python.write_bytes(b"python")
    with tarfile.open(directory / "wheels-x86_64-gnu.tar.gz", "w:gz"):
        pass
    monkeypatch.setitem(bootstrap.PYTHON_SHA256, ("x86_64", "gnu"), hashlib.sha256(b"python").hexdigest())
    return directory


class TestOfflineSetup:
    def test_offline_assets_need_every_file_and_the_pinned_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _offline_dir(tmp_path, monkeypatch)
        uv, python, wheels = bootstrap.offline_assets(directory, "x86_64", "gnu")
        assert wheels.name == "wheels-x86_64-gnu.tar.gz"
        python.write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="checksum"):
            bootstrap.offline_assets(directory, "x86_64", "gnu")
        with pytest.raises(RuntimeError, match="missing"):
            bootstrap.offline_assets(directory, "x86_64", "musl")

    async def test_install_uses_the_uploaded_wheelhouse_and_no_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _offline_dir(tmp_path, monkeypatch)
        sandbox = _RecordingSandbox()
        harness = MiniSWEHarness(
            sandbox=sandbox,
            context=HarnessContext(session_id="s", instruction="task"),
            config=MiniSWEConfig(),
            params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            model_base_url="http://gateway:1/v1",
            model_name="policy_model",
            directory=tmp_path / "artifacts",
            bootstrap_dir=directory,
        )
        await harness._install_miniswe()
        install = sandbox.commands[-1]
        assert any(remote.endswith("/wheels.tar.gz") for remote in sandbox.uploads)
        assert "--no-index --find-links" in install and "pip install --no-index" in install
        assert "mini-swe-agent==2.4.6" in install

    def test_prepare_writes_a_wheelhouse_per_platform_and_a_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(url, directory, *, sha256=None):
            path = Path(directory) / url.rsplit("/", 1)[1]
            path.write_bytes(b"asset")
            return path

        def fake_pip(command, check):
            destination = Path(command[command.index("--dest") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "mini_swe_agent-2.4.6-py3-none-any.whl").write_bytes(b"wheel")
            assert "--only-binary=:all:" in command and "--python-version" in command

        monkeypatch.setattr(bootstrap, "_fetch", fake_download)
        monkeypatch.setattr(bootstrap.subprocess, "run", fake_pip)
        manifest = bootstrap.prepare(tmp_path / "out", platforms=[("x86_64", "gnu")], pip=["pip"])
        entry = manifest["platforms"]["x86_64-gnu"]
        assert entry["wheel_files"] == ["mini_swe_agent-2.4.6-py3-none-any.whl"]
        assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == manifest
        with tarfile.open(tmp_path / "out" / entry["wheels"]) as tar:
            assert tar.getnames() == ["mini_swe_agent-2.4.6-py3-none-any.whl"]
