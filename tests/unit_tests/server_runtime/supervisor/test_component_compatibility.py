# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real offline installer regressions for unchanged custom/parent environments."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.cli import _locked_env
from nemo_gym.cli.env import _resolve_server_dir


@pytest.mark.parametrize("manifest", ["requirements.txt", "pyproject.toml"])
def test_custom_component_discovery_keeps_existing_manifest(tmp_path, monkeypatch, manifest):
    server = tmp_path / "resources_servers/custom"
    server.mkdir(parents=True)
    (server / manifest).touch()
    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
    assert _resolve_server_dir(Path("resources_servers/custom")) == server


def _wheel(directory: Path, name: str, version: str, requires: tuple[str, ...] = ()) -> Path:
    stem = name.replace("-", "_")
    info = f"{stem}-{version}.dist-info"
    path = directory / f"{stem}-{version}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {value}\n" for value in requires)
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{stem}.py", f"VERSION = {version!r}\n")
        wheel.writestr(f"{info}/METADATA", metadata)
        wheel.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        wheel.writestr(f"{info}/RECORD", "")
    return path


@pytest.mark.parametrize("ray_dependency", [False, True])
@pytest.mark.parametrize("parent_openai", ["2.43.0", "2.44.0"])
def test_parent_only_provider_and_differing_pins_are_installed_without_lock_mutation(
    tmp_path, monkeypatch, ray_dependency, parent_openai
):
    if shutil.which("uv") is None:
        pytest.skip("uv is required for the real compatibility installer regression")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name, version in [("openai", "2.43.0"), ("openai", "2.44.0"), ("ray", "2.50.0"), ("ray", "2.51.2")]:
        _wheel(wheels, name, version)
    # The checker runs inside the component interpreter; include its real
    # packaging dependency without a registry or a mock checker implementation.
    packaging = importlib.metadata.distribution("packaging")
    packaging_wheel = wheels / f"packaging-{packaging.version}-py3-none-any.whl"
    with zipfile.ZipFile(packaging_wheel, "w") as archive:
        for file in packaging.files:
            if str(file).startswith(("packaging/", f"packaging-{packaging.version}.dist-info/")):
                archive.write(packaging.locate_file(file), str(file))
    _wheel(wheels, "provider-helper", "1.0")
    provider = _wheel(wheels, "custom-provider", "1.0", ("provider-helper==1.0",))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_NO_INDEX", "true")
    monkeypatch.setenv("UV_OFFLINE", "true")
    monkeypatch.setenv("UV_FIND_LINKS", str(wheels))
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    project = tmp_path / "component"
    project.mkdir()
    dependencies = ["openai>=2.43.0,<3", "packaging"] + (["ray==2.51.2"] if ray_dependency else [])
    metadata = '[project]\nname="sample"\nversion="0"\ndependencies=' + json.dumps(dependencies)
    metadata += "\n[tool.uv]\npackage=false\n"
    (project / "pyproject.toml").write_text(metadata)
    (project / "uv.toml").write_text("no-index=true\nfind-links=" + json.dumps([str(wheels)]) + "\n")
    subprocess.run(
        [
            "uv",
            "lock",
            "--offline",
            "--no-index",
            "--find-links",
            str(wheels),
            "--project",
            str(project),
            "--python",
            sys.executable,
        ],
        check=True,
    )
    original = {name: (project / name).read_bytes() for name in ("pyproject.toml", "uv.lock")}
    # Model the already-installed parent's PEP 610 provider requirement, without
    # installing test wheels into the test runner's real environment.
    parent = tmp_path / "parent"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(parent)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(parent / "bin/python"), str(provider)], check=True)
    inherited = subprocess.check_output(
        [
            str(parent / "bin/python"),
            "-c",
            'import importlib.metadata,json; d=importlib.metadata.distribution("custom-provider"); '
            'print("custom-provider @ " + json.loads(d.read_text("direct_url.json"))["url"])',
        ],
        text=True,
    ).strip()
    venv = tmp_path / "server"
    _locked_env.sync_environment(
        project, venv, sys.executable, [f"openai=={parent_openai}", inherited], ["ray==2.50.0"]
    )
    script = """import importlib.metadata as m, json
result = {name: m.version(name) for name in ('openai', 'custom-provider', 'provider-helper')}
try:
    result['ray'] = m.version('ray')
except m.PackageNotFoundError:
    result['ray'] = None
print(json.dumps(result))
"""
    installed = json.loads(subprocess.check_output([str(venv / "bin/python"), "-c", script], text=True))
    assert installed == {
        "openai": parent_openai,
        "custom-provider": "1.0",
        "provider-helper": "1.0",
        "ray": "2.50.0" if ray_dependency else None,
    }
    assert {name: (project / name).read_bytes() for name in original} == original


def test_parent_checker_crash_does_not_enable_compatibility(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="sample"\nversion="0"\n')
    (tmp_path / "uv.lock").touch()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--check" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(_locked_env.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        _locked_env.sync_environment(tmp_path, tmp_path / "venv", sys.executable, [], [])
    assert len(calls) == 2
    assert calls[0][:3] == ["uv", "sync", "--locked"]
