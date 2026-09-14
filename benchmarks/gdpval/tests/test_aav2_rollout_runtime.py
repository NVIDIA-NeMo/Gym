# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.gdpval.hsg.aav2 import rollout_runtime as runtime


def _write(path: Path, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fixture\n", encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


@pytest.mark.parametrize("escape", ["file", "parent", "prefix-lookalike"])
def test_node_local_path_rejects_resolved_shared_files(tmp_path: Path, escape: str) -> None:
    local = tmp_path / "local"
    local.mkdir()
    shared = _write(tmp_path / "local-shared" / "bin" / "python")
    if escape == "file":
        candidate = local / "python"
        candidate.symlink_to(shared)
    elif escape == "parent":
        (local / "python-home").symlink_to(shared.parent, target_is_directory=True)
        candidate = local / "python-home" / shared.name
    else:
        candidate = shared
    with pytest.raises(ValueError, match="outside node-local storage"):
        runtime.assert_node_local(candidate, local_root=local)


@pytest.fixture
def local_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    local = tmp_path / "node-local"
    gym, venvs = local / "gym", local / "component-venvs"
    gym.mkdir(parents=True)
    venvs.mkdir()
    python = _write(local / "python/bin/python", executable=True)
    venv_python = gym / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(python)
    interpreter = SimpleNamespace(
        executable=str(venv_python),
        _base_executable=str(python),
        prefix=str(gym / ".venv"),
        base_prefix=str(python.parent.parent),
    )
    monkeypatch.setattr(runtime, "sys", interpreter)
    for name in runtime.LOCAL_ENV_PATHS:
        path = local / "cache" / name.lower()
        path.mkdir(parents=True)
        monkeypatch.setenv(name, str(path))
    uv = _write(local / "bin/uv", executable=True)
    _write(local / "bin/apptainer", executable=True)
    sif = _write(local / "assets/agent.sif")
    monkeypatch.setenv("MARS_UV", str(uv))
    monkeypatch.setenv("PATH", str(uv.parent))
    monkeypatch.setenv("APPTAINER_BIN", str(uv.parent))
    monkeypatch.setenv("GDPVAL_CONTAINER_PATH", str(sif))
    monkeypatch.chdir(gym)
    _write(gym / "fixture_rollout_probe/__init__.py")
    _write(gym / "fixture_rollout_probe/main.py")
    monkeypatch.syspath_prepend(str(gym))
    try:
        yield SimpleNamespace(root=local, gym=gym, venvs=venvs, interpreter=interpreter, python=python)
    finally:
        for name in ("fixture_rollout_probe.main", "fixture_rollout_probe"):
            sys.modules.pop(name, None)


def _verify(fixture: SimpleNamespace) -> dict:
    return runtime.verify_runtime(
        fixture.gym,
        fixture.venvs,
        local_root=fixture.root,
        modules=["fixture_rollout_probe", "fixture_rollout_probe.main"],
    )


def test_runtime_reports_resolved_interpreter_and_requested_module_paths(local_runtime) -> None:
    report = _verify(local_runtime)
    assert report["executable"] == str(local_runtime.python.resolve())
    assert report["module:fixture_rollout_probe.main"] == str(local_runtime.gym / "fixture_rollout_probe/main.py")
    assert report["UV_CACHE_DIR"] == os.environ["UV_CACHE_DIR"]
    assert report["GDPVAL_CONTAINER_PATH"] == os.environ["GDPVAL_CONTAINER_PATH"]


@pytest.mark.parametrize("attribute", ["executable", "_base_executable", "prefix", "base_prefix"])
def test_runtime_rejects_shared_python_even_when_the_venv_is_local(local_runtime, tmp_path, attribute) -> None:
    setattr(local_runtime.interpreter, attribute, str(_write(tmp_path / "shared/python")))
    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


@pytest.mark.parametrize(
    "variable",
    ["TMPDIR", "UV_CACHE_DIR", "HF_HOME", "HF_DATASETS_CACHE", "GDPVAL_REF_FILES_DIR", "GDPVAL_CONTAINER_PATH"],
)
def test_runtime_rejects_shared_cache_and_asset_paths(local_runtime, tmp_path, monkeypatch, variable) -> None:
    monkeypatch.setenv(variable, str(_write(tmp_path / "shared/asset")))
    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


def test_runtime_rejects_missing_cache_contract(local_runtime, monkeypatch) -> None:
    monkeypatch.delenv("UV_CACHE_DIR")
    with pytest.raises(ValueError, match="UV_CACHE_DIR"):
        _verify(local_runtime)


@pytest.mark.parametrize("location", ["shared", "other-local-source", "source-symlink"])
def test_runtime_rejects_imports_outside_the_staged_source(local_runtime, tmp_path, monkeypatch, location) -> None:
    directory = local_runtime.root / "other" if location == "other-local-source" else tmp_path / "shared"
    source = _write(directory / "fixture_rollout_probe/__init__.py")
    if location == "source-symlink":
        local = local_runtime.gym / "fixture_rollout_probe/__init__.py"
        local.unlink()
        local.symlink_to(source)
    else:
        monkeypatch.syspath_prepend(str(directory))
    with pytest.raises(ValueError, match="outside (node-local storage|the staged Gym source)"):
        _verify(local_runtime)


def test_runtime_requires_a_concrete_source_file_for_requested_modules(local_runtime) -> None:
    (local_runtime.gym / "fixture_rollout_probe/__init__.py").unlink()
    with pytest.raises(ValueError, match="module has no source file"):
        _verify(local_runtime)


def test_runtime_rejects_shared_cwd_and_component_root(local_runtime, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)
    monkeypatch.chdir(local_runtime.gym)
    local_runtime.venvs.rmdir()
    local_runtime.venvs.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


@pytest.mark.parametrize("asset", ["bin/uv", "bin/apptainer", "assets/agent.sif"])
def test_runtime_rejects_execution_asset_symlinks_and_missing_files(local_runtime, tmp_path, asset) -> None:
    local = local_runtime.root / asset
    local.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _verify(local_runtime)
    local.symlink_to(_write(tmp_path / "shared/asset", executable=True))
    with pytest.raises(ValueError, match="outside node-local storage"):
        _verify(local_runtime)


def test_runtime_requires_path_to_select_the_staged_uv(local_runtime, monkeypatch) -> None:
    alternate = _write(local_runtime.root / "alternate/uv", executable=True)
    monkeypatch.setenv("PATH", str(alternate.parent))
    with pytest.raises(ValueError, match="uv on PATH differs"):
        _verify(local_runtime)
