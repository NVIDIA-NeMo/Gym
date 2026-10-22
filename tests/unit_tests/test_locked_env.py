# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import json
import shutil
import tomllib
from email.message import Message
from pathlib import Path
from unittest.mock import Mock

import pytest

from nemo_gym.cli import _locked_env


def _distribution(
    version: str, *, source: dict | None = None, extras: tuple[str, ...] = (), requires: tuple[str, ...] = ()
) -> Mock:
    metadata = Message()
    for extra in extras:
        metadata["Provides-Extra"] = extra
    return Mock(
        version=version,
        metadata=metadata,
        requires=requires,
        read_text=lambda _: json.dumps(source) if source else None,
    )


def test_parent_requirement_and_constraint_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = {"openai": _distribution("2.44.0")}

    def distribution(name: str) -> Mock:
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    _locked_env.check_parent_requirements(["openai==2.44.0"], ["ray==2.58.0"])
    with pytest.raises(RuntimeError, match="absent from the component lock"):
        _locked_env.check_parent_requirements(["missing==1"], [])
    with pytest.raises(RuntimeError, match="Parent requires openai==2.43.0"):
        _locked_env.check_parent_requirements(["openai==2.43.0"], [])
    installed["ray"] = _distribution("2.59.0")
    with pytest.raises(RuntimeError, match="Parent requires ray==2.58.0"):
        _locked_env.check_parent_requirements([], ["ray==2.58.0"])


def test_parent_source_and_extras_are_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    source = {"url": "https://github.com/example/plugin.git", "vcs_info": {"commit_id": "abc123"}}
    installed = {
        "plugin": _distribution("1", source=source, extras=("sdk",), requires=('transport==2; extra == "sdk"',)),
        "transport": _distribution("1"),
    }
    monkeypatch.setattr(importlib.metadata, "distribution", installed.__getitem__)
    with pytest.raises(RuntimeError, match="Parent requires transport==2"):
        _locked_env.check_parent_requirements(["plugin[sdk] @ git+https://github.com/example/plugin.git@abc123"], [])
    installed["transport"] = _distribution("2")
    _locked_env.check_parent_requirements(["plugin[sdk] @ git+https://github.com/example/plugin.git@abc123"], [])
    with pytest.raises(RuntimeError, match="Parent source"):
        _locked_env.check_parent_requirements(["plugin @ git+https://github.com/example/plugin.git@other"], [])


def test_sync_uses_locked_project_and_exact_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "resources_servers/sample"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname="sample"\nversion="0"\n')
    (project / "uv.lock").touch()
    venv = tmp_path / "external venv"
    run = Mock()
    monkeypatch.setattr(_locked_env.subprocess, "run", run)
    _locked_env.sync_environment(project, venv, "3.13", ["nemo-lens==1"], ["ray==2"], verbose=True)
    first, second = run.call_args_list
    assert first.args[0] == [
        "uv",
        "sync",
        "--locked",
        "--project",
        str(project),
        "--python",
        "3.13",
        "--no-dev",
        "--extra",
        "telemetry",
        "--verbose",
    ]
    assert first.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(venv)
    assert first.kwargs["check"] is True
    assert second.args[0][0] == str(venv / "bin/python")
    assert second.args[0][-4:] == ["--requirement", "nemo-lens==1", "--constraint", "ray==2"]
    assert second.kwargs["check"] is True


def test_sync_forwards_selected_component_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname="sample"\nversion="0"\n')
    (tmp_path / "uv.lock").touch()
    run = Mock()
    monkeypatch.setattr(_locked_env.subprocess, "run", run)
    _locked_env.sync_environment(tmp_path, tmp_path / ".venv", "3.13", [], [], extras=("lexmount",))
    assert run.call_args_list[0].args[0][-2:] == ["--extra", "lexmount"]
    assert "--inexact" not in run.call_args_list[0].args[0]


def test_sync_failure_never_runs_parent_check_or_unlocked_fallback(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname="sample"\nversion="0"\n')
    (tmp_path / "uv.lock").touch()
    run = Mock(side_effect=_locked_env.subprocess.CalledProcessError(1, ["uv", "sync"]))
    monkeypatch.setattr(_locked_env.subprocess, "run", run)
    with pytest.raises(_locked_env.subprocess.CalledProcessError):
        _locked_env.sync_environment(tmp_path, tmp_path / ".venv", "3.13", [], [])
    assert run.call_count == 1


@pytest.mark.parametrize("relative", ["resources_servers/sample", "benchmarks/indic/sample"])
@pytest.mark.parametrize("external", [False, True])
def test_wheel_project_restores_metadata_without_modifying_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str, external: bool
) -> None:
    import nemo_gym

    installed = tmp_path / "site-packages"
    package = installed / "nemo_gym"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    snapshot = package / "resources/locked_project"
    snapshot.mkdir(parents=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (snapshot / name).write_text(name)
    component_root = tmp_path / "plugins" if external else installed
    project = component_root / relative
    project.mkdir(parents=True)
    source = "/".join([".."] * len(Path(relative).parts))
    metadata = f'[tool.uv.sources]\n"nemo-gym" = {{path = "{source}", editable = true}}\n'
    (project / "pyproject.toml").write_text(metadata)
    for name in ("uv.lock", "app.py"):
        (project / name).write_text(name)
    monkeypatch.setattr(nemo_gym, "__file__", str(package / "__init__.py"))
    venv = tmp_path / "writable/sample/.venv"
    restored = _locked_env._wheel_project(project, venv)
    root = restored.parents[len(Path(relative).parts) - 1]
    assert (root / "pyproject.toml").read_text() == "pyproject.toml"
    assert (root / "nemo_gym").resolve() == package
    assert (restored / "uv.lock").read_bytes() == (project / "uv.lock").read_bytes()
    assert (restored / "app.py").resolve() == project / "app.py"
    assert not (installed / "pyproject.toml").exists()
    timestamps = {path: path.stat().st_mtime_ns for path in (root / "pyproject.toml", restored / "uv.lock")}
    assert _locked_env._wheel_project(project, venv) == restored
    assert timestamps == {path: path.stat().st_mtime_ns for path in timestamps}
    (project / "uv.lock").write_text("updated lock")
    _locked_env._wheel_project(project, venv)
    assert (restored / "uv.lock").read_text() == "updated lock"

    # Reusing the venv with a different installation must not retain links to the old one.
    replacement = tmp_path / "new-site-packages"
    shutil.copytree(installed, replacement)
    monkeypatch.setattr(nemo_gym, "__file__", str(replacement / "nemo_gym/__init__.py"))
    new_project = project if external else replacement / relative
    rebuilt = _locked_env._wheel_project(new_project, venv)
    assert (root / "nemo_gym").resolve() == replacement / "nemo_gym"
    assert (rebuilt / "app.py").resolve() == new_project / "app.py"
    assert (rebuilt / "pyproject.toml").read_text() == metadata


def test_real_uv_locked_sync_and_stale_lock_rejection(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    project = tmp_path / "project"
    project.mkdir()
    manifest = project / "pyproject.toml"
    manifest.write_text(
        '[project]\nname="locked-smoke"\nversion="0"\nrequires-python=">=3.13"\ndependencies=[]\n[tool.uv]\npackage=false\n[tool.uv.workspace]\nmembers=[]\n'
    )
    subprocess.run(["uv", "lock", "--project", str(project), "--python", sys.executable, "--offline"], check=True)
    lock_bytes = (project / "uv.lock").read_bytes()
    venv = tmp_path / "custom venv"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "wrong-environment"))
    subprocess.run(
        ["uv", "sync", "--locked", "--offline", "--project", str(project), "--python", sys.executable],
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(venv)},
        check=True,
    )
    assert (venv / "bin/python").exists()
    assert not (tmp_path / "wrong-environment").exists()
    assert not (project / ".venv").exists()
    assert (project / "uv.lock").read_bytes() == lock_bytes
    manifest.write_text(manifest.read_text().replace("dependencies=[]", 'dependencies=["nemo-gym-lock-missing==1"]'))
    result = subprocess.run(
        ["uv", "sync", "--locked", "--offline", "--project", str(project), "--python", sys.executable],
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(venv)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (project / "uv.lock").read_bytes() == lock_bytes


def test_bixbench_mcp_uses_its_lock_without_mutating_scientific_kernels():
    root = Path(__file__).resolve().parents[2] / "benchmarks/bixbench_hypothesis"
    dockerfile = (root / "Dockerfile").read_text()
    assert "UV_PROJECT_ENVIRONMENT=/opt/envs/bbh-mcp uv sync --locked" in dockerfile
    assert "COPY pkgs/bbh-mcp /opt/bbh-mcp" in dockerfile
    assert 'uv pip install -p "${KERNEL_ENV_PATH}/bin/python" /pkgs/bbh-mcp' not in dockerfile
    setup = (root / "harbor/task-template/steps/rollout/workdir/setup.sh").read_text()
    assert '"${BBH_MCP_PYTHON}" -m bbh_mcp.server' in setup
    assert 'nohup env PATH="$(dirname "${BBH_MCP_PYTHON}"):${PATH}"' in setup


@pytest.mark.parametrize("failed_stage", ["sync", "parent_check"])
def test_main_preserves_command_failure_without_prescribing_a_lock_update(tmp_path, monkeypatch, capsys, failed_stage):
    project = tmp_path / "component"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname="sample"\nversion="0"\n')
    (project / "uv.lock").write_text("unchanged lock")
    venv = tmp_path / "venv"
    monkeypatch.setattr(
        _locked_env.sys,
        "argv",
        ["locked-env", "--project", str(project), "--venv", str(venv), "--python", "3.13"],
    )
    error = _locked_env.subprocess.CalledProcessError(1, ["uv", "sync", "--locked"])
    run = Mock(side_effect=[error] if failed_stage == "sync" else [Mock(), error])
    monkeypatch.setattr(_locked_env.subprocess, "run", run)

    with pytest.raises(SystemExit) as result:
        _locked_env.main()

    assert result.value.code == 1
    diagnostic = capsys.readouterr().err
    assert "Locked environment command failed:" in diagnostic
    assert "dependency access, cache, build, or lock errors" in diagnostic
    assert "will not fall back to an unlocked install" in diagnostic
    assert "Update and commit" not in diagnostic
    assert run.call_count == (1 if failed_stage == "sync" else 2)
    assert (project / "uv.lock").read_text() == "unchanged lock"


def test_main_parent_contract_failure_signals_compatibility_resolution(monkeypatch, capsys):
    monkeypatch.setattr(_locked_env.sys, "argv", ["locked-env", "--check", "--requirement", "openai==0"])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: _distribution("1"))

    with pytest.raises(SystemExit) as result:
        _locked_env.main()

    assert result.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "Parent requires openai==0, but the component installed openai==1" in diagnostic
    assert "Parent mismatches use compatibility resolution" in diagnostic
    assert "Update and commit" not in diagnostic


@pytest.mark.parametrize("url", ["file:///opt/source%20tree", "https://example.invalid/source.tar.gz"])
def test_source_matching_honors_pep610_subdirectory(url):
    metadata = {"url": url.replace("%20", " "), "subdirectory": "python/sdk"}
    assert _locked_env._source_matches(url + "#subdirectory=python%2Fsdk", metadata)
    assert not _locked_env._source_matches(url + "#subdirectory=other", metadata)
    assert not _locked_env._source_matches(url, metadata)
    assert not _locked_env._source_matches(url + "#subdirectory=python/sdk", {"url": metadata["url"]})
    assert _locked_env._source_matches(url, {"url": metadata["url"]})


@pytest.mark.parametrize("scheme", ["file", "https"])
def test_parent_source_checks_accept_matching_non_git_subdirectory_and_reject_skew(monkeypatch, scheme):
    url = "file:///opt/plugin" if scheme == "file" else "https://example.invalid/plugin.tar.gz"
    metadata = {"url": url, "subdirectory": "python"}
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: _distribution("1", source=metadata))
    _locked_env.check_parent_requirements(["plugin @ " + url + "#subdirectory=python"], [])
    with pytest.raises(RuntimeError, match="Parent source"):
        _locked_env.check_parent_requirements(["plugin @ " + url + "#subdirectory=other"], [])


def test_git_source_subdirectory_matching_retains_exact_revision_checks():
    metadata = {
        "url": "https://github.com/example/plugin",
        "vcs_info": {"commit_id": "abc123"},
        "subdirectory": "python/sdk",
    }
    url = "git+https://github.com/example/plugin.git@abc123#subdirectory=python%2Fsdk"
    assert _locked_env._source_matches(url, metadata)
    assert not _locked_env._source_matches(url.replace("abc123", "other"), metadata)
    assert not _locked_env._source_matches(url.replace("python%2Fsdk", "other"), metadata)


def test_osworld_ray_lock_matches_parent_process_constraint() -> None:
    root = Path(__file__).resolve().parents[2]
    project = root / "responses_api_agents/osworld_agent"
    parent = tomllib.loads((root / "uv.lock").read_text())
    lock = tomllib.loads((project / "uv.lock").read_text())
    versions = {package["version"] for package in parent["package"] if package["name"] == "ray"}
    assert versions
    assert {package["version"] for package in lock["package"] if package["name"] == "ray"} == versions
    constraint = f"ray=={next(iter(versions))}"
    manifest = tomllib.loads((project / "pyproject.toml").read_text())
    config = tomllib.loads((project / "uv.toml").read_text())
    assert constraint in manifest["tool"]["uv"]["constraint-dependencies"]
    assert constraint in config["constraint-dependencies"]


@pytest.mark.parametrize("version", ["1.1.dev1", "1.1rc1"])
def test_parent_source_accepts_installed_prerelease_without_relaxing_bounds(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    source = {"url": "https://github.com/example/plugin.git", "vcs_info": {"commit_id": "abc123"}}
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: _distribution(version, source=source))
    _locked_env.check_parent_requirements(["plugin @ git+https://github.com/example/plugin.git@abc123"], [])
    _locked_env.check_parent_requirements(["plugin>=1,<2"], [])
    with pytest.raises(RuntimeError, match="Parent source"):
        _locked_env.check_parent_requirements(["plugin @ git+https://github.com/example/plugin.git@other"], [])
    with pytest.raises(RuntimeError, match="Parent requires plugin>=2"):
        _locked_env.check_parent_requirements(["plugin>=2"], [])


def test_sibling_venvs_keep_independent_wheel_projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_gym

    restored = []
    for name in ("first", "second"):
        package = tmp_path / name / "site-packages/nemo_gym"
        snapshot = package / "resources/locked_project"
        snapshot.mkdir(parents=True)
        (package / "__init__.py").write_text(name)
        for filename in ("pyproject.toml", "README.md", "LICENSE"):
            (snapshot / filename).write_text(name)
        project = tmp_path / name / "plugins/resources_servers/sample"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text('[tool.uv.sources]\n"nemo-gym" = {path = "../..", editable = true}\n')
        (project / "uv.lock").write_text(name)
        (project / "app.py").write_text(name)
        monkeypatch.setattr(nemo_gym, "__file__", str(package / "__init__.py"))
        destination = _locked_env._wheel_project(project, tmp_path / "cache" / name)
        restored.append((destination, package, name))

    for destination, package, name in restored:
        assert (destination / "uv.lock").read_text() == name
        assert (destination / "app.py").read_text() == name
        root = destination.parents[1]
        assert (root / "pyproject.toml").read_text() == name
        assert (root / "nemo_gym").resolve() == package
        assert (root / "nemo_gym/__init__.py").read_text() == name
    assert restored[0][0] != restored[1][0]


def test_environment_servers_retain_locked_local_gym_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    servers = sorted((root / "environment_servers").glob("*/app.py"))
    assert servers
    for app in servers:
        project = app.parent
        manifest = tomllib.loads((project / "pyproject.toml").read_text())
        lock = tomllib.loads((project / "uv.lock").read_text())
        source = manifest["tool"]["uv"]["sources"]["nemo-gym"]
        assert source["editable"]
        assert (project / source["path"]).resolve() == root
        assert not (project / "requirements.txt").exists()
        packages = {package["name"]: package for package in lock["package"]}
        component = packages[manifest["project"]["name"]]
        assert component["source"] == {"virtual": "."}
        gym = next(dependency for dependency in component["dependencies"] if dependency["name"] == "nemo-gym")
        assert "dev" in gym["extra"]
        if project.name == "harbor":
            assert "sandbox" in gym["extra"]
            assert {"tenacity", "opensandbox", "daytona", "e2b", "boto3"} <= packages.keys()
