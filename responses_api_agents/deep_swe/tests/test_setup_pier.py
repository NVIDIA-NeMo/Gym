# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from responses_api_agents.deep_swe import setup_pier
from responses_api_agents.deep_swe.tests import conftest as pier_test_config


def _valid_runtime(runtime: Path, gym_root: Path) -> None:
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "pier").write_text(f"#!/bin/sh\nprintf '{setup_pier.PIER_VERSION}\\n'\n")
    os.chmod(runtime / "bin" / "pier", 0o755)
    (runtime / "bin" / "python").write_text("python")
    site_packages = (
        runtime
        / "lib"
        / f"python{setup_pier.sys.version_info.major}.{setup_pier.sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    (site_packages / "nemo_gym_current_install.pth").write_text(str(gym_root.resolve()) + "\n")
    dist_info = site_packages / f"datacurve_pier-{setup_pier.PIER_VERSION}.dist-info"
    dist_info.mkdir()
    (dist_info / "direct_url.json").write_text(json.dumps(setup_pier._expected_pier_direct_url()))
    (runtime / "runtime.json").write_text(
        json.dumps(
            {
                "pier_version": setup_pier.PIER_VERSION,
                "pier_source_url": setup_pier.PIER_SOURCE_URL,
                "pier_source_commit": setup_pier.PIER_SOURCE_COMMIT,
                "pier_direct_url_sha256": setup_pier.pier_direct_url_sha256(),
                "modal_version": setup_pier.PIER_RUNTIME_MODAL_VERSION,
                "runtime_layout_version": setup_pier.PIER_RUNTIME_LAYOUT_VERSION,
                "constraints_sha256": setup_pier.pier_constraints_sha256(),
                "gym_root": str(gym_root.resolve()),
                "python": f"{setup_pier.sys.version_info.major}.{setup_pier.sys.version_info.minor}",
                "relocatable": True,
            }
        )
    )


def test_runtime_validation(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    gym_root = tmp_path / "gym"
    gym_root.mkdir()
    assert setup_pier._runtime_is_valid(runtime, gym_root) is False
    _valid_runtime(runtime, gym_root)
    assert setup_pier._runtime_is_valid(runtime, gym_root) is True
    direct_url_path = setup_pier._pier_direct_url_path(runtime)
    assert direct_url_path is not None
    direct_url_path.write_text(json.dumps({"url": "https://example.invalid/forged"}))
    assert setup_pier._runtime_is_valid(runtime, gym_root) is False
    direct_url_path.write_text(json.dumps(setup_pier._expected_pier_direct_url()))
    (runtime / "runtime.json").write_text("not json")
    assert setup_pier._runtime_is_valid(runtime, gym_root) is False


def test_run_reports_sanitized_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup_pier.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 3, "out", "err"),
    )
    with pytest.raises(RuntimeError, match="failed with code 3"):
        setup_pier._run(["uv", "pip", "install", "secret-package"])


def test_install_runtime_builds_versioned_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = tmp_path / "runtime"
    gym_root = tmp_path / "gym"
    gym_root.mkdir()
    monkeypatch.setattr(setup_pier.shutil, "which", lambda _: "/usr/bin/uv")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> None:
        commands.append(command)
        if command[1] == "venv":
            candidate = Path(command[-1])
            (candidate / "bin").mkdir(parents=True)
            (candidate / "bin" / "python").write_text("python")
        elif command[1:3] == ["pip", "install"] and "datacurve-pier" in " ".join(command):
            python = Path(command[command.index("--python") + 1])
            (python.parent / "pier").write_text(f"#!/bin/sh\nprintf '{setup_pier.PIER_VERSION}\\n'\n")
            os.chmod(python.parent / "pier", 0o755)
            site_packages = (
                python.parent.parent
                / "lib"
                / f"python{setup_pier.sys.version_info.major}.{setup_pier.sys.version_info.minor}"
                / "site-packages"
            )
            dist_info = site_packages / f"datacurve_pier-{setup_pier.PIER_VERSION}.dist-info"
            dist_info.mkdir(parents=True)
            (dist_info / "direct_url.json").write_text(json.dumps(setup_pier._expected_pier_direct_url()))

    monkeypatch.setattr(setup_pier, "_run", fake_run)
    pier = setup_pier._install_runtime(runtime, gym_root)

    assert pier == runtime / "bin" / "pier"
    assert setup_pier._runtime_is_valid(runtime, gym_root)
    assert any(command[1] == "venv" and "--relocatable" in command for command in commands)
    assert any("--constraint" in command and str(setup_pier.PIER_CONSTRAINTS_PATH) in command for command in commands)
    assert any(setup_pier.PIER_REQUIREMENT in command for command in commands)
    assert not any("--editable" in command for command in commands)
    site_packages = (
        runtime
        / "lib"
        / f"python{setup_pier.sys.version_info.major}.{setup_pier.sys.version_info.minor}"
        / "site-packages"
    )
    assert (site_packages / "nemo_gym_current_install.pth").read_text() == str(gym_root.resolve()) + "\n"
    assert setup_pier._install_runtime(runtime, gym_root) == pier


def test_install_runtime_refuses_mismatch_and_missing_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    with pytest.raises(RuntimeError, match="does not match"):
        setup_pier._install_runtime(runtime, tmp_path)

    monkeypatch.setattr(setup_pier.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires the 'uv'"):
        setup_pier._install_runtime(tmp_path / "other", tmp_path)


def test_runtime_validation_executes_pier_and_rejects_broken_script(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    gym_root = tmp_path / "gym"
    gym_root.mkdir()
    _valid_runtime(runtime, gym_root)
    assert setup_pier._runtime_is_valid(runtime, gym_root)

    pier = runtime / "bin" / "pier"
    pier.write_text("#!/missing/python\n")
    os.chmod(pier, 0o755)
    assert setup_pier._runtime_is_valid(runtime, gym_root) is False

    assert setup_pier._executable_is_valid(tmp_path / "missing") is False
    non_executable = tmp_path / "not-executable"
    non_executable.write_text("not executable")
    assert setup_pier._executable_is_valid(non_executable) is False
    wrong_version = tmp_path / "wrong-version"
    wrong_version.write_text("#!/bin/sh\nprintf '9.9.9\\n'\n")
    os.chmod(wrong_version, 0o755)
    assert setup_pier._executable_is_valid(wrong_version) is False

    forged = tmp_path / "forged"
    _valid_runtime(forged, gym_root)
    os.chmod(forged, 0o777)
    assert setup_pier._runtime_is_valid(forged, gym_root) is False

    symlink = tmp_path / "runtime-link"
    symlink.symlink_to(runtime, target_is_directory=True)
    assert setup_pier._runtime_is_valid(symlink, gym_root) is False


@pytest.mark.asyncio
async def test_ensure_runtime_delegates_to_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = tmp_path / "runtime" / "bin" / "pier"

    def install(runtime_dir: Path, gym_root: Path) -> Path:
        assert runtime_dir == tmp_path / "runtime"
        assert gym_root == tmp_path / "gym"
        return expected

    monkeypatch.setattr(setup_pier, "_install_runtime", install)
    assert await setup_pier.ensure_pier_runtime(tmp_path / "runtime", tmp_path / "gym") == expected


def test_pinned_test_environment_discards_python_and_pytest_injection(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    gym_root = tmp_path / "gym"
    overlay = tmp_path / "overlay"
    env = pier_test_config._pinned_test_environment(
        runtime=runtime,
        gym_root=gym_root,
        test_overlay=overlay,
        base_environment={
            "PATH": "/bin",
            "PYTHONPATH": "/poison",
            "PYTHONHOME": "/poison-home",
            "PYTHONSTARTUP": "/poison-startup",
            "PYTEST_ADDOPTS": "--pdb",
            "PYTEST_PLUGINS": "poison_plugin",
        },
    )

    assert env["PATH"] == "/bin"
    assert env["PYTHONPATH"] == str(overlay)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env[pier_test_config._PINNED_PREFIX_ENV] == str(runtime)
    assert env[pier_test_config._PINNED_GYM_ROOT_ENV] == str(gym_root)
    for removed in ("PYTHONHOME", "PYTHONSTARTUP", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        assert removed not in env
