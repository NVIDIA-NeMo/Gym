# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME
from resources_servers.cube import bootstrap


_MINIMAL_CFG = """
cube_x_resources_server:
  resources_servers:
    cube:
      entrypoint: app.py
      environment: osworld
"""


def _fake_venv_python(tmp_path: Path) -> Path:
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    return py


def test_maybe_install_skips_without_config_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
    monkeypatch.delenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, raising=False)
    run = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    bootstrap.maybe_install_environment_extras()
    run.assert_not_called()


def test_maybe_install_runs_uv_when_stamp_stale(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    cube = tmp_path / "cube"
    (cube / "environments/osworld").mkdir(parents=True)
    req = cube / "environments/osworld/requirements.txt"
    req.write_text("some_pkg\n")
    py = _fake_venv_python(tmp_path)
    monkeypatch.setattr(bootstrap, "_cube_root", lambda: cube)
    monkeypatch.setattr(bootstrap.sys, "executable", str(py.resolve()))
    monkeypatch.setattr(bootstrap.sys, "prefix", str(py.parent.parent))
    monkeypatch.setattr(bootstrap.sys, "base_prefix", "/usr")
    monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, _MINIMAL_CFG)
    monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, "cube_x_resources_server")

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool) -> None:
        captured.append(cmd)
        assert check is True

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _x: "/usr/bin/uv")

    bootstrap.maybe_install_environment_extras()

    assert len(captured) == 1
    assert captured[0][:4] == ["/usr/bin/uv", "pip", "install", "--python"]
    assert captured[0][4] == str(py.resolve())
    assert captured[0][5:7] == ["-r", str(req.resolve())]

    stamp = py.parent.parent / bootstrap._STAMP_FILENAME
    assert stamp.is_file()


def test_maybe_install_skips_when_stamp_matches(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    cube = tmp_path / "cube"
    (cube / "environments/osworld").mkdir(parents=True)
    req = cube / "environments/osworld/requirements.txt"
    req.write_text("same\n")
    py = _fake_venv_python(tmp_path)
    monkeypatch.setattr(bootstrap, "_cube_root", lambda: cube)
    monkeypatch.setattr(bootstrap.sys, "executable", str(py.resolve()))
    monkeypatch.setattr(bootstrap.sys, "prefix", str(py.parent.parent))
    monkeypatch.setattr(bootstrap.sys, "base_prefix", "/usr")
    monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, _MINIMAL_CFG)
    monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, "cube_x_resources_server")

    stamp = py.parent.parent / bootstrap._STAMP_FILENAME
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(f"osworld\n{bootstrap._requirements_fingerprint(req)}\n")

    run = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    bootstrap.maybe_install_environment_extras()
    run.assert_not_called()


def test_maybe_install_skips_unsafe_environment_name(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    bad_cfg = """
cube_x_resources_server:
  resources_servers:
    cube:
      entrypoint: app.py
      environment: "../evil"
"""
    cube = tmp_path / "cube"
    (cube / "environments/osworld").mkdir(parents=True)
    (cube / "environments/osworld/requirements.txt").write_text("x\n")
    py = _fake_venv_python(tmp_path)
    monkeypatch.setattr(bootstrap, "_cube_root", lambda: cube)
    monkeypatch.setattr(bootstrap.sys, "executable", str(py.resolve()))
    monkeypatch.setenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, bad_cfg)
    monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, "cube_x_resources_server")

    run = MagicMock()
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    bootstrap.maybe_install_environment_extras()
    run.assert_not_called()
