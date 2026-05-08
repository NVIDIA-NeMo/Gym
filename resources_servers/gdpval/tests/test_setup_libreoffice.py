# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
import sys
from unittest.mock import patch

from resources_servers.gdpval import setup_libreoffice as setup


def _which_from(table: dict[str, str | None]):
    """Return a side_effect for shutil.which that looks up names in ``table``."""

    def _impl(name: str) -> str | None:
        return table.get(name)

    return _impl


def test_returns_true_when_libreoffice_and_java_already_on_path() -> None:
    table = {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java"}
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(setup, "_run") as run_mock:
            assert setup.ensure_libreoffice() is True
    run_mock.assert_not_called()


def test_runs_apt_install_when_libreoffice_present_but_java_missing() -> None:
    """The deployment image bakes libreoffice in without a JRE; we must still apt-install."""
    # Pre-install: libreoffice yes, java no, apt-get yes. Post-install: both yes.
    table = {"libreoffice": "/usr/bin/libreoffice", "java": None, "apt-get": "/usr/bin/apt-get"}
    install_done = {"v": False}

    def _which(name: str) -> str | None:
        if install_done["v"] and name == "java":
            return "/usr/bin/java"
        return table.get(name)

    captured: list[list[str]] = []

    def _capture(cmd, **kw):
        captured.append(cmd)
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            install_done["v"] = True
            return 0, "", ""
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_capture):
                assert setup.ensure_libreoffice() is True

    install_cmd = next(c for c in captured if c[:3] == ["apt-get", "install", "-y"])
    assert "default-jre-headless" in install_cmd


def test_returns_false_when_libreoffice_present_but_java_missing_and_install_does_not_provide_java() -> None:
    table = {"libreoffice": "/usr/bin/libreoffice", "java": None, "apt-get": "/usr/bin/apt-get"}
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", return_value=(0, "", "")):
                assert setup.ensure_libreoffice() is False


def test_returns_false_on_non_linux_when_missing() -> None:
    with patch.object(shutil, "which", return_value=None):
        with patch.object(sys, "platform", "darwin"):
            with patch.object(setup, "_run") as run_mock:
                assert setup.ensure_libreoffice() is False
    run_mock.assert_not_called()


def test_returns_false_when_apt_get_unavailable() -> None:
    with patch.object(shutil, "which", side_effect=_which_from({})):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run") as run_mock:
                assert setup.ensure_libreoffice() is False
    run_mock.assert_not_called()


def test_returns_false_when_apt_get_update_fails() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", return_value=(1, "", "Network down")) as run_mock:
                assert setup.ensure_libreoffice() is False
    # Only the apt-get update call before bailing
    assert run_mock.call_count == 1
    assert run_mock.call_args_list[0][0][0][:2] == ["apt-get", "update"]


def test_returns_false_when_apt_install_fails() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    runs = iter([(0, "", ""), (100, "", "E: Unable to fetch some archives")])
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=lambda *a, **kw: next(runs)) as run_mock:
                assert setup.ensure_libreoffice() is False
    # update + install were both attempted
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[1][0][0][:3] == ["apt-get", "install", "-y"]


def test_returns_false_when_install_succeeds_but_binary_still_missing() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    runs = iter([(0, "", ""), (0, "", "")])
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=lambda *a, **kw: next(runs)):
                assert setup.ensure_libreoffice() is False


def test_full_success_path() -> None:
    """Initial state: nothing present. After apt install: both libreoffice and java present."""
    install_done = {"v": False}

    def _which(name: str) -> str | None:
        if not install_done["v"]:
            return "/usr/bin/apt-get" if name == "apt-get" else None
        return {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java", "apt-get": "/usr/bin/apt-get"}.get(name)

    def _run(cmd, **kw):
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            install_done["v"] = True
            return 0, "", ""
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_run):
                assert setup.ensure_libreoffice() is True


def test_handles_subprocess_timeout_gracefully() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}

    def _raise_timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="apt-get", timeout=1)

    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_raise_timeout):
                assert setup.ensure_libreoffice() is False


def test_install_command_uses_no_install_recommends() -> None:
    install_done = {"v": False}

    def _which(name: str) -> str | None:
        if not install_done["v"]:
            return "/usr/bin/apt-get" if name == "apt-get" else None
        return {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java", "apt-get": "/usr/bin/apt-get"}.get(name)

    captured: list[list[str]] = []

    def _capture(cmd, **kw):
        captured.append(cmd)
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            install_done["v"] = True
            return 0, "", ""
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_capture):
                assert setup.ensure_libreoffice() is True

    install_cmd = next(c for c in captured if c[:3] == ["apt-get", "install", "-y"])
    assert "--no-install-recommends" in install_cmd
    assert "libreoffice" in install_cmd
    assert "fonts-liberation" in install_cmd
    assert "default-jre-headless" in install_cmd
