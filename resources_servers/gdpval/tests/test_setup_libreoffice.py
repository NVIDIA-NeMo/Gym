# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
import sys
from unittest.mock import patch

from resources_servers.gdpval import setup_libreoffice as setup


def test_returns_true_when_libreoffice_already_on_path() -> None:
    with patch.object(shutil, "which", return_value="/usr/bin/libreoffice") as which:
        with patch.object(setup, "_run") as run_mock:
            assert setup.ensure_libreoffice() is True
    which.assert_called_once_with("libreoffice")
    run_mock.assert_not_called()


def test_returns_false_on_non_linux_when_missing() -> None:
    with patch.object(shutil, "which", return_value=None):
        with patch.object(sys, "platform", "darwin"):
            with patch.object(setup, "_run") as run_mock:
                assert setup.ensure_libreoffice() is False
    run_mock.assert_not_called()


def test_returns_false_when_apt_get_unavailable() -> None:
    def _which(name: str) -> str | None:
        return None

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run") as run_mock:
                assert setup.ensure_libreoffice() is False
    run_mock.assert_not_called()


def test_returns_false_when_apt_get_update_fails() -> None:
    which_calls = {"libreoffice": None, "apt-get": "/usr/bin/apt-get"}

    def _which(name: str) -> str | None:
        return which_calls[name]

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", return_value=(1, "", "Network down")) as run_mock:
                assert setup.ensure_libreoffice() is False
    # Only the apt-get update call before bailing
    assert run_mock.call_count == 1
    assert run_mock.call_args_list[0][0][0][:2] == ["apt-get", "update"]


def test_returns_false_when_apt_install_fails() -> None:
    which_seq = iter([None, "/usr/bin/apt-get"])

    def _which(name: str) -> str | None:
        return next(which_seq)

    runs = iter([(0, "", ""), (100, "", "E: Unable to fetch some archives")])
    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=lambda *a, **kw: next(runs)) as run_mock:
                assert setup.ensure_libreoffice() is False
    # update + install were both attempted
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[1][0][0][:3] == ["apt-get", "install", "-y"]


def test_returns_false_when_install_succeeds_but_binary_still_missing() -> None:
    # which returns: 1) initial check -> None, 2) apt-get -> /usr/bin, 3) post-install -> None
    which_seq = iter([None, "/usr/bin/apt-get", None])

    def _which(name: str) -> str | None:
        return next(which_seq)

    runs = iter([(0, "", ""), (0, "", "")])
    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=lambda *a, **kw: next(runs)):
                assert setup.ensure_libreoffice() is False


def test_full_success_path() -> None:
    # which: initial None, apt-get yes, post-install yes; --version returns 0
    which_seq = iter([None, "/usr/bin/apt-get", "/usr/bin/libreoffice"])

    def _which(name: str) -> str | None:
        return next(which_seq)

    runs = iter([(0, "", ""), (0, "", ""), (0, "LibreOffice 24.2", "")])
    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=lambda *a, **kw: next(runs)):
                assert setup.ensure_libreoffice() is True


def test_handles_subprocess_timeout_gracefully() -> None:
    which_seq = iter([None, "/usr/bin/apt-get"])

    def _which(name: str) -> str | None:
        return next(which_seq)

    def _raise_timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="apt-get", timeout=1)

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_raise_timeout):
                assert setup.ensure_libreoffice() is False


def test_install_command_uses_no_install_recommends() -> None:
    which_seq = iter([None, "/usr/bin/apt-get", "/usr/bin/libreoffice"])

    def _which(name: str) -> str | None:
        return next(which_seq)

    captured: list[list[str]] = []

    def _capture(cmd, **kw):
        captured.append(cmd)
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
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
