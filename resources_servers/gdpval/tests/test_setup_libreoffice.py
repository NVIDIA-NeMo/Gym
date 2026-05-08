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


def _make_run(java_runs_pre: bool, java_runs_post: bool, install_succeeds: bool = True):
    """Build a `setup._run` side_effect that:
    - returns rc=0 for `java -version` based on `java_runs_pre` (before install)
      and `java_runs_post` (after the apt-install install command runs);
    - models apt-get update / install / libreoffice --version outcomes per
      `install_succeeds`.
    """
    state = {"installed": False}

    def _impl(cmd, **_kw):
        if cmd == ["java", "-version"]:
            ok = java_runs_post if state["installed"] else java_runs_pre
            return (0 if ok else 1), "", ""
        if cmd[:2] == ["apt-get", "update"]:
            return (0 if install_succeeds else 1), "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            if install_succeeds:
                state["installed"] = True
                return 0, "", ""
            return 100, "", "E: Unable to fetch"
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    return _impl, state


def test_returns_true_when_libreoffice_and_functional_java_present() -> None:
    """Early-exit fires only when libreoffice is on PATH AND `java -version` rc=0."""
    table = {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java"}
    run_impl, _ = _make_run(java_runs_pre=True, java_runs_post=True)
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
            assert setup.ensure_libreoffice() is True
    # Only `java -version` is invoked; no apt-get
    assert run_mock.call_count == 1
    assert run_mock.call_args_list[0][0][0] == ["java", "-version"]


def test_runs_apt_install_when_java_on_path_but_not_functional() -> None:
    """The deployment-image regression: `which("java")` returns a path, but
    `java -version` exits non-zero — apt-install must still run."""
    table = {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java", "apt-get": "/usr/bin/apt-get"}
    run_impl, _ = _make_run(java_runs_pre=False, java_runs_post=True)
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
                assert setup.ensure_libreoffice() is True

    cmds = [c.args[0] for c in run_mock.call_args_list]
    install = next(c for c in cmds if c[:3] == ["apt-get", "install", "-y"])
    assert "default-jre-headless" in install


def test_runs_apt_install_when_libreoffice_present_but_java_missing() -> None:
    """The other half: libreoffice is on PATH but java isn't (which → None)."""
    table = {"libreoffice": "/usr/bin/libreoffice", "java": None, "apt-get": "/usr/bin/apt-get"}
    run_impl, state = _make_run(java_runs_pre=False, java_runs_post=True)

    def _which(name: str) -> str | None:
        # java appears on PATH only after the apt install has run
        if name == "java" and state["installed"]:
            return "/usr/bin/java"
        return table.get(name)

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
                assert setup.ensure_libreoffice() is True

    cmds = [c.args[0] for c in run_mock.call_args_list]
    install = next(c for c in cmds if c[:3] == ["apt-get", "install", "-y"])
    assert "default-jre-headless" in install


def test_returns_false_when_install_does_not_provide_functional_java() -> None:
    """Install reports success but post-install `java -version` still fails."""
    table = {"libreoffice": "/usr/bin/libreoffice", "java": None, "apt-get": "/usr/bin/apt-get"}
    run_impl, _ = _make_run(java_runs_pre=False, java_runs_post=False)
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl):
                assert setup.ensure_libreoffice() is False


def test_returns_false_on_non_linux_when_missing() -> None:
    with patch.object(shutil, "which", return_value=None):
        with patch.object(sys, "platform", "darwin"):
            with patch.object(setup, "_run", return_value=(1, "", "")) as run_mock:
                assert setup.ensure_libreoffice() is False
    # Only `java -version` was probed; apt-get never invoked
    assert all(c.args[0] == ["java", "-version"] for c in run_mock.call_args_list)


def test_returns_false_when_apt_get_unavailable() -> None:
    """libreoffice + java probe both fail and apt-get isn't on PATH."""
    table: dict[str, str | None] = {}
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", return_value=(1, "", "")) as run_mock:
                assert setup.ensure_libreoffice() is False
    # No apt-get commands attempted (java -version may have been probed)
    for call in run_mock.call_args_list:
        assert call.args[0][0] != "apt-get"


def test_returns_false_when_apt_get_update_fails() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    run_impl, _ = _make_run(java_runs_pre=False, java_runs_post=False, install_succeeds=False)
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
                assert setup.ensure_libreoffice() is False
    cmds = [c.args[0] for c in run_mock.call_args_list]
    # apt-get update was attempted; install was NOT (we bailed after update failed)
    assert any(c[:2] == ["apt-get", "update"] for c in cmds)
    assert not any(c[:3] == ["apt-get", "install", "-y"] for c in cmds)


def test_returns_false_when_apt_install_fails() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}

    def _impl(cmd, **_kw):
        if cmd == ["java", "-version"]:
            return 1, "", ""
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            return 100, "", "E: Unable to fetch some archives"
        raise AssertionError(f"unexpected cmd: {cmd}")

    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_impl) as run_mock:
                assert setup.ensure_libreoffice() is False

    cmds = [c.args[0] for c in run_mock.call_args_list]
    assert any(c[:3] == ["apt-get", "install", "-y"] for c in cmds)


def test_returns_false_when_install_succeeds_but_libreoffice_still_missing() -> None:
    """Install command returns rc=0 but libreoffice still not on PATH afterwards."""
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    # libreoffice/java which always returns None (table never updated)
    run_impl, _ = _make_run(java_runs_pre=False, java_runs_post=False, install_succeeds=True)
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl):
                assert setup.ensure_libreoffice() is False


def test_full_success_path() -> None:
    """Initial state: nothing present. After apt install: libreoffice + working java."""
    table_state = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}

    def _which(name: str) -> str | None:
        return table_state.get(name)

    run_impl, state = _make_run(java_runs_pre=False, java_runs_post=True, install_succeeds=True)

    def _run(cmd, **kw):
        result = run_impl(cmd, **kw)
        if cmd[:3] == ["apt-get", "install", "-y"] and state["installed"]:
            table_state["libreoffice"] = "/usr/bin/libreoffice"
            table_state["java"] = "/usr/bin/java"
        return result

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_run):
                assert setup.ensure_libreoffice() is True


def test_handles_subprocess_timeout_gracefully() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}

    def _impl(cmd, **_kw):
        if cmd == ["java", "-version"]:
            return 1, "", ""
        raise subprocess.TimeoutExpired(cmd="apt-get", timeout=1)

    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_impl):
                assert setup.ensure_libreoffice() is False


def test_install_command_uses_no_install_recommends_and_includes_jre() -> None:
    table_state = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}

    def _which(name: str) -> str | None:
        return table_state.get(name)

    run_impl, state = _make_run(java_runs_pre=False, java_runs_post=True, install_succeeds=True)

    def _run(cmd, **kw):
        result = run_impl(cmd, **kw)
        if cmd[:3] == ["apt-get", "install", "-y"] and state["installed"]:
            table_state["libreoffice"] = "/usr/bin/libreoffice"
            table_state["java"] = "/usr/bin/java"
        return result

    captured: list[list[str]] = []

    def _capture(cmd, **kw):
        captured.append(cmd)
        return _run(cmd, **kw)

    with patch.object(shutil, "which", side_effect=_which):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_capture):
                assert setup.ensure_libreoffice() is True

    install_cmd = next(c for c in captured if c[:3] == ["apt-get", "install", "-y"])
    assert "--no-install-recommends" in install_cmd
    assert "libreoffice" in install_cmd
    assert "fonts-liberation" in install_cmd
    assert "default-jre-headless" in install_cmd


def test_java_runs_returns_false_when_java_not_on_path() -> None:
    with patch.object(shutil, "which", return_value=None):
        assert setup._java_runs() is False


def test_java_runs_returns_false_when_java_version_exits_nonzero() -> None:
    """Regression test: a non-functional `java` on PATH must NOT count as usable."""
    with patch.object(shutil, "which", return_value="/usr/bin/java"):
        with patch.object(setup, "_run", return_value=(1, "", "java: error while loading shared libraries")):
            assert setup._java_runs() is False


def test_java_runs_returns_true_when_java_version_exits_zero() -> None:
    with patch.object(shutil, "which", return_value="/usr/bin/java"):
        with patch.object(setup, "_run", return_value=(0, "", 'openjdk version "21.0.1"')):
            assert setup._java_runs() is True
