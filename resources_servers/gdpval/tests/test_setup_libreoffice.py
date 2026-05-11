# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `setup_libreoffice.ensure_libreoffice`.

Behavior under test (v4): the function always runs `apt-get update` +
`apt-get install -y` regardless of what's already on PATH. Earlier
versions tried to short-circuit when libreoffice and/or `java` were
already present, but every version of that gate was satisfied by the
deployment image's broken JRE that libreoffice's `javaldx` helper
couldn't actually use. Apt-install is idempotent on already-installed
packages so the cost of always running it is small.
"""

import shutil
import subprocess
import sys
from unittest.mock import patch

from resources_servers.gdpval import setup_libreoffice as setup


def _which_from(table: dict[str, str | None]):
    def _impl(name: str) -> str | None:
        return table.get(name)

    return _impl


def _make_run(install_succeeds: bool = True, java_works_after_install: bool = True):
    """Side effect for `setup._run` modeling apt + libreoffice + java behaviour.

    Tracks an internal "installed" flag flipped by a successful apt install;
    `["java", "-version"]` then returns rc=0, modelling the post-install
    JRE coming online.
    """
    state = {"installed": False}

    def _impl(cmd, **_kw):
        if cmd == ["java", "-version"]:
            ok = state["installed"] and java_works_after_install
            return (0 if ok else 1), "", ""
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            if install_succeeds:
                state["installed"] = True
                return 0, "", ""
            return 100, "", "E: Unable to fetch"
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    return _impl, state


def _which_with_state(
    table_static: dict[str, str | None], state: dict, post_install_extra: dict[str, str | None] | None = None
):
    """Side effect for `shutil.which` that flips post-install paths once `state["installed"]` is True."""
    extra = post_install_extra or {}

    def _impl(name: str) -> str | None:
        if state.get("installed") and name in extra:
            return extra[name]
        return table_static.get(name)

    return _impl


def test_always_runs_apt_install_even_when_libreoffice_and_java_already_on_path() -> None:
    """The v4 contract: even when both binaries already resolve, run apt-install anyway.

    The deployment image had `libreoffice` + a broken-but-callable `java` baked
    in; previous versions skipped the install and the JRE that libreoffice's
    `javaldx` needs was never landed.
    """
    table = {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java", "apt-get": "/usr/bin/apt-get"}
    run_impl, _ = _make_run()
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
                assert setup.ensure_libreoffice() is True
    cmds = [c.args[0] for c in run_mock.call_args_list]
    assert any(c[:2] == ["apt-get", "update"] for c in cmds), "apt-get update must run"
    assert any(c[:3] == ["apt-get", "install", "-y"] for c in cmds), "apt-get install must run"


def test_runs_apt_install_when_libreoffice_present_but_java_missing() -> None:
    table = {"libreoffice": "/usr/bin/libreoffice", "java": None, "apt-get": "/usr/bin/apt-get"}
    run_impl, state = _make_run()
    with patch.object(shutil, "which", side_effect=_which_with_state(table, state, {"java": "/usr/bin/java"})):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
                assert setup.ensure_libreoffice() is True
    cmds = [c.args[0] for c in run_mock.call_args_list]
    install = next(c for c in cmds if c[:3] == ["apt-get", "install", "-y"])
    assert "default-jre-headless" in install


def test_runs_apt_install_when_java_on_path_but_not_functional() -> None:
    """Regression test for v3 → v4 transition: `java` resolves but `java -version` fails.

    The v3 fix gated on `_java_runs()` — but the deployment container had a
    callable-enough `java` to satisfy `java -version` while still failing
    `javaldx`. v4 doesn't even try to detect this; it just always installs.
    """
    table = {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java", "apt-get": "/usr/bin/apt-get"}
    run_impl, _ = _make_run()
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
                assert setup.ensure_libreoffice() is True
    cmds = [c.args[0] for c in run_mock.call_args_list]
    assert any(c[:3] == ["apt-get", "install", "-y"] for c in cmds)


def test_returns_false_on_non_linux() -> None:
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


def test_proceeds_to_install_even_when_apt_update_fails() -> None:
    """A stale apt index is still satisfiable by cached packages."""
    table = {"libreoffice": "/usr/bin/libreoffice", "java": None, "apt-get": "/usr/bin/apt-get"}
    state = {"installed": False}

    def _impl(cmd, **_kw):
        if cmd == ["java", "-version"]:
            return (0 if state["installed"] else 1), "", ""
        if cmd[:2] == ["apt-get", "update"]:
            return 1, "", "Network down"  # update fails
        if cmd[:3] == ["apt-get", "install", "-y"]:
            state["installed"] = True
            return 0, "", ""
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    with patch.object(shutil, "which", side_effect=_which_with_state(table, state, {"java": "/usr/bin/java"})):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_impl) as run_mock:
                assert setup.ensure_libreoffice() is True
    # update was attempted but failed; install was still attempted and succeeded
    cmds = [c.args[0] for c in run_mock.call_args_list]
    assert any(c[:2] == ["apt-get", "update"] for c in cmds)
    assert any(c[:3] == ["apt-get", "install", "-y"] for c in cmds)


def test_returns_false_when_apt_install_fails() -> None:
    table = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    run_impl, _ = _make_run(install_succeeds=False)
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl) as run_mock:
                assert setup.ensure_libreoffice() is False
    cmds = [c.args[0] for c in run_mock.call_args_list]
    assert any(c[:3] == ["apt-get", "install", "-y"] for c in cmds)


def test_returns_false_when_install_succeeds_but_libreoffice_still_missing() -> None:
    """Install reports rc=0 but the binary doesn't materialise (rare; broken package)."""
    table_static = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    state = {"installed": False}

    def _impl(cmd, **_kw):
        if cmd == ["java", "-version"]:
            return (0 if state["installed"] else 1), "", ""
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            state["installed"] = True
            return 0, "", ""
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    # `which` never returns a libreoffice path even after "install"
    with patch.object(shutil, "which", side_effect=_which_from(table_static)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_impl):
                assert setup.ensure_libreoffice() is False


def test_returns_false_when_install_succeeds_but_java_still_broken() -> None:
    """Install reports rc=0; libreoffice resolves; `java -version` keeps failing."""
    table = {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java", "apt-get": "/usr/bin/apt-get"}
    run_impl, _ = _make_run(java_works_after_install=False)
    with patch.object(shutil, "which", side_effect=_which_from(table)):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=run_impl):
                assert setup.ensure_libreoffice() is False


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
    table_static = {"libreoffice": None, "java": None, "apt-get": "/usr/bin/apt-get"}
    state = {"installed": False}

    def _impl(cmd, **_kw):
        if cmd == ["java", "-version"]:
            return (0 if state["installed"] else 1), "", ""
        if cmd[:2] == ["apt-get", "update"]:
            return 0, "", ""
        if cmd[:3] == ["apt-get", "install", "-y"]:
            state["installed"] = True
            return 0, "", ""
        if cmd == ["libreoffice", "--version"]:
            return 0, "LibreOffice 24.2", ""
        raise AssertionError(f"unexpected cmd: {cmd}")

    with patch.object(
        shutil,
        "which",
        side_effect=_which_with_state(
            table_static,
            state,
            {"libreoffice": "/usr/bin/libreoffice", "java": "/usr/bin/java"},
        ),
    ):
        with patch.object(sys, "platform", "linux"):
            with patch.object(setup, "_run", side_effect=_impl) as run_mock:
                assert setup.ensure_libreoffice() is True

    install_cmd = next(c.args[0] for c in run_mock.call_args_list if c.args[0][:3] == ["apt-get", "install", "-y"])
    assert "--no-install-recommends" in install_cmd
    assert "libreoffice" in install_cmd
    assert "fonts-liberation" in install_cmd
    assert "default-jre-headless" in install_cmd
    # libreoffice-java-common ships javaldx, the helper libreoffice uses
    # to find the JRE; without it the JRE-install does nothing useful.
    assert "libreoffice-java-common" in install_cmd


def test_java_runs_returns_false_when_java_not_on_path() -> None:
    with patch.object(shutil, "which", return_value=None):
        assert setup._java_runs() is False


def test_java_runs_returns_false_when_java_version_exits_nonzero() -> None:
    """A non-functional `java` on PATH must NOT count as usable."""
    with patch.object(shutil, "which", return_value="/usr/bin/java"):
        with patch.object(setup, "_run", return_value=(1, "", "java: error while loading shared libraries")):
            assert setup._java_runs() is False


def test_java_runs_returns_true_when_java_version_exits_zero() -> None:
    with patch.object(shutil, "which", return_value="/usr/bin/java"):
        with patch.object(setup, "_run", return_value=(0, "", 'openjdk version "21.0.1"')):
            assert setup._java_runs() is True
