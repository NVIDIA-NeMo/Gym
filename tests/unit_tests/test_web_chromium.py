# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from nemo_gym.web import chromium


def test_wrong_cached_revision_and_shell_do_not_admit_full_browser(tmp_path, monkeypatch):
    expected = tmp_path / "chromium-123" / "chrome-linux" / "chrome"
    old = tmp_path / "chromium-122" / "chrome"
    old.parent.mkdir()
    old.write_text("old")
    shell = tmp_path / "chromium_headless_shell-123" / "chrome-headless-shell"
    shell.parent.mkdir()
    shell.write_text("shell")
    monkeypatch.setattr(chromium, "_chromium_executable", lambda: expected)
    run = Mock()
    monkeypatch.setattr(chromium.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="required Chromium is missing"):
        chromium.ensure_chromium(allow_install=False)
    run.assert_not_called()


def test_install_is_serialized_and_checks_selected_executable(tmp_path, monkeypatch):
    expected = tmp_path / "chromium-123" / "chrome-linux" / "chrome"
    monkeypatch.setattr(chromium, "_chromium_executable", lambda: expected)

    def install(*args, **kwargs):
        expected.parent.mkdir(parents=True)
        expected.write_text("test executable")
        expected.chmod(0o700)
        return SimpleNamespace(returncode=0, stderr="")

    run = Mock(side_effect=install)
    monkeypatch.setattr(chromium.subprocess, "run", run)
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda _: chromium.ensure_chromium(), range(8)))
    run.assert_called_once()
    assert run.call_args.args[0] == [chromium.sys.executable, "-m", "playwright", "install", "chromium"]


def test_install_error_retains_stderr_and_system_channel_does_not_install(tmp_path, monkeypatch):
    monkeypatch.setattr(chromium, "_chromium_executable", lambda: tmp_path / "chromium-123" / "chrome")
    run = Mock(return_value=SimpleNamespace(returncode=1, stderr="download failed: offline"))
    monkeypatch.setattr(chromium.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="download failed: offline"):
        chromium.ensure_chromium()
    run.reset_mock()
    chromium.ensure_chromium(channel="chrome")
    run.assert_not_called()


def test_install_lock_wait_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(chromium, "_chromium_executable", lambda: tmp_path / "chromium-123" / "chrome")
    monkeypatch.setattr(chromium.fcntl, "flock", Mock(side_effect=BlockingIOError()))
    monkeypatch.setattr(chromium.subprocess, "run", Mock())
    with pytest.raises(RuntimeError, match="timed out waiting for Chromium install lock"):
        chromium.ensure_chromium(timeout_seconds=0)
    chromium.subprocess.run.assert_not_called()


def test_existing_driver_selected_executable_needs_no_install(tmp_path, monkeypatch):
    executable = tmp_path / "chrome"
    executable.touch(mode=0o700)
    runtime = MagicMock()
    runtime.__enter__.return_value.chromium.executable_path = str(executable)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=lambda: runtime))
    monkeypatch.setattr(chromium.subprocess, "run", Mock())
    chromium.ensure_chromium(allow_install=False)
    chromium.subprocess.run.assert_not_called()
    runtime.__exit__.assert_called_once()


def test_unrecognized_cache_layout_and_read_only_cache_fail_with_context(tmp_path, monkeypatch):
    monkeypatch.setattr(chromium, "_chromium_executable", lambda: tmp_path / "unrecognized" / "chrome")
    with pytest.raises(RuntimeError, match="cannot identify Playwright Chromium cache"):
        chromium.ensure_chromium()
    cache_file = tmp_path / "not-a-directory"
    cache_file.touch()
    monkeypatch.setattr(chromium, "_chromium_executable", lambda: cache_file / "chromium-123" / "chrome")
    with pytest.raises(RuntimeError, match="cannot prepare Chromium.*preinstall"):
        chromium.ensure_chromium()
