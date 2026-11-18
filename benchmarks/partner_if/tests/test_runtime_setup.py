# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required search tooling is available before Resources accepts a recorded-context episode."""

import os
import shutil
import subprocess

import pytest

from resources_servers.partner_if.recorded_context import setup_ripgrep


def test_startup_adopts_installed_component_binary_with_empty_path(monkeypatch):
    monkeypatch.setenv("PATH", "")
    executable = setup_ripgrep.ensure_ripgrep()
    assert shutil.which("rg") == executable
    assert subprocess.check_output([executable, "--version"], text=True).startswith("ripgrep ")


def test_existing_binary_is_reused_without_changing_path(monkeypatch):
    previous = os.environ["PATH"]
    executable = shutil.which("rg")
    assert setup_ripgrep.ensure_ripgrep() == executable
    assert os.environ["PATH"] == previous


def test_missing_component_install_fails_before_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(setup_ripgrep.sysconfig, "get_path", lambda _: str(tmp_path))
    with pytest.raises(RuntimeError, match="uv sync --locked"):
        setup_ripgrep.ensure_ripgrep()
