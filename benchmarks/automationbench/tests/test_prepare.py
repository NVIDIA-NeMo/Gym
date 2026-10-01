# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib.util
import sys
from pathlib import Path


def _SENTINEL():  # noqa: N802 - stands in for the real load_environment
    raise AssertionError("not called")


def _stub_env_module():
    module = type(sys)("automationbench_env")
    module.load_environment = _SENTINEL
    return module


def _prepare_module():
    """Load prepare.py directly: it is a script, not an installed package."""
    path = Path(__file__).parents[1] / "prepare.py"
    spec = importlib.util.spec_from_file_location("automationbench_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_env_package_is_installed_when_the_harness_venv_lacks_it(monkeypatch):
    """Gym does not depend on automation-bench, so the venv running
    `gym eval prepare` normally cannot import it. Prepare used to exit telling
    the operator to install it by hand."""
    prepare = _prepare_module()
    calls = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    real_import = builtins.__import__
    attempts = []

    def flaky_import(name, *args, **kwargs):
        if name == "automationbench_env":
            attempts.append(name)
            if len(attempts) == 1:
                raise ImportError("no automationbench_env")
            return _stub_env_module()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", flaky_import)
    assert prepare._load_environment() is _SENTINEL

    assert len(calls) == 1
    # --python sys.executable: installing into whatever interpreter is on PATH
    # would put the package where this process cannot import it.
    assert calls[0][:5] == ["uv", "pip", "install", "--python", sys.executable]
    assert calls[0][5].endswith("automationbench")


def test_nothing_is_installed_when_the_package_is_already_importable(monkeypatch):
    prepare = _prepare_module()
    calls = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    real_import = builtins.__import__

    def ok_import(name, *args, **kwargs):
        if name == "automationbench_env":
            return _stub_env_module()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", ok_import)
    assert prepare._load_environment() is _SENTINEL

    assert calls == []
