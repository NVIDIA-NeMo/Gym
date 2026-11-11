# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-scoped namespace requirements for native database readback."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).parents[1] / "benchmark/scripts/nrt/docker_enroot_shim.py"
spec = importlib.util.spec_from_file_location("sqlcmd_exec_shim", path)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


@pytest.mark.parametrize(
    "binary,needs_pid", [("/opt/mssql-tools18/bin/sqlcmd", True), ("sqlcmd", True), ("psql", False), ("php", False)]
)
def test_sqlcmd_joins_target_root_and_pid_without_changing_other_commands(monkeypatch, binary, needs_pid):
    commands = []
    monkeypatch.setattr(shim, "resolve_exec_pids", lambda _: (100, 101))
    monkeypatch.setattr(shim.os, "chdir", lambda _: None)
    monkeypatch.setattr(shim.Path, "read_bytes", lambda _: b"")
    monkeypatch.setattr(
        shim.subprocess, "run", lambda command, **kw: commands.append(command) or SimpleNamespace(returncode=0)
    )
    assert shim.docker_exec(["fixture", binary, "-Q", "SELECT 1"]) == 0
    cmd = commands[0]
    assert ("--pid" in cmd) == needs_pid
    assert ("--root" in cmd) == needs_pid
    assert ("--wd=/" in cmd) != needs_pid
    assert cmd[-3:] == [binary, "-Q", "SELECT 1"]


def test_context_failure_is_an_unavailable_grade_not_a_model_zero():
    path = Path(__file__).parents[1] / "benchmark/saas_bench/verify_runner.py"
    spec = importlib.util.spec_from_file_location("native_verifier_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parsed = runner._parse_verify_output(
        "WARNING: context load failed: Unable to load SQLCMD resource file(s)\n[FAIL] (3pt) product  (found 0)\nSCORE: 0.000 PASS: False (0/3)\n"
    )
    assert parsed["score"] == 0.0
    assert parsed["errors"] == ["WARNING: context load failed: Unable to load SQLCMD resource file(s)"]
    assert len(parsed["checks"]) == 1
