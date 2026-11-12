# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-scoped namespace requirements for native database readback."""

import datetime
import importlib.util
import os
import subprocess
import sys
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
    assert ("--root" in cmd) == (needs_pid or binary == "php")
    assert ("--wd=/" in cmd) == (not needs_pid and binary != "php")
    if needs_pid or binary == "php":
        assert "--wd=/proc/101/root" in cmd
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


def test_inspect_reports_real_process_birth_not_query_time(monkeypatch, capsys):
    monkeypatch.setattr(shim, "resolve_exec_pids", lambda _: (os.getpid(), os.getpid()))
    assert shim.main(["inspect", "--format", "{{.State.StartedAt}}", "fixture"]) == 0
    first = capsys.readouterr().out.strip()
    started = datetime.datetime.fromisoformat(first.replace("Z", "+00:00"))
    assert started <= datetime.datetime.now(datetime.timezone.utc)
    assert shim.main(["inspect", "-f", "{{ .State.StartedAt }}", "fixture"]) == 0
    assert capsys.readouterr().out.strip() == first


def test_inspect_dead_service_does_not_invent_timestamp(monkeypatch, capsys):
    monkeypatch.setattr(shim, "resolve_exec_pids", lambda _: "no live Enroot app PID")
    assert shim.main(["inspect", "--format", "{{.State.StartedAt}}", "fixture"]) != 0
    assert capsys.readouterr().out == ""
    assert shim.main(["inspect", "--format", "{{.Config.Env}}", "fixture"]) != 0


def test_php_namespace_root_cwd_survives_deleted_host_cwd(tmp_path):
    # Execute the actual shim in a deleted cwd; the fake nsenter records the
    # complete command instead of requiring privileged namespaces in unit CI.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    nsenter = bindir / "nsenter"
    nsenter.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    nsenter.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    for suffix in ["enroot", "app"]:
        (state / f"fixture.{suffix}.pid").write_text(str(os.getpid()))
    cwd = tmp_path / "deleted"
    cwd.mkdir()
    # Remove only our temporary child directory after entering it.
    program = "import os,runpy,sys; script=sys.argv.pop(1); os.chdir(sys.argv.pop(1)); os.rmdir(os.getcwd()); sys.argv[0]=script; runpy.run_path(script,run_name='__main__')"
    env = os.environ | {"PATH": str(bindir) + ":" + os.environ["PATH"], "SAAS_ENROOT_STATE_DIR": str(state)}
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(path), str(cwd), "exec", "fixture", "php", "-r", "echo 1;"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--root" in result.stdout.splitlines()
    assert f"--wd=/proc/{os.getpid()}/root" in result.stdout.splitlines()
    assert "getcwd" not in result.stderr


@pytest.mark.parametrize("number", ["003", "011", "016", "031"])
def test_farmos_reads_register_shipped_drupal_collation_and_preserve_sql(number):
    import ast
    import base64
    import json

    source = path.parents[2] / f"tasks/multi-m/Agriculture/agriculture_{number}/verify.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "farmos_query"]
    commands = []
    namespace = {
        "base64": base64,
        "json": json,
        "FARMOS_DB": "/fixture.sqlite",
        "FARMOS_CONTAINER": "fixture",
        "docker_exec": lambda *args: commands.append(args) or (0, '[{"id":41}]', ""),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    sql = "SELECT id FROM log_field_data WHERE name='Évidence'"
    assert namespace["farmos_query"](sql) == [{"id": 41}]
    args = commands[0]
    php = args[args.index("-r") + 1]
    assert "require '/opt/drupal/vendor/autoload.php';" in php
    assert "'NOCASE_UTF8',['Drupal\\Component\\Utility\\Unicode','strcasecmp']" in php
    if "--" in args:
        assert args[-1] == sql
    else:
        assert base64.b64encode(sql.encode()).decode() in php
    assert args[args.index("-d") + 1] == "display_errors=stderr"
