# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool shells keep the task's home and XDG directories while Kilo's own state stays private."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from responses_api_agents.kilocode_agent import sandbox_runner


PLUGIN = Path(sandbox_runner.__file__).with_name("task_environment.mjs")
TASK_DIRECTORIES = ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the Kilo plugin")
@pytest.mark.parametrize("custom_xdg", [False, True])
def test_shell_environment_keeps_task_files_after_session_close(tmp_path: Path, custom_xdg: bool) -> None:
    session, workdir, home = tmp_path / "session", tmp_path / "task", tmp_path / "task-home"
    for directory in (session, workdir, home):
        directory.mkdir()
    shutil.copyfile(PLUGIN, session / PLUGIN.name)
    # The probe stands in for Kilo: it loads the plugin, applies its shell.env hook, and runs a tool shell.
    (session / "probe.mjs").write_text(
        """
import { TaskEnvironment } from "./task_environment.mjs";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
const hooks = await TaskEnvironment();
const output = {env: {UNRELATED: "retained"}};
await hooks["shell.env"]({}, output);
mkdirSync(process.env.XDG_DATA_HOME + "/kilo", {recursive: true});
writeFileSync(process.env.XDG_DATA_HOME + "/kilo/cli-state", "private");
execFileSync("/bin/sh", ["-c", 'mkdir -p "$XDG_CONFIG_HOME"; printf task > "$HOME/home-marker"; printf config > "$XDG_CONFIG_HOME/config-marker"'],
  {env: {...process.env, ...output.env}});
console.log(JSON.stringify({cliHome: process.env.HOME, cliConfig: process.env.XDG_CONFIG_HOME, tool: output.env}));
"""
    )
    (session / "input.json").write_text(json.dumps({"command": [shutil.which("node"), str(session / "probe.mjs")]}))
    # No task directories leak in from the agent-server host.
    env = {key: value for key, value in os.environ.items() if not key.startswith("XDG_")}
    env["HOME"] = str(home)
    if custom_xdg:
        env["XDG_CONFIG_HOME"] = str(tmp_path / "task-config")
    result = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, str(session), str(workdir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + (session / "stderr.log").read_text()
    assert (session / "task-environment.ready").read_text() == "ready\n"
    observed = json.loads((session / "stdout.jsonl").read_text())
    assert (observed["cliHome"], observed["cliConfig"]) == (str(session / "home"), str(session / "config"))
    assert (session / "data/kilo/cli-state").read_text() == "private"
    task_config = tmp_path / "task-config" if custom_xdg else home / ".config"
    assert observed["tool"]["UNRELATED"] == "retained"
    assert observed["tool"]["HOME"] == str(home)
    assert observed["tool"]["XDG_CONFIG_HOME"] == str(task_config)
    assert observed["tool"]["XDG_DATA_HOME"] == str(home / ".local/share")
    shutil.rmtree(session)
    assert (home / "home-marker").read_text() == "task"
    assert (task_config / "config-marker").read_text() == "config"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the Kilo plugin")
@pytest.mark.parametrize("failure", ["missing", "malformed", "missing_home", "relative_home", "invalid_xdg"])
def test_invalid_task_environment_cannot_acknowledge_hook_initialization(tmp_path: Path, failure: str) -> None:
    shutil.copyfile(PLUGIN, tmp_path / PLUGIN.name)
    env = {key: f"/task/{key.lower()}" for key in TASK_DIRECTORIES}
    if failure == "missing_home":
        del env["HOME"]
    elif failure == "relative_home":
        env["HOME"] = "relative"
    elif failure == "invalid_xdg":
        env["XDG_CONFIG_HOME"] = None
    if failure != "missing":
        (tmp_path / "task-env.json").write_text("{" if failure == "malformed" else json.dumps(env))
    result = subprocess.run(
        [
            shutil.which("node"),
            "--input-type=module",
            "-e",
            'import { TaskEnvironment } from "./task_environment.mjs"; await TaskEnvironment();',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert not (tmp_path / "task-environment.ready").exists()


def test_runner_removes_stale_task_environment_receipt_before_launch(tmp_path: Path) -> None:
    session, workdir = tmp_path / "session", tmp_path / "task"
    session.mkdir()
    workdir.mkdir()
    receipt = session / "task-environment.ready"
    receipt.write_text("ready\n")
    command = [sys.executable, "-c", "print('Kilo continued without loading the hook')"]
    (session / "input.json").write_text(json.dumps({"command": command}))
    result = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, str(session), str(workdir)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Kilo continued" in (session / "stdout.jsonl").read_text()
    assert not receipt.exists()


@pytest.mark.parametrize("task_environment", ["valid", "missing", "malformed", "plugin_error"])
def test_pinned_cli_reports_whether_required_plugin_initialized(tmp_path: Path, task_environment: str) -> None:
    binary = os.environ.get("GYM_TEST_KILO_BINARY")
    if binary is None:
        pytest.skip("Set GYM_TEST_KILO_BINARY to the pinned Kilo 7.4.15 executable")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
    assert version.returncode == 0 and version.stdout.strip() == "7.4.15"
    plugin = tmp_path / PLUGIN.name
    shutil.copyfile(PLUGIN, plugin)
    if task_environment == "plugin_error":
        plugin.write_text(
            plugin.read_text().replace(
                "const hooks = {", 'throw new Error("hook initialization failed");\n  const hooks = {'
            )
        )
    if task_environment != "missing":
        directories = {key: str(tmp_path / "task" / key.lower()) for key in TASK_DIRECTORIES}
        (tmp_path / "task-env.json").write_text("{" if task_environment == "malformed" else json.dumps(directories))
    (tmp_path / "kilo.json").write_text(json.dumps({"plugin": [plugin.as_uri()]}))
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "KILO_CONFIG": str(tmp_path / "kilo.json"),
        "KILO_DISABLE_PROJECT_CONFIG": "1",
        "KILO_NO_DAEMON": "1",
        "KILO_DB": ":memory:",
    }
    # The real plugin loader, without a model request. Kilo logs a plugin failure and still exits 0.
    result = subprocess.run(
        [binary, "debug", "agent", "build", "--print-logs"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    receipt = tmp_path / "task-environment.ready"
    if task_environment == "valid":
        assert receipt.read_text() == "ready\n"
    else:
        assert not receipt.exists()
