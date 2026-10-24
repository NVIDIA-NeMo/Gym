# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task shell directories survive removal of OpenCode's private CLI state."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from responses_api_agents.opencode_agent import sandbox_runner


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the OpenCode plugin")
@pytest.mark.parametrize("custom_xdg", [False, True])
def test_shell_environment_keeps_task_files_after_session_close(tmp_path: Path, custom_xdg: bool) -> None:
    session = tmp_path / "session"
    session.mkdir()
    home = tmp_path / "task-home"
    home.mkdir()
    plugin = session / "task_environment.mjs"
    shutil.copyfile(Path(sandbox_runner.__file__).with_name(plugin.name), plugin)
    # Use a subprocess-local environment: no task directories come from the agent-server host.
    env = {key: value for key, value in os.environ.items() if not key.startswith("XDG_")}
    env["HOME"] = str(home)
    if custom_xdg:
        env["XDG_CONFIG_HOME"] = str(tmp_path / "task-config")
    cli_env = {
        "HOME": str(session / "home"),
        "XDG_CONFIG_HOME": str(session / "config"),
        "XDG_DATA_HOME": str(session / "data"),
    }
    script = """
import { TaskEnvironment } from "./task_environment.mjs";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
const hooks = await TaskEnvironment();
const output = {env: {UNRELATED: "retained"}};
await hooks["shell.env"]({}, output);
// The CLI's private storage remains isolated; only shell env is changed.
mkdirSync(process.env.XDG_DATA_HOME + "/opencode", {recursive: true});
writeFileSync(process.env.XDG_DATA_HOME + "/opencode/cli-state", "private");
execFileSync("/bin/sh", ["-c", 'mkdir -p "$XDG_CONFIG_HOME"; printf task > "$HOME/home-marker"; printf config > "$XDG_CONFIG_HOME/config-marker"'],
  {env: {...process.env, ...output.env}});
console.log(JSON.stringify({cli: process.env.XDG_CONFIG_HOME, cliHome: process.env.HOME, tool: output.env}));
"""
    (session / "probe.mjs").write_text(script)
    params = {
        "directory": str(session),
        "command": [shutil.which("node"), str(session / "probe.mjs")],
        "cwd": str(session),
        "env": cli_env,
        "prompt": "",
    }
    input_path = session / "input.json"
    input_path.write_text(json.dumps(params))
    result = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, str(input_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (session / "task-environment.ready").read_text() == "ready\n"
    observed = json.loads((session / "stdout.jsonl").read_text())
    assert observed["cli"] == str(session / "config")
    assert observed["cliHome"] == str(session / "home")
    assert observed["tool"]["UNRELATED"] == "retained"
    assert observed["tool"]["HOME"] == str(home)
    assert (session / "data/opencode/cli-state").read_text() == "private"
    task_config = tmp_path / "task-config" if custom_xdg else home / ".config"
    assert observed["tool"]["XDG_CONFIG_HOME"] == str(task_config)
    assert observed["tool"]["XDG_DATA_HOME"] == str(home / ".local/share")
    shutil.rmtree(session)
    assert (home / "home-marker").read_text() == "task"
    assert (task_config / "config-marker").read_text() == "config"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the OpenCode plugin")
@pytest.mark.parametrize("failure", ["missing", "malformed", "missing_home", "relative_home", "invalid_xdg"])
def test_invalid_task_environment_cannot_acknowledge_hook_initialization(tmp_path: Path, failure: str) -> None:
    shutil.copyfile(Path(sandbox_runner.__file__).with_name("task_environment.mjs"), tmp_path / "task_environment.mjs")
    env = {
        "HOME": "/task/home",
        "XDG_CONFIG_HOME": "/task/config",
        "XDG_DATA_HOME": "/task/data",
        "XDG_STATE_HOME": "/task/state",
        "XDG_CACHE_HOME": "/task/cache",
    }
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
        timeout=10,
    )
    assert result.returncode != 0
    assert not (tmp_path / "task-environment.ready").exists()


def test_runner_removes_stale_task_environment_receipt_before_launch(tmp_path: Path) -> None:
    receipt = tmp_path / "task-environment.ready"
    receipt.write_text("ready\n")
    params = {
        "directory": str(tmp_path),
        "command": [sys.executable, "-c", "print('CLI continued without loading the hook')"],
        "cwd": str(tmp_path),
        "env": {},
        "prompt": "",
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(params))
    result = subprocess.run(
        [sys.executable, "-I", sandbox_runner.__file__, str(input_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "CLI continued" in (tmp_path / "stdout.jsonl").read_text()
    assert not receipt.exists()


@pytest.mark.parametrize("task_environment", ["valid", "missing", "malformed", "plugin_error"])
def test_pinned_cli_reports_whether_required_plugin_initialized(tmp_path: Path, task_environment: str) -> None:
    configured_binary = os.environ.get("GYM_TEST_OPENCODE_BINARY")
    binary = configured_binary or shutil.which("opencode")
    if binary is None:
        pytest.skip("Set GYM_TEST_OPENCODE_BINARY to the pinned OpenCode 1.17.11 executable")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    if configured_binary is None and (version.returncode != 0 or version.stdout.strip() != "1.17.11"):
        pytest.skip("PATH OpenCode is not 1.17.11; set GYM_TEST_OPENCODE_BINARY to the pinned executable")
    assert version.returncode == 0 and version.stdout.strip() == "1.17.11"
    plugin = tmp_path / "task_environment.mjs"
    shutil.copyfile(Path(sandbox_runner.__file__).with_name(plugin.name), plugin)
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "OPENCODE_CONFIG_CONTENT": json.dumps({"plugin": [plugin.as_uri()]}),
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
    }
    if task_environment != "missing":
        directories = {key: value for key, value in env.items() if key == "HOME" or key.startswith("XDG_")}
        (tmp_path / "task-env.json").write_text("{" if task_environment == "malformed" else json.dumps(directories))
    if task_environment == "plugin_error":
        plugin.write_text(
            plugin.read_text().replace(
                "const hooks = {", 'throw new Error("hook initialization failed");\n  const hooks = {'
            )
        )
    # This uses the real plugin loader without a model request. The pinned CLI
    # logs failures but returns successfully with the required hook absent.
    result = subprocess.run(
        [binary, "debug", "agent", "build", "--print-logs"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["name"] == "build"
    receipt = tmp_path / "task-environment.ready"
    if task_environment == "valid":
        assert receipt.read_text() == "ready\n"
        assert "failed to load plugin" not in result.stderr
    else:
        assert "failed to load plugin" in result.stderr
        assert not receipt.exists()
